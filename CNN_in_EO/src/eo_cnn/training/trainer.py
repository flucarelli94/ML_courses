"""
Training loops for classification and segmentation.

Design goals:
  - Explicit, readable training loops (no magic abstractions like pytorch-lightning)
  - Compatible with any PyTorch model, dataset, and optimizer
  - Tracks and plots loss + metrics in-notebook
  - Supports early stopping and checkpointing
  - FAST_MODE flag for CPU-only environments (reduced epochs/data)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    """Accumulates per-epoch metrics for later plotting."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_metric: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoch_times: list[float] = field(default_factory=list)

    def best_val_metric(self) -> float:
        return max(self.val_metric) if self.val_metric else 0.0

    def best_epoch(self) -> int:
        return int(torch.tensor(self.val_metric).argmax().item()) if self.val_metric else 0


class EarlyStopping:
    """
    Stop training when validation metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after last improvement
    min_delta : float
        Minimum change to qualify as improvement
    mode : str
        'max' (e.g. accuracy, IoU) or 'min' (e.g. loss)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.triggered = False

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            (value > self.best + self.min_delta) if self.mode == "max"
            else (value < self.best - self.min_delta)
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False


class ClassificationTrainer:
    """
    Training loop for image classification (Chapter 5-7).

    Usage
    -----
    trainer = ClassificationTrainer(model, optimizer, criterion, device="cuda")
    history = trainer.fit(train_loader, val_loader, epochs=30)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str | torch.device = "cpu",
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_path: Optional[str | Path] = None,
        early_stopping: Optional[EarlyStopping] = None,
        verbose: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.history = TrainingHistory()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
    ) -> TrainingHistory:
        """Run training loop for `epochs` epochs."""
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            val_loss, val_acc = self._run_epoch(val_loader, training=False)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            elapsed = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.train_metric.append(train_acc)
            self.history.val_metric.append(val_acc)
            self.history.lr.append(current_lr)
            self.history.epoch_times.append(elapsed)

            # Save best checkpoint
            if val_acc > best_val_acc and self.checkpoint_path is not None:
                best_val_acc = val_acc
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                }, self.checkpoint_path)

            if self.verbose:
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
                    f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} | "
                    f"LR: {current_lr:.2e} | {elapsed:.1f}s"
                )

            if self.early_stopping is not None and self.early_stopping.step(val_acc):
                if self.verbose:
                    print(f"Early stopping at epoch {epoch} (best val acc: {best_val_acc:.4f})")
                break

        return self.history

    def _run_epoch(
        self, loader: DataLoader, training: bool
    ) -> tuple[float, float]:
        """Run one epoch (train or eval) and return (avg_loss, accuracy)."""
        self.model.train(training)
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.set_grad_enabled(training):
            for batch in loader:
                images, labels = self._unpack_batch(batch)
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        return total_loss / total, correct / total

    @staticmethod
    def _unpack_batch(batch) -> tuple[Tensor, Tensor]:
        """Handle both (image, label) tuples and torchgeo dict batches."""
        if isinstance(batch, dict):
            return batch["image"], batch["label"]
        return batch[0], batch[1]


class SegmentationTrainer:
    """
    Training loop for semantic segmentation (Chapters 9-11).

    Differences from ClassificationTrainer:
      - Uses IoU (Jaccard index) as the primary metric instead of accuracy
      - Expects (image, mask) pairs where mask is (B, H, W) long tensor
      - Supports mixed-precision training via torch.amp
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str | torch.device = "cpu",
        num_classes: int = 7,
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_path: Optional[str | Path] = None,
        early_stopping: Optional[EarlyStopping] = None,
        use_amp: bool = True,
        verbose: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.early_stopping = early_stopping
        self.use_amp = use_amp and (device != "cpu")
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        self.verbose = verbose
        self.history = TrainingHistory()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
    ) -> TrainingHistory:
        best_miou = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_miou = self._run_epoch(train_loader, training=True)
            val_loss, val_miou = self._run_epoch(val_loader, training=False)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_miou)
                else:
                    self.scheduler.step()

            elapsed = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.train_metric.append(train_miou)
            self.history.val_metric.append(val_miou)
            self.history.lr.append(current_lr)
            self.history.epoch_times.append(elapsed)

            if val_miou > best_miou and self.checkpoint_path is not None:
                best_miou = val_miou
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_miou": val_miou,
                }, self.checkpoint_path)

            if self.verbose:
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train loss: {train_loss:.4f} mIoU: {train_miou:.4f} | "
                    f"Val loss: {val_loss:.4f} mIoU: {val_miou:.4f} | "
                    f"LR: {current_lr:.2e} | {elapsed:.1f}s"
                )

            if self.early_stopping is not None and self.early_stopping.step(val_miou):
                if self.verbose:
                    print(f"Early stopping at epoch {epoch} (best mIoU: {best_miou:.4f})")
                break

        return self.history

    def _run_epoch(
        self, loader: DataLoader, training: bool
    ) -> tuple[float, float]:
        self.model.train(training)
        total_loss = 0.0
        all_preds: list[Tensor] = []
        all_masks: list[Tensor] = []

        ctx = torch.amp.autocast(device_type=str(self.device).split(":")[0]) if self.use_amp else _null_ctx()

        with torch.set_grad_enabled(training):
            for batch in loader:
                images, masks = self._unpack_batch(batch)
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True).long()

                with ctx:
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_masks.append(masks.cpu())

        all_preds_t = torch.cat(all_preds, dim=0)
        all_masks_t = torch.cat(all_masks, dim=0)
        miou = _compute_miou_simple(all_preds_t, all_masks_t, self.num_classes)

        n = sum(len(b) for b in all_preds)
        return total_loss / max(n, 1), miou

    @staticmethod
    def _unpack_batch(batch) -> tuple[Tensor, Tensor]:
        if isinstance(batch, dict):
            return batch["image"], batch["mask"]
        return batch[0], batch[1]


# ─── Private helpers ──────────────────────────────────────────────────────────

def _compute_miou_simple(preds: Tensor, targets: Tensor, num_classes: int) -> float:
    """Efficient mean IoU computation over a batch."""
    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        tgt_cls = targets == cls
        intersection = (pred_cls & tgt_cls).sum().item()
        union = (pred_cls | tgt_cls).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return float(sum(ious) / len(ious)) if ious else 0.0


class _null_ctx:
    """No-op context manager for CPU fallback."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
