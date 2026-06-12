"""
Evaluation metrics for EO segmentation and classification.

Metrics covered:
  - Per-class IoU (Intersection over Union / Jaccard index)
  - mIoU (mean IoU over all classes)
  - Pixel accuracy
  - Per-class precision, recall, F1
  - Confusion matrix (count and normalised)
  - Dice coefficient (equivalent to F1 at pixel level)

Mathematical definitions:
  IoU_c = |pred_c ∩ true_c| / |pred_c ∪ true_c|
  mIoU  = (1/C) * Σ IoU_c
  Dice_c = 2|pred_c ∩ true_c| / (|pred_c| + |true_c|)
  Dice ≡ F1 score at pixel level
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor


def compute_iou(
    preds: Tensor,
    targets: Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> np.ndarray:
    """
    Compute per-class IoU.

    Parameters
    ----------
    preds : (N, H, W) long tensor — predicted class indices
    targets : (N, H, W) long tensor — ground truth class indices
    num_classes : int
    ignore_index : class index to exclude from computation (-1 to disable)

    Returns
    -------
    iou : (num_classes,) float32 array — NaN for absent classes
    """
    iou = np.full(num_classes, np.nan, dtype=np.float32)
    preds_np = preds.cpu().numpy().ravel()
    targets_np = targets.cpu().numpy().ravel()

    valid = targets_np != ignore_index
    preds_np = preds_np[valid]
    targets_np = targets_np[valid]

    for cls in range(num_classes):
        pred_c = preds_np == cls
        true_c = targets_np == cls
        intersection = np.logical_and(pred_c, true_c).sum()
        union = np.logical_or(pred_c, true_c).sum()
        if union > 0:
            iou[cls] = intersection / union
    return iou


def compute_miou(
    preds: Tensor,
    targets: Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> float:
    """
    Mean IoU over all classes with at least one ground truth pixel.
    NaN classes (absent in GT) are excluded from the mean.
    """
    iou = compute_iou(preds, targets, num_classes, ignore_index)
    valid = ~np.isnan(iou)
    return float(iou[valid].mean()) if valid.any() else 0.0


def compute_dice(
    preds: Tensor,
    targets: Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> np.ndarray:
    """
    Per-class Dice coefficient (= F1 score at pixel level).
    Dice = 2*IoU / (1 + IoU)
    """
    iou = compute_iou(preds, targets, num_classes, ignore_index)
    return np.where(np.isnan(iou), np.nan, 2 * iou / (1 + iou))


def compute_confusion_matrix(
    preds: Tensor,
    targets: Tensor,
    num_classes: int,
    ignore_index: int = -1,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute (num_classes × num_classes) confusion matrix.

    Entry [i, j] = number of pixels where true class = i, predicted class = j.

    Parameters
    ----------
    normalize : if True, normalize each row to sum to 1 (recall-based)
    """
    preds_np = preds.cpu().numpy().ravel().astype(np.int64)
    targets_np = targets.cpu().numpy().ravel().astype(np.int64)

    if ignore_index >= 0:
        valid = targets_np != ignore_index
        preds_np = preds_np[valid]
        targets_np = targets_np[valid]

    # Flatten and bin into confusion matrix
    idx = targets_np * num_classes + preds_np
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    cm = cm.reshape(num_classes, num_classes)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sums > 0, cm / row_sums, 0).astype(np.float32)

    return cm


def pixel_accuracy(
    preds: Tensor,
    targets: Tensor,
    ignore_index: int = -1,
) -> float:
    """Overall pixel accuracy: fraction of correctly classified pixels."""
    preds_np = preds.cpu().numpy().ravel()
    targets_np = targets.cpu().numpy().ravel()
    if ignore_index >= 0:
        valid = targets_np != ignore_index
        preds_np = preds_np[valid]
        targets_np = targets_np[valid]
    return float((preds_np == targets_np).mean())


def classification_report_eo(
    preds: Tensor,
    targets: Tensor,
    class_names: list[str],
    ignore_index: int = -1,
) -> str:
    """
    Human-readable per-class metrics report.

    Returns a formatted string similar to sklearn's classification_report,
    but operating on pixel-level predictions (not sample-level).

    Example output:
        Class              IoU    Dice    Precision  Recall
        ─────────────────────────────────────────────────────
        Urban            0.612   0.759      0.78      0.74
        Agriculture      0.741   0.851      0.88      0.83
        Forest           0.834   0.909      0.91      0.90
        ...
        ─────────────────────────────────────────────────────
        mIoU: 0.653
    """
    num_classes = len(class_names)
    iou = compute_iou(preds, targets, num_classes, ignore_index)
    dice = compute_dice(preds, targets, num_classes, ignore_index)
    cm = compute_confusion_matrix(preds, targets, num_classes, ignore_index)

    lines = [
        f"\n{'Class':<22} {'IoU':>6} {'Dice':>6} {'Prec':>6} {'Rec':>6}",
        "─" * 50,
    ]

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        iou_str = f"{iou[i]:.3f}" if not np.isnan(iou[i]) else "  N/A"
        dice_str = f"{dice[i]:.3f}" if not np.isnan(dice[i]) else "  N/A"
        lines.append(f"{name:<22} {iou_str:>6} {dice_str:>6} {prec:>6.3f} {rec:>6.3f}")

    lines.append("─" * 50)
    valid_iou = iou[~np.isnan(iou)]
    miou = float(valid_iou.mean()) if len(valid_iou) > 0 else 0.0
    pa = pixel_accuracy(preds, targets, ignore_index)
    lines.append(f"{'mIoU':>22}: {miou:.4f}")
    lines.append(f"{'Pixel Accuracy':>22}: {pa:.4f}")

    return "\n".join(lines)
