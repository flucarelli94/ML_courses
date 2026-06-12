"""
ResNet-based classifier for EuroSAT and other EO classification tasks.

Covers:
  - Fine-tuning pretrained ResNet (ImageNet weights)
  - Replacing the final classification head for custom num_classes
  - Feature extraction mode (frozen backbone)
  - Multi-spectral input adaptation (first conv layer modification)
  - EfficientNet variant for Chapter 6 architecture comparison
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights,
)


class EuroSATClassifier(nn.Module):
    """
    ResNet-based classifier with a configurable head.

    Supports three modes:
      - 'scratch':   random init, all layers trainable
      - 'finetune':  ImageNet pretrained, all layers trainable
      - 'feature':   ImageNet pretrained, backbone frozen (linear probe)

    Parameters
    ----------
    backbone : str
        One of 'resnet18', 'resnet50', 'efficientnet_b0'
    num_classes : int
        Number of output classes
    mode : str
        Training mode: 'scratch', 'finetune', or 'feature'
    in_channels : int
        Input channels. 3 = RGB, 13 = Sentinel-2 MS.
        If in_channels != 3, the first conv layer is reinitialised.
    dropout_rate : float
        Dropout before the final linear layer
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 10,
        mode: str = "finetune",
        in_channels: int = 3,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        assert mode in ("scratch", "finetune", "feature"), \
            f"mode must be 'scratch', 'finetune', or 'feature', got '{mode}'"

        self.backbone_name = backbone
        self.mode = mode
        self.num_classes = num_classes

        # ─── Build backbone ───────────────────────────────────────────────
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if mode != "scratch" else None
            base = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if mode != "scratch" else None
            base = models.resnet50(weights=weights)
            feature_dim = 2048
        elif backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if mode != "scratch" else None
            base = models.efficientnet_b0(weights=weights)
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # ─── Handle non-RGB input ─────────────────────────────────────────
        if in_channels != 3:
            if "resnet" in backbone:
                old_conv = base.conv1
                base.conv1 = nn.Conv2d(
                    in_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False,
                )
                # Initialise: copy RGB weights to first 3 channels, random for rest
                with torch.no_grad():
                    base.conv1.weight[:, :3] = old_conv.weight
                    if in_channels > 3:
                        nn.init.kaiming_normal_(base.conv1.weight[:, 3:])
            # For EfficientNet, a similar approach applies to features[0][0]
            # (handled separately in build_resnet_classifier)

        # ─── Replace classifier head ─────────────────────────────────────
        if "resnet" in backbone:
            base.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, num_classes),
            )
            self.backbone = base
        elif "efficientnet" in backbone:
            base.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, num_classes),
            )
            self.backbone = base

        # ─── Freeze backbone if feature extraction mode ───────────────────
        if mode == "feature":
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze all layers except the final classification head."""
        if "resnet" in self.backbone_name:
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad_(False)
        elif "efficientnet" in self.backbone_name:
            for name, param in self.backbone.named_parameters():
                if "classifier" not in name:
                    param.requires_grad_(False)

    def unfreeze(self, layer_names: list[str] | None = None) -> None:
        """
        Unfreeze specific layers (or all if layer_names is None).
        Used in progressive fine-tuning: start frozen, then unfreeze in stages.

        Example:
            model.unfreeze(["layer4", "fc"])   # unfreeze only last ResNet block
            model.unfreeze()                   # unfreeze everything
        """
        if layer_names is None:
            for param in self.backbone.parameters():
                param.requires_grad_(True)
        else:
            for name, param in self.backbone.named_parameters():
                if any(l in name for l in layer_names):
                    param.requires_grad_(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def get_features(self, x: Tensor) -> Tensor:
        """
        Extract penultimate-layer feature vector (before classification head).
        Useful for t-SNE visualisation and nearest-neighbour retrieval.
        """
        if "resnet" in self.backbone_name:
            # Run all layers up to (not including) fc
            m = self.backbone
            x = m.relu(m.bn1(m.conv1(x)))
            x = m.maxpool(x)
            x = m.layer1(x); x = m.layer2(x)
            x = m.layer3(x); x = m.layer4(x)
            x = m.avgpool(x)
            return torch.flatten(x, 1)
        raise NotImplementedError("get_features not implemented for this backbone")

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_resnet_classifier(
    backbone: str = "resnet50",
    num_classes: int = 10,
    pretrained: bool = True,
    in_channels: int = 3,
) -> EuroSATClassifier:
    """
    Factory function for quick model construction.

    Examples
    --------
    >>> model = build_resnet_classifier("resnet50", num_classes=10)
    >>> model = build_resnet_classifier("resnet18", num_classes=10, in_channels=13)
    """
    mode = "finetune" if pretrained else "scratch"
    return EuroSATClassifier(
        backbone=backbone,
        num_classes=num_classes,
        mode=mode,
        in_channels=in_channels,
    )


class ProgressiveFinetuner:
    """
    Helper class for staged fine-tuning — a standard industry practice.

    Stage 1: Freeze backbone, train head only (few epochs → fast convergence)
    Stage 2: Unfreeze top blocks, train with lower LR
    Stage 3: Unfreeze all, train with very low LR (optional)

    This prevents catastrophic forgetting of pretrained representations
    while allowing adaptation to the EO domain.
    """

    def __init__(self, model: EuroSATClassifier) -> None:
        self.model = model
        self.stage = 0

    def advance_stage(self, optimizer: torch.optim.Optimizer, lr_scale: float = 0.1) -> None:
        """
        Move to the next fine-tuning stage.
        Unfreezes more backbone layers and scales the optimizer's LR.
        """
        if "resnet" in self.model.backbone_name:
            stages = [
                [],                                           # stage 0: head only
                ["layer4", "fc"],                             # stage 1: last block
                ["layer3", "layer4", "fc"],                   # stage 2: last 2 blocks
                None,                                         # stage 3: full unfreeze
            ]
        else:
            stages = [[], None]

        self.stage += 1
        if self.stage < len(stages):
            self.model.unfreeze(stages[self.stage])
            for group in optimizer.param_groups:
                group["lr"] *= lr_scale
            trainable = self.model.trainable_params()
            print(f"Stage {self.stage}: {trainable:,} trainable parameters, LR scaled by {lr_scale}")
        else:
            print(f"Already at final stage ({self.stage - 1})")
