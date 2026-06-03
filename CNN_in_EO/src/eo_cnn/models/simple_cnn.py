"""
Simple CNN implementations for teaching CNN fundamentals (Chapters 2-3).

These models are intentionally transparent — every architectural choice
is explicit and pedagogically motivated. They are NOT optimised for
performance; they are optimised for UNDERSTANDING.

Architecture progression:
  ConvBlock     → single conv + activation + pooling unit
  SimpleCNN     → 3-block deep classifier (the "hello world" of CNNs)
  DeepCNN       → 5-block version showing depth effects
  CNNWithBN     → adds BatchNorm to show its effect on training stability
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBlock(nn.Module):
    """
    A single convolutional block: Conv2d → BatchNorm → ReLU → MaxPool.

    This is the fundamental building block of most CNNs. Each component
    serves a distinct purpose:
      - Conv2d:    spatial feature extraction via learned kernels
      - BatchNorm: normalises activations → faster, more stable training
      - ReLU:      introduces non-linearity, sparse activations
      - MaxPool:   spatial downsampling, translation invariance

    Parameters
    ----------
    in_channels  : number of input feature maps (e.g. 3 for RGB)
    out_channels : number of output feature maps (filters to learn)
    kernel_size  : spatial extent of each filter (typically 3×3 or 5×5)
    use_bn       : whether to include BatchNorm (default True)
    use_pool     : whether to include MaxPool2d (default True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_bn: bool = True,
        use_pool: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # "same" padding: keeps spatial dims

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not use_bn,  # bias is redundant when BN is present
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    Three-block CNN for EuroSAT patch classification (64×64 input).

    Architecture:
      Input: (B, 3, 64, 64)
      Block 1: Conv(3→32)  + BN + ReLU + MaxPool  → (B, 32,  32, 32)
      Block 2: Conv(32→64) + BN + ReLU + MaxPool  → (B, 64,  16, 16)
      Block 3: Conv(64→128)+ BN + ReLU + MaxPool  → (B, 128,  8,  8)
      Flatten:                                     → (B, 128*8*8)
      FC1: 128*8*8 → 256 + ReLU + Dropout(0.5)
      FC2: 256 → num_classes

    ~1.8M parameters total.

    This architecture deliberately mirrors LeNet-5's design philosophy:
    alternating conv and pool layers, followed by fully-connected layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # 64 → 32
            ConvBlock(32, 64),            # 32 → 16
            ConvBlock(64, 128),           # 16 → 8
        )

        # After 3 maxpool operations on 64×64: spatial dims = 64 / 2^3 = 8
        feature_dim = 128 * 8 * 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.classifier(x)

    def feature_maps(self, x: Tensor) -> list[Tensor]:
        """Return intermediate feature maps for visualization (Chapter 2)."""
        maps = []
        for block in self.features:
            x = block(x)
            maps.append(x)
        return maps


class DeepCNN(nn.Module):
    """
    Five-block CNN to demonstrate the effect of depth vs width.
    Used in Chapter 3 ablation experiments.

    Architecture: 3→32→64→128→256→256 with MaxPool after each block.
    Input: (B, 3, 64, 64)  Output: (B, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),         # 64 → 32
            ConvBlock(32, 64),                  # 32 → 16
            ConvBlock(64, 128),                 # 16 → 8
            ConvBlock(128, 256, use_pool=False), # 8 → 8 (no pool)
            ConvBlock(256, 256),                # 8 → 4
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, 256, 1, 1)
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


class CNNWithGAP(nn.Module):
    """
    CNN with Global Average Pooling instead of fully-connected layers.
    Demonstrates: fewer parameters, better generalisation, CAM-compatible.

    Used in Chapter 3 to introduce the concept of class activation maps.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128, use_pool=False),
            nn.Conv2d(128, num_classes, kernel_size=1),  # 1×1 conv = channel projection
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.gap(x)
        return x.squeeze(-1).squeeze(-1)  # (B, num_classes)

    def cam(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Return (logits, CAMs) for Class Activation Map visualization.
        CAMs shape: (B, num_classes, H, W)
        """
        feature_map = self.features(x)           # (B, num_classes, H, W)
        logits = self.gap(feature_map).squeeze(-1).squeeze(-1)
        return logits, feature_map


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters. Useful for architecture comparisons."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def receptive_field_size(num_conv_layers: int, kernel_size: int = 3) -> int:
    """
    Compute theoretical receptive field of a stack of same-kernel conv layers
    (no dilation, no pooling). Each layer adds (kernel_size - 1) pixels.

    receptive_field = 1 + num_layers * (kernel_size - 1)
    """
    return 1 + num_conv_layers * (kernel_size - 1)
