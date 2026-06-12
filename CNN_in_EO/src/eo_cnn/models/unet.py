"""
U-Net implementation from scratch (Chapter 9).

U-Net (Ronneberger et al., 2015) is the dominant architecture for
semantic segmentation in medical imaging and Earth Observation.

Key architectural features:
  - Encoder path: progressively halves spatial resolution, doubles channels
  - Bottleneck: captures global context
  - Decoder path: progressively restores spatial resolution
  - Skip connections: concatenate encoder features to decoder inputs
    (preserves fine spatial detail that would otherwise be lost)

Each block in this implementation is labelled to make the data flow
through the architecture as legible as possible for teaching.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d → BatchNorm → ReLU blocks.

    This is U-Net's fundamental feature extraction unit. Applying
    convolution twice at the same scale:
      - First conv: aggregates neighbourhood information
      - Second conv: refines the representation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Encoder step: MaxPool2d → DoubleConv.
    Halves spatial resolution, doubles channels.
    Each Down block increases the model's receptive field.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Decoder step: Upsample → DoubleConv.

    Restores spatial resolution via bilinear interpolation (preferred
    over transposed convolution for stability in segmentation tasks).
    The skip connection is concatenated here: this is what makes U-Net
    a U-Net rather than a plain encoder-decoder.

    in_channels = channels from decoder + channels from skip connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()

        if bilinear:
            # Bilinear upsampling: no learned parameters, computationally cheap
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Transposed convolution: learnable upsampling (more VRAM)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        x1: upsampled feature map from decoder
        x2: skip connection from corresponding encoder layer

        We must handle potential size mismatch (when input is not a
        power of 2) by padding x1 to match x2.
        """
        x1 = self.up(x1)
        # Spatial alignment: pad x1 if x2 is larger
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])
        # Concatenate along channel dimension (this IS the skip connection)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1×1 convolution: maps feature channels to class scores.
    Output shape: (B, num_classes, H, W) — one score per class per pixel.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation.

    Default configuration (n_channels=3, base_channels=64):

    Encoder:
      Input   (B, 3, H, W)
      enc1:   DoubleConv(3→64)        → (B, 64,  H,    W   )
      down1:  MaxPool+DoubleConv(64→128) → (B, 128, H/2,  W/2 )
      down2:  MaxPool+DoubleConv(128→256)→ (B, 256, H/4,  W/4 )
      down3:  MaxPool+DoubleConv(256→512)→ (B, 512, H/8,  W/8 )
      down4:  MaxPool+DoubleConv(512→1024)→(B,1024, H/16, W/16)

    Decoder (with skip connections from encoder):
      up1:  UpSample+DoubleConv(1024+512→512) → (B, 512, H/8, W/8)
      up2:  UpSample+DoubleConv(512+256→256)  → (B, 256, H/4, W/4)
      up3:  UpSample+DoubleConv(256+128→128)  → (B, 128, H/2, W/2)
      up4:  UpSample+DoubleConv(128+64→64)    → (B, 64,  H,   W  )
      out:  Conv1×1(64→num_classes)           → (B, num_classes, H, W)

    Total parameters with n_channels=3, base=64: ~31M

    Parameters
    ----------
    n_channels : int
        Number of input image channels (3 for RGB, 10+ for multispectral)
    n_classes : int
        Number of segmentation classes
    base_channels : int
        Number of filters in the first layer. All subsequent layers scale
        proportionally (×2 per down, ÷2 per up). Reduce to 32 for lighter
        models that fit on 6GB VRAM.
    bilinear : bool
        Use bilinear upsampling (True) or transposed convolution (False)
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 7,
        base_channels: int = 64,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        b = base_channels
        factor = 2 if bilinear else 1

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # ─── Encoder ─────────────────────────────────────────────────────
        self.inc = DoubleConv(n_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16 // factor)   # bottleneck

        # ─── Decoder ─────────────────────────────────────────────────────
        self.up1 = Up(b * 16, b * 8 // factor, bilinear)
        self.up2 = Up(b * 8, b * 4 // factor, bilinear)
        self.up3 = Up(b * 4, b * 2 // factor, bilinear)
        self.up4 = Up(b * 2, b, bilinear)
        self.outc = OutConv(b, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder (save feature maps for skip connections)
        x1 = self.inc(x)      # full resolution
        x2 = self.down1(x1)   # 1/2
        x3 = self.down2(x2)   # 1/4
        x4 = self.down3(x3)   # 1/8
        x5 = self.down4(x4)   # 1/16 (bottleneck)

        # Decoder (use skip connections)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)   # (B, n_classes, H, W)

    def encoder_features(self, x: Tensor) -> list[Tensor]:
        """Return all encoder feature maps (for skip connection visualization)."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]


def build_smp_unet(
    encoder_name: str = "resnet50",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 7,
) -> nn.Module:
    """
    Build a U-Net using segmentation-models-pytorch with a pretrained encoder.

    This is the production-grade approach: the encoder is a full ResNet/
    EfficientNet pretrained on ImageNet, giving a massive head-start over
    random initialization.

    Parameters
    ----------
    encoder_name : e.g. 'resnet50', 'efficientnet-b4', 'mobilenet_v2'
    encoder_weights : 'imagenet' or None
    in_channels : 3 for RGB, more for multispectral
    num_classes : number of segmentation classes
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation-models-pytorch required: pip install segmentation-models-pytorch"
        ) from exc

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # raw logits; apply softmax in loss or inference
    )
