"""
Augmentation pipelines for Earth Observation imagery using albumentations.

EO-specific augmentation considerations:
  1. Geometric flips/rotations are label-safe (land cover is rotation-invariant)
  2. Color jitter must be applied PER BAND (not as RGB triplet transform)
  3. Brightness/contrast shifts simulate different atmospheric conditions
  4. Elastic distortions simulate subtle terrain warping
  5. Cutout / GridDropout simulates cloud shadows and sensor artefacts
  6. NO horizontal-only assumptions — satellite imagery has no canonical "up"
"""

from __future__ import annotations

from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: int = 64,
    mean: list[float] = (0.485, 0.456, 0.406),
    std: list[float] = (0.229, 0.224, 0.225),
    additional_targets: Optional[dict] = None,
) -> A.Compose:
    """
    Augmentation pipeline for EuroSAT classification training.
    Input: H×W×C numpy uint8 image (0-255 RGB).
    Output: C×H×W float32 tensor.
    """
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.CoarseDropout(
                max_holes=4, max_height=8, max_width=8,
                fill_value=0, p=0.2,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        additional_targets=additional_targets or {},
    )


def get_val_transforms(
    image_size: int = 64,
    mean: list[float] = (0.485, 0.456, 0.406),
    std: list[float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Validation/test transforms — normalize only, no augmentation.
    """
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_seg_train_transforms(
    image_size: int = 512,
    mean: list[float] = (0.485, 0.456, 0.406),
    std: list[float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Augmentation pipeline for semantic segmentation training.
    Applies the SAME geometric transform to both image and mask.

    Input: H×W×C numpy uint8 image + H×W numpy uint8 mask.
    Output: C×H×W float32 image tensor + H×W long tensor.

    Usage with albumentations:
        result = transform(image=img_np, mask=mask_np)
        image = result["image"]   # C×H×W float32 tensor
        mask  = result["mask"]    # H×W int64 tensor
    """
    return A.Compose(
        [
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                border_mode=0, p=0.4,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=50, sigma=10, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                ],
                p=0.2,
            ),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                fill_value=0, mask_fill_value=0, p=0.3,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_seg_val_transforms(
    image_size: int = 512,
    mean: list[float] = (0.485, 0.456, 0.406),
    std: list[float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Validation transforms for segmentation — normalize + center crop only.
    """
    return A.Compose([
        A.CenterCrop(height=image_size, width=image_size, p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_multispectral_seg_transforms(
    image_size: int = 512,
    n_bands: int = 10,
    band_means: Optional[list[float]] = None,
    band_stds: Optional[list[float]] = None,
) -> A.Compose:
    """
    Segmentation transforms for multispectral (non-RGB) imagery.
    Geometric augmentations only (no color jitter — irrelevant for MS).

    Note: albumentations ColorJitter only works on 3-channel images.
    For MS imagery, normalize manually before calling this pipeline.
    """
    return A.Compose(
        [
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                border_mode=0, p=0.4,
            ),
            A.GaussNoise(var_limit=(0.01, 0.05), p=0.2),
            A.CoarseDropout(
                max_holes=6, max_height=32, max_width=32,
                fill_value=0, mask_fill_value=0, p=0.3,
            ),
            ToTensorV2(),
        ]
    )
