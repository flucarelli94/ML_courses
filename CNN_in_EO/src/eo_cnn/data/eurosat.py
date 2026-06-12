"""
EuroSAT dataset utilities.

EuroSAT is a land-use / land-cover classification dataset derived from
Sentinel-2 imagery. It contains 27,000 geo-referenced image patches
(64×64 pixels) across 10 classes in both RGB and 13-band MS variants.

Dataset paper: Helber et al. (2019) — EuroSAT: A Novel Dataset and
Deep Learning Benchmark for Land Use and Land Cover Classification.
DOI: 10.1109/JSTARS.2019.2918242

Access: https://github.com/phelber/EuroSAT
        Also available via torchgeo (auto-download).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# EuroSAT class names and their class IDs
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

EUROSAT_CLASS_TO_IDX = {name: idx for idx, name in enumerate(EUROSAT_CLASSES)}

# Per-channel mean and std for EuroSAT RGB (ImageNet-normalised variant)
# These values are computed over the full EuroSAT RGB training set.
EUROSAT_RGB_MEAN = [0.3444, 0.3803, 0.4078]  # R, G, B
EUROSAT_RGB_STD = [0.2025, 0.1368, 0.1156]

# Per-channel stats for EuroSAT MS (13 bands, Sentinel-2 ordering)
# Band order: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12
EUROSAT_MS_MEAN = [
    1354.4, 1118.2, 1042.9, 947.6, 1199.5, 2003.0,
    2374.0, 2301.2, 732.1, 12.1, 1819.0, 1118.2, 2655.6,
]
EUROSAT_MS_STD = [
    245.7, 333.0, 395.1, 593.8, 566.4, 861.5,
    1086.0, 1117.9, 404.9, 4.8, 1002.6, 761.3, 1231.2,
]


class EuroSATDataModule:
    """
    Convenience wrapper that loads EuroSAT via torchgeo and returns
    DataLoaders ready for training.

    Usage
    -----
    dm = EuroSATDataModule(root="./data", batch_size=64)
    train_loader, val_loader, test_loader = dm.get_loaders()
    """

    def __init__(
        self,
        root: str | Path = "./data/eurosat",
        batch_size: int = 64,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        use_ms: bool = False,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.use_ms = use_ms
        self.seed = seed

        # Default transforms if none provided
        if train_transform is None:
            train_transform = _default_train_transform(use_ms)
        if val_transform is None:
            val_transform = _default_val_transform(use_ms)

        self.train_transform = train_transform
        self.val_transform = val_transform

    def get_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Download (if needed) and return train/val/test DataLoaders."""
        try:
            from torchgeo.datasets import EuroSAT, EuroSATSpatial
        except ImportError as exc:
            raise ImportError(
                "torchgeo is required. Install with: pip install torchgeo"
            ) from exc

        # torchgeo's EuroSAT uses predefined splits
        train_ds = EuroSAT(root=self.root, split="train", transforms=self.train_transform, download=True)
        val_ds = EuroSAT(root=self.root, split="val", transforms=self.val_transform, download=True)
        test_ds = EuroSAT(root=self.root, split="test", transforms=self.val_transform, download=True)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        return train_loader, val_loader, test_loader


def get_eurosat_splits(
    root: str | Path = "./data/eurosat",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Load EuroSAT and return (train_dataset, val_dataset, test_dataset) using
    random splits from the full dataset. Useful when you want full control
    over transforms and DataLoader construction.
    """
    try:
        from torchgeo.datasets import EuroSAT
    except ImportError as exc:
        raise ImportError("torchgeo is required.") from exc

    full_ds = EuroSAT(root=root, download=True)
    n = len(full_ds)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    return random_split(full_ds, [n_train, n_val, n_test], generator=generator)


# ─── Private helpers ──────────────────────────────────────────────────────────

def _default_train_transform(use_ms: bool) -> Callable:
    mean = EUROSAT_MS_MEAN if use_ms else EUROSAT_RGB_MEAN
    std = EUROSAT_MS_STD if use_ms else EUROSAT_RGB_STD
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1) if not use_ms else _identity,
        transforms.ToTensor() if not use_ms else _to_float_tensor,
        transforms.Normalize(mean=mean, std=std),
    ])


def _default_val_transform(use_ms: bool) -> Callable:
    mean = EUROSAT_MS_MEAN if use_ms else EUROSAT_RGB_MEAN
    std = EUROSAT_MS_STD if use_ms else EUROSAT_RGB_STD
    return transforms.Compose([
        transforms.ToTensor() if not use_ms else _to_float_tensor,
        transforms.Normalize(mean=mean, std=std),
    ])


def _identity(x):
    return x


def _to_float_tensor(x: np.ndarray) -> torch.Tensor:
    """Convert HxWxC numpy array (uint16 Sentinel-2) to CxHxW float32 tensor."""
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if t.ndim == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3:
        t = t.permute(2, 0, 1)
    return t
