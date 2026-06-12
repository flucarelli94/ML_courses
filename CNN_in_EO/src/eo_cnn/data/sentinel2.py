"""
Sentinel-2 tile loading and scene tiling utilities.

Sentinel-2 Level-2A (Bottom-of-Atmosphere) products provide surface
reflectance in 13 spectral bands at 10m, 20m, and 60m resolution.

This module handles:
  - Loading multi-band GeoTIFFs (rasterio)
  - Band selection and resampling to a common 10m grid
  - Percentile-based normalization
  - Spectral index computation (NDVI, NDWI, NDBI, EVI)
  - Tiling a full scene into 512×512 (or custom) patches
  - Reconstructing a prediction map from tiles with optional blending
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

# Sentinel-2 band order and spatial resolutions
S2_BANDS = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10,
    "B05": 20, "B06": 20, "B07": 20, "B08": 10,
    "B8A": 20, "B09": 60, "B10": 60, "B11": 20, "B12": 20,
}

# Default band selection for 10m-upsampled 10-band stack
# (B01, B09, B10 excluded — 60m resolution, limited information for land cover)
DEFAULT_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# Approximate surface reflectance stats (scaled to [0, 10000] range)
# Computed over a representative European Sentinel-2 tile subset
S2_MEAN_SR = {
    "B02": 426.0, "B03": 680.0, "B04": 613.0, "B05": 724.0,
    "B06": 2141.0, "B07": 2551.0, "B08": 2444.0, "B8A": 2704.0,
    "B11": 1536.0, "B12": 880.0,
}
S2_STD_SR = {
    "B02": 427.0, "B03": 397.0, "B04": 514.0, "B05": 499.0,
    "B06": 795.0, "B07": 883.0, "B08": 904.0, "B8A": 963.0,
    "B11": 818.0, "B12": 656.0,
}


class Sentinel2Tile:
    """
    Loads a Sentinel-2 Level-2A GeoTIFF (multi-band or single-band per file)
    and exposes spectral index computation and normalization utilities.

    Parameters
    ----------
    path : str | Path
        Path to a multi-band GeoTIFF stacked in DEFAULT_BANDS order,
        OR the directory of individual band files (B02.tif, B03.tif, …).
    bands : list[str]
        Bands to load. Defaults to DEFAULT_BANDS.
    """

    def __init__(
        self,
        path: str | Path,
        bands: list[str] = DEFAULT_BANDS,
    ) -> None:
        self.path = Path(path)
        self.bands = bands
        self._data: Optional[np.ndarray] = None  # (C, H, W) float32
        self._meta: Optional[dict] = None

    def load(self) -> "Sentinel2Tile":
        """Load raster data into memory. Returns self for chaining."""
        try:
            import rasterio
            from rasterio.enums import Resampling
        except ImportError as exc:
            raise ImportError("rasterio is required: pip install rasterio") from exc

        if self.path.is_dir():
            self._data, self._meta = self._load_from_directory(self.path)
        else:
            with rasterio.open(self.path) as src:
                self._meta = src.meta.copy()
                self._data = src.read().astype(np.float32)  # (C, H, W)
        return self

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self.load()
        return self._data

    @property
    def meta(self) -> dict:
        if self._meta is None:
            self.load()
        return self._meta

    def band(self, name: str) -> np.ndarray:
        """Return a single band as (H, W) array."""
        if name not in self.bands:
            raise ValueError(f"Band '{name}' not loaded. Available: {self.bands}")
        idx = self.bands.index(name)
        return self.data[idx]

    # ─── Spectral Indices ────────────────────────────────────────────────────

    def ndvi(self) -> np.ndarray:
        """
        Normalised Difference Vegetation Index.
        NDVI = (NIR - Red) / (NIR + Red)
        Range: [-1, 1]. High values → dense vegetation.
        """
        nir = self.band("B08").astype(np.float64)
        red = self.band("B04").astype(np.float64)
        return _safe_ratio(nir - red, nir + red).astype(np.float32)

    def ndwi(self) -> np.ndarray:
        """
        Normalised Difference Water Index (McFeeters, 1996).
        NDWI = (Green - NIR) / (Green + NIR)
        Range: [-1, 1]. High values → open water.
        """
        green = self.band("B03").astype(np.float64)
        nir = self.band("B08").astype(np.float64)
        return _safe_ratio(green - nir, green + nir).astype(np.float32)

    def ndbi(self) -> np.ndarray:
        """
        Normalised Difference Built-up Index.
        NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        Range: [-1, 1]. High values → built-up / urban areas.
        """
        swir = self.band("B11").astype(np.float64)
        nir = self.band("B08").astype(np.float64)
        return _safe_ratio(swir - nir, swir + nir).astype(np.float32)

    def evi(self) -> np.ndarray:
        """
        Enhanced Vegetation Index.
        EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        Less susceptible to atmospheric and soil background effects than NDVI.
        """
        nir = self.band("B08").astype(np.float64) / 10000.0
        red = self.band("B04").astype(np.float64) / 10000.0
        blue = self.band("B02").astype(np.float64) / 10000.0
        denom = nir + 6.0 * red - 7.5 * blue + 1.0
        return np.clip(2.5 * (nir - red) / (denom + 1e-8), -1, 1).astype(np.float32)

    # ─── Normalization ───────────────────────────────────────────────────────

    def normalize_percentile(self, low: float = 2.0, high: float = 98.0) -> np.ndarray:
        """
        Percentile stretch normalization: clips to [p_low, p_high] then scales to [0, 1].
        Robust to outliers (clouds, sensor saturations). Returns (C, H, W) float32.
        """
        out = np.zeros_like(self.data, dtype=np.float32)
        for i in range(self.data.shape[0]):
            band = self.data[i]
            p_low = np.percentile(band, low)
            p_high = np.percentile(band, high)
            out[i] = np.clip((band - p_low) / (p_high - p_low + 1e-8), 0.0, 1.0)
        return out

    def normalize_global(self, band_names: Optional[list[str]] = None) -> np.ndarray:
        """
        Z-score normalize using precomputed per-band statistics from S2_MEAN_SR.
        Appropriate when training on standardized datasets.
        Returns (C, H, W) float32.
        """
        bands = band_names or self.bands
        out = np.zeros_like(self.data, dtype=np.float32)
        for i, b in enumerate(bands):
            mean = S2_MEAN_SR.get(b, 1000.0)
            std = S2_STD_SR.get(b, 500.0)
            out[i] = (self.data[i] - mean) / (std + 1e-8)
        return out

    def true_color_rgb(self, gamma: float = 2.2) -> np.ndarray:
        """
        Return a display-ready (H, W, 3) uint8 RGB image using B04, B03, B02.
        """
        r = self.band("B04")
        g = self.band("B03")
        b = self.band("B02")
        rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
        for i in range(3):
            p2, p98 = np.percentile(rgb[..., i], [2, 98])
            rgb[..., i] = np.clip((rgb[..., i] - p2) / (p98 - p2 + 1e-8), 0, 1)
        rgb = (rgb ** (1.0 / gamma) * 255).astype(np.uint8)
        return rgb

    def _load_from_directory(self, directory: Path) -> tuple[np.ndarray, dict]:
        """Load individual band GeoTIFFs from a directory and stack them."""
        import rasterio
        from rasterio.enums import Resampling

        arrays = []
        meta = None
        target_h, target_w = None, None

        for band_name in self.bands:
            candidates = list(directory.glob(f"*{band_name}*.tif"))
            if not candidates:
                raise FileNotFoundError(f"No .tif file found for band {band_name} in {directory}")
            with rasterio.open(candidates[0]) as src:
                if meta is None:
                    meta = src.meta.copy()
                    target_h, target_w = src.height, src.width
                arr = src.read(
                    1,
                    out_shape=(target_h, target_w),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
                arrays.append(arr)

        return np.stack(arrays, axis=0), meta


def tile_scene(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """
    Split a (C, H, W) image array into overlapping (C, tile_size, tile_size) patches.

    Returns
    -------
    tiles : list of (C, tile_size, tile_size) arrays
    positions : list of (row_start, col_start) for each tile
    """
    _, h, w = image.shape
    stride = tile_size - overlap
    tiles = []
    positions = []

    row = 0
    while row + tile_size <= h + overlap:
        r0 = min(row, h - tile_size)
        col = 0
        while col + tile_size <= w + overlap:
            c0 = min(col, w - tile_size)
            tile = image[:, r0:r0 + tile_size, c0:c0 + tile_size]
            tiles.append(tile.copy())
            positions.append((r0, c0))
            col += stride
        row += stride

    return tiles, positions


def reconstruct_from_tiles(
    predictions: list[np.ndarray],
    positions: list[tuple[int, int]],
    scene_h: int,
    scene_w: int,
    tile_size: int = 512,
    num_classes: int = 7,
    blend: bool = True,
) -> np.ndarray:
    """
    Reconstruct a full-scene probability map from tile-level predictions.

    Parameters
    ----------
    predictions : list of (num_classes, tile_size, tile_size) float32 arrays
        Softmax probability maps per tile.
    positions : list of (row_start, col_start)
    scene_h, scene_w : full scene dimensions
    blend : if True, use Gaussian weighting to blend overlapping regions

    Returns
    -------
    (num_classes, scene_h, scene_w) float32 probability map
    """
    accumulator = np.zeros((num_classes, scene_h, scene_w), dtype=np.float64)
    weight_map = np.zeros((scene_h, scene_w), dtype=np.float64)

    if blend:
        # Gaussian weight kernel for smooth blending at tile boundaries
        kernel = _gaussian_kernel(tile_size)
    else:
        kernel = np.ones((tile_size, tile_size), dtype=np.float64)

    for pred, (r0, c0) in zip(predictions, positions):
        accumulator[:, r0:r0 + tile_size, c0:c0 + tile_size] += pred * kernel
        weight_map[r0:r0 + tile_size, c0:c0 + tile_size] += kernel

    weight_map = np.maximum(weight_map, 1e-8)
    return (accumulator / weight_map[np.newaxis]).astype(np.float32)


def _gaussian_kernel(size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """Create a 2D Gaussian kernel for tile blending."""
    sigma = size * sigma_ratio
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = np.outer(gauss_1d, gauss_1d)
    return (kernel / kernel.max()).astype(np.float64)


def _safe_ratio(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Compute numerator/denominator with zero-division protection."""
    return np.where(np.abs(denom) < 1e-8, 0.0, num / (denom + 1e-8))
