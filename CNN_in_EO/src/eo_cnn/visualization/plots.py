"""
Visualization utilities for EO deep learning results.

All functions return matplotlib Figure objects so they can be
displayed in Jupyter notebooks, saved to disk, or logged to W&B.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch import Tensor

# Default land cover colormap (LoveDA 7-class palette)
LOVEDA_COLORS = [
    "#000000",  # 0: Background
    "#E30B0B",  # 1: Urban
    "#F5E642",  # 2: Agriculture
    "#1A7A1A",  # 3: Forest
    "#8FD68F",  # 4: Grassland
    "#1E56C8",  # 5: Water
    "#A0754A",  # 6: Barren
]

LOVEDA_CLASSES = ["Background", "Urban", "Agriculture", "Forest", "Grassland", "Water", "Barren"]
LOVEDA_CMAP = ListedColormap(LOVEDA_COLORS)

EUROSAT_CLASSES = [
    "Annual Crop", "Forest", "Herbaceous Veg.", "Highway",
    "Industrial", "Pasture", "Permanent Crop", "Residential", "River", "Sea/Lake",
]


def show_batch(
    images: Tensor,
    labels: Tensor,
    class_names: list[str] = EUROSAT_CLASSES,
    n: int = 16,
    figsize: tuple[int, int] = (14, 8),
    unnormalize: bool = True,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> plt.Figure:
    """
    Display a grid of images with their class labels.
    Works for both EuroSAT classification and segmentation preview.
    """
    n = min(n, images.shape[0])
    cols = min(8, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).ravel()

    for i in range(n):
        img = images[i].cpu().float()
        if unnormalize:
            img = _unnormalize(img, mean, std)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = img_np[..., :3]  # take first 3 channels if multispectral
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        label = labels[i].item() if labels[i].ndim == 0 else labels[i].cpu().numpy()
        if isinstance(label, (int, np.integer)) and class_names:
            axes[i].set_title(class_names[label], fontsize=8)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Sample Batch ({n} images)", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def show_segmentation_predictions(
    images: Tensor,
    gt_masks: Tensor,
    pred_masks: Tensor,
    n: int = 4,
    class_names: list[str] = LOVEDA_CLASSES,
    colors: list[str] = LOVEDA_COLORS,
    figsize: tuple[int, int] = (15, 5),
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> plt.Figure:
    """
    Show (RGB input | Ground Truth mask | Predicted mask) triplets.
    """
    cmap = ListedColormap(colors[:len(class_names)])
    n = min(n, images.shape[0])

    fig, axes = plt.subplots(n, 3, figsize=(figsize[0], figsize[1] * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["RGB Input", "Ground Truth", "Prediction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for i in range(n):
        img = _unnormalize(images[i].cpu().float(), mean, std)
        img_np = img.permute(1, 2, 0).numpy()[..., :3]
        img_np = np.clip(img_np, 0, 1)

        gt = gt_masks[i].cpu().numpy()
        pred = pred_masks[i].cpu().numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 1].imshow(gt, cmap=cmap, vmin=0, vmax=len(class_names) - 1, interpolation="nearest")
        axes[i, 2].imshow(pred, cmap=cmap, vmin=0, vmax=len(class_names) - 1, interpolation="nearest")

        for ax in axes[i]:
            ax.axis("off")

    # Add legend
    patches = [mpatches.Patch(color=colors[j], label=class_names[j]) for j in range(len(class_names))]
    fig.legend(handles=patches, loc="lower center", ncol=len(class_names), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    return fig


def plot_training_history(
    history,
    metric_name: str = "Accuracy",
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot training curves: loss (left) and primary metric (right).
    `history` should have .train_loss, .val_loss, .train_metric, .val_metric.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(history.train_loss) + 1)

    ax1.plot(epochs, history.train_loss, label="Train", color="#2196F3")
    ax1.plot(epochs, history.val_loss, label="Val", color="#F44336")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.train_metric, label="Train", color="#2196F3")
    ax2.plot(epochs, history.val_metric, label="Val", color="#F44336")
    ax2.set_title(metric_name)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(metric_name)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if history.lr:
        ax1_lr = ax1.twinx()
        ax1_lr.plot(epochs, history.lr, color="gray", linestyle="--", alpha=0.5, label="LR")
        ax1_lr.set_ylabel("Learning Rate", color="gray")
        ax1_lr.tick_params(axis="y", labelcolor="gray")

    best_epoch = history.best_epoch()
    ax2.axvline(x=best_epoch + 1, color="green", linestyle=":", alpha=0.8, label=f"Best (ep {best_epoch + 1})")
    ax2.legend()

    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    figsize: tuple[int, int] = (10, 8),
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot a (num_classes × num_classes) confusion matrix as a heatmap.
    """
    import seaborn as sns

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_class_distribution(
    labels: Tensor | np.ndarray,
    class_names: list[str],
    title: str = "Class Distribution",
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Bar chart of class frequencies. Highlights imbalanced datasets.
    """
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    labels = labels.ravel()

    counts = np.bincount(labels.astype(int), minlength=len(class_names))
    percentages = 100.0 * counts / counts.sum()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(class_names, counts, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    return fig


def show_augmentation_examples(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    transform,
    n: int = 8,
    figsize: tuple[int, int] = (16, 4),
) -> plt.Figure:
    """
    Show `n` augmented versions of a single image (+ optional mask).
    Useful for verifying augmentation pipelines.
    """
    rows = 2 if mask is not None else 1
    fig, axes = plt.subplots(rows, n + 1, figsize=figsize)
    if rows == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].imshow(image[..., :3] if image.ndim == 3 else image, cmap="gray")
    axes[0, 0].set_title("Original", fontsize=9)
    axes[0, 0].axis("off")
    if mask is not None:
        axes[1, 0].imshow(mask, cmap=LOVEDA_CMAP, vmin=0, vmax=len(LOVEDA_CLASSES) - 1)
        axes[1, 0].axis("off")

    for i in range(1, n + 1):
        if mask is not None:
            out = transform(image=image, mask=mask)
            aug_img = out["image"]
            aug_mask = out["mask"]
        else:
            aug_img = transform(image=image)["image"]
            aug_mask = None

        # Convert tensor to numpy for display
        if hasattr(aug_img, "numpy"):
            aug_img = aug_img.permute(1, 2, 0).numpy() if aug_img.ndim == 3 else aug_img.numpy()
        aug_img = np.clip(aug_img[..., :3] if aug_img.ndim == 3 else aug_img, 0, 1)
        axes[0, i].imshow(aug_img)
        axes[0, i].set_title(f"Aug {i}", fontsize=9)
        axes[0, i].axis("off")

        if aug_mask is not None and mask is not None:
            if hasattr(aug_mask, "numpy"):
                aug_mask = aug_mask.numpy()
            axes[1, i].imshow(aug_mask, cmap=LOVEDA_CMAP, vmin=0, vmax=len(LOVEDA_CLASSES) - 1)
            axes[1, i].axis("off")

    fig.suptitle("Augmented Samples", fontsize=11)
    fig.tight_layout()
    return fig


def visualize_feature_maps(
    feature_maps: list[Tensor],
    n_maps: int = 8,
    figsize: tuple[int, int] = (16, 3),
    layer_names: Optional[list[str]] = None,
) -> plt.Figure:
    """
    Visualize activation maps from CNN encoder layers.
    Shows the first `n_maps` channels from each feature map tensor.
    """
    n_layers = len(feature_maps)
    fig, axes = plt.subplots(n_layers, n_maps, figsize=(figsize[0], figsize[1] * n_layers))
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    for row, fmap in enumerate(feature_maps):
        fmap_np = fmap[0].cpu().float().numpy()  # take first batch element
        n_show = min(n_maps, fmap_np.shape[0])
        for col in range(n_show):
            ax = axes[row, col]
            channel = fmap_np[col]
            ax.imshow(channel, cmap="viridis")
            ax.axis("off")
            if col == 0 and layer_names:
                ax.set_ylabel(layer_names[row], fontsize=9)
        for col in range(n_show, n_maps):
            axes[row, col].axis("off")

    fig.suptitle("CNN Feature Maps (first 8 channels per layer)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_land_cover_map(
    prediction: np.ndarray,
    rgb_image: Optional[np.ndarray] = None,
    class_names: list[str] = LOVEDA_CLASSES,
    colors: list[str] = LOVEDA_COLORS,
    title: str = "Land Cover Prediction",
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Display a full-scene land cover prediction map with optional RGB comparison.

    Parameters
    ----------
    prediction : (H, W) integer array of class indices
    rgb_image : (H, W, 3) uint8 array for comparison panel (optional)
    """
    cmap = ListedColormap(colors[:len(class_names)])
    n_panels = 2 if rgb_image is not None else 1

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    if rgb_image is not None:
        axes[0].imshow(rgb_image)
        axes[0].set_title("True Colour (RGB)", fontsize=11)
        axes[0].axis("off")

    im = axes[-1].imshow(
        prediction, cmap=cmap,
        vmin=0, vmax=len(class_names) - 1,
        interpolation="nearest",
    )
    axes[-1].set_title(title, fontsize=11)
    axes[-1].axis("off")

    patches = [
        mpatches.Patch(color=colors[j], label=class_names[j])
        for j in range(len(class_names))
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=min(7, len(class_names)),
        fontsize=9, bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout()
    return fig


# ─── Private helpers ──────────────────────────────────────────────────────────

def _unnormalize(
    img: Tensor,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> Tensor:
    """Reverse ImageNet normalization for display."""
    mean_t = torch.tensor(mean, dtype=img.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype).view(-1, 1, 1)
    n_channels = img.shape[0]
    if n_channels > len(mean):
        mean_t = mean_t.expand(n_channels, -1, -1)[:n_channels]
        std_t = std_t.expand(n_channels, -1, -1)[:n_channels]
    return img * std_t[:n_channels] + mean_t[:n_channels]
