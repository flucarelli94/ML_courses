"""Generate all matplotlib figures for the CNNs-for-Earth-Observation textbook.

Run with the project virtualenv:
    .venv/bin/python docs/textbook/make_figures.py

All figures are written to docs/textbook/figures/ as PNG at 200 dpi.
The figures are deterministic (fixed RNG seeds) so the textbook is reproducible.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, FancyArrow

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.grid": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "serif",
})

RNG = np.random.default_rng(42)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 1. Spectral signatures of land cover types
# ---------------------------------------------------------------------------
def fig_spectral_signatures():
    bands = ["B02\nBlue", "B03\nGreen", "B04\nRed", "B05\nRE1", "B06\nRE2",
             "B07\nRE3", "B08\nNIR", "B11\nSWIR1", "B12\nSWIR2"]
    wl = [0.49, 0.56, 0.66, 0.70, 0.74, 0.78, 0.83, 1.61, 2.20]
    veg = [0.04, 0.08, 0.05, 0.20, 0.35, 0.42, 0.46, 0.22, 0.10]
    water = [0.06, 0.05, 0.035, 0.025, 0.02, 0.015, 0.012, 0.008, 0.006]
    soil = [0.10, 0.14, 0.20, 0.24, 0.27, 0.29, 0.31, 0.36, 0.32]
    urban = [0.18, 0.19, 0.20, 0.21, 0.22, 0.225, 0.23, 0.26, 0.25]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(wl, veg, "o-", color="#1A7A1A", label="Vegetation / Forest", lw=2)
    ax.plot(wl, water, "s-", color="#1E56C8", label="Water", lw=2)
    ax.plot(wl, soil, "^-", color="#A0754A", label="Bare soil", lw=2)
    ax.plot(wl, urban, "d-", color="#E30B0B", label="Urban / built-up", lw=2)
    ax.axvspan(0.40, 0.70, color="orange", alpha=0.06)
    ax.text(0.55, 0.49, "Visible", ha="center", fontsize=9, color="0.4")
    ax.text(0.83, 0.49, "NIR", ha="center", fontsize=9, color="0.4")
    ax.text(1.9, 0.49, "SWIR", ha="center", fontsize=9, color="0.4")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Surface reflectance")
    ax.set_title("Spectral reflectance signatures (Sentinel-2 bands)")
    ax.set_ylim(0, 0.52)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "spectral_signatures.png")


# ---------------------------------------------------------------------------
# 2. Convolution operation illustration
# ---------------------------------------------------------------------------
def fig_convolution():
    img = np.array([
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9],
    ], dtype=float)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)

    out = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            out[i, j] = np.sum(img[i:i + 3, j:j + 3] * kernel)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8),
                             gridspec_kw={"width_ratios": [3, 1.1, 2.2]})

    ax = axes[0]
    ax.imshow(img, cmap="Blues", vmin=0, vmax=9)
    for (i, j), v in np.ndenumerate(img):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=11)
    rect = Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor="red", lw=2.5)
    ax.add_patch(rect)
    ax.set_title("Input $I$ (6×6)  with 3×3 window")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.imshow(kernel, cmap="RdBu", vmin=-1, vmax=1)
    for (i, j), v in np.ndenumerate(kernel):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=12)
    ax.set_title("Kernel $K$\n(vertical edge)")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    ax.imshow(out, cmap="PuOr", vmin=-np.abs(out).max(), vmax=np.abs(out).max())
    for (i, j), v in np.ndenumerate(out):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=10)
    hl = Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="red", lw=2.5)
    ax.add_patch(hl)
    ax.set_title("Feature map (4×4)")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("2D cross-correlation: slide kernel, sum element-wise products", y=1.02)
    save(fig, "convolution.png")


# ---------------------------------------------------------------------------
# 3. Classic kernels applied to a synthetic satellite-like image
# ---------------------------------------------------------------------------
def _synthetic_scene(n=96):
    y, x = np.mgrid[0:n, 0:n]
    img = 0.3 + 0.1 * np.sin(x / 5.0)
    img += 0.25 * ((x // 12 + y // 12) % 2)  # field grid
    img[10:30, 60:90] = 0.85  # bright urban block
    img[55:85, 10:40] += 0.2 * np.sin(y[55:85, 10:40] / 2.0)
    img = np.clip(img + 0.03 * RNG.standard_normal((n, n)), 0, 1)
    return img


def _conv(img, k):
    from numpy.lib.stride_tricks import sliding_window_view
    pad = k.shape[0] // 2
    p = np.pad(img, pad, mode="reflect")
    w = sliding_window_view(p, k.shape)
    return np.einsum("ijkl,kl->ij", w, k)


def fig_kernels():
    img = _synthetic_scene()
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)
    sobel_y = sobel_x.T
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
    blur = np.ones((3, 3)) / 9.0
    sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float)

    outs = {
        "Original": img,
        "Sobel-x (vertical edges)": _conv(img, sobel_x),
        "Sobel-y (horizontal edges)": _conv(img, sobel_y),
        "Laplacian": _conv(img, lap),
        "Gaussian blur": _conv(img, blur),
        "Sharpen": _conv(img, sharp),
    }
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.4))
    for ax, (name, o) in zip(axes.ravel(), outs.items()):
        cmap = "gray" if name in ("Original", "Gaussian blur", "Sharpen") else "RdBu"
        ax.imshow(o, cmap=cmap)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Hand-designed kernels detect edges, texture and structure", y=1.0)
    save(fig, "kernels.png")


# ---------------------------------------------------------------------------
# 4. Activation functions
# ---------------------------------------------------------------------------
def fig_activations():
    x = np.linspace(-5, 5, 400)
    relu = np.maximum(0, x)
    lrelu = np.where(x > 0, x, 0.1 * x)
    sig = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color="0.7", lw=0.8); ax.axvline(0, color="0.7", lw=0.8)
    ax.plot(x, relu, label="ReLU", lw=2)
    ax.plot(x, lrelu, label="LeakyReLU (0.1)", lw=2, ls="--")
    ax.plot(x, gelu, label="GELU", lw=2)
    ax.plot(x, sig, label="Sigmoid", lw=2)
    ax.plot(x, tanh, label="Tanh", lw=2)
    ax.set_ylim(-1.5, 5)
    ax.set_xlabel("$z$"); ax.set_ylabel(r"$\phi(z)$")
    ax.set_title("Activation functions")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    save(fig, "activations.png")


# ---------------------------------------------------------------------------
# 5. Pooling
# ---------------------------------------------------------------------------
def fig_pooling():
    a = np.array([
        [1, 3, 2, 4],
        [5, 6, 8, 1],
        [2, 9, 4, 7],
        [3, 1, 5, 2],
    ], float)
    maxp = np.array([[a[0:2, 0:2].max(), a[0:2, 2:4].max()],
                     [a[2:4, 0:2].max(), a[2:4, 2:4].max()]])
    avgp = np.array([[a[0:2, 0:2].mean(), a[0:2, 2:4].mean()],
                     [a[2:4, 0:2].mean(), a[2:4, 2:4].mean()]])

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
    for ax, (mat, title, cmap) in zip(axes, [
        (a, "Input (4×4)", "Greens"),
        (maxp, "Max pool 2×2", "Oranges"),
        (avgp, "Avg pool 2×2", "Purples"),
    ]):
        ax.imshow(mat, cmap=cmap)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:.0f}" if title == "Input (4×4)" or "Max" in title
                    else f"{v:.1f}", ha="center", va="center", fontsize=12)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    for k in range(2):
        for l in range(2):
            axes[0].add_patch(Rectangle((2 * l - 0.5, 2 * k - 0.5), 2, 2,
                                        fill=False, ec="red", lw=2))
    fig.suptitle("Pooling downsamples by a factor of 2 (stride 2)", y=1.02)
    save(fig, "pooling.png")


# ---------------------------------------------------------------------------
# 6. Receptive field growth
# ---------------------------------------------------------------------------
def fig_receptive_field():
    layers = np.arange(1, 21)
    rf_nopool = 1 + layers * 2  # 3x3, stride 1
    # with a pool (stride2) every 2 conv layers
    rf_pool = []
    rf = 1
    jump = 1
    for n in range(1, 21):
        rf = rf + (3 - 1) * jump
        if n % 2 == 0:
            rf = rf + (2 - 1) * jump
            jump *= 2
        rf_pool.append(rf)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(layers, rf_nopool, "o-", label="3×3 conv, stride 1 (no pool)")
    ax.plot(layers, rf_pool, "s-", label="3×3 conv + 2×2 pool every 2 layers")
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] * 10, ax.get_ylim()[1] * 10)
    ax2.set_ylabel("Ground coverage at 10 m/px (m)")
    ax.set_xlabel("Network depth (number of conv layers)")
    ax.set_ylabel("Receptive field (pixels)")
    ax.set_title("Receptive field grows with depth")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "receptive_field.png")


# ---------------------------------------------------------------------------
# 7. Cross-entropy loss
# ---------------------------------------------------------------------------
def fig_crossentropy():
    p = np.linspace(1e-3, 1, 300)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(p, -np.log(p), color="#c0392b", lw=2.2)
    ax.set_xlabel("Predicted probability of the true class $p_y$")
    ax.set_ylabel(r"Loss $-\log p_y$")
    ax.set_title("Cross-entropy penalises confident wrong predictions")
    ax.set_ylim(0, 7)
    for pv in (0.1, 0.5, 0.9):
        ax.plot([pv], [-np.log(pv)], "o", color="#2c3e50")
        ax.annotate(f"$p_y={pv}$\nloss={-np.log(pv):.2f}", (pv, -np.log(pv)),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)
    save(fig, "crossentropy.png")


# ---------------------------------------------------------------------------
# 8. NDVI heatmap
# ---------------------------------------------------------------------------
def fig_ndvi():
    n = 120
    y, x = np.mgrid[0:n, 0:n]
    red = 0.2 + 0.05 * RNG.standard_normal((n, n))
    nir = 0.25 + 0.05 * RNG.standard_normal((n, n))
    # forest patch (high NIR)
    forest = ((x - 35)**2 + (y - 40)**2) < 600
    nir[forest] = 0.55
    # water body (low NIR)
    water = ((x - 85)**2 + (y - 80)**2) < 500
    nir[water] = 0.05; red[water] = 0.06
    # crops
    crop = (x > 70) & (y < 45)
    nir[crop] = 0.4; red[crop] = 0.18
    ndvi = (nir - red) / (nir + red + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    rgb = np.dstack([np.clip(red * 2.5, 0, 1),
                     np.clip((red + nir) / 2 * 2, 0, 1),
                     np.clip((1 - nir) * 0.6, 0, 1)])
    axes[0].imshow(rgb)
    axes[0].set_title("Pseudo true-colour"); axes[0].axis("off")
    im = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.3, vmax=0.8)
    axes[1].set_title("NDVI = (NIR − Red)/(NIR + Red)"); axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    save(fig, "ndvi.png")


# ---------------------------------------------------------------------------
# 9. Confusion matrix example
# ---------------------------------------------------------------------------
def fig_confusion():
    classes = ["Urban", "Agri", "Forest", "Grass", "Water", "Soil"]
    cm_ = np.array([
        [0.82, 0.05, 0.01, 0.02, 0.02, 0.08],
        [0.04, 0.78, 0.05, 0.10, 0.01, 0.02],
        [0.01, 0.06, 0.86, 0.06, 0.00, 0.01],
        [0.03, 0.14, 0.09, 0.70, 0.01, 0.03],
        [0.02, 0.01, 0.00, 0.01, 0.95, 0.01],
        [0.10, 0.05, 0.01, 0.04, 0.01, 0.79],
    ])
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm_, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(classes)
    for (i, j), v in np.ndenumerate(cm_):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Row-normalised confusion matrix")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    save(fig, "confusion.png")


# ---------------------------------------------------------------------------
# 10. IoU / Dice illustration
# ---------------------------------------------------------------------------
def fig_iou():
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    gt = Rectangle((1, 1), 3, 2.4, alpha=0.45, color="#1E56C8", label="Ground truth")
    pr = Rectangle((2.2, 1.6), 3, 2.4, alpha=0.45, color="#E30B0B", label="Prediction")
    ax.add_patch(gt); ax.add_patch(pr)
    ax.text(2.5, 2.2, "GT", fontsize=12, color="#1E56C8")
    ax.text(4.4, 3.4, "Pred", fontsize=12, color="#E30B0B")
    ax.text(3.0, 2.6, "∩", fontsize=18)
    ax.set_xlim(0, 7); ax.set_ylim(0, 5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(r"IoU $=\dfrac{|A\cap B|}{|A\cup B|}$"
                 "      "
                 r"Dice $=\dfrac{2|A\cap B|}{|A|+|B|}$")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "iou.png")


# ---------------------------------------------------------------------------
# 11. Augmentation grid
# ---------------------------------------------------------------------------
def fig_augmentation():
    img = _synthetic_scene(64)
    rgb = np.dstack([img, np.roll(img, 3, 0), 1 - img])
    rgb = np.clip(rgb, 0, 1)
    variants = {
        "Original": rgb,
        "H-flip": rgb[:, ::-1],
        "V-flip": rgb[::-1],
        "Rot 90°": np.rot90(rgb),
        "Brightness": np.clip(rgb * 1.4, 0, 1),
        "+ Gaussian noise": np.clip(rgb + 0.08 * RNG.standard_normal(rgb.shape), 0, 1),
    }
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.6))
    for ax, (name, im) in zip(axes.ravel(), variants.items()):
        ax.imshow(im); ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Label-preserving augmentations for satellite imagery", y=1.0)
    save(fig, "augmentation.png")


# ---------------------------------------------------------------------------
# 12. Land cover map (synthetic prediction)
# ---------------------------------------------------------------------------
LC_COLORS = ["#000000", "#E30B0B", "#F5E642", "#1A7A1A", "#8FD68F", "#1E56C8", "#A0754A"]
LC_NAMES = ["Background", "Urban", "Agriculture", "Forest", "Grass/Shrub", "Water", "Bare soil"]


def fig_landcover():
    n = 160
    y, x = np.mgrid[0:n, 0:n]
    lab = np.full((n, n), 4, dtype=int)  # grass background
    lab[(y < 70)] = 2  # agriculture top
    lab[((x - 110)**2 + (y - 50)**2) < 900] = 3  # forest
    lab[(x > 120) & (y > 90)] = 1  # urban
    lab[((x - 40)**2 + (y - 120)**2) < 700] = 5  # water
    lab[(y > 130) & (x < 70)] = 6  # soil
    cmap = ListedColormap(LC_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 7.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    # fake RGB
    rgb = np.zeros((n, n, 3))
    for c in range(7):
        m = lab == c
        col = np.array([int(LC_COLORS[c][i:i + 2], 16) / 255 for i in (1, 3, 5)])
        rgb[m] = col * 0.6 + 0.2
    rgb = np.clip(rgb + 0.04 * RNG.standard_normal(rgb.shape), 0, 1)
    axes[0].imshow(rgb); axes[0].set_title("Input scene (true-colour)")
    axes[0].axis("off")
    axes[1].imshow(lab, cmap=cmap, norm=norm)
    axes[1].set_title("Predicted land-cover map"); axes[1].axis("off")
    handles = [Rectangle((0, 0), 1, 1, color=LC_COLORS[i]) for i in range(7)]
    axes[1].legend(handles, LC_NAMES, loc="center left",
                   bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    save(fig, "landcover.png")


# ---------------------------------------------------------------------------
# 13. Normalization strategies
# ---------------------------------------------------------------------------
def fig_normalization():
    raw = RNG.gamma(2.0, 800, 20000)
    raw = np.clip(raw, 0, 10000)
    minmax = raw / raw.max()
    p2, p98 = np.percentile(raw, [2, 98])
    perc = np.clip((raw - p2) / (p98 - p2), 0, 1)
    z = (raw - raw.mean()) / raw.std()

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, (d, t) in zip(axes, [
        (raw, "Raw DN (0–10000)"),
        (minmax, "Min–max [0,1]"),
        (perc, "2–98% percentile"),
        (z, "Z-score (µ=0, σ=1)"),
    ]):
        ax.hist(d, bins=50, color="#2c7fb8")
        ax.set_title(t, fontsize=10)
        ax.set_yticks([])
    fig.suptitle("Normalization strategies for multispectral reflectance", y=1.04)
    save(fig, "normalization.png")


# ---------------------------------------------------------------------------
# 14. Training curves
# ---------------------------------------------------------------------------
def fig_training_curves():
    ep = np.arange(1, 31)
    tr = 1.6 * np.exp(-ep / 6) + 0.18 + 0.01 * RNG.standard_normal(30)
    va = 1.6 * np.exp(-ep / 6) + 0.30 + 0.04 * RNG.standard_normal(30)
    va_over = 1.6 * np.exp(-ep / 5) + 0.25 + 0.02 * ep / 10 + 0.03 * RNG.standard_normal(30)
    miou = 0.65 * (1 - np.exp(-ep / 7)) + 0.02 * RNG.standard_normal(30)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(ep, tr, label="train loss")
    axes[0].plot(ep, va, label="val loss (good fit)")
    axes[0].plot(ep, va_over, "--", label="val loss (overfitting)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curves"); axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(ep, np.clip(miou, 0, 1), color="#16a085")
    axes[1].axhline(miou.max(), ls=":", color="0.5")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation mIoU")
    axes[1].set_title("Segmentation metric over training")
    save(fig, "training_curves.png")


if __name__ == "__main__":
    fig_spectral_signatures()
    fig_convolution()
    fig_kernels()
    fig_activations()
    fig_pooling()
    fig_receptive_field()
    fig_crossentropy()
    fig_ndvi()
    fig_confusion()
    fig_iou()
    fig_augmentation()
    fig_landcover()
    fig_normalization()
    fig_training_curves()
    print("All figures generated.")
