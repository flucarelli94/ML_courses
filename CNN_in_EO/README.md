# CNNs for Earth Observation
### A Production-Grade Course in Convolutional Neural Networks Applied to Satellite Imagery

---

## Course Overview

This course teaches **Convolutional Neural Networks through real satellite-imagery applications**, progressing from CNN fundamentals to a complete geospatial deep learning pipeline. The backbone project is **Sentinel-2 Multiclass Land Cover Segmentation** — a task that is simultaneously industry-relevant, pedagogically rich, and executable on accessible hardware.

**Target learner:** Advanced Python programmer (7+ years) with ML experience but limited CNN and EO background.

**Total estimated duration:** ~60 hours (theory + practicals)

---

## Backbone Project: Sentinel-2 Land Cover Segmentation

The course builds toward a single coherent goal: producing pixel-level land cover maps from multispectral Sentinel-2 imagery using deep CNNs. This problem:

- Requires understanding convolution at a pixel level (segmentation, not just classification)
- Demands multi-band geospatial preprocessing (13 spectral bands, CRS handling, tiling)
- Rewards transfer learning (ImageNet → EO domain adaptation)
- Produces visually compelling outputs (full-scene prediction maps)
- Mirrors real workflows at ESA, Planet, Maxar, and national mapping agencies

**Progression:** EuroSAT patch classification → U-Net binary segmentation → multiclass Sentinel-2 land cover → full geospatial inference pipeline

---

## Course Structure

| # | Chapter | Key Concepts | Practical | Duration | Level |
|---|---------|-------------|-----------|----------|-------|
| 01 | Intro: EO + Deep Learning | Satellites, spectral bands, CNN motivation | EO data exploration with rasterio | 4h | Beginner |
| 02 | Convolution Fundamentals | Kernels, feature maps, receptive field | Convolution from scratch in NumPy + PyTorch | 5h | Beginner |
| 03 | CNN Building Blocks | Pooling, activations, BatchNorm, Dropout | Build a CNN block-by-block | 4h | Beginner |
| 04 | Geospatial Preprocessing | CRS, tiling, normalization, bands | Sentinel-2 preprocessing pipeline | 5h | Beginner→Intermediate |
| 05 | Classification Pipeline | Dataset loaders, training loop, metrics | EuroSAT 10-class classifier | 5h | Intermediate |
| 06 | CNN Architectures | VGG, ResNet, EfficientNet, skip connections | Replace backbone, compare results | 4h | Intermediate |
| 07 | Transfer Learning for EO | Fine-tuning, frozen layers, domain shift | Fine-tune ResNet50 on EuroSAT | 5h | Intermediate |
| 08 | Augmentation for Satellite Data | Geometric, spectral, CutMix, MixUp | albumentations pipeline for EO | 4h | Intermediate |
| 09 | Semantic Segmentation + U-Net | Encoder-decoder, skip connections, U-Net | U-Net on LoveDA dataset | 6h | Intermediate→Advanced |
| 10 | Advanced Segmentation | DeepLabV3+, FPN, PSPNet, attention | SMP library, multi-architecture benchmark | 5h | Advanced |
| 11 | Full EO Pipeline | Scene tiling, sliding-window inference, GeoTIFF | End-to-end Sentinel-2 prediction map | 6h | Advanced |
| 12 | Evaluation + Interpretability | IoU, mIoU, GradCAM, confusion matrix | Full metrics dashboard + saliency maps | 4h | Advanced |
| Cap | Capstone: Land Cover Mapping | Full pipeline integration | Independent project deliverable | 10h | Advanced |

---

## Repository Structure

```
CNN_projects/
├── README.md
├── pyproject.toml              ← uv project config
├── environment.yml             ← conda environment
├── requirements.txt            ← pip requirements
│
├── docs/
│   ├── course_design.md        ← Parts 1-3: candidate analysis + backbone selection
│   ├── infrastructure.md       ← Part 5: environment, stack, repo guide
│   ├── capstone.md             ← Part 6: capstone brief + rubric
│   └── resources.md            ← Part 7: papers, tools, advanced topics
│
├── src/
│   └── eo_cnn/
│       ├── __init__.py
│       ├── data/               ← dataset loaders + transforms
│       │   ├── eurosat.py
│       │   ├── sentinel2.py
│       │   └── transforms.py
│       ├── models/             ← CNN architectures
│       │   ├── simple_cnn.py
│       │   ├── resnet.py
│       │   └── unet.py
│       ├── training/           ← training loops + metrics
│       │   ├── trainer.py
│       │   └── metrics.py
│       └── visualization/      ← plotting utilities
│           └── plots.py
│
└── notebooks/
    ├── chapter_01_intro_eo_dl/
    ├── chapter_02_convolution_fundamentals/
    ├── chapter_03_cnn_building_blocks/
    ├── chapter_04_geospatial_preprocessing/
    ├── chapter_05_classification_pipeline/
    ├── chapter_06_cnn_architectures/
    ├── chapter_07_transfer_learning/
    ├── chapter_08_augmentation/
    ├── chapter_09_semantic_segmentation/
    ├── chapter_10_advanced_segmentation/
    ├── chapter_11_full_eo_pipeline/
    ├── chapter_12_evaluation_interpretability/
    └── capstone/
```

---

## Quick Start

### Option A: uv (recommended)

```bash
git clone <repo>
cd CNN_projects
uv sync
uv run jupyter lab
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate eo-cnn
jupyter lab
```

### Option C: Google Colab

Each notebook contains a `# COLAB SETUP` cell at the top that installs all required packages and mounts Google Drive. Open any notebook directly in Colab.

---

## Hardware Requirements

| Mode | Hardware | Training time (ch05 baseline) |
|------|----------|-------------------------------|
| Recommended | NVIDIA GPU (8GB+ VRAM) | ~10 min |
| Acceptable | CPU-only | ~60-90 min (use `FAST_MODE=True`) |
| Cloud | Google Colab T4 | ~15 min |

All notebooks include a `DEVICE` auto-detection cell and `FAST_MODE` flag for CPU fallback with reduced dataset size.

---

## Software Stack

| Category | Library | Version |
|----------|---------|---------|
| Deep learning | PyTorch | ≥2.2 |
| Segmentation | segmentation-models-pytorch | ≥0.3 |
| EO datasets | torchgeo | ≥0.6 |
| Raster I/O | rasterio | ≥1.3 |
| Raster xarray | rioxarray | ≥0.15 |
| Geospatial | geopandas | ≥0.14 |
| Multi-dim arrays | xarray | ≥2024.1 |
| Augmentation | albumentations | ≥1.4 |
| Visualization | matplotlib, seaborn | latest |
| Notebooks | jupyter, ipywidgets | latest |

---

## Datasets Used

| Chapter | Dataset | Access | Size |
|---------|---------|--------|------|
| Ch01, Ch05, Ch06, Ch07 | EuroSAT (RGB + MS) | torchgeo auto-download | ~90MB |
| Ch04, Ch11 | Sentinel-2 sample tiles | via sentinelsat or manual | ~500MB |
| Ch09, Ch10 | LoveDA | torchgeo auto-download | ~3GB |
| Ch12, Cap | BigEarthNet (subset) | via torchgeo | ~65GB full / 1GB subset |
| Supplementary | OpenStreetMap labels | osmnx | on-demand |

---

## Learning Outcomes

By the end of this course you will be able to:

1. Implement CNNs from scratch and explain every architectural decision
2. Load, preprocess, tile and normalize multispectral satellite imagery
3. Train classification and segmentation models on real EO datasets
4. Apply transfer learning from ImageNet to the EO domain
5. Design augmentation pipelines appropriate for satellite imagery
6. Evaluate models with IoU, mIoU, F1, confusion matrices and GradCAM
7. Run sliding-window inference over full Sentinel-2 scenes
8. Export prediction maps as georeferenced GeoTIFFs
9. Build a portfolio-ready end-to-end EO deep learning system

---

## License

Course materials: MIT License. Datasets are subject to their respective licenses (see `docs/resources.md`).
