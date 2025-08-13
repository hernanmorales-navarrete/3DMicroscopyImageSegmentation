# 3D Microscopy Image Segmentation

This software provides tools for training and evaluating 3D microscopy image segmentation models using both classical and deep learning approaches.

## Command Line Interface (CLI) Guide

This guide explains how to use each command-line tool in detail. Before running any command, make sure you're in the project's root directory.

### Note About Boolean Options

Throughout this CLI, boolean options follow these conventions:
- Use `--option/--no-option` to enable/disable a feature
- Short versions use `-o/-O` format when available
- Default values are always shown in help text
- Some options may have custom names (e.g. `--accept/--reject`)

### 1. Generate Patches (`generate_patches.py`)

This tool splits your 3D microscopy images into smaller patches for processing.

```bash
python src/generate_patches.py DATASET_DIR FOR_RECONSTRUCTION

Required arguments:
- DATASET_DIR: Directory containing the dataset with 'images' and 'masks' subdirectories
- FOR_RECONSTRUCTION: Whether to generate overlapping patches suitable for image reconstruction
```

Example commands:
```bash
# For training data (without overlap)
python src/generate_patches.py data/processed/BC/training_data False

# For test data (with overlap for reconstruction)
python src/generate_patches.py data/processed/BC/test_data True

# For test data (without overlap for evaluation)
python src/generate_patches.py data/processed/BC/test_data False
```

Your input dataset must have this structure:
```
dataset/
├── images/
│   ├── image1.tif
│   └── image2.tif
└── masks/
    ├── image1.tif
    └── image2.tif
```

### 2. Train Models (`train.py`)

Train a segmentation model with various options:

```bash
python src/modeling/train.py MODEL_NAME DATA_DIR DATASET_NAME [OPTIONS]

Required arguments:
- MODEL_NAME: Name of the model to train
- DATA_DIR: Directory containing the dataset
- DATASET_NAME: Name of the dataset for model organization

Optional arguments:
- --augmentation, -a: Type of augmentation to use [default: NONE]
  - NONE: No augmentation
  - STANDARD: Basic augmentation
  - OURS: Advanced microscopy-specific augmentation
- --psf, -p PATH: Path to PSF file for microscopy augmentations
- --enable-reproducibility/--no-enable-reproducibility: Enable reproducibility by setting random seeds [default: True]
```

Example commands:
```bash
# Basic training without augmentation
python src/modeling/train.py UNet3D data/processed/BC/training_data/regular_patches BC --augmentation NONE

# Training with standard augmentation
python src/modeling/train.py UNet3D data/processed/BC/training_data/regular_patches BC --augmentation STANDARD

# Training with advanced augmentation and PSF
python src/modeling/train.py UNet3D data/processed/BC/training_data/regular_patches BC \
    --augmentation OURS \
    --psf data/external/PSF.tif

# For mouse dataset, use specific PSF
python src/modeling/train.py UNet3D data/processed/mouse/training_data/regular_patches mouse \
    --augmentation OURS \
    --psf data/external/PSF_mouse.tif
```

### 3. Generate Predictions (`predict.py`)

Generate segmentation predictions using trained models:

```bash
python src/modeling/predict.py PATCHES_DIR COMPLETE_IMAGES_DIR DATASET_NAME MODELS_DIR [OPTIONS]

Required arguments:
- PATCHES_DIR: Directory containing image patches for deep learning methods
- COMPLETE_IMAGES_DIR: Directory containing complete images for classical methods
- DATASET_NAME: Identifier to organize different sets of images and select appropriate models
- MODELS_DIR: Directory containing trained models [default: models/]

Optional arguments:
- --output-dir: Directory to save predictions [default: reports/]
```

Example command:
```bash
python src/modeling/predict.py \
    data/processed/BC/test_data/reconstruction_patches \
    data/processed/BC/test_data \
    BC \
    models/
```

### 4. Generate Plots (`plots.py`)

Generate comparison plots and metrics between different methods:

```bash
python src/plots.py RECONSTRUCTION_PATCHES_DIR REGULAR_PATCHES_DIR COMPLETE_IMAGES_DIR DATASET_NAME MODELS_DIR [OPTIONS]

Required arguments:
- RECONSTRUCTION_PATCHES_DIR: Directory containing reconstruction patches (with overlap) for evaluating deep learning methods on complete images
- REGULAR_PATCHES_DIR: Directory containing regular patches (no overlap) for patch-level evaluation of all methods
- COMPLETE_IMAGES_DIR: Directory containing complete images for classical methods
- DATASET_NAME: Identifier to distinguish and organize different sets of images (only models trained on this dataset will be used)
- MODELS_DIR: Directory containing trained models [default: models/]

Optional arguments:
- --output-dir: Directory to save plots [default: reports/figures/]
```

The tool evaluates methods in two ways:
- For deep learning methods:
  - Complete image evaluation uses reconstruction patches (with overlap)
  - Patch-level evaluation uses regular patches (no overlap)
- For classical methods:
  - Complete image evaluation uses the complete images
  - Patch-level evaluation uses regular patches (no overlap)

Example command:
```bash
python src/plots.py \
    data/processed/BC/test_data/reconstruction_patches \
    data/processed/BC/test_data/regular_patches \
    data/processed/BC/test_data \
    BC \
    models/
```

## Configuration

All default parameters can be modified in `config.py`. Here are the key parameters you might want to adjust:

```python
# Patch generation settings
PATCH_SIZE = (64, 64, 64)  # Size of each 3D patch
PATCH_STEP = 64           # Step size between patches

# Training parameters
LEARNING_RATE = 1e-4      # Learning rate for training
BATCH_SIZE = 1           # Number of samples per training batch
NUM_EPOCHS = 50          # Number of training epochs
VALIDATION_SPLIT = 0.2   # Fraction of data used for validation

# Early stopping settings
EARLY_STOPPING_PATIENCE = 10  # Number of epochs to wait before stopping
EARLY_STOPPING_MIN_DELTA = 0  # Minimum change to qualify as improvement

# Model options
AVAILABLE_MODELS = ["UNet3D", "AttentionUNet3D"]

# Intensity augmentation settings
INTENSITY_PARAMS = {
    "background_level": 0.1,        # Background intensity
    "local_variation_scale": 5,     # Local intensity variations
    "z_decay_rate": 0.999,         # Z-axis intensity decay
    "noise_std": 0.1,              # Gaussian noise level
    "poisson_scale": 1.0,          # Poisson noise scaling
    "intensity_scale": 1000.0,      # Overall intensity scaling
    "snr_targets": [15, 10, 5, 4, 3, 2, 1]  # Target signal-to-noise ratios
}
```

## Common Issues and Solutions

1. If you get "out of memory" errors:
   - Reduce `BATCH_SIZE` in config.py
   - Try smaller `PATCH_SIZE`

2. If reconstruction looks incorrect:
   - Make sure you used padded patches (generated with `USE_PADDING=true`)
   - Verify patch size matches training patch size

3. For GPU-related errors:
   - Make sure your GPU has enough memory
   - Try reducing model size or batch size

## Directory Structure

Your dataset should follow this structure:
```
dataset/
├── images/
│   └── *.tif
└── masks/
    └── *.tif
```

## Available Datasets and PSF Files

The project includes several datasets for 3D microscopy image segmentation:

### Datasets
- `BC.zip`: Bile Canaliculi dataset
- `Sinusoids.zip`: Sinusoids dataset
- `Sinusoids_filled.zip`: Filled Sinusoids dataset
- `mouse.zip`: Mouse tissue dataset

### Point Spread Functions (PSF)
Two PSF files are provided for different datasets:

- `PSF.tif`: Use this PSF file for:
  - BC dataset
  - Sinusoids dataset
  - Sinusoids_filled dataset
- `PSF_mouse.tif`: Use this PSF file for:
  - Mouse dataset

When training models with microscopy-specific augmentation (using `--augmentation OURS`), make sure to use the correct PSF file for your dataset:

```bash
# For BC, Sinusoids, or Sinusoids_filled datasets
python src/modeling/train.py UNet3D /path/to/training_patches -a OURS --psf data/external/PSF.tif

# For Mouse dataset
python src/modeling/train.py UNet3D /path/to/training_patches -a OURS --psf data/external/PSF_mouse.tif
```
