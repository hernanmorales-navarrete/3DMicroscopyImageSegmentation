# 3D Microscopy Image Segmentation

This software provides tools for training and evaluating 3D microscopy image segmentation models using both classical and deep learning approaches.

## Command Line Interface (CLI) Guide

This guide explains how to use each command-line tool in detail. Before running any command, make sure you're in the project's root directory.

### Note About Boolean Options

Throughout this CLI, boolean options follow standard conventions:
- Use `--flag` to enable a feature (sets it to true)
- Use `--no-flag` to disable a feature (sets it to false)
- If neither is specified, the default value is used

### 1. Generate Patches (`generate_patches.py`)

This tool splits your 3D microscopy images into smaller patches for processing.

```bash
python src/generate_patches.py PATH_TO_DATASET [OPTIONS]

Required arguments:
- PATH_TO_DATASET: Full path to your dataset directory

Optional arguments:
- --padding/--no-padding: Whether to use padding (default: --no-padding)
```

**Important Note About Padding:**
- For training: Use `--no-padding` (no padding needed)
- For prediction/reconstruction: Use `--padding` (padding required for proper reconstruction)

Example commands:
```bash
# For training data
python src/generate_patches.py /home/user/microscopy_data --no-padding

# For prediction data
python src/generate_patches.py /home/user/microscopy_data --padding
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
python src/modeling/train.py MODEL_NAME DATA_DIR [OPTIONS]

Required arguments:
- MODEL_NAME: Name of the model to use (UNet3D or AttentionUNet3D)
- DATA_DIR: Path to directory containing training patches

Optional arguments:
- --augmentation, -a: Type of data augmentation
  - NONE: No augmentation (default)
  - STANDARD: Basic augmentation
  - OURS: Advanced microscopy-specific augmentation
- --psf PATH: Path to Point Spread Function file for microscopy augmentation
```

Example commands:
```bash
# Basic training without augmentation
python src/modeling/train.py UNet3D /path/to/training_patches

# Training with advanced augmentation and PSF
python src/modeling/train.py UNet3D /path/to/training_patches -a OURS --psf /path/to/psf.tif
```

### 3. Generate Predictions (`predict.py`)

Generate segmentation predictions using trained models:

```bash
python src/modeling/predict.py PATCHES_DIR COMPLETE_IMAGES_DIR MODELS_DIR [OUTPUT_DIR]

Required arguments:
- PATCHES_DIR: Directory containing padded patches for deep learning
- COMPLETE_IMAGES_DIR: Directory containing full images for classical methods
- MODELS_DIR: Directory containing trained models

Optional arguments:
- --output_dir: Directory to save predictions (default: reports/)
```

Example command:
```bash
python src/modeling/predict.py /path/to/padded_patches /path/to/complete_images /path/to/models --output_dir /path/to/predictions
```

### 4. Generate Plots (`plots.py`)

Generate comparison plots and metrics between different methods:

```bash
python src/plots.py PATCHES_DIR COMPLETE_IMAGES_DIR MODELS_DIR [OUTPUT_DIR]

Required arguments:
- PATCHES_DIR: Directory containing patches with ground truth
- COMPLETE_IMAGES_DIR: Directory containing complete images
- MODELS_DIR: Directory containing trained models

Optional arguments:
- --output_dir: Directory to save plots (default: reports/figures/)
```

Example command:
```bash
python src/plots.py /path/to/patches /path/to/complete_images /path/to/models --output_dir /path/to/plots
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