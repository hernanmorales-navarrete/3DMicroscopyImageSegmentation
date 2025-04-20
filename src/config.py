from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import tensorflow as tf

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
LOGS_DIR = MODELS_DIR / "logs"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# VARIABLES

PATCH_SIZE = (64, 64, 64)
PATCH_BATCH = 1

# Augmentation flags
STANDARD_AUGMENTATION = True
OURS_AUGMENTATION = False

# Intensity augmentation parameters
INTENSITY_PARAMS = {
    "background_level": 0.1,  # Background intensity level
    "local_variation_scale": 5,  # Scale of local variations
    "z_decay_rate": 0.999,  # Rate of intensity decay along z-axis
    "noise_std": 0.1,  # Standard deviation for Gaussian noise
    "poisson_scale": 1.0,  # Scaling factor for Poisson noise
    "use_psf": False,  # Whether to apply PSF convolution
    "psf_path": None,  # Path to PSF file (required if use_psf is True)
    "intensity_scale": 1000.0,  # Scale factor for image intensity before augmentation
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# Training Configuration
RANDOM_SEED = 42
AVAILABLE_MODELS = ["UNet3D", "AttentionUNet3D"]

# Training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32  # Default Keras batch size
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.2

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0  # Use Keras default

# Model checkpoint parameters
CHECKPOINT_MONITOR = "val_loss"
CHECKPOINT_MODE = "min"
SAVE_BEST_ONLY = True
CHECKPOINT_SAVE_FREQ = "epoch"
CHECKPOINT_SAVE_WEIGHTS_ONLY = False
CHECKPOINT_SAVE_FORMAT = "h5"  # or 'tf' for SavedModel format

# Loss and metrics
LOSS_FUNCTION = "binary_crossentropy"
METRICS = ["accuracy"]

# Tensorboard
TENSORBOARD_UPDATE_FREQ = "epoch"
PROFILE_BATCH = 0  # Disable profiling
