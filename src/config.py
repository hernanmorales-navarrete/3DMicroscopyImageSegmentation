from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

ALLOWED_EXTENSIONS = (".tiff", ".tif")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
LOGS_DIR = PROJ_ROOT / "logs"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# VARIABLES

PATCH_SIZE = (64, 64, 64)
PATCH_STEP = 64  

MAX_WORKERS = 16 # Number of cores used in multiprocessing tasks
THRESHOLD = 0.5 #Thresholding for 

# Intensity augmentation parameters
INTENSITY_PARAMS = {
    "background_level": 0.1,  # Background intensity level
    "local_variation_scale": 5,  # Scale of local variations
    "z_decay_rate": 0.999,  # Rate of intensity decay along z-axis
    "noise_std": 0.1,  # Standard deviation for Gaussian noise
    "poisson_scale": 1.0,  # Scaling factor for Poisson noise
    "intensity_scale": 1000.0,  # Scale factor for image intensity before augmentation,
    "snr_tolerance": 0.1,  # Tolerance for SNR
    "max_iterations": 200,  # Maximum number of iterations for intensity augmentation
    "std_dev": 10,  # Standard deviation for Gaussian noise
    "snr_targets": [
        15,
        10,
        5,
        4,
        3,
        2,
    ],  # Target SNR values for augmentation
}

# Training Configuration
RANDOM_SEED = 42
AVAILABLE_MODELS = ["UNet3D", "AttentionUNet3D"]

# Training hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.2

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 5
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

#Available classical methods
CLASSICAL_METHODS = [
    "otsu",
    "adaptive_gaussian", 
    "adaptive_mean", 
    "frangi"
]

# Visualization settings
METHOD_ORDER = [
    "Classical_otsu",  # Otsu
    "Classical_adaptive_gaussian",  # Adaptive Gaussian
    "Classical_adaptive_mean",  # Adaptive Mean
    "Classical_frangi",  # Frangi
    "UNet3D_NONE",  # UNets without augmentation
    "UNet3D_STANDARD",  # UNets with standard augmentation
    "UNet3D_OURS",  # UNets with our augmentation
    "AttentionUNet3D_NONE",  # UNets+attention without augmentation
    "AttentionUNet3D_STANDARD",  # UNets+attention with standard augmentation
    "AttentionUNet3D_OURS",  # UNets+attention with our augmentation
]
