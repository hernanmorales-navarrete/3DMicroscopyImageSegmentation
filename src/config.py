from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

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
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
