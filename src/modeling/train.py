from pathlib import Path
import tensorflow as tf
import keras
from loguru import logger
import typer
from datetime import datetime
import inspect
from enum import Enum
import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    MODELS_DIR,
    LOGS_DIR,
    RANDOM_SEED,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    CHECKPOINT_MONITOR,
    CHECKPOINT_MODE,
    SAVE_BEST_ONLY,
    LOSS_FUNCTION,
    METRICS,
    TENSORBOARD_UPDATE_FREQ,
)
from src.data_loader import ImageDataset
import src.models as models_module


class AugmentationType(str, Enum):
    NONE = "NONE"
    STANDARD = "STANDARD"
    OURS = "OURS"


app = typer.Typer()


def set_random_seed():
    """Set random seed for reproducibility."""
    keras.utils.set_random_seed(RANDOM_SEED)
    tf.config.experimental.enable_op_determinism()


def get_available_models():
    """Dynamically get all model classes from models.py."""
    model_classes = {}
    for name, obj in inspect.getmembers(models_module):
        if inspect.isclass(obj) and hasattr(obj, "build_model"):
            model_classes[name] = obj
    return model_classes


def get_model_class(model_name):
    """Get model class by name."""
    model_classes = get_available_models()
    if model_name not in model_classes:
        available_models = list(model_classes.keys())
        raise ValueError(f"Model {model_name} not found. Available models: {available_models}")
    return model_classes[model_name]


def create_callbacks(model_name: str, augmentation: AugmentationType):
    """Create training callbacks."""
    callbacks = []

    # Create directories for logs and checkpoints with augmentation info
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir_name = f"{model_name}_{augmentation.value}"

    log_dir = LOGS_DIR / model_dir_name / timestamp
    checkpoint_dir = MODELS_DIR / model_dir_name / timestamp

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        update_freq=TENSORBOARD_UPDATE_FREQ,
    )
    callbacks.append(tensorboard_callback)

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=CHECKPOINT_MONITOR,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        patience=EARLY_STOPPING_PATIENCE,
        mode=CHECKPOINT_MODE,
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)

    # Model checkpoint callback
    checkpoint_path = checkpoint_dir / "{epoch:02d}_{val_loss:.4f}.h5"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint_path),
        monitor=CHECKPOINT_MONITOR,
        save_best_only=SAVE_BEST_ONLY,
        mode=CHECKPOINT_MODE,
    )
    callbacks.append(checkpoint)

    return callbacks


def load_and_split_data(data_dir: Path, validation_split: float = 0.2, random_state: int = None):
    """Load and split data paths into train and validation sets.

    Args:
        data_dir: Directory containing the dataset
        validation_split: Fraction of data to use for validation
        random_state: Random state for reproducibility

    Returns:
        Tuple of (train_image_paths, val_image_paths, train_mask_paths, val_mask_paths)
    """
    # Get all file paths
    image_paths = sorted(tf.io.gfile.glob(os.path.join(data_dir, "images/**/*.tif")))
    mask_paths = sorted(tf.io.gfile.glob(os.path.join(data_dir, "masks/**/*.tif")))

    if not image_paths or not mask_paths:
        raise ValueError(
            f"No .tif files found in {data_dir}/images/ or {data_dir}/masks/ subdirectories"
        )

    if len(image_paths) != len(mask_paths):
        raise ValueError("Number of images does not match number of masks")

    # Split the data
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=validation_split, random_state=random_state
    )

    logger.info(
        f"Found {len(train_image_paths)} training samples and {len(val_image_paths)} validation samples"
    )

    return train_image_paths, val_image_paths, train_mask_paths, val_mask_paths


@app.command()
def main(
    model_name: str = typer.Argument(..., help="Name of the model to train"),
    data_dir: Path = typer.Argument(..., help="Directory containing the dataset"),
    validation_split: float = typer.Option(0.2, help="Fraction of data to use for validation"),
    augmentation: AugmentationType = typer.Option(
        AugmentationType.NONE,
        "--augmentation",
        "-a",
        help="Type of augmentation to use",
    ),
    enable_reproducibility: bool = typer.Option(
        False, help="Enable reproducibility by setting random seeds"
    ),
):
    """Train a 3D segmentation model with optional augmentation."""

    if enable_reproducibility:
        logger.info(f"Setting random seed to {RANDOM_SEED}")
        set_random_seed()

    logger.info(f"Using {augmentation.value} augmentation")

    # Load and split the data
    logger.info(f"Loading and splitting data from {data_dir}")
    random_state = RANDOM_SEED if enable_reproducibility else None
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = load_and_split_data(
        data_dir, validation_split=validation_split, random_state=random_state
    )

    logger.info("Creating datasets...")
    train_dataset = ImageDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        batch_size=BATCH_SIZE,
        augmentation=augmentation.value,
    )

    val_dataset = ImageDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        batch_size=BATCH_SIZE,
        augmentation="NONE",  # No augmentation for validation
    )

    logger.info(f"Creating {model_name} model...")
    model_class = get_model_class(model_name)
    model = model_class().build_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=LOSS_FUNCTION,
        metrics=METRICS,
    )

    callbacks = create_callbacks(model_name, augmentation)

    logger.info("Starting training...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    logger.success("Training complete!")


if __name__ == "__main__":
    app()
