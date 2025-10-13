import datetime
from enum import Enum
import inspect
import os
from pathlib import Path
import types
from typing import Tuple

import keras
from loguru import logger
from sklearn.model_selection import train_test_split
import tensorflow


class AugmentationType(Enum):
    "Enumeration class for augmentation options"
    NONE = "NONE"
    STANDARD = "STANDARD"
    OURS = "OURS"

def configure_gpu():
    """Configure GPU memory growth to avoid taking all memory.

    This should be called at the beginning of scripts that use GPU.
    """
    gpus = tensorflow.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tensorflow.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tensorflow.config.list_logical_devices("GPU")
            logger.info(f"Using {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.error(e)
    else:
        logger.warning("No GPUs found. Running on CPU only.")

def set_random_seed(seed: int):
    keras.utils.set_random_seed(seed)
    tensorflow.config.experimental.enable_op_determinism()

def get_available_models(module: types.ModuleType): 
    "Get models from a module. A model should be a class and have the method attribute build_model"
    model_classes = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and hasattr(obj, "build_model"):
            model_classes[name] = obj
    return model_classes

def get_model_from_class(model_name: str, module: types.ModuleType):
    "Get a model class from a module"
    model_classes = get_available_models(module)
    if model_name not in model_classes:
        available_models = list(model_classes.keys())
        raise ValueError(f"Model {model_name} not found. Available models: {available_models}")
    return model_classes[model_name]

def read_images_from_dir_and_create_dataset(data_dir: Path, allowed_extensions: Tuple[str, str], validation_split: int, random_state: int = None):
    image_paths = []
    mask_paths = []

    #Get all file paths from allowed extensions
    for ext in allowed_extensions: 
        image_paths.extend(tensorflow.io.gfile.glob(os.path.join(data_dir, f"images/**/*{ext}")))
        mask_paths.extend(tensorflow.io.gfile.glob(os.path.join(data_dir, f"images/**/*{ext}")))

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

def create_callbacks(model_name: str, augmentation: AugmentationType, dataset_name: str, dir_to_save_logs: Path, dir_to_save_checkpoints: Path, tensorboard_update_freq: str, checkpoint_monitor: str, early_stopping_min_delta: int, early_stopping_patience: int, checkpoint_mode: str, save_best_only: bool):
    callbacks = []

    # Create directories for logs and checkpoints with dataset and augmentation info
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir_name = f"{dataset_name}/{model_name}_{augmentation.value}"

    log_dir = dir_to_save_logs / model_dir_name / timestamp
    checkpoint_dir = dir_to_save_checkpoints / model_dir_name / timestamp

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard callback
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        update_freq=tensorboard_update_freq,
    )
    callbacks.append(tensorboard_callback)

    # Early stopping callback
    early_stopping = tensorflow.keras.callbacks.EarlyStopping(
        monitor=checkpoint_monitor,
        min_delta=early_stopping_min_delta,
        patience=early_stopping_patience,
        mode=checkpoint_mode,
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)

    # Model checkpoint callback
    checkpoint_path = checkpoint_dir / "best_model.h5"

    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(
        str(checkpoint_path),
        monitor=checkpoint_monitor,
        save_best_only=save_best_only,
        mode=checkpoint_mode,
    )
    callbacks.append(checkpoint)

    return callbacks
