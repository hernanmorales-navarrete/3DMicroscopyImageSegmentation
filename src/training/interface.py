

import datetime
from pathlib import Path
from typing import Annotated

from loguru import logger
import models
import tensorflow
import typer

from src.config import (
    ALLOWED_EXTENSIONS,
    BATCH_SIZE,
    CHECKPOINT_MODE,
    CHECKPOINT_MONITOR,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    INTENSITY_PARAMS,
    LEARNING_RATE,
    LOGS_DIR,
    LOSS_FUNCTION,
    METRICS,
    MODELS_DIR,
    NUM_EPOCHS,
    RANDOM_SEED,
    SAVE_BEST_ONLY,
    TENSORBOARD_UPDATE_FREQ,
    VALIDATION_SPLIT,
)
from src.training.dataset import ImageDataset
from src.training.utils import (
    AugmentationType,
    configure_gpu,
    create_callbacks,
    get_model_from_class,
    read_images_from_dir_and_create_dataset,
    set_random_seed,
)

configure_gpu()
app = typer.Typer()

def interface(
    model_name: Annotated[str, typer.Argument(help="Name of the model")], 
    data_dir: Annotated[str, typer.Argument(help="Directory containing the dataset")], 
    dataset_name: Annotated[str, typer.Argument(help="Name of the dataset for model organization")], 
    augmentation: Annotated[str, typer.Option(AugmentationType.NONE, help="Type of augmentation to use. Available: NONE, STANDARD or OURS")],
    psf_path: Annotated[Path, typer.Option(
        None, 
        help="Path to PSF file for microscopy augmentations"
    )], 
    enable_reproducibility: Annotated[bool, typer.Option(True, help="Enable reproducibility")]
):
    #Enable reproducibility
    if enable_reproducibility: 
        logger.info(f"Setting random seed to {RANDOM_SEED}")
        set_random_seed(RANDOM_SEED)

    #Set PSF file if provided
    if psf_path and psf_path.exists(): 
        INTENSITY_PARAMS.update({"use_psf": True, "psf_path": str(psf_path)})
        logger.info(f"PSF file activated: {psf_path}")
    else: 
        INTENSITY_PARAMS.update({"use_psf": False})
        logger.info("No PSF file provided")

    #Create dataset
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = read_images_from_dir_and_create_dataset(data_dir, ALLOWED_EXTENSIONS, VALIDATION_SPLIT, RANDOM_SEED)

    train_dataset = ImageDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        batch_size=BATCH_SIZE,
        augmentation=augmentation.value,
        intensity_params=INTENSITY_PARAMS,
    )

    val_dataset = ImageDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        batch_size=BATCH_SIZE,
        augmentation="NONE",  # No augmentation for validation
    )

    logger.info(f"Creating {model_name}d model...")
    model_class = get_model_from_class(model_name, models)
    #Build model
    model = model_class().build_model()

    #Create optimizer
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    #Compile model
    model.compile(
        optimizer=optimizer, 
        loss=LOSS_FUNCTION, 
        metrics=METRICS
    )

    #Create directories for logs and checkpoints with dataset and augmentation info. We save the directory as BC/UNet3D_NONE/timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    particular_model_directory_name = f"{dataset_name}/{model_name}_{augmentation.value}"
    log_dir = LOGS_DIR / particular_model_directory_name / timestamp
    checkpoint_dir = MODELS_DIR / particular_model_directory_name / timestamp
    log_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    callbacks = create_callbacks(model_name, augmentation, dataset_name, log_dir, checkpoint_dir, TENSORBOARD_UPDATE_FREQ, CHECKPOINT_MONITOR, EARLY_STOPPING_MIN_DELTA, EARLY_STOPPING_PATIENCE, CHECKPOINT_MODE, SAVE_BEST_ONLY)

    logger.info("Starting training")
    model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=NUM_EPOCHS, 
        callbacks=callbacks, 
        verbose=1
    )

    logger.success("Training complete!")

if __name__ == '__main__': 
    typer.run(interface)