from pathlib import Path
from typing import Annotated

from loguru import logger
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
import src.training.models as models_module
from src.training.utils import (
    configure_gpu,
    create_callbacks,
    get_model_from_class,
    read_images_from_dir_and_create_dataset,
    set_random_seed,
)

configure_gpu()


def interface(
    model_name: Annotated[str, typer.Argument(help="Name of the model")],
    data_dir: Annotated[str, typer.Argument(help="Directory containing the dataset")],
    dataset_name: Annotated[
        str, typer.Argument(help="Name of the dataset for model organization")
    ],
    augmentation: Annotated[
        str, typer.Argument(help="Type of augmentation to use. Available: NONE, STANDARD or OURS")
    ],
    psf_path: Annotated[
        Path, typer.Argument(help="Path to PSF file for microscopy augmentations")
    ],
    enable_reproducibility: Annotated[bool, typer.Option(help="Enable reproducibility")] = True,
):
    # Enable reproducibility
    if enable_reproducibility:
        logger.info(f"Setting random seed to {RANDOM_SEED}")
        set_random_seed(RANDOM_SEED)

    # Set PSF file if provided
    if psf_path and psf_path.exists():
        INTENSITY_PARAMS.update({"use_psf": True, "psf_path": str(psf_path)})
        logger.info(f"PSF file activated: {psf_path}")
    else:
        INTENSITY_PARAMS.update({"use_psf": False})
        logger.info("No PSF file provided")

    # Create dataset
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = (
        read_images_from_dir_and_create_dataset(
            data_dir, ALLOWED_EXTENSIONS, VALIDATION_SPLIT, RANDOM_SEED
        )
    )

    train_dataset = ImageDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        batch_size=BATCH_SIZE,
        augmentation=augmentation,
        intensity_params=INTENSITY_PARAMS,
    )

    val_dataset = ImageDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        batch_size=BATCH_SIZE,
        augmentation="NONE",  # No augmentation for validation
    )

    logger.info(f"Creating {model_name}d model...")
    model_class = get_model_from_class(model_name, models_module)
    # Build model
    model = model_class().build_model()

    # Create optimizer
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Compile model
    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION, metrics=METRICS)

    callbacks = create_callbacks(
        model_name,
        augmentation,
        dataset_name,
        LOGS_DIR,
        MODELS_DIR,
        TENSORBOARD_UPDATE_FREQ,
        CHECKPOINT_MONITOR,
        EARLY_STOPPING_MIN_DELTA,
        EARLY_STOPPING_PATIENCE,
        CHECKPOINT_MODE,
        SAVE_BEST_ONLY,
    )

    logger.info("Starting training")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    logger.success("Training complete!")


if __name__ == "__main__":
    typer.run(interface)
