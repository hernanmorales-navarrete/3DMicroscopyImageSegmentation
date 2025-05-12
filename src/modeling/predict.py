from pathlib import Path
import re
import numpy as np
from loguru import logger
import typer
from patchify import unpatchify
import tensorflow as tf

from src.config import MODELS_DIR, REPORTS_DIR, PATCH_SIZE, PATCH_STEP
from src.processors import Predictor
from src.utils import configure_gpu

# Configure GPU at startup
configure_gpu()

app = typer.Typer()


def extract_patch_info(filename):
    """Extract information from patch filename.

    Args:
        filename: Patch filename (e.g. 'image1_orig_512_512_128_pad_520_520_130_npatches_4_8_8_patch_0000.tif')

    Returns:
        tuple: (image_name, original_shape, padded_shape, n_patches)
    """
    # Extract information using regex
    pattern = r"(.+)_orig_(\d+)_(\d+)_(\d+)(?:_pad_(\d+)_(\d+)_(\d+))?_npatches_(\d+)_(\d+)_(\d+)_patch_\d+\.tiff?"
    match = re.match(pattern, filename.name)

    if not match:
        raise ValueError(f"Invalid patch filename format: {filename}")

    image_name = match.group(1)
    orig_shape = (int(match.group(2)), int(match.group(3)), int(match.group(4)))

    # Get padded shape if it exists, otherwise use original shape
    if match.group(5):
        padded_shape = (int(match.group(5)), int(match.group(6)), int(match.group(7)))
        n_patches = (int(match.group(8)), int(match.group(9)), int(match.group(10)))
    else:
        padded_shape = orig_shape
        n_patches = (int(match.group(8)), int(match.group(9)), int(match.group(10)))

    return image_name, orig_shape, padded_shape, n_patches


def predict_patches(image_paths, predictor, model_path):
    """Predict segmentation for all patches using deep learning model.

    Args:
        image_paths: List of patch image paths
        predictor: Predictor instance
        model_path: Path to the deep learning model

    Returns:
        Dictionary mapping image names to (original_shape, padded_shape, n_patches, predictions) tuple
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Dictionary to store patches for each original image
    image_predictions = {}

    # Sequential processing for deep learning models
    logger.info("Processing patches with deep learning model")
    for img_path in image_paths:
        # Extract patch information and patch index
        image_name, orig_shape, padded_shape, n_patches = extract_patch_info(img_path)
        patch_idx = int(re.search(r"patch_(\d+)\.tiff?$", img_path.name).group(1))

        # Initialize predictions dictionary for this image if needed
        if image_name not in image_predictions:
            image_predictions[image_name] = (orig_shape, padded_shape, n_patches, [])

        # Read and predict patch
        patch = predictor.load_image(img_path)
        pred = predictor.predict_patch(patch, model=model)

        # Store prediction
        image_predictions[image_name][3].append((patch_idx, pred))

    # Sort predictions by patch index for each image
    for image_name in image_predictions:
        image_predictions[image_name][3].sort(key=lambda x: x[0])
        # Convert to just the predictions array
        image_predictions[image_name] = (
            image_predictions[image_name][0],  # orig_shape
            image_predictions[image_name][1],  # padded_shape
            image_predictions[image_name][2],  # n_patches
            np.array([p[1] for p in image_predictions[image_name][3]]),  # predictions array
        )

    return image_predictions


@app.command()
def main(
    patches_dir: Path = typer.Argument(
        ..., help="Directory containing image patches for deep learning methods"
    ),
    complete_images_dir: Path = typer.Argument(
        ..., help="Directory containing complete images for classical methods"
    ),
    dataset_name: str = typer.Argument(
        ...,
        help="Identifier/name to distinguish and organize different sets of images - all predictions will be saved in a subdirectory with this name. Only models trained on this dataset will be used.",
    ),
    models_dir: Path = typer.Argument(MODELS_DIR, help="Directory containing trained models"),
    output_dir: Path = typer.Option(REPORTS_DIR, help="Directory to save predictions"),
):
    """Generate predictions using classical methods on complete images and then deep learning on patches."""

    # Create output directory
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize predictor
    predictor = Predictor()

    # Process classical methods with complete images first
    complete_image_paths = sorted(complete_images_dir.glob("images/**/*.tif*"))
    if not complete_image_paths:
        raise ValueError(f"No .tif or .tiff files found in {complete_images_dir}/images/")

    logger.info(f"Found {len(complete_image_paths)} complete images for classical methods")

    # Process each complete image with classical methods
    for image_path in complete_image_paths:
        image_name = image_path.stem
        logger.info(f"Processing image: {image_name}")

        # Load image
        image = predictor.load_image(image_path)

        for method in predictor.classical_methods:
            logger.info(f"Applying {method} method")
            prediction = predictor.predict_patch(image, method=method)

            # Save prediction without augmentation type for classical methods
            output_path = dataset_output_dir / f"{image_name}_{method}.tif"
            predictor.save_image(prediction, output_path)

    # Then process deep learning methods with patches
    patch_paths = sorted(patches_dir.glob("images/**/*.tif*"))
    if not patch_paths:
        raise ValueError(f"No .tif or .tiff files found in {patches_dir}/images/")

    logger.info(f"Found {len(patch_paths)} patches for deep learning")

    # Load and apply deep learning models
    deep_models = predictor.load_deep_models(models_dir, dataset_name=dataset_name)

    for model_name, (model_path, augmentation_type) in deep_models.items():
        logger.info(f"Processing deep learning model: {model_name} ({augmentation_type})")
        predictions = predict_patches(patch_paths, predictor, model_path)

        # Reconstruct and save each image
        for image_name, (
            orig_shape,
            padded_shape,
            n_patches,
            patches_array,
        ) in predictions.items():
            patches_reshaped = patches_array.reshape(
                n_patches[0], n_patches[1], n_patches[2], *PATCH_SIZE
            )

            # Reconstruct full image using unpatchify with padded shape
            reconstructed = unpatchify(patches_reshaped, padded_shape)

            # Crop back to original size
            padding = [
                        ( (p - o) // 2, p - o - ( (p - o) // 2 ) )
                        for p, o in zip(padded_shape, orig_shape)
                      ]
            reconstructed = reconstructed[
                                padding[0][0] : orig_shape[0]+padding[0][0],
                                padding[1][0] : orig_shape[1]+padding[1][0],
                                padding[2][0] : orig_shape[1]+padding[2][0]
                            ]  
        
            # Save reconstructed image with augmentation type only for deep learning
            output_path = dataset_output_dir / f"{image_name}_{model_name}_{augmentation_type}.tif"
            predictor.save_image(reconstructed, output_path)

    logger.success(
        f"Prediction and reconstruction complete! Results saved in {dataset_output_dir}"
    )


if __name__ == "__main__":
    app()
