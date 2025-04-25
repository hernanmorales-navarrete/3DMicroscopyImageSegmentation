from pathlib import Path
import re
import numpy as np
from loguru import logger
import typer
import tifffile
from collections import defaultdict

from src.config import MODELS_DIR, REPORTS_DIR, PATCH_SIZE, PATCH_STEP
from src.plots import load_deep_models
from src.metrics import predict_patch

app = typer.Typer()


def extract_patch_info(filename):
    """Extract information from patch filename.

    Args:
        filename: Patch filename (e.g. 'image1_512_512_128_z50_y100_x150.tif')

    Returns:
        tuple: (image_name, original_shape, (z, y, x) position)
    """
    # Extract information using regex
    pattern = r"(.+)_(\d+)_(\d+)_(\d+)_z(\d+)_y(\d+)_x(\d+)\.tif"
    match = re.match(pattern, filename.name)

    if not match:
        raise ValueError(f"Invalid patch filename format: {filename}")

    image_name = match.group(1)
    orig_shape = (int(match.group(2)), int(match.group(3)), int(match.group(4)))
    position = (int(match.group(5)), int(match.group(6)), int(match.group(7)))

    return image_name, orig_shape, position


def calculate_padded_shape(original_shape):
    """Calculate shape that will be evenly divisible by step size.

    Args:
        original_shape: Original image shape (z, y, x)

    Returns:
        tuple: Shape padded to next size that's compatible with patch size and step
    """
    padded_shape = []
    for dim, patch_size in zip(original_shape, PATCH_SIZE):
        # If dimension is not compatible with patch size and step
        if (dim - patch_size) % PATCH_STEP != 0:
            # Calculate next compatible size
            n_steps = (dim - patch_size + PATCH_STEP) // PATCH_STEP
            new_dim = patch_size + n_steps * PATCH_STEP
            padded_shape.append(new_dim)
        else:
            padded_shape.append(dim)

    return tuple(padded_shape)


def reconstruct_image(patches_dict, original_shape):
    """Reconstruct full image from patches.

    Args:
        patches_dict: Dictionary mapping (z,y,x) position to patch data
        original_shape: Shape of the original image

    Returns:
        Reconstructed image array
    """
    # Calculate padded shape that will be compatible with patch size and step
    padded_shape = calculate_padded_shape(original_shape)

    # Create padded output array
    reconstructed = np.zeros(padded_shape, dtype=np.uint8)

    # Place each patch in its position
    for position, patch in patches_dict.items():
        z, y, x = position
        reconstructed[z : z + PATCH_SIZE[0], y : y + PATCH_SIZE[1], x : x + PATCH_SIZE[2]] = patch

    # Cut back to original shape
    return reconstructed[: original_shape[0], : original_shape[1], : original_shape[2]]


def predict_patches(image_paths, method="otsu", model=None):
    """Predict segmentation for all patches using specified method.

    Args:
        image_paths: List of patch image paths
        method: Classical method name or None for deep learning
        model: Deep learning model (if method is None)

    Returns:
        Dictionary mapping image names to (original_shape, predictions) tuple
    """
    # Dictionary to store patches for each original image
    image_predictions = {}

    for img_path in image_paths:
        # Extract patch information
        image_name, orig_shape, position = extract_patch_info(img_path)

        # Initialize predictions dictionary for this image if needed
        if image_name not in image_predictions:
            image_predictions[image_name] = (orig_shape, {})

        # Read patch
        patch = tifffile.imread(str(img_path))

        # Get prediction using predict_patch
        pred = predict_patch(patch, model=model, method=method)

        # Store prediction
        image_predictions[image_name][1][position] = pred

    return image_predictions


@app.command()
def main(
    test_dir: Path = typer.Argument(..., help="Directory containing test patches"),
    models_dir: Path = typer.Argument(MODELS_DIR, help="Directory containing trained models"),
    output_dir: Path = typer.Option(REPORTS_DIR, help="Directory to save predictions"),
):
    """Generate predictions and reconstruct full images from patches."""

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image paths
    logger.info("Getting file paths...")
    image_paths = sorted(test_dir.glob("images/**/*.tif"))

    if not image_paths:
        raise ValueError(f"No .tif files found in {test_dir}/images/")

    logger.info(f"Found {len(image_paths)} patches")

    # Load deep learning models
    logger.info("Loading deep learning models...")
    deep_models = load_deep_models(models_dir)

    # Classical methods to try
    classical_methods = ["binary", "otsu", "adaptive_mean", "adaptive_gaussian", "frangi"]

    # Process classical methods
    for method in classical_methods:
        logger.info(f"Processing classical method: {method}")
        predictions = predict_patches(image_paths, method=method)

        # Reconstruct and save each image
        for image_name, (orig_shape, patches) in predictions.items():
            # Reconstruct full image
            full_image = reconstruct_image(patches, orig_shape)

            # Save reconstructed image
            output_path = output_dir / f"{image_name}_{method}.tif"
            tifffile.imwrite(str(output_path), full_image)

    # Process deep learning models
    for model_name, model in deep_models.items():
        logger.info(f"Processing deep learning model: {model_name}")
        predictions = predict_patches(image_paths, model=model)

        # Reconstruct and save each image
        for image_name, (orig_shape, patches) in predictions.items():
            # Reconstruct full image
            full_image = reconstruct_image(patches, orig_shape)

            # Save reconstructed image
            output_path = output_dir / f"{image_name}_{model_name}.tif"
            tifffile.imwrite(str(output_path), full_image)

    logger.success("Prediction and reconstruction complete!")


if __name__ == "__main__":
    app()
