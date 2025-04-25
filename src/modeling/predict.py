from pathlib import Path
import re
import numpy as np
from loguru import logger
import typer
import tifffile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from patchify import unpatchify

from src.config import MODELS_DIR, REPORTS_DIR, PATCH_SIZE, PATCH_STEP
from src.plots import load_deep_models
from src.metrics import predict_patch

app = typer.Typer()


def extract_patch_info(filename):
    """Extract information from patch filename.

    Args:
        filename: Patch filename (e.g. 'image1_orig_512_512_128_pad_520_520_130_npatches_4_8_8_patch_0000.tif')

    Returns:
        tuple: (image_name, original_shape, padded_shape, n_patches)
    """
    # Extract information using regex
    pattern = r"(.+)_orig_(\d+)_(\d+)_(\d+)(?:_pad_(\d+)_(\d+)_(\d+))?_npatches_(\d+)_(\d+)_(\d+)_patch_\d+\.tif"
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


def process_single_patch(args):
    """Process a single patch for classical methods in parallel execution.

    Args:
        args: Tuple of (img_path, patch_idx, method)

    Returns:
        Tuple of (image_name, orig_shape, padded_shape, n_patches, patch_idx, prediction)
    """
    img_path, patch_idx, method = args

    # Extract patch information
    image_name, orig_shape, padded_shape, n_patches = extract_patch_info(img_path)

    # Read patch
    patch = tifffile.imread(str(img_path))

    # Get prediction using predict_patch (classical method only)
    pred = predict_patch(patch, model=None, method=method)

    return image_name, orig_shape, padded_shape, n_patches, patch_idx, pred


def predict_patches(image_paths, method="otsu", model=None):
    """Predict segmentation for all patches using specified method.

    Args:
        image_paths: List of patch image paths
        method: Classical method name or None for deep learning
        model: Deep learning model (if method is None)

    Returns:
        Dictionary mapping image names to (original_shape, padded_shape, n_patches, predictions) tuple
    """
    # Dictionary to store patches for each original image
    image_predictions = {}

    if model is not None:
        # Sequential processing for deep learning models
        logger.info("Using sequential processing for deep learning model")
        for img_path in image_paths:
            # Extract patch information and patch index
            image_name, orig_shape, padded_shape, n_patches = extract_patch_info(img_path)
            patch_idx = int(re.search(r"patch_(\d+)\.tif$", img_path.name).group(1))

            # Initialize predictions dictionary for this image if needed
            if image_name not in image_predictions:
                image_predictions[image_name] = (orig_shape, padded_shape, n_patches, [])

            # Read patch
            patch = tifffile.imread(str(img_path))

            # Get prediction using predict_patch
            pred = predict_patch(patch, model=model, method=method)

            # Store prediction
            image_predictions[image_name][3].append((patch_idx, pred))
    else:
        # Parallel processing for classical methods
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using {n_workers} workers for parallel processing with {method} method")

        # Prepare arguments for parallel processing
        process_args = []
        for img_path in image_paths:
            patch_idx = int(re.search(r"patch_(\d+)\.tif$", img_path.name).group(1))
            process_args.append((img_path, patch_idx, method))

        # Process patches in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all patches for processing
            futures = []
            for args in process_args:
                future = executor.submit(process_single_patch, args)
                futures.append(future)

            # Process results as they complete
            for future in futures:
                image_name, orig_shape, padded_shape, n_patches, patch_idx, pred = future.result()

                # Initialize predictions dictionary for this image if needed
                if image_name not in image_predictions:
                    image_predictions[image_name] = (orig_shape, padded_shape, n_patches, [])

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
        for image_name, (
            orig_shape,
            padded_shape,
            n_patches,
            patches_array,
        ) in predictions.items():
            # Reshape patches array to match unpatchify requirements (n_patches_z, n_patches_y, n_patches_x, *patch_size)
            patches_reshaped = patches_array.reshape(
                n_patches[0], n_patches[1], n_patches[2], *PATCH_SIZE
            )

            # Reconstruct full image using unpatchify with padded shape
            reconstructed = unpatchify(patches_reshaped, padded_shape)

            # Crop back to original size
            reconstructed = reconstructed[: orig_shape[0], : orig_shape[1], : orig_shape[2]]

            # Save reconstructed image
            output_path = output_dir / f"{image_name}_{method}.tif"
            tifffile.imwrite(str(output_path), reconstructed)

    # Process deep learning models
    for model_name, model in deep_models.items():
        logger.info(f"Processing deep learning model: {model_name}")
        predictions = predict_patches(image_paths, model=model)

        # Reconstruct and save each image
        for image_name, (
            orig_shape,
            padded_shape,
            n_patches,
            patches_array,
        ) in predictions.items():
            # Reshape patches array to match unpatchify requirements (n_patches_z, n_patches_y, n_patches_x, *patch_size)
            patches_reshaped = patches_array.reshape(
                n_patches[0], n_patches[1], n_patches[2], *PATCH_SIZE
            )

            # Reconstruct full image using unpatchify with padded shape
            reconstructed = unpatchify(patches_reshaped, padded_shape)

            # Crop back to original size
            reconstructed = reconstructed[: orig_shape[0], : orig_shape[1], : orig_shape[2]]

            # Save reconstructed image
            output_path = output_dir / f"{image_name}_{model_name}.tif"
            tifffile.imwrite(str(output_path), reconstructed)

    logger.success("Prediction and reconstruction complete!")


if __name__ == "__main__":
    app()
