from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer
from loguru import logger
from patchify import patchify

from src.config import FIGURES_DIR, BATCH_SIZE, PATCH_SIZE, PATCH_STEP
from src.processors import Metrics, Predictor, Visualizer
from src.utils import configure_gpu

# Configure GPU at startup
configure_gpu()

app = typer.Typer()


def evaluate_methods(patch_paths, patch_masks, complete_image_paths, complete_masks, deep_models):
    """Evaluate all methods on the patches and complete images.

    Args:
        patch_paths: List of paths to patch image files
        patch_masks: List of paths to patch mask files
        complete_image_paths: List of paths to complete image files
        complete_masks: List of paths to complete mask files
        deep_models: Dictionary of deep learning models

    Returns:
        DataFrame with results
    """
    # Initialize our classes
    metrics = Metrics()
    predictor = Predictor()
    all_results = []

    # Process deep learning models
    if deep_models:
        logger.info("Processing deep learning models...")
        for model_name, model in deep_models.items():
            for img_path, mask_path in tqdm(
                zip(patch_paths, patch_masks),
                total=len(patch_paths),
                desc=f"Processing {model_name}",
            ):
                try:
                    # Load and process single image
                    image = predictor.load_image(img_path)
                    mask = predictor.load_image(mask_path)

                    # Predict on single image
                    pred = predictor.predict_patch(image, model=model)

                    # Compute metrics
                    mask_binary = metrics.ensure_binary_mask(mask)
                    result = metrics.compute_metrics(mask_binary, pred)
                    result["method"] = f"Deep_{model_name}"
                    result["image_path"] = str(img_path)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {img_path} for {model_name}: {e}")
                    continue

    # Process classical methods on complete images
    logger.info("Processing classical methods on complete images...")
    for img_path, mask_path in tqdm(
        zip(complete_image_paths, complete_masks),
        total=len(complete_image_paths),
        desc="Processing classical methods",
    ):
        logger.info(f"Processing image: {img_path.name}")

        # Load image and mask
        image = predictor.load_image(img_path)
        mask = predictor.load_image(mask_path)
        mask_binary = metrics.ensure_binary_mask(mask)

        for method in predictor.classical_methods:
            logger.info(f"Applying {method} method")
            try:
                # Get prediction for the whole image
                pred = predictor.predict_patch(image, method=method)

                # Break down prediction and mask into patches for evaluation
                pred_patches = patchify(pred, PATCH_SIZE, PATCH_STEP)
                mask_patches = patchify(mask_binary, PATCH_SIZE, PATCH_STEP)

                # Reshape to match the evaluation format
                pred_patches = pred_patches.reshape(-1, *PATCH_SIZE)
                mask_patches = mask_patches.reshape(-1, *PATCH_SIZE)

                # Evaluate each patch
                for patch_idx in range(len(pred_patches)):
                    result = metrics.compute_metrics(
                        mask_patches[patch_idx], pred_patches[patch_idx]
                    )
                    result["method"] = f"Classical_{method}"
                    result["image_path"] = f"{str(img_path)}_patch_{patch_idx}"
                    all_results.append(result)

            except Exception as e:
                logger.error(f"Error processing {img_path.name} with {method}: {e}")
                continue

    return pd.DataFrame(all_results)


@app.command()
def main(
    patches_dir: Path = typer.Argument(
        ..., help="Directory containing patches for deep learning methods"
    ),
    complete_images_dir: Path = typer.Argument(
        ..., help="Directory containing complete images for classical methods"
    ),
    models_dir: Path = typer.Argument(..., help="Directory containing trained models"),
    output_dir: Path = typer.Option(FIGURES_DIR, help="Directory to save plots"),
):
    """Generate plots comparing classical and deep learning models."""

    # Initialize classes
    predictor = Predictor()
    visualizer = Visualizer(output_dir)

    # Get patch paths for deep learning
    logger.info("Getting patch file paths...")
    patch_paths = sorted(patches_dir.glob("images/**/*.tif"))
    patch_masks = sorted(patches_dir.glob("masks/**/*.tif"))

    if not patch_paths or not patch_masks:
        raise ValueError(f"No .tif files found in {patches_dir}/images/ or {patches_dir}/masks/")

    if len(patch_paths) != len(patch_masks):
        raise ValueError("Number of patch images does not match number of patch masks")

    logger.info(f"Found {len(patch_paths)} patch image-mask pairs")

    # Get complete image paths for classical methods
    logger.info("Getting complete image file paths...")
    complete_image_paths = sorted(complete_images_dir.glob("images/**/*.tif"))
    complete_masks = sorted(complete_images_dir.glob("masks/**/*.tif"))

    if not complete_image_paths or not complete_masks:
        raise ValueError(
            f"No .tif files found in {complete_images_dir}/images/ or {complete_images_dir}/masks/"
        )

    if len(complete_image_paths) != len(complete_masks):
        raise ValueError("Number of complete images does not match number of complete masks")

    logger.info(f"Found {len(complete_image_paths)} complete image-mask pairs")

    # Load deep learning models
    logger.info("Loading deep learning models...")
    deep_models = predictor.load_deep_models(models_dir)

    # Evaluate all methods
    logger.info("Evaluating methods...")
    results_df = evaluate_methods(
        patch_paths, patch_masks, complete_image_paths, complete_masks, deep_models
    )

    # List of metrics to plot
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "dice",
        "iou",
        "volume_similarity",
    ]

    # Generate violin plots
    logger.info("Generating violin plots...")
    visualizer.plot_violin(results_df, metrics)

    # Generate radar chart
    logger.info("Generating radar chart...")
    visualizer.plot_radar_chart(results_df)

    # Generate summary table
    logger.info("Generating summary table...")
    summary_df = visualizer.create_summary_table(results_df)
    summary_df.to_csv(output_dir / "metrics_summary.csv")

    logger.success("Plot generation complete!")


if __name__ == "__main__":
    app()
