from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer
from loguru import logger
from patchify import patchify
import tensorflow as tf

from src.config import FIGURES_DIR, BATCH_SIZE, PATCH_SIZE, PATCH_STEP, MODELS_DIR
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
        deep_models: Dictionary mapping model names to tuples of (model_path, augmentation_type)

    Returns:
        DataFrame with results
    """
    # Initialize our classes
    metrics = Metrics()
    predictor = Predictor()
    all_results = []
    logger.info(f"Deep models: {deep_models}")
    # Process deep learning models
    if deep_models:
        logger.info("Processing deep learning models...")
        for model_name, (model_path, augmentation_type) in deep_models.items():
            # Load all images and masks for batch processing
            images = []
            masks = []
            for img_path, mask_path in tqdm(
                zip(patch_paths, patch_masks),
                total=len(patch_paths),
                desc=f"Loading data for {model_name} ({augmentation_type})",
            ):
                try:
                    images.append(predictor.load_image(img_path))
                    masks.append(predictor.load_image(mask_path))
                except Exception as e:
                    logger.error(f"Error loading {img_path}: {e}")
                    continue

            try:
                # Load model only when needed
                logger.info(f"Loading model {model_name}")
                model = tf.keras.models.load_model(model_path)

                # Process all patches in batches
                logger.info(f"Running batch predictions for {model_name}")
                predictions = predictor.predict_batch_patches(
                    images, model=model, batch_size=BATCH_SIZE
                )

                # Clear model from GPU memory
                del model
                tf.keras.backend.clear_session()

                # Evaluate predictions
                for img_path, mask, pred in zip(patch_paths, masks, predictions):
                    try:
                        mask_binary = metrics.ensure_binary_mask(mask)
                        result = metrics.compute_metrics(mask_binary, pred)
                        result["method"] = (
                            model_name  # The model_name already includes augmentation type
                        )
                        result["augmentation"] = augmentation_type
                        result["image_path"] = str(img_path)
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error evaluating {img_path} for {model_name}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error processing model {model_name}: {e}")
                # Ensure model is cleared even if an error occurs
                if "model" in locals():
                    del model
                    tf.keras.backend.clear_session()
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
                    result["augmentation"] = "Classical"  # Mark as classical method
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
    dataset_name: str = typer.Argument(
        ...,
        help="Identifier/name to distinguish and organize different sets of images",
    ),
    models_dir: Path = typer.Argument(MODELS_DIR, help="Directory containing trained models"),
    use_matching_models: bool = typer.Option(
        True,
        help="If True, only use models trained on the same dataset. If False, use all available models.",
    ),
    output_dir: Path = typer.Option(FIGURES_DIR, help="Directory to save plots"),
):
    """Generate plots comparing classical and deep learning models."""

    # Create output directory
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

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
    dataset_filter = dataset_name if use_matching_models else None
    deep_models = predictor.load_deep_models(models_dir, dataset_name=dataset_filter)

    if not deep_models:
        logger.warning(
            f"No {'matching ' if use_matching_models else ''}deep learning models found in {models_dir}"
        )

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

    # Generate plots combining all methods
    logger.info("Generating combined plots for all methods...")
    visualizer.plot_violin(results_df, metrics, f"{dataset_name}_combined")
    visualizer.plot_radar_chart(results_df, f"{dataset_name}_combined")

    # Generate summary table
    logger.info("Generating summary table...")
    summary_df = visualizer.create_summary_table(results_df)
    summary_df.to_csv(dataset_output_dir / f"metrics_summary_{dataset_name}_combined.csv")

    logger.success("Plot generation complete!")


if __name__ == "__main__":
    app()
