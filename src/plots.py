from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import typer
import tensorflow as tf
from loguru import logger
import tifffile

from src.utils import configure_gpu
from src.config import FIGURES_DIR, BATCH_SIZE
from src.metrics import evaluate_patch, compute_metrics

app = typer.Typer()

# Configure GPU at startup
configure_gpu()


def load_deep_models(models_path):
    """Load all deep learning models from the models directory.

    Args:
        models_path: Path to models directory

    Returns:
        Dictionary of model name to model object
    """
    models = {}
    for model_dir in models_path.glob("*"):
        if not model_dir.is_dir():
            continue

        # Get latest timestamp directory
        latest_model = sorted(model_dir.glob("*"))[-1]

        # Get model file (*.h5)
        model_file = sorted(latest_model.glob("*.h5"))[-1]

        # Load model
        model = tf.keras.models.load_model(str(model_file))
        models[model_dir.name] = model

    return models


def evaluate_complete_image(image_path, mask_path, method):
    """Evaluate a single complete image with a classical method.

    Args:
        image_path: Path to image file
        mask_path: Path to mask file
        method: Classical method to use

    Returns:
        Dictionary with metrics
    """
    # Read image and mask
    image = tifffile.imread(str(image_path))
    mask = tifffile.imread(str(mask_path))

    # Evaluate using classical method
    metrics = evaluate_patch(image, mask, method=method)
    metrics["method"] = f"Classical_{method}"
    metrics["image_path"] = str(image_path)

    return metrics


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
    all_results = []

    # Process deep learning models in batches
    if deep_models:
        logger.info("Processing deep learning models...")
        n_samples = len(patch_paths)

        for model_name, model in deep_models.items():
            for i in tqdm(range(0, n_samples, BATCH_SIZE), desc=f"Processing {model_name}"):
                batch_slice = slice(i, min(i + BATCH_SIZE, n_samples))
                batch_images = [tifffile.imread(str(p)) for p in patch_paths[batch_slice]]
                batch_masks = [tifffile.imread(str(p)) for p in patch_masks[batch_slice]]

                try:
                    # Stack images into a batch
                    batch_data = np.stack(batch_images)[..., np.newaxis]
                    # Predict on batch
                    batch_preds = model.predict(batch_data, verbose=0)
                    batch_preds = (batch_preds > 0.5).astype(np.uint8)

                    # Evaluate each prediction in the batch
                    for j, (pred, mask, img_path) in enumerate(
                        zip(batch_preds, batch_masks, patch_paths[batch_slice])
                    ):
                        # Ensure both pred and mask are binary (0 or 1)
                        pred_binary = (pred[..., 0] > 0).astype(np.uint8)
                        mask_binary = (mask > 0).astype(np.uint8)
                        metrics = compute_metrics(mask_binary, pred_binary)
                        metrics["method"] = f"Deep_{model_name}"
                        metrics["image_path"] = str(img_path)
                        all_results.append(metrics)
                except Exception as e:
                    logger.error(f"Error processing batch for {model_name}: {e}")
                    continue

    # Process classical methods on complete images
    classical_methods = ["binary", "otsu", "adaptive_mean", "adaptive_gaussian", "frangi"]

    logger.info("Processing classical methods on complete images...")
    for img_path, mask_path in tqdm(
        zip(complete_image_paths, complete_masks),
        total=len(complete_image_paths),
        desc="Processing images",
    ):
        logger.info(f"Processing image: {img_path.name}")
        for method in classical_methods:
            logger.info(f"Applying {method} method")
            try:
                result = evaluate_complete_image(img_path, mask_path, method)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {img_path.name} with {method}: {e}")
                continue

    return pd.DataFrame(all_results)


def plot_violin(df, metrics, output_path):
    """Create violin plots for metrics.

    Args:
        df: DataFrame with results
        metrics: List of metrics to plot
        output_path: Path to save plots
    """
    # Set style
    sns.set_style("whitegrid")

    # Calculate number of rows and columns for subplots
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.violinplot(data=df, x="method", y=metric, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")

    # Remove empty subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_path / "metrics_violin_plots.png", bbox_inches="tight", dpi=300)
    plt.close()


def create_summary_table(df):
    """Create summary table with mean and std metrics for each method.

    Args:
        df: DataFrame with results

    Returns:
        DataFrame with summary statistics
    """
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

    # Calculate mean and std for each metric
    mean_df = df.groupby("method")[metrics].mean()
    std_df = df.groupby("method")[metrics].std()

    # Format as mean ± std
    summary = pd.DataFrame(index=mean_df.index, columns=metrics)
    for metric in metrics:
        summary[metric] = (
            mean_df[metric].round(3).astype(str) + " ± " + std_df[metric].round(3).astype(str)
        )

    return summary


def plot_radar_chart(df, output_path):
    """Create radar chart comparing methods.

    Args:
        df: DataFrame with mean results
        output_path: Path to save plot
    """
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

    # Get mean values for radar plot
    means = df.groupby("method")[metrics].mean()

    # Number of variables
    num_vars = len(metrics)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Plot data
    for idx, method in enumerate(means.index):
        values = means.loc[method].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle="solid", label=method)
        ax.fill(angles, values, alpha=0.1)

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.title("Methods Comparison - Radar Chart")
    plt.tight_layout()
    plt.savefig(output_path / "methods_radar_chart.png", bbox_inches="tight", dpi=300)
    plt.close()


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

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

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
    deep_models = load_deep_models(models_dir)

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
    plot_violin(results_df, metrics, output_dir)

    # Generate radar chart
    logger.info("Generating radar chart...")
    plot_radar_chart(results_df, output_dir)

    # Generate summary table
    logger.info("Generating summary table...")
    summary_df = create_summary_table(results_df)
    summary_df.to_csv(output_dir / "metrics_summary.csv")

    logger.success("Plot generation complete!")


if __name__ == "__main__":
    app()
