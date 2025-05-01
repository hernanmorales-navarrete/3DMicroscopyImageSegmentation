from pathlib import Path
import typer
from loguru import logger

from src.config import FIGURES_DIR, MODELS_DIR, METHOD_ORDER
from src.processors import Predictor, Visualizer
from src.processors.evaluator import Evaluator
from src.utils import configure_gpu

# Configure GPU at startup
configure_gpu()

app = typer.Typer()


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
        help="Identifier/name to distinguish and organize different sets of images. Only models trained on this dataset will be used.",
    ),
    models_dir: Path = typer.Argument(MODELS_DIR, help="Directory containing trained models"),
    output_dir: Path = typer.Option(FIGURES_DIR, help="Directory to save plots"),
):
    """Generate plots comparing classical and deep learning models."""
    try:
        # Initialize processors
        predictor = Predictor()
        visualizer = Visualizer(output_dir, method_order=METHOD_ORDER)
        evaluator = Evaluator()

        # Get and validate image paths
        patch_paths = sorted(patches_dir.glob("images/**/*.tif"))
        patch_masks = sorted(patches_dir.glob("masks/**/*.tif"))
        if not patch_paths or not patch_masks:
            raise ValueError(
                f"No .tif files found in {patches_dir}/images/ or {patches_dir}/masks/"
            )
        if len(patch_paths) != len(patch_masks):
            raise ValueError("Number of patch images does not match number of patch masks")
        logger.info(f"Found {len(patch_paths)} patch image-mask pairs")

        complete_image_paths = sorted(complete_images_dir.glob("images/**/*.tif"))
        complete_masks = sorted(complete_images_dir.glob("masks/**/*.tif"))
        if not complete_image_paths or not complete_masks:
            raise ValueError(
                f"No .tif files found in {complete_images_dir}/images/ or {complete_images_dir}/masks/"
            )
        if len(complete_image_paths) != len(complete_masks):
            raise ValueError("Number of complete images does not match number of complete masks")
        logger.info(f"Found {len(complete_image_paths)} complete image-mask pairs")

        # Load models
        logger.info("Loading deep learning models...")
        deep_models = predictor.load_deep_models(models_dir, dataset_name=dataset_name)

        # Evaluate all methods
        logger.info("Evaluating methods...")
        results_df = evaluator.evaluate_all_methods(
            patch_paths, patch_masks, complete_image_paths, complete_masks, deep_models
        )

        # Define metrics to plot
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

        # Generate plots and summaries
        visualizer.generate_plots(results_df, dataset_name, metrics)

        logger.success("Plot generation complete!")

    except Exception as e:
        logger.error(f"Error during plot generation: {e}")
        raise


if __name__ == "__main__":
    app()
