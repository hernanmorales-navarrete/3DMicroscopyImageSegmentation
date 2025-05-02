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
    reconstruction_patches_dir: Path = typer.Argument(
        ...,
        help="Directory containing reconstruction patches (with overlap) for evaluating deep learning methods on complete images",
    ),
    regular_patches_dir: Path = typer.Argument(
        ...,
        help="Directory containing regular patches (no overlap) for patch-level evaluation of all methods",
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
    """Generate plots comparing classical and deep learning models.

    For deep learning methods:
    - Complete image evaluation uses reconstruction patches (with overlap)
    - Patch-level evaluation uses regular patches (no overlap)

    For classical methods:
    - Complete image evaluation uses the complete images
    - Patch-level evaluation uses regular patches (no overlap)
    """
    try:
        # Initialize processors
        predictor = Predictor()
        visualizer = Visualizer(output_dir, method_order=METHOD_ORDER)
        evaluator = Evaluator()

        # Get and validate reconstruction patches (for deep learning complete image evaluation)
        reconstruction_patch_paths = sorted(reconstruction_patches_dir.glob("images/**/*.tif"))
        reconstruction_patch_masks = sorted(reconstruction_patches_dir.glob("masks/**/*.tif"))
        if not reconstruction_patch_paths or not reconstruction_patch_masks:
            raise ValueError(
                f"No .tif files found in {reconstruction_patches_dir}/images/ or {reconstruction_patches_dir}/masks/"
            )
        if len(reconstruction_patch_paths) != len(reconstruction_patch_masks):
            raise ValueError(
                "Number of reconstruction patch images does not match number of patch masks"
            )
        logger.info(
            f"Found {len(reconstruction_patch_paths)} reconstruction patch image-mask pairs"
        )

        # Get and validate regular patches (for patch-level evaluation)
        regular_patch_paths = sorted(regular_patches_dir.glob("images/**/*.tif"))
        regular_patch_masks = sorted(regular_patches_dir.glob("masks/**/*.tif"))
        if not regular_patch_paths or not regular_patch_masks:
            raise ValueError(
                f"No .tif files found in {regular_patches_dir}/images/ or {regular_patches_dir}/masks/"
            )
        if len(regular_patch_paths) != len(regular_patch_masks):
            raise ValueError("Number of regular patch images does not match number of patch masks")
        logger.info(f"Found {len(regular_patch_paths)} regular patch image-mask pairs")

        # Get and validate complete images (for classical methods)
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
            regular_patch_paths=regular_patch_paths,
            regular_patch_masks=regular_patch_masks,
            reconstruction_patch_paths=reconstruction_patch_paths,
            reconstruction_patch_masks=reconstruction_patch_masks,
            complete_image_paths=complete_image_paths,
            complete_masks=complete_masks,
            deep_models=deep_models,
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
