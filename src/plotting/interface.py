from pathlib import Path
from typing import Annotated

import typer

from src.plotting.plot import Plotter


def interface(
    predictions_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing the image_level and patch_level predictions"),
    ],
    ground_truth_mask_patches_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing ground truth patches of masks (reconstruction patches from masks folder)"
        ),
    ],
    ground_truth_masks_dir: Annotated[
        Path, typer.Argument(help="Directory containing the ground truth masks of complete images")
    ],
    dataset_name: Annotated[str, typer.Argument(help="Name of the dataset")],
    output_dir: Annotated[Path, typer.Argument(help="Directory to save patches")],
):
    plotter = Plotter(
        predictions_dir,
        ground_truth_mask_patches_dir,
        ground_truth_masks_dir,
        dataset_name,
        output_dir,
    )

    plotter.plot_images_metrics()


if __name__ == "__main__":
    typer.run(interface)
