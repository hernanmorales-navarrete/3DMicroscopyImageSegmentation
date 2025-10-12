from pathlib import Path

import typer
from typing_extensions import Annotated

from .dataset import Dataset


def interface(
    dataset_dir: Annotated[
        Path, typer.Argument(help="Dataset containing directories for images and masks")
    ],
    for_reconstruction: Annotated[
        bool, typer.Argument(help="Whether to pad image in order to reconstruct it")
    ],
):
    """
    CLI for generating patches from a dataset
    """
    dataset = Dataset(dataset_dir, for_reconstruction)
    dataset.create_patch_dataset()


if __name__ == "__main__":
    typer.run(interface)
