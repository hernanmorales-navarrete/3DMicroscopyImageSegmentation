from pathlib import Path
from typing import Annotated

import typer

from src.inference.predict import Prediction


def interface(
    patches_dir: Annotated[Path, typer.Argument(help="Directory containing image patches")],
    images_dir: Annotated[Path, typer.Argument(help="Directory containing complete images")], 
    dataset_name: Annotated[str, typer.Argument(help="Identifier/name to distinguish different predictions from different datasts. All prediction from a dataset will be saved here")],
    models_dir: Annotated[Path, typer.Argument(help="Path to models' folder")],
    output_dir: Annotated[Path, typer.Argument(help="Directory to save predictions")]
):
    Prediction(patches_dir, images_dir, output_dir, dataset_name, models_dir).predict()


if __name__ == '__main__': 
    typer.run(interface)