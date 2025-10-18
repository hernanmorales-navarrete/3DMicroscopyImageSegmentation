from pathlib import Path
from typing import Annotated
import typer
from src.utils import overwrite_and_create_directory


def interface(
    patches_dir: Annotated[Path, typer.Argument(help="Directory containing image patches")],
    dataset_name: Annotated[str, typer.Argument(help="Identifier/name to distinguish different predictions from different datasts. All prediction from a dataset will be saved here")],
    output_dir: Annotated[str, typer.Argument(help="Directory to save predictions")]
):
    overwrite_and_create_directory(output_dir / dataset_name)

    



if __name__ == '__main__': 
    typer.run(interface)