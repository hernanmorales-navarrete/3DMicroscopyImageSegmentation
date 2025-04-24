import typer
import os
import numpy as np
from pathlib import Path
import tifffile
from typing import Union, Tuple
from patchify import patchify
from loguru import logger
from tqdm.auto import tqdm

from src.config import PATCH_SIZE, PATCH_STEP

app = typer.Typer(help="CLI tool for generating 3D patches from microscopy datasets")


def generate_patches(
    dataset_dir: Union[str, Path],
    patch_size: Tuple[int, int, int] = PATCH_SIZE,
    step_size: int = PATCH_STEP,
    output_subdir: str = "patches",
) -> None:
    """
    Generate 3D patches from TIFF/TIF microscopy images and their corresponding masks.
    Maintains the dataset structure where images and masks are in parallel folders.

    Args:
        dataset_dir: Root directory containing 'images' and 'masks' folders
        patch_size: Size of 3D patches (tuple of 3 ints for x,y,z dimensions)
        step_size: Step size for patch generation
        output_subdir: Name of subdirectory to save patches
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    # Verify directory structure
    if not images_dir.exists() or not masks_dir.exists():
        raise ValueError(
            f"Dataset directory must contain both 'images' and 'masks' folders. Check {dataset_dir}"
        )

    # Create output directories
    patches_dir = dataset_dir / output_subdir
    patches_dir.mkdir(exist_ok=True, parents=True)
    patches_images_dir = patches_dir / "images"
    patches_masks_dir = patches_dir / "masks"
    patches_images_dir.mkdir(exist_ok=True, parents=True)
    patches_masks_dir.mkdir(exist_ok=True, parents=True)

    # Get all TIFF image files
    image_extensions = (".tiff", ".tif")
    image_files = sorted([f for f in images_dir.glob("*") if f.suffix.lower() in image_extensions])

    if not image_files:
        logger.warning(f"No TIFF files found in {images_dir}")
        return

    logger.info(f"Found {len(image_files)} images to process")
    logger.info(f"Using patch size: {patch_size}")

    # Process each image and its corresponding mask
    for img_path in tqdm(image_files, desc="Processing images", position=0):
        # Find corresponding mask file
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            logger.warning(f"No corresponding mask found for {img_path.name}, skipping...")
            continue

        try:
            # Read image and mask
            img = tifffile.imread(str(img_path))
            mask = tifffile.imread(str(mask_path))

            # Verify 3D dimensions
            if len(img.shape) != 3 or len(mask.shape) != 3:
                logger.warning(
                    f"Skipping {img_path.name}: Expected 3D image but got shape {img.shape}"
                )
                continue

            # Generate 3D patches
            img_patches = patchify(img, patch_size, step=step_size)
            mask_patches = patchify(mask, patch_size, step=step_size)

            # Get original image shape
            orig_shape = img.shape

            # Create output directories for this image-mask pair
            img_patches_subdir = patches_images_dir / img_path.stem
            mask_patches_subdir = patches_masks_dir / img_path.stem
            img_patches_subdir.mkdir(exist_ok=True, parents=True)
            mask_patches_subdir.mkdir(exist_ok=True, parents=True)

            # Calculate number of patches in each dimension
            n_z, n_y, n_x = img_patches.shape[:3]

            # Save patches with nested progress bars
            patch_idx = 0
            # Z-axis progress bar
            z_pbar = tqdm(range(n_z), desc=f"Z-axis ({img_path.stem})", position=1, leave=False)
            for z in z_pbar:
                # Y-axis progress bar
                y_pbar = tqdm(range(n_y), desc="Y-axis", position=2, leave=False)
                for y in y_pbar:
                    # X-axis progress bar
                    x_pbar = tqdm(range(n_x), desc="X-axis", position=3, leave=False)
                    for x in x_pbar:
                        # Calculate actual position in original image
                        pos_z = z * step_size
                        pos_y = y * step_size
                        pos_x = x * step_size

                        # Create filename with position and original size info
                        # Format: name_origsize_z_y_x_pos_z_y_x.tif
                        # Example: image1_512_512_128_z50_y100_x150.tif
                        patch_filename = (
                            f"{img_path.stem}_"
                            f"{orig_shape[0]}_{orig_shape[1]}_{orig_shape[2]}_"
                            f"z{pos_z}_y{pos_y}_x{pos_x}.tif"
                        )

                        # Save image patch
                        img_patch_path = img_patches_subdir / patch_filename
                        tifffile.imwrite(str(img_patch_path), img_patches[z, y, x])

                        # Save mask patch
                        mask_patch_path = mask_patches_subdir / patch_filename
                        tifffile.imwrite(str(mask_patch_path), mask_patches[z, y, x])

                        patch_idx += 1

                        # Update progress bar descriptions with current positions
                        x_pbar.set_description(f"X-axis (pos: {pos_x})")
                    y_pbar.set_description(f"Y-axis (pos: {pos_y})")
                z_pbar.set_description(f"Z-axis (pos: {pos_z})")

            logger.info(f"Generated {patch_idx} patches of size {patch_size} for {img_path.name}")

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue

    logger.success("Patch generation complete!")


@app.command()
def main(
    dataset_dir: Path = typer.Argument(
        ...,
        help="Directory containing the dataset with 'images' and 'masks' subdirectories",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
) -> None:
    """
    Generate 3D patches from microscopy images and their corresponding masks.

    The dataset directory should contain:
    - An 'images' subdirectory with 3D TIFF files
    - A 'masks' subdirectory with corresponding mask files

    All patch generation parameters are configured in config.py
    """
    # Generate patches using config values
    generate_patches(dataset_dir=dataset_dir)


if __name__ == "__main__":
    app()
