import typer
import os
import numpy as np
from pathlib import Path
import tifffile
from typing import Union, Tuple
from patchify import patchify
from loguru import logger
from tqdm.auto import tqdm
import shutil

from src.config import PATCH_SIZE, PATCH_STEP, PATCH_STEP_RECONSTRUCTION

app = typer.Typer(help="CLI tool for generating 3D patches from microscopy datasets")


def calculate_padding(
    image_shape: Tuple[int, int, int], patch_size: Tuple[int, int, int], step_size: int
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Calculate required padding for each dimension to ensure proper reconstruction with overlapping patches.

    Args:
        image_shape: Original image shape (z, y, x)
        patch_size: Size of patches (z, y, x)
        step_size: Step size for patch generation (smaller step size means more overlap)

    Returns:
        Tuple of padding values for each dimension ((z_before, z_after), (y_before, y_after), (x_before, x_after))
    """
    padding = []
    for dim, patch_dim in zip(image_shape, patch_size):
        remainder = (dim - patch_dim) % step_size
        padding.append((0, step_size - remainder))
    return tuple(padding)


def generate_patches(
    dataset_dir: Union[str, Path],
    patch_size: Tuple[int, int, int] = PATCH_SIZE,
    step_size: int = PATCH_STEP,
    output_subdir: str = "patches",
    for_reconstruction: bool = False,
) -> None:
    """
    Generate 3D patches from TIFF/TIF microscopy images and their corresponding masks.
    Maintains the dataset structure where images and masks are in parallel folders.

    Args:
        dataset_dir: Root directory containing 'images' and 'masks' folders
        patch_size: Size of 3D patches (tuple of 3 ints for x,y,z dimensions)
        step_size: Step size for patch generation (if for_reconstruction is True, uses PATCH_STEP_RECONSTRUCTION)
        output_subdir: Name of subdirectory to save patches
        for_reconstruction: Whether patches should be generated with overlap for later reconstruction
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    # Use reconstruction step size if specified
    if for_reconstruction:
        step_size = PATCH_STEP_RECONSTRUCTION

    # Verify directory structure
    if not images_dir.exists() or not masks_dir.exists():
        raise ValueError(
            f"Dataset directory must contain both 'images' and 'masks' folders. Check {dataset_dir}"
        )

    # Create output directories
    prefix = "reconstruction" if for_reconstruction else "regular"
    patches_dir = dataset_dir / f"{prefix}_{output_subdir}"

    try:
        patches_dir.mkdir(exist_ok=False, parents=True)
    except:
        logger.info(f"Existing directory {patches_dir}. Deleting it")
        shutil.rmtree(patches_dir)

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
    logger.info(
        f"Using step size: {step_size} ({'with overlap for reconstruction' if for_reconstruction else 'without overlap'})"
    )

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

            # Store original shape before any padding
            orig_shape = img.shape
            logger.info(f"Image File: {img_path}, Original Image Size {orig_shape}")

            if for_reconstruction:
                # Calculate required padding for reconstruction
                padding = calculate_padding(img.shape, patch_size, step_size)
                logger.info(f"Applying padding for reconstruction to {img_path.name}: {padding}")

                # Apply padding to both image and mask
                img = np.pad(img, padding, mode="reflect")
                mask = np.pad(mask, padding, mode="reflect")

                logger.info(f"Padded Image Size {img.shape}")

            # Generate 3D patches
            img_patches = patchify(img, patch_size, step=step_size)
            mask_patches = patchify(mask, patch_size, step=step_size)

            # Store number of patches in each dimension
            n_patches_z, n_patches_y, n_patches_x = img_patches.shape[:3]

            # Create output directories for this image-mask pair
            img_patches_subdir = patches_images_dir / img_path.stem
            mask_patches_subdir = patches_masks_dir / img_path.stem
            img_patches_subdir.mkdir(exist_ok=True, parents=True)
            mask_patches_subdir.mkdir(exist_ok=True, parents=True)

            # Flatten the patches arrays
            img_patches_flat = img_patches.reshape(-1, *patch_size)
            mask_patches_flat = mask_patches.reshape(-1, *patch_size)
            total_patches = len(img_patches_flat)

            # Save patches with a simple progress bar
            with tqdm(
                total=total_patches,
                desc=f"Saving patches for {img_path.stem}",
                position=1,
                leave=False,
            ) as pbar:
                for patch_idx in range(total_patches):
                    # Create filename with patch number, original size, and number of patches info
                    patch_filename = (
                        f"{img_path.stem}_"
                        f"orig_{orig_shape[0]}_{orig_shape[1]}_{orig_shape[2]}_"
                        f"{'' if not for_reconstruction else f'pad_{img.shape[0]}_{img.shape[1]}_{img.shape[2]}_'}"
                        f"npatches_{n_patches_z}_{n_patches_y}_{n_patches_x}_"
                        f"patch_{patch_idx:04d}.tif"
                    )

                    # Save image patch
                    img_patch_path = img_patches_subdir / patch_filename
                    tifffile.imwrite(str(img_patch_path), img_patches_flat[patch_idx])

                    # Save mask patch
                    mask_patch_path = mask_patches_subdir / patch_filename
                    tifffile.imwrite(str(mask_patch_path), mask_patches_flat[patch_idx])

                    pbar.update(1)

            logger.info(
                f"Generated {total_patches} patches of size {patch_size} for {img_path.name}"
            )

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
    for_reconstruction: bool = typer.Argument(
        ...,
        help="Whether to generate overlapping patches suitable for image reconstruction",
    ),
) -> None:
    """
    Generate 3D patches from microscopy images and their corresponding masks.

    The dataset directory should contain:
    - An 'images' subdirectory with 3D TIFF files
    - A 'masks' subdirectory with corresponding mask files

    When for_reconstruction is True:
    - Patches will be generated with overlap (smaller step size)
    - Images will be padded if needed to ensure proper reconstruction
    - Output will be saved in 'reconstruction_patches' directory

    When for_reconstruction is False:
    - Patches will be generated without overlap
    - No padding will be applied
    - Output will be saved in 'regular_patches' directory

    All patch generation parameters are configured in config.py
    """
    # Generate patches using config values
    generate_patches(dataset_dir=dataset_dir, for_reconstruction=for_reconstruction)


if __name__ == "__main__":
    app()
