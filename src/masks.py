import os
import numpy as np
from pathlib import Path
import tifffile
from typing import Union, Tuple
from patchify import patchify
from loguru import logger
from tqdm.auto import tqdm
from src.config import PATCH_SIZE


def generate_and_save_patches(
    dataset_dir: Union[str, Path],
    patch_size: Union[int, Tuple[int, int, int]] = PATCH_SIZE,
    step_size: int = 1,
    output_subdir: str = "patches",
) -> None:
    """
    Generate 3D patches from TIFF/TIF microscopy images and their corresponding masks.
    Maintains the dataset structure where images and masks are in parallel folders.

    Args:
        dataset_dir: Root directory containing 'images' and 'masks' folders
        patch_size: Size of 3D patches (tuple of 3 ints for x,y,z dimensions), defaults to config.PATCH_SIZE
        step_size: Step size for patch generation (default: 1)
        output_subdir: Name of subdirectory to save patches (default: "patches")
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
    patches_images_dir = patches_dir / "images"
    patches_masks_dir = patches_dir / "masks"
    patches_images_dir.mkdir(exist_ok=True, parents=True)
    patches_masks_dir.mkdir(exist_ok=True, parents=True)

    # Ensure patch_size is a 3D tuple
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    elif len(patch_size) != 3:
        raise ValueError(
            "patch_size must be either an integer or a tuple of 3 integers for 3D patches"
        )

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

            # Reshape to (N, patch_x, patch_y, patch_z)
            img_patches_reshaped = img_patches.reshape(-1, *patch_size)
            mask_patches_reshaped = mask_patches.reshape(-1, *patch_size)

            # Create output directories for this image-mask pair
            img_patches_subdir = patches_images_dir / img_path.stem
            mask_patches_subdir = patches_masks_dir / img_path.stem
            img_patches_subdir.mkdir(exist_ok=True, parents=True)
            mask_patches_subdir.mkdir(exist_ok=True, parents=True)

            # Save patches with progress bar
            patch_desc = f"Saving patches for {img_path.stem}"
            for idx, (img_patch, mask_patch) in enumerate(
                tqdm(
                    zip(img_patches_reshaped, mask_patches_reshaped),
                    desc=patch_desc,
                    total=len(img_patches_reshaped),
                    position=1,
                    leave=False,
                )
            ):
                # Save image patch
                img_patch_filename = f"{img_path.stem}_patch_{idx:04d}.tif"
                img_patch_path = img_patches_subdir / img_patch_filename
                tifffile.imwrite(str(img_patch_path), img_patch)

                # Save mask patch
                mask_patch_filename = f"{img_path.stem}_patch_{idx:04d}.tif"
                mask_patch_path = mask_patches_subdir / mask_patch_filename
                tifffile.imwrite(str(mask_patch_path), mask_patch)

            logger.info(
                f"Generated {len(img_patches_reshaped)} patches of size {patch_size} for {img_path.name}"
            )

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue

    logger.success("Patch generation complete!")
