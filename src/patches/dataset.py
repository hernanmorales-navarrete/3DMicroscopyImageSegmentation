from pathlib import Path

import tifffile
from tqdm import tqdm

from src.config import (
    ALLOWED_EXTENSIONS,
    MAX_WORKERS,
    PATCH_SIZE,
    PATCH_STEP,
)

from .utils import (
    overwrite_and_create_directory,
    validate_mask,
)

from src.utils import(
    overwrite_and_create_directory
)


class Dataset:
    """
    Given a datset_folder, an output_folder and whether or not it is for reconstruction to add padding, it returns a new dataset that consists of volumnes.

    Therefore, ALL THE FUNCTIONS FOR INFERENCE, TRAINING ARE BASED ON 3D IMAGES. 
    """
    def __init__(
        self, dataset_folder: Path, for_reconstruction: bool, output_dir: Path = "patches"
    ) -> None:
        images_dir = dataset_folder / "images"
        masks_dir = dataset_folder / "masks"
        self.for_reconstruction = for_reconstruction

        # Ensure that folder has images and masks folders
        if not (images_dir.is_dir() and masks_dir.is_dir()):
            raise Exception(f"No images or masks folder found  in {dataset_folder}")

        # Define a prefix to distinguish regular and reconstruction patches
        prefix = "regular" if not for_reconstruction else "reconstruction"

        # Create patches dir; if it exists, delete it
        patches_dir = dataset_folder / f"{prefix}_{output_dir}"
        overwrite_and_create_directory(patches_dir)

        # Create directory for patches from images folder and masks folder
        self.patches_images_dir = patches_dir / "images"
        self.patches_masks_dir = patches_dir / "masks"
        overwrite_and_create_directory(self.patches_images_dir)
        overwrite_and_create_directory(self.patches_masks_dir)

        # Get corresponding pairs of images and masks and create a generator
        self.image_mask_path_generator = (
            (image, masks_dir / image.name)
            for image in images_dir.glob("*")
            if image.suffix.lower() in ALLOWED_EXTENSIONS
            if validate_mask(image, masks_dir / image.name)
        )

    def create_patch_dataset(self):
        for image_path, mask_path in tqdm(
            self.image_mask_path_generator, desc="Proccessing images", position=0
        ):
            try:
                image = tifffile.imread(str(image_path))
                mask = tifffile.imread(str(mask_path))

                # Verify 3D dimensions
                if len(image.shape) != 3 or len(mask.shape) != 3:
                    raise Exception(
                        f"Check your dataset. Expected 3D image {image.name} but got shape {image.shape}"
                    )

                # Create output directories to store the patches of a single image and its corresponding mask
                dir_to_store_patches_for_single_image = self.patches_images_dir / image_path.stem
                dir_to_store_patches_for_single_mask = self.patches_masks_dir / image_path.stem

                overwrite_and_create_directory(dir_to_store_patches_for_single_image)
                overwrite_and_create_directory(dir_to_store_patches_for_single_mask)

                create_and_save_patches_from_image_and_mask(
                    image_path,
                    image,
                    mask,
                    PATCH_SIZE,
                    PATCH_STEP,
                    dir_to_store_patches_for_single_image,
                    dir_to_store_patches_for_single_mask,
                    self.for_reconstruction,
                    MAX_WORKERS,
                )

            except Exception as e:
                raise Exception(f"Failed to save patches for image-mask pair {image_path}: {e}")
