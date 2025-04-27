import numpy as np
import tensorflow as tf
import tifffile
from src.config import (
    BATCH_SIZE,
    INTENSITY_PARAMS,
)
from src.microscopy_augmentations import (
    augment_patch_intensity,
    create_standard_augmentation_pipeline,
)


class ImageDataset(tf.keras.utils.PyDataset):
    """Dataset class for loading and augmenting 3D microscopy images.

    This class handles:
    1. Loading 3D TIF images and masks from provided paths
    2. Applying standard spatial augmentations (if enabled)
    3. Applying microscopy-specific intensity augmentations (if enabled)
    4. Batching the data
    """

    def __init__(
        self,
        image_paths,
        mask_paths,
        batch_size=BATCH_SIZE,
        augmentation="NONE",
        intensity_params=None,
        **kwargs,
    ):
        """Initialize the dataset.

        Args:
            image_paths: List of paths to image files
            mask_paths: List of corresponding mask files
            batch_size: Number of samples per batch
            augmentation: Type of augmentation to use
            intensity_params: Parameters for microscopy-specific intensity augmentations
            **kwargs: Additional arguments passed to tf.keras.utils.PyDataset
        """
        super().__init__(**kwargs)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.intensity_params = intensity_params or INTENSITY_PARAMS

        # Create standard augmentation pipeline if needed
        if self.augmentation == "STANDARD":
            self.transform = create_standard_augmentation_pipeline()

        if not self.image_paths or not self.mask_paths:
            raise ValueError("No image or mask paths provided")

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Number of image paths must match number of mask paths")

    def _augment_3d_patch(self, image, mask):
        """Apply augmentations to a patch and its mask.

        Args:
            image: Input image patch of shape (z, y, x, 1)
            mask: Input mask patch of shape (z, y, x, 1)

        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        if self.augmentation == "NONE":
            return image, mask

        if self.augmentation == "STANDARD":
            # Remove channel dimension for Albumentations
            image_no_channel = np.squeeze(image)
            mask_no_channel = np.squeeze(mask)

            # Apply spatial augmentations
            transformed = self.transform(volume=image_no_channel, mask3d=mask_no_channel)

            # Add channel dimension back
            image = transformed["volume"][..., np.newaxis]
            mask = transformed["mask3d"][..., np.newaxis]

        if self.augmentation == "OURS":
            image = augment_patch_intensity(image, mask, params=self.intensity_params)

        return image, mask

    def _load_3d_tif(self, path):
        """Load and normalize a 3D TIF file.

        Args:
            path: Path to the TIF file

        Returns:
            Normalized volume of shape (z, y, x, 1) with values in [0, 1]
        """
        # Read TIFF file using tifffile
        volume = tifffile.imread(str(path))

        # Add channel dimension if not present
        if len(volume.shape) == 3:
            volume = volume[..., np.newaxis]

        # Convert to float32
        volume = volume.astype(np.float32)

        # Min-max normalization
        volume_min = np.min(volume)
        volume_max = np.max(volume)
        volume_normalized = (volume - volume_min) / (volume_max - volume_min + np.finfo(float).eps)

        return volume_normalized

    def __len__(self):
        """Return the number of batches in the dataset."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        """Get batch at position idx.

        Args:
            idx: Batch index

        Returns:
            Tuple of (images, masks) where each is a numpy array of shape
            (batch_size, z, y, x, 1)
        """
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_paths))
        batch_image_paths = self.image_paths[start_idx:end_idx]
        batch_mask_paths = self.mask_paths[start_idx:end_idx]

        images = []
        masks = []

        for img_path, mask_path in zip(batch_image_paths, batch_mask_paths):
            image = self._load_3d_tif(img_path)
            mask = self._load_3d_tif(mask_path)

            image, mask = self._augment_3d_patch(image, mask)

            images.append(image)
            masks.append(mask)

        return np.array(images), np.array(masks)


# Example usage
if __name__ == "__main__":
    dataset = ImageDataset(
        image_paths=["path/to/image1.tif", "path/to/image2.tif"],
        mask_paths=["path/to/mask1.tif", "path/to/mask2.tif"],
        batch_size=32,
        workers=4,
        use_multiprocessing=True,
        max_queue_size=10,
    )
