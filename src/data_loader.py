import os
import numpy as np
import tensorflow as tf
from src.config import (
    PATCH_SIZE,
    PATCH_BATCH,
)
from src.microscopy_augmentations import (
    augment_patch_intensity,
    create_standard_augmentation_pipeline,
)


class ImageDataset(tf.keras.utils.PyDataset):
    """Dataset class for loading and augmenting 3D microscopy images.

    This class handles:
    1. Loading 3D TIF images and masks
    2. Applying standard spatial augmentations (if enabled)
    3. Applying microscopy-specific intensity augmentations (if enabled)
    4. Batching the data
    """

    def __init__(self, data_dir, batch_size=PATCH_BATCH, augmentation="NONE", **kwargs):
        """Initialize the dataset.

        Args:
            data_dir: Directory containing the dataset with 'images' and 'masks' subdirectories
            batch_size: Number of samples per batch
            augmentation: Type of augmentation to use
            **kwargs: Additional arguments passed to tf.keras.utils.PyDataset
        """
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augmentation = augmentation
        # Create standard augmentation pipeline if needed
        if self.augmentation == "STANDARD":
            self.transform = create_standard_augmentation_pipeline()

        # Get file paths
        self.image_paths = sorted(tf.io.gfile.glob(os.path.join(data_dir, "images/*.tif")))
        self.mask_paths = sorted(tf.io.gfile.glob(os.path.join(data_dir, "masks/*.tif")))

        if not self.image_paths or not self.mask_paths:
            raise ValueError("No data found in specified directory")

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
            image = augment_patch_intensity(image)

        return image, mask

    def _load_3d_tif(self, path):
        """Load and normalize a 3D TIF file.

        Args:
            path: Path to the TIF file

        Returns:
            Normalized volume of shape (z, y, x, 1) with values in [0, 1]
        """
        raw_data = tf.io.read_file(path)
        volume = tf.io.decode_raw(raw_data, tf.uint16)
        volume = tf.reshape(volume, PATCH_SIZE)
        volume = volume[..., tf.newaxis]
        volume = tf.cast(volume, tf.float32)

        # Min-max normalization
        volume_min = tf.reduce_min(volume)
        volume_max = tf.reduce_max(volume)
        volume_normalized = (volume - volume_min) / (
            volume_max - volume_min + tf.keras.backend.epsilon()
        )

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
        data_dir="path/to/data",
        batch_size=32,
        workers=4,
        use_multiprocessing=True,
        max_queue_size=10,
    )
