import tensorflow as tf
import numpy as np
import os
import albumentations as A
from config import PATCH_SIZE, PATCH_BATCH


def create_3d_augmentation_pipeline():
    """Create an augmentation pipeline for 3D patches using supported transforms.
    Only includes transforms that preserve patch size.

    Returns:
        Albumentations Compose object with 3D transforms
    """
    return A.Compose(
        [
            # Spatial transform - flips and rotations
            A.CubicSymmetry(p=0.7),
            # Intensity transform - random dropout regions
            A.CoarseDropout3D(
                max_holes=8,
                max_height=8,
                max_width=8,
                max_depth=8,
                min_holes=4,
                min_height=4,
                min_width=4,
                min_depth=4,
                fill_value=0,
                p=0.3,
            ),
        ]
    )


class ImageDataset(tf.keras.utils.PyDataset):
    def __init__(self, data_dir, batch_size=PATCH_BATCH, augment=False, **kwargs):
        """Initialize the dataset.

        Args:
            data_dir: Directory containing the dataset
            batch_size: Number of samples per batch
            augment: Whether to apply data augmentation
        """
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augment = augment

        # Create augmentation pipeline if needed
        if self.augment:
            self.transform = create_3d_augmentation_pipeline()

        # Get file paths
        self.image_paths = sorted(tf.io.gfile.glob(os.path.join(data_dir, "images/*.tif")))
        self.mask_paths = sorted(tf.io.gfile.glob(os.path.join(data_dir, "masks/*.tif")))

        if not self.image_paths or not self.mask_paths:
            raise ValueError("No data found in specified directory")

    def _augment_3d_patch(self, image, mask):
        """Apply 3D augmentations to a patch and its mask.

        Args:
            image: 3D image patch of shape [H, W, D, 1]
            mask: 3D mask patch of shape [H, W, D, 1]

        Returns:
            Tuple of augmented (image, mask)
        """
        if not self.augment:
            return image, mask

        # Remove channel dimension for Albumentations
        image = np.squeeze(image)
        mask = np.squeeze(mask)

        # Apply augmentations
        transformed = self.transform(volume=image, mask3d=mask)

        # Add channel dimension back
        image = transformed["volume"][..., np.newaxis]
        mask = transformed["mask3d"][..., np.newaxis]

        return image, mask

    def _load_3d_tif(self, path):
        """Load a 3D TIF file as a numpy array."""
        # Read raw bytes
        raw_data = tf.io.read_file(path)
        # Decode as uint16
        volume = tf.io.decode_raw(raw_data, tf.uint16)
        # Reshape to 3D
        volume = tf.reshape(volume, PATCH_SIZE)
        # Add channel dimension
        volume = volume[..., tf.newaxis]

        # Convert to float32 for normalization
        volume = tf.cast(volume, tf.float32)

        # Min-max normalization using TensorFlow ops
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
            idx: Position of the batch in the dataset.

        Returns:
            Tuple (images, masks) containing the batch data.
        """
        # Get batch file paths
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_paths))
        batch_image_paths = self.image_paths[start_idx:end_idx]
        batch_mask_paths = self.mask_paths[start_idx:end_idx]

        # Load 3D patches
        images = []
        masks = []

        for img_path, mask_path in zip(batch_image_paths, batch_mask_paths):
            image = self._load_3d_tif(img_path)
            mask = self._load_3d_tif(mask_path)

            # Apply augmentations if enabled
            if self.augment:
                image, mask = self._augment_3d_patch(image, mask)

            images.append(image)
            masks.append(mask)

        return np.array(images), np.array(masks)


# Example usage
if __name__ == "__main__":
    # Create dataset with augmentation
    dataset = ImageDataset(
        data_dir="path/to/data",
        batch_size=32,
        augment=True,  # Enable augmentation
        workers=4,
        use_multiprocessing=True,
        max_queue_size=10,
    )

    # Example iteration
    for batch_idx in range(len(dataset)):
        images, masks = dataset[batch_idx]
        print(f"Batch {batch_idx} shapes:")
        print(f"Images: {images.shape}")  # Should be (batch_size, 64, 64, 64, 1)
        print(f"Masks: {masks.shape}")  # Should be (batch_size, 64, 64, 64, 1)
