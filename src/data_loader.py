import tensorflow as tf
import numpy as np
import os

from config import PATCH_SIZE, PATCH_BATCH


class ImageDataset(tf.keras.utils.PyDataset):
    def __init__(self, data_dir, batch_size=PATCH_BATCH, **kwargs):
        """Initialize the dataset.

        Args:
            data_dir: Directory containing the dataset
            batch_size: Number of samples per batch
        """
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Get file paths
        self.image_paths = sorted(tf.io.gfile.glob(os.path.join(data_dir, "images/*.tif")))
        self.mask_paths = sorted(tf.io.gfile.glob(os.path.join(data_dir, "masks/*.tif")))

        if not self.image_paths or not self.mask_paths:
            raise ValueError("No data found in specified directory")

    def __len__(self):
        """Return the number of batches in the dataset."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def _load_3d_tif(self, path):
        """Load a 3D TIF file as a numpy array."""
        # Read raw bytes
        raw_data = tf.io.read_file(path)
        # Decode as uint16
        volume = tf.io.decode_raw(raw_data, tf.uint16)
        # Reshape to 3D (assuming 64x64x64)
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
        images = [self._load_3d_tif(path) for path in batch_image_paths]
        masks = [self._load_3d_tif(path) for path in batch_mask_paths]

        return np.array(images), np.array(masks)


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = ImageDataset(
        data_dir="path/to/data",
        batch_size=32,
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
