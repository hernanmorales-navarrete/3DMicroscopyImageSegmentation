import tensorflow as tf
import numpy as np
import os
import albumentations as A
from config import (
    PATCH_SIZE,
    PATCH_BATCH,
    STANDARD_AUGMENTATION,
    OURS_AUGMENTATION,
    INTENSITY_PARAMS,
)
from scipy.signal import fftconvolve
from scipy.ndimage import convolve, zoom


def create_3d_augmentation_pipeline():
    """Create an augmentation pipeline for 3D patches using supported transforms."""
    return A.Compose(
        [
            # Spatial transform - flips and rotations
            A.CubicSymmetry(p=0.7),
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


def simulate_local_variations(shape, binnings=[1, 2, 4, 8], scale=5):
    """Simulate local variations in staining intensity.

    Args:
        shape: Shape of the patch
        binnings: List of binning factors for multi-scale variations
        scale: Scale of the smoothing kernel

    Returns:
        Array of local intensity variations
    """
    result = np.ones(shape)
    for binning in binnings:
        # Calculate smaller shape based on binning
        small_shape = tuple(dim // binning for dim in shape)
        zoom_factors = [orig_dim / small_dim for orig_dim, small_dim in zip(shape, small_shape)]

        # Generate and smooth local variations
        local_var = np.random.normal(size=small_shape) + 1.0
        smoothed = convolve(local_var, np.ones((scale, scale, scale)) / scale**3)

        # Zoom back to original size and combine
        variation = zoom(smoothed, zoom_factors, order=1)
        result *= variation

    return result


def augment_patch_intensity(patch, params=INTENSITY_PARAMS):
    """Apply intensity-based augmentations to a 3D patch.

    Args:
        patch: Input 3D patch
        params: Dict with augmentation parameters:
            - background_level: Background intensity level
            - local_variation_scale: Scale of local variations
            - z_decay_rate: Rate of intensity decay along z-axis
            - noise_std: Standard deviation for Gaussian noise
            - poisson_scale: Scaling factor for Poisson noise

    Returns:
        Augmented patch
    """
    # Add local variations
    local_var = simulate_local_variations(
        patch.shape[:-1],  # Remove channel dim
        binnings=[1, 2, 4, 8],
        scale=params["local_variation_scale"],
    )
    patch = patch * local_var[..., np.newaxis]

    # Add z-axis intensity decay
    if 0 < params["z_decay_rate"] < 1:
        z_profile = np.exp(-np.arange(patch.shape[0]) * (1 - params["z_decay_rate"]))
        patch = patch * z_profile[:, np.newaxis, np.newaxis, np.newaxis]

    # Add background
    if params["background_level"] > 0:
        bg = np.ones_like(patch) * params["background_level"]
        bg_noise = (
            bg
            * simulate_local_variations(
                patch.shape[:-1], binnings=[1, 2], scale=params["local_variation_scale"]
            )[..., np.newaxis]
        )
        bg_noise[patch > 0] = 0
        patch = patch + bg_noise

    # Add Poisson noise
    if params["poisson_scale"] > 0:
        scaled = patch * params["poisson_scale"]
        patch = np.random.poisson(scaled) / params["poisson_scale"]

    # Add Gaussian noise
    if params["noise_std"] > 0:
        noise = np.random.normal(0, params["noise_std"], patch.shape)
        patch = patch + noise

    # Ensure non-negative values
    patch = np.clip(patch, 0, None)

    return patch


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

        # Create standard augmentation pipeline if needed
        if STANDARD_AUGMENTATION:
            self.transform = create_3d_augmentation_pipeline()

        # Get file paths
        self.image_paths = sorted(tf.io.gfile.glob(os.path.join(data_dir, "images/*.tif")))
        self.mask_paths = sorted(tf.io.gfile.glob(os.path.join(data_dir, "masks/*.tif")))

        if not self.image_paths or not self.mask_paths:
            raise ValueError("No data found in specified directory")

    def _augment_3d_patch(self, image, mask):
        """Apply augmentations to a patch and its mask."""
        if not STANDARD_AUGMENTATION and not OURS_AUGMENTATION:
            return image, mask

        if STANDARD_AUGMENTATION:
            # Remove channel dimension for Albumentations
            image_no_channel = np.squeeze(image)
            mask_no_channel = np.squeeze(mask)

            # Apply spatial augmentations
            transformed = self.transform(volume=image_no_channel, mask3d=mask_no_channel)

            # Add channel dimension back
            image = transformed["volume"][..., np.newaxis]
            mask = transformed["mask3d"][..., np.newaxis]

        if OURS_AUGMENTATION:
            image = augment_patch_intensity(image)

        return image, mask

    def _load_3d_tif(self, path):
        """Load a 3D TIF file as a numpy array."""
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
        """Get batch at position idx."""
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
