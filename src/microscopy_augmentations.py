import numpy as np
import tifffile as tiff
from scipy.signal import fftconvolve
from scipy.ndimage import convolve, zoom
from src.config import INTENSITY_PARAMS
import albumentations as A


def normalize_psf(psf):
    """Normalize PSF to sum to 1."""
    psf_sum = np.sum(psf)
    if not np.isclose(psf_sum, 1.0):
        psf = psf / psf_sum
    return psf


def convolve_with_psf(image, psf):
    """Convolve 3D image with PSF."""
    # Normalize PSF
    psf = normalize_psf(psf)

    # Calculate padding
    pad_size = psf.shape

    # Pad image
    padded_image = np.pad(image, [(p, p) for p in pad_size] + [(0, 0)], mode="reflect")

    # Convolve
    convolved_padded = fftconvolve(padded_image, psf[..., np.newaxis], mode="same")

    # Remove padding
    slices = tuple(slice(p, -p) for p in pad_size) + (slice(None),)
    convolved_image = convolved_padded[slices]

    # Ensure non-negative values
    convolved_image = np.clip(convolved_image, 0, None)

    return convolved_image


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


def augment_patch_intensity(patch, params):
    """Apply intensity-based augmentations to a 3D patch.

    This function applies various microscopy-specific augmentations:
    1. Local intensity variations (uneven staining)
    2. Z-axis intensity decay
    3. Background noise
    4. PSF convolution (optional)
    5. Poisson noise
    6. Gaussian noise with target SNR

    Args:
        patch: Input 3D patch with shape (z, y, x, channels)
        params: Dictionary of augmentation parameters

    Returns:
        Augmented patch with same shape as input
    """
    # Scale intensity before augmentation
    patch = patch * params["intensity_scale"]

    # Step 1: Add local variations and staining effects
    local_var = simulate_local_variations(
        patch.shape[:-1],
        binnings=[1, 2, 4, 8],
        scale=params["local_variation_scale"],
    )
    patch = patch * local_var[..., np.newaxis]

    # Step 2: Add z-axis intensity decay
    if 0 < params["z_decay_rate"] < 1:
        z_profile = np.exp(-np.arange(patch.shape[0]) * (1 - params["z_decay_rate"]))
        patch = patch * z_profile[:, np.newaxis, np.newaxis, np.newaxis]

    # Step 3: Add background
    if params["background_level"] > 0:
        bg = np.ones_like(patch) * params["background_level"] * params["intensity_scale"]
        bg_noise = (
            bg
            * simulate_local_variations(
                patch.shape[:-1], binnings=[1, 2], scale=params["local_variation_scale"]
            )[..., np.newaxis]
        )
        bg_noise[patch > 0] = 0
        patch = patch + bg_noise

    # Step 4: Apply PSF convolution if specified
    if params["use_psf"] and params["psf_path"]:
        psf = tiff.imread(params["psf_path"])
        patch = convolve_with_psf(patch, psf)

    # Step 5: Add Poisson noise
    if params["poisson_scale"] > 0:
        scaled = patch * params["poisson_scale"]
        patch = np.random.poisson(scaled) / params["poisson_scale"]

    # Step 6: Add Gaussian noise to achieve target SNR
    if params["snr_targets"] and len(params["snr_targets"]) > 0:
        # Randomly select a target SNR
        target_snr = np.random.choice(params["snr_targets"])

        # Calculate current SNR
        signal = patch.copy()
        signal[signal == 0] = np.nan  # Ignore background
        signal_mean = np.nanmean(signal)

        # Start with initial noise std
        noise_std = signal_mean / (target_snr * params["intensity_scale"])

        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, patch.shape)
        patch = patch + noise

    # Ensure non-negative values
    patch = np.clip(patch, 0, None)

    # Scale back to original range
    patch = patch / params["intensity_scale"]

    return patch


def create_standard_augmentation_pipeline():
    """Create an augmentation pipeline for 3D patches using supported transforms."""
    return A.Compose(
        [
            # Spatial transform - flips and rotations
            A.CubicSymmetry(p=0.7),
            A.CoarseDropout3D(
                p=0.3,
            ),
        ]
    )
