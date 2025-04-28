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


def calculate_snr(signal, noise , mask = None):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for a given signal and noise.

    Args:
        signal (np.ndarray): The reference (ground truth) signal image.
        noise (np.ndarray): The noise image (typically difference between final and convolved image).
        mask (np.ndarray, optional): Binary mask to restrict SNR calculation to a region of interest. Defaults to None.

    Returns:
        float: The calculated SNR (signal mean divided by noise standard deviation).
    """
    if mask is not None:
        signal = signal * mask
    # Consider only FG values for mean
    non_zero_values = signal[signal != 0]
    signal_mean = np.mean(non_zero_values)
    # Calculate noise std
    noise_std = np.std(noise)
    return signal_mean / noise_std
    
def generate_SNR_image(noisy_poisson_image, convolved_image, binary_mask, snr_target, snr_tolerance, std_dev, max_iterations):
    """
    Generate an image with a target Signal-to-Noise Ratio (SNR) by iteratively gadding Gaussian noise.

    Args:
        noisy_poisson_image (np.ndarray): Input image with initial Poisson noise.
        convolved_image (np.ndarray): Reference image used for SNR calculation.
        binary_mask (np.ndarray): Binary mask defining the region to calculate SNR.
        snr_target (float): Desired SNR value to achieve.
        snr_tolerance (float): Acceptable deviation from the target SNR.
        std_dev (float): Initial standard deviation for Gaussian noise.
        max_iterations (int): Maximum number of iterations allowed for adjustment.

    Returns:
        np.ndarray: Final image with noise adjusted to achieve the target SNR within the specified tolerance.
    """
    iteration = 0
    snr = 0

    while iteration < max_iterations:
      # Add Gaussian noise with the current std_dev
      final_image =  noisy_poisson_image + np.random.normal(0, std_dev, noisy_poisson_image.shape)
      final_image = np.clip(final_image, 0, None)
        
      # Calculate noise (difference from convolved image) and SNR
      noise = final_image - convolved_image
      snr = calculate_snr(convolved_image, noise, binary_mask)

      # Check if we are close to the target SNR
      if np.abs(snr - snr_target) < snr_tolerance or std_dev < 1e-6:
          return final_image

      # Adjust std_dev to bring SNR closer to the target
      if snr < snr_target:
          std_dev *= 0.9  # Reduce noise to increase SNR
      else:
          std_dev *= 1.1  # Increase noise to decrease SNR

      iteration += 1

     # Return the last version even if SNR was not reached
    return final_image 





def augment_patch_intensity(patch, mask, params):
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

    if params["snr_targets"] and len(params["snr_targets"]) > 0:
        # Randomly select a target SNR, add 0 for chossing the originl image
        target_snr = np.random.choice(params["snr_targets"].append(0))
        if target_snr == 0:
            return patch
        else:
            # Scale intensity before augmentation
            mask = mask.astype(np.uint16)
            mask[mask>0] = 1  # make it is binary
            patch = mask * params["intensity_scale"]
        
            # Step 1: Add z-axis intensity decay
            if 0 < params["z_decay_rate"] < 1:
                z_profile = np.exp(-np.arange(patch.shape[0]) * (1 - params["z_decay_rate"]))
                patch = patch * z_profile[:, np.newaxis, np.newaxis, np.newaxis]
            
            # Step 2: Add local variations and staining effects
            local_var = simulate_local_variations(
                patch.shape[:-1],
                binnings=[1, 2, 4, 8],
                scale=params["local_variation_scale"],
            )
            patch = patch * local_var[..., np.newaxis]
            patch = np.clip(patch, 0, None)
            
            # Step 3: Add background
            if params["background_level"] > 0:
                bg = np.ones_like(patch) * params["background_level"] * params["intensity_scale"]
                bg_noise = (
                    bg
                    * simulate_local_variations(
                        patch.shape[:-1], binnings=[1, 2], scale=params["local_variation_scale"]
                    )[..., np.newaxis]
                )
                bg_noise[patch != 0] = 0 # set 0 to places with FG
                bg_noise = np.clip(bg_noise, 0, None)
                patch = patch + bg_noise
        
            # Step 4: Apply PSF convolution if specified
            if params["use_psf"] and params["psf_path"]:
                psf = tiff.imread(params["psf_path"])
                convolved_image = convolve_with_psf(patch, psf)
                patch = convolved_image
        
            # Step 5: Add Poisson noise
            if params["poisson_scale"] > 0:
                scaled = patch * params["poisson_scale"]
                patch = np.random.poisson(scaled) / params["poisson_scale"]
                patch = np.clip(patch, 0, None)
        
            # Step 6: Add Gaussian noise to achieve target SNR
            if convolved_image:
                patch = generate_SNR_image(patch, convolved_image, mask, target_snr, params["snr_tolerance"], params["std_dev"], params["max_iterations"])
            
            return patch
            
    else
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
