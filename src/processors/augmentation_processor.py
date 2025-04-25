import numpy as np
import tifffile as tiff
from scipy.signal import fftconvolve
from scipy.ndimage import convolve, zoom
import albumentations as A
from typing import List, Dict, Any
from .base import ImageProcessor


class Augmentor(ImageProcessor):
    """Class for handling image augmentations."""

    def __init__(self, intensity_params: Dict[str, Any]):
        self.intensity_params = intensity_params

    @staticmethod
    def normalize_psf(psf: np.ndarray) -> np.ndarray:
        """Normalize PSF to sum to 1."""
        psf_sum = np.sum(psf)
        if not np.isclose(psf_sum, 1.0):
            psf = psf / psf_sum
        return psf

    def convolve_with_psf(self, image: np.ndarray, psf: np.ndarray) -> np.ndarray:
        """Convolve 3D image with PSF."""
        psf = self.normalize_psf(psf)
        pad_size = psf.shape
        padded_image = np.pad(image, [(p, p) for p in pad_size] + [(0, 0)], mode="reflect")
        convolved_padded = fftconvolve(padded_image, psf[..., np.newaxis], mode="same")
        slices = tuple(slice(p, -p) for p in pad_size) + (slice(None),)
        convolved_image = convolved_padded[slices]
        return np.clip(convolved_image, 0, None)

    def simulate_local_variations(
        self, shape: tuple, binnings: List[int] = [1, 2, 4, 8], scale: int = 5
    ) -> np.ndarray:
        """Simulate local variations in staining intensity."""
        result = np.ones(shape)
        for binning in binnings:
            small_shape = tuple(dim // binning for dim in shape)
            zoom_factors = [
                orig_dim / small_dim for orig_dim, small_dim in zip(shape, small_shape)
            ]
            local_var = np.random.normal(size=small_shape) + 1.0
            smoothed = convolve(local_var, np.ones((scale, scale, scale)) / scale**3)
            variation = zoom(smoothed, zoom_factors, order=1)
            result *= variation
        return result

    def augment_patch_intensity(self, patch: np.ndarray) -> np.ndarray:
        """Apply intensity-based augmentations to a 3D patch."""
        params = self.intensity_params
        patch = patch * params["intensity_scale"]

        # Add local variations
        local_var = self.simulate_local_variations(
            patch.shape[:-1],
            binnings=[1, 2, 4, 8],
            scale=params["local_variation_scale"],
        )
        patch = patch * local_var[..., np.newaxis]

        # Add z-axis decay
        if 0 < params["z_decay_rate"] < 1:
            z_profile = np.exp(-np.arange(patch.shape[0]) * (1 - params["z_decay_rate"]))
            patch = patch * z_profile[:, np.newaxis, np.newaxis, np.newaxis]

        # Add background
        if params["background_level"] > 0:
            bg = np.ones_like(patch) * params["background_level"] * params["intensity_scale"]
            bg_noise = (
                bg
                * self.simulate_local_variations(
                    patch.shape[:-1], binnings=[1, 2], scale=params["local_variation_scale"]
                )[..., np.newaxis]
            )
            bg_noise[patch > 0] = 0
            patch = patch + bg_noise

        # Apply PSF convolution
        if params["use_psf"] and params["psf_path"]:
            psf = tiff.imread(params["psf_path"])
            patch = self.convolve_with_psf(patch, psf)

        # Add Poisson noise
        if params["poisson_scale"] > 0:
            scaled = patch * params["poisson_scale"]
            patch = np.random.poisson(scaled) / params["poisson_scale"]

        # Add Gaussian noise
        if params["snr_targets"] and len(params["snr_targets"]) > 0:
            target_snr = np.random.choice(params["snr_targets"])
            signal = patch.copy()
            signal[signal == 0] = np.nan
            signal_mean = np.nanmean(signal)
            noise_std = signal_mean / (target_snr * params["intensity_scale"])
            noise = np.random.normal(0, noise_std, patch.shape)
            patch = patch + noise

        patch = np.clip(patch, 0, None)
        patch = patch / params["intensity_scale"]
        return patch

    @staticmethod
    def create_standard_augmentation_pipeline() -> A.Compose:
        """Create an augmentation pipeline for 3D patches."""
        return A.Compose(
            [
                A.CubicSymmetry(p=0.7),
                A.CoarseDropout3D(p=0.3),
            ]
        )
