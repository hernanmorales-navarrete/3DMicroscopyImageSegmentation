import numpy as np
import tifffile as tiff
from typing import Union, Tuple, Dict, Any
from pathlib import Path


class ImageProcessor:
    """Base class for image processing operations."""

    @staticmethod
    def load_image(image_path: Union[str, Path]) -> np.ndarray:
        """Load an image from file."""
        return tiff.imread(str(image_path))

    @staticmethod
    def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save an image to file."""
        tiff.imwrite(str(output_path), image)

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0,1] range."""
        return (image - image.min()) / (image.max() - image.min() + np.finfo(float).eps)

    @staticmethod
    def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
        """Ensure mask is binary (0 or 1)."""
        return (mask > 0).astype(np.uint8)
