import numpy as np
import cv2
from skimage.filters import frangi
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import tensorflow as tf
from .base import ImageProcessor


class Predictor(ImageProcessor):
    """Class for making predictions using both classical and deep learning methods."""

    def __init__(self):
        self.classical_methods = ["otsu", "adaptive_mean", "adaptive_gaussian", "frangi"]

    def apply_classical_threshold(self, image: np.ndarray, method: str = "otsu") -> np.ndarray:
        """Apply classical thresholding methods."""
        # Normalize and scale to [0, 255]
        image = (self.normalize_image(image) * 255).astype(np.uint8)

        if method == "otsu":
            _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive_mean":
            mask = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 2
            )
        elif method == "adaptive_gaussian":
            mask = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
            )
        elif method == "frangi":
            mask = frangi(image)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            raise ValueError(f"Unknown thresholding method: {method}")

        return (mask > 0).astype(np.uint8)

    def predict_patch(
        self, patch: np.ndarray, model: Optional[tf.keras.Model] = None, method: str = "otsu"
    ) -> np.ndarray:
        """Generate prediction for a single patch."""
        # Normalize input patch
        patch_norm = self.normalize_image(patch)

        if model is not None:
            # Deep learning prediction
            patch_input = patch_norm[np.newaxis, ..., np.newaxis]
            pred = model.predict(patch_input, verbose=0)
            pred = pred[0, ..., 0]
            pred = (pred > 0.5).astype(np.uint8)
        else:
            # Classical thresholding
            pred = np.zeros_like(patch, dtype=np.uint8)
            for z in range(patch.shape[0]):
                slice_norm = (patch_norm[z, :, :] * 255).astype(np.uint8)
                pred[z, :, :] = self.apply_classical_threshold(slice_norm, method)

        return pred

    @staticmethod
    def load_deep_models(models_dir: Path, dataset_name: str = None) -> Dict[str, tf.keras.Model]:
        """Load deep learning models from the models directory.

        Args:
            models_dir: Base directory containing all models
            dataset_name: Optional name of dataset to filter models. If provided,
                         only loads models trained on this dataset.

        Returns:
            Dictionary mapping model names to loaded models
        """
        models = {}

        # Get all model directories for the specified dataset
        if dataset_name:
            model_dirs = list((models_dir / dataset_name).glob("*"))
        else:
            # If no dataset specified, get all models from all datasets
            model_dirs = []
            for dataset_path in models_dir.glob("*"):
                if dataset_path.is_dir():
                    model_dirs.extend(dataset_path.glob("*"))

        for model_dir in model_dirs:
            if not model_dir.is_dir():
                continue

            # Get latest timestamp directory
            latest_model = sorted(model_dir.glob("*"))[-1]

            # Get model file (*.h5)
            model_file = sorted(latest_model.glob("*.h5"))[-1]

            # Extract model name from directory structure
            # Path format: models_dir/dataset_name/model_name_augmentation/timestamp/model.h5
            model_name = model_dir.name.split("_")[0]  # Get base model name without augmentation

            # Load model
            model = tf.keras.models.load_model(str(model_file))
            models[model_name] = model

        return models
