import numpy as np
import cv2
from skimage.filters import frangi
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import tensorflow as tf
from .base import ImageProcessor
from ..config import BATCH_SIZE


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
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 2
            )
        elif method == "adaptive_gaussian":
            mask = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 2
            )
        elif method == "frangi":
            mask = frangi(image)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = (mask > 0).astype(np.uint8) * 255
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

    def predict_batch_patches(
        self,
        patches: List[np.ndarray],
        model: Optional[tf.keras.Model] = None,
        method: str = "otsu",
        batch_size: int = BATCH_SIZE,
    ) -> List[np.ndarray]:
        """Generate predictions for a batch of patches efficiently.

        Args:
            patches: List of patches to predict on
            model: Optional deep learning model
            method: Classical method to use if no model provided
            batch_size: Batch size for deep learning predictions

        Returns:
            List of predictions corresponding to input patches
        """
        predictions = []

        if model is not None:
            # Process patches in batches for deep learning
            for i in range(0, len(patches), batch_size):
                batch = patches[i : i + batch_size]

                # Normalize and prepare batch
                batch_norm = np.stack([self.normalize_image(p) for p in batch])
                batch_input = batch_norm[..., np.newaxis]

                # Get predictions for batch
                batch_preds = model.predict(batch_input, verbose=0)
                batch_preds = (batch_preds[..., 0] > 0.5).astype(np.uint8)

                # Add individual predictions to results
                predictions.extend([pred for pred in batch_preds])
        else:
            # For classical methods, process each patch individually
            for patch in patches:
                pred = self.predict_patch(patch, method=method)
                predictions.append(pred)

        return predictions

    @staticmethod
    def load_deep_models(models_dir: Path, dataset_name: str = None) -> Dict[str, tuple[str, str]]:
        """Get paths to deep learning models from the models directory.

        Args:
            models_dir: Base directory containing all models
            dataset_name: Optional name of dataset to filter models. If provided,
                         only includes models trained on this dataset.

        Returns:
            Dictionary mapping model names to tuples of (model_path, augmentation_type)
        """
        models_info = {}

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

            # Extract model name and augmentation type from directory structure
            # Path format: models_dir/dataset_name/model_name_augmentation/timestamp/model.h5
            dir_parts = model_dir.name.split("_")
            model_name = dir_parts[0]  # Get base model name
            augmentation_type = (
                dir_parts[-1] if len(dir_parts) > 1 else "NONE"
            )  # Get augmentation type

            models_info[model_name] = (str(model_file), augmentation_type)

        return models_info
