from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from .base import ImageProcessor


class Metrics(ImageProcessor):
    """Class for computing segmentation metrics."""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute segmentation metrics between ground truth and prediction."""
        # Flatten arrays
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Basic metrics from sklearn
        metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "iou": jaccard_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        }

        # Calculate additional metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Add more metrics
        metrics.update(
            {
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                "sensitivity": metrics["recall"],
                "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                "volume_similarity": 1 - abs((fn - fp) / (2 * tp + fp + fn))
                if (2 * tp + fp + fn) > 0
                else 0,
            }
        )

        return metrics

    def evaluate_patch(
        self, patch: np.ndarray, mask: np.ndarray, model=None, method: str = "otsu"
    ) -> Dict[str, float]:
        """Evaluate a single patch by comparing prediction with ground truth."""
        from src.processors.prediction_processor import (
            Predictor,
        )  # Import here to avoid circular imports

        # Get prediction
        predictor = Predictor()
        pred = predictor.predict_patch(patch, model, method)

        # Ensure mask is binary
        mask = self.ensure_binary_mask(mask)

        return self.compute_metrics(mask, pred)
