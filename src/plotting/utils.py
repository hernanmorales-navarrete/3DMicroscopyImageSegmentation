from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from tifffile import tifffile

from src.inference.utils import apply_threshold_to_image_and_convert_to_dtype


def compute_metrics_with_true_and_pred(
    image_true: np.ndarray, image_pred: np.ndarray, metrics_to_compute: List[str]
) -> Dict[str, float]:
    """Compute segmentation metrics between ground truth and prediction."""

    # Ensure that masks are binary
    image_true = apply_threshold_to_image_and_convert_to_dtype(image_true, 0, int)
    image_pred = apply_threshold_to_image_and_convert_to_dtype(image_pred, 0, int)

    # Flatten arrays
    y_true = image_true.flatten()
    y_pred = image_pred.flatten()

    metrics = {}

    # Compute confusion matrix elements if needed for custom metrics
    need_confusion_matrix = any(
        m in metrics_to_compute for m in ["specificity", "dice", "volume_similarity"]
    )

    if need_confusion_matrix:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

    # Compute only requested metrics
    if "precision" in metrics_to_compute:
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)

    if "recall" in metrics_to_compute:
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

    if "f1" in metrics_to_compute:
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    if "iou" in metrics_to_compute:
        metrics["iou"] = jaccard_score(y_true, y_pred, zero_division=0)

    if "accuracy" in metrics_to_compute:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

    if "specificity" in metrics_to_compute:
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    if "sensitivity" in metrics_to_compute:
        if "recall" not in metrics:
            metrics["sensitivity"] = recall_score(y_true, y_pred, zero_division=0)
        else:
            metrics["sensitivity"] = metrics["recall"]

    if "dice" in metrics_to_compute:
        metrics["dice"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    if "volume_similarity" in metrics_to_compute:
        metrics["volume_similarity"] = (
            1 - abs((fn - fp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0
        )

    return metrics


def read_and_compute_metrics(
    image_true_path: Path, image_pred_path: Path, metrics_to_compute: List[str]
):
    image_true = tifffile.imread(str(image_true_path))
    image_pred = tifffile.imread(str(image_pred_path))

    return compute_metrics_with_true_and_pred(image_true, image_pred, metrics_to_compute)


def results_to_dataframe(results):
    return pd.DataFrame(results)


def plot_metrics_boxplots(df, save_path=None):
    available_metrics = [m for m in df.columns if m != "method"]

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        df.boxplot(column=metric, by="method", ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")
        ax.set_xlabel("method")
        ax.set_ylabel(metric)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    for idx in range(n_metrics, 9):
        fig.delaxes(axes[idx])

    plt.suptitle("")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
