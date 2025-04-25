import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score
import cv2
from skimage.filters import frangi
from loguru import logger


def compute_metrics(y_true, y_pred):
    """Compute segmentation metrics between ground truth and prediction.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask

    Returns:
        Dictionary with metrics
    """
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Basic metrics from sklearn
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Same as sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    iou = jaccard_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate true positives, false positives, true negatives, false negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Dice coefficient (equivalent to F1-score but calculated differently)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    # Volume similarity
    volume_similarity = 1 - abs((fn - fp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,  # Same as sensitivity
        "sensitivity": recall,  # Added for clarity
        "specificity": specificity,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "volume_similarity": volume_similarity,
    }


def apply_classical_threshold(image, method="otsu"):
    """Apply classical thresholding methods.

    Args:
        image: Input grayscale image
        method: Thresholding method ('otsu', 'adaptive_mean', 'adaptive_gaussian', 'binary', 'frangi')

    Returns:
        Binary mask with values 0 and 1
    """
    if method == "otsu":
        _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive_mean":
        mask = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == "adaptive_gaussian":
        mask = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == "binary":
        _, mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    elif method == "frangi":
        # Apply Frangi filter for vessel enhancement
        mask = frangi(image)
        # Normalize and threshold
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = (mask > 0.5).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    # Convert from [0, 255] to [0, 1]
    return (mask > 0).astype(np.uint8)


def predict_patch(patch, model=None, method="otsu"):
    """Generate prediction for a single patch using either classical or deep learning method.

    Args:
        patch: Input image patch (3D volume of shape [z, y, x])
        model: Deep learning model (if None, uses classical method)
        method: Classical thresholding method (ignored if model is provided)

    Returns:
        Binary prediction mask with values 0 and 1
    """
    # Normalize input patch consistently for both methods
    patch_norm = (patch - patch.min()) / (patch.max() - patch.min() + np.finfo(float).eps)

    if model is not None:
        # Deep learning prediction
        # Add batch and channel dimensions
        patch_input = patch_norm[np.newaxis, ..., np.newaxis]

        # Get prediction
        pred = model.predict(patch_input, verbose=0)
        pred = pred[0, ..., 0]  # Remove batch and channel dimensions

        # Apply consistent thresholding
        pred = (pred > 0.5).astype(np.uint8)
    else:
        # Classical thresholding
        # For 3D volumes, process each z-slice separately
        pred = np.zeros_like(patch, dtype=np.uint8)
        for z in range(patch.shape[0]):
            # Get 2D slice and scale to [0, 255] for OpenCV
            slice_norm = (patch_norm[z, :, :] * 255).astype(np.uint8)
            # Apply threshold to 2D slice
            pred[z, :, :] = apply_classical_threshold(slice_norm, method)

    return pred  # Already binary (0 or 1)


def evaluate_patch(patch, mask, model=None, method="otsu"):
    """Evaluate a single patch by comparing prediction with ground truth.

    Args:
        patch: Input image patch (3D volume of shape [z, y, x])
        mask: Ground truth mask (3D volume of shape [z, y, x])
        model: Deep learning model (if None, uses classical method)
        method: Classical thresholding method (ignored if model is provided)

    Returns:
        Dictionary with metrics
    """
    # Get prediction
    pred = predict_patch(patch, model, method)

    # Ensure mask is binary (0 or 1)
    mask = (mask > 0).astype(np.uint8)

    return compute_metrics(mask, pred)
