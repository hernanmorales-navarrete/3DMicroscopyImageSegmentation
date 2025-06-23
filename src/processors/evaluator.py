from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
import tensorflow as tf
from patchify import patchify, unpatchify
import re
from typing import List, Dict, Tuple, Any

from src.config import BATCH_SIZE, PATCH_SIZE, PATCH_STEP
from src.processors import Metrics, Predictor


class Evaluator:
    """Class for evaluating segmentation methods on both patches and complete images."""

    def __init__(self):
        self.metrics = Metrics()
        self.predictor = Predictor()

    def extract_patch_info(self, filename: Path) -> Tuple[str, tuple, tuple, tuple]:
        """Extract information from patch filename.

        Args:
            filename: Patch filename (e.g. 'image1_orig_512_512_128_pad_520_520_130_npatches_4_8_8_patch_0000.tif')

        Returns:
            tuple: (image_name, original_shape, padded_shape, n_patches)
        """
        pattern = r"(.+)_orig_(\d+)_(\d+)_(\d+)(?:_pad_(\d+)_(\d+)_(\d+))?_npatches_(\d+)_(\d+)_(\d+)_patch_\d+\.tif"
        match = re.match(pattern, filename.name)

        if not match:
            raise ValueError(f"Invalid patch filename format: {filename}")

        image_name = match.group(1)
        orig_shape = (int(match.group(2)), int(match.group(3)), int(match.group(4)))

        # Get padded shape if it exists, otherwise use original shape
        if match.group(5):
            padded_shape = (int(match.group(5)), int(match.group(6)), int(match.group(7)))
            n_patches = (int(match.group(8)), int(match.group(9)), int(match.group(10)))
        else:
            padded_shape = orig_shape
            n_patches = (int(match.group(8)), int(match.group(9)), int(match.group(10)))

        return image_name, orig_shape, padded_shape, n_patches

    def reconstruct_complete_image(
        self,
        patch_predictions: List[np.ndarray],
        orig_shape: tuple,
        padded_shape: tuple,
        n_patches: tuple,
    ) -> np.ndarray:
        """Reconstruct complete image from patch predictions.

        Args:
            patch_predictions: List of patch predictions
            orig_shape: Original image shape (z, y, x)
            padded_shape: Padded image shape (z, y, x)
            n_patches: Number of patches in each dimension (z, y, x)

        Returns:
            Reconstructed image of original shape
        """
        patches_reshaped = np.array(patch_predictions).reshape(
            n_patches[0], n_patches[1], n_patches[2], *PATCH_SIZE
        )
        reconstructed = unpatchify(patches_reshaped, padded_shape)

        reconstructed = reconstructed[
            0: orig_shape[0],
            0: orig_shape[1], 
            0: orig_shape[2]
        ]
        return reconstructed

    def process_deep_learning_batch(
        self,
        current_batch: List[Tuple[str, np.ndarray, np.ndarray, int]],
        model: tf.keras.Model,
        model_name: str,
        augmentation_type: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple[int, np.ndarray]]]]:
        """Process a batch of patches for deep learning evaluation.

        Args:
            current_batch: List of tuples (image_name, image, mask, patch_idx)
            model: Deep learning model
            model_name: Name of the model
            augmentation_type: Type of augmentation used

        Returns:
            Tuple of (results, predictions_dict)
        """
        batch_results = []
        predictions_dict = {}

        # Prepare batch data
        images = [item[1] for item in current_batch]
        masks = [item[2] for item in current_batch]

        # Get predictions for batch
        predictions = self.predictor.predict_batch_patches(
            images, model=model, batch_size=len(images)
        )

        # Process each prediction
        for (img_name, _, mask, p_idx), pred in zip(current_batch, predictions):
            # Store prediction
            if img_name not in predictions_dict:
                predictions_dict[img_name] = []
            predictions_dict[img_name].append((p_idx, pred))

            # Evaluate patch-level metrics
            mask_binary = self.metrics.ensure_binary_mask(mask)
            result = self.metrics.compute_metrics(mask_binary, pred)
            result.update(
                {
                    "method": model_name,
                    "augmentation": augmentation_type,
                    "image_path": f"{img_name}_patch_{p_idx}",
                    "evaluation_type": "patch",
                }
            )
            batch_results.append(result)

        return batch_results, predictions_dict

    def evaluate_deep_learning_model(
        self,
        model_name: str,
        model_path: str,
        augmentation_type: str,
        patch_paths: List[Path],
        patch_masks: List[Path],
        evaluation_type: str = "complete",
    ) -> List[Dict[str, Any]]:
        """Evaluate a deep learning model on patches and reconstructed images.

        Args:
            model_name: Name of the model
            model_path: Path to the model file
            augmentation_type: Type of augmentation used
            patch_paths: List of paths to patch images
            patch_masks: List of paths to patch mask files
            evaluation_type: Type of evaluation (complete or patch)

        Returns:
            List of evaluation results
        """
        all_results = []
        image_predictions = {}
        image_data = {}  # Store metadata for each image
        current_batch = []

        # Load model
        logger.info(f"Loading model {model_name}")
        model = tf.keras.models.load_model(model_path)

        try:
            # Process patches
            for img_path, mask_path in tqdm(
                zip(patch_paths, patch_masks),
                total=len(patch_paths),
                desc=f"Processing patches for {model_name}",
            ):
                try:
                    if evaluation_type == "complete":
                        # For complete image evaluation, we need patch info for reconstruction
                        image_name, orig_shape, padded_shape, n_patches = self.extract_patch_info(
                            img_path
                        )
                        patch_idx = int(re.search(r"patch_(\d+)\.tif$", img_path.name).group(1))

                        # Store image metadata
                        if image_name not in image_data:
                            image_data[image_name] = {
                                "orig_shape": orig_shape,
                                "padded_shape": padded_shape,
                                "n_patches": n_patches,
                                "mask_patches": [],
                            }

                        # Load image and mask
                        image = self.predictor.load_image(img_path)
                        mask = self.predictor.load_image(mask_path)
                        image_data[image_name]["mask_patches"].append((patch_idx, mask))

                        # Add to current batch
                        current_batch.append((image_name, image, mask, patch_idx))
                    else:  # patch evaluation
                        # For patch evaluation, we don't need reconstruction info
                        image = self.predictor.load_image(img_path)
                        mask = self.predictor.load_image(mask_path)
                        current_batch.append((str(img_path), image, mask, 0))

                    # Process batch when it reaches BATCH_SIZE
                    if len(current_batch) == BATCH_SIZE:
                        batch_results, batch_predictions = self.process_deep_learning_batch(
                            current_batch, model, model_name, augmentation_type
                        )

                        if evaluation_type == "complete":
                            # For complete evaluation, store predictions for reconstruction
                            for img_name, preds in batch_predictions.items():
                                if img_name not in image_predictions:
                                    image_predictions[img_name] = []
                                image_predictions[img_name].extend(preds)
                        else:  # patch evaluation
                            # For patch evaluation, store patch-level results
                            all_results.extend(batch_results)

                        current_batch = []

                except Exception as e:
                    logger.error(f"Error processing patch {img_path}: {e}")
                    continue

            # Process remaining patches
            if current_batch:
                batch_results, batch_predictions = self.process_deep_learning_batch(
                    current_batch, model, model_name, augmentation_type
                )

                if evaluation_type == "complete":
                    # For complete evaluation, store predictions for reconstruction
                    for img_name, preds in batch_predictions.items():
                        if img_name not in image_predictions:
                            image_predictions[img_name] = []
                        image_predictions[img_name].extend(preds)
                else:  # patch evaluation
                    # For patch evaluation, store patch-level results
                    all_results.extend(batch_results)

            # For complete evaluation, reconstruct and evaluate complete images
            if evaluation_type == "complete":
                logger.info(f"Evaluating complete images for {model_name}")
                for image_name, data in image_data.items():
                    if image_name not in image_predictions:
                        continue

                    # Sort predictions and masks by patch index
                    sorted_preds = [
                        p[1] for p in sorted(image_predictions[image_name], key=lambda x: x[0])
                    ]
                    sorted_masks = [m[1] for m in sorted(data["mask_patches"], key=lambda x: x[0])]

                    # Reconstruct complete image and mask
                    reconstructed_pred = self.reconstruct_complete_image(
                        sorted_preds, data["orig_shape"], data["padded_shape"], data["n_patches"]
                    )
                    reconstructed_mask = self.reconstruct_complete_image(
                        sorted_masks, data["orig_shape"], data["padded_shape"], data["n_patches"]
                    )

                    # Evaluate complete image
                    mask_binary = self.metrics.ensure_binary_mask(reconstructed_mask)
                    result = self.metrics.compute_metrics(mask_binary, reconstructed_pred)
                    result.update(
                        {
                            "method": model_name,
                            "augmentation": augmentation_type,
                            "image_path": image_name,
                            "evaluation_type": evaluation_type,
                        }
                    )
                    all_results.append(result)

        finally:
            # Clean up
            del model
            tf.keras.backend.clear_session()

        return all_results

    def evaluate_classical_method(
        self,
        method: str,
        image_path: Path,
        mask_path: Path,
    ) -> List[Dict[str, Any]]:
        """Evaluate a classical method on both complete image and patches.

        Args:
            method: Classical method name
            image_path: Path to the complete image
            mask_path: Path to the complete mask

        Returns:
            List of evaluation results
        """
        results = []

        try:
            # Load image and mask
            image = self.predictor.load_image(image_path)
            mask = self.predictor.load_image(mask_path)
            mask_binary = self.metrics.ensure_binary_mask(mask)

            # Get prediction for complete image
            pred = self.predictor.predict_patch(image, method=method)

            # Evaluate complete image
            result = self.metrics.compute_metrics(mask_binary, pred)
            result.update(
                {
                    "method": f"Classical_{method}",
                    "augmentation": "Classical",
                    "image_path": str(image_path),
                    "evaluation_type": "complete",
                }
            )
            results.append(result)

            # Break down prediction and mask into patches
            pred_patches = patchify(pred, PATCH_SIZE, PATCH_STEP)
            mask_patches = patchify(mask_binary, PATCH_SIZE, PATCH_STEP)

            # Evaluate each patch
            pred_patches = pred_patches.reshape(-1, *PATCH_SIZE)
            mask_patches = mask_patches.reshape(-1, *PATCH_SIZE)

            for patch_idx in range(len(pred_patches)):
                result = self.metrics.compute_metrics(
                    mask_patches[patch_idx], pred_patches[patch_idx]
                )
                result.update(
                    {
                        "method": f"Classical_{method}",
                        "augmentation": "Classical",
                        "image_path": f"{str(image_path)}_patch_{patch_idx}",
                        "evaluation_type": "patch",
                    }
                )
                results.append(result)

        except Exception as e:
            logger.error(f"Error processing {image_path} with {method}: {e}")

        return results

    def evaluate_all_methods(
        self,
        regular_patch_paths: List[Path],
        regular_patch_masks: List[Path],
        reconstruction_patch_paths: List[Path],
        reconstruction_patch_masks: List[Path],
        complete_image_paths: List[Path],
        complete_masks: List[Path],
        deep_models: Dict[str, Tuple[str, str]],
    ) -> pd.DataFrame:
        """Evaluate all methods on patches and complete images.

        Args:
            regular_patch_paths: List of paths to regular patch image files (for patch-level evaluation)
            regular_patch_masks: List of paths to regular patch mask files (for patch-level evaluation)
            reconstruction_patch_paths: List of paths to reconstruction patch image files (for complete image evaluation)
            reconstruction_patch_masks: List of paths to reconstruction patch mask files (for complete image evaluation)
            complete_image_paths: List of paths to complete image files
            complete_masks: List of paths to complete mask files
            deep_models: Dictionary mapping model names to tuples of (model_path, augmentation_type)

        Returns:
            DataFrame with results for both patch-level and complete image evaluations
        """
        all_results = []

        # Process deep learning models
        logger.info("Processing deep learning models...")
        for model_name, (model_path, augmentation_type) in deep_models.items():
            try:
                # Evaluate on complete images using reconstruction patches
                results = self.evaluate_deep_learning_model(
                    model_name,
                    model_path,
                    augmentation_type,
                    reconstruction_patch_paths,
                    reconstruction_patch_masks,
                    evaluation_type="complete",
                )
                all_results.extend(results)

                # Evaluate on regular patches
                results = self.evaluate_deep_learning_model(
                    model_name,
                    model_path,
                    augmentation_type,
                    regular_patch_paths,
                    regular_patch_masks,
                    evaluation_type="patch",
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                continue

        # Process classical methods
        logger.info("Processing classical methods...")
        for img_path, mask_path in tqdm(
            zip(complete_image_paths, complete_masks),
            total=len(complete_image_paths),
            desc="Processing classical methods",
        ):
            for method in self.predictor.classical_methods:
                try:
                    results = self.evaluate_classical_method(method, img_path, mask_path)
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error evaluating {method} on {img_path}: {e}")
                    continue

        return pd.DataFrame(all_results)
