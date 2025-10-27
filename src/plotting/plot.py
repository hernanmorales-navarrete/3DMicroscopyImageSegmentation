import concurrent.futures
from pathlib import Path

from src.config import (
    AVAILABLE_METRICS,
    MAX_WORKERS,
    OUTPUT_EXTENSION,
)
from src.plotting.utils import (
    plot_metrics_boxplots,
    read_and_compute_metrics,
    results_to_dataframe,
)


class Plotter:
    def __init__(
        self,
        predictions_dir: Path,
        ground_truth_mask_patches_dir: Path,
        ground_truth_masks_dir: Path,
        dataset_name: Path,
        output_dir: Path,
    ):
        self.image_level_dir = predictions_dir / "image_level"
        self.patch_level_dir = predictions_dir / "patch_level"
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.ground_truth_masks_dir = ground_truth_masks_dir

        if not (self.image_level_dir.exists() and self.patch_level_dir.exists()):
            raise Exception(
                f"{predictions_dir} does not contain predictions at patch_level or image_level"
            )

        # Create a dictionary lookup to search for files given an image name
        self.patches_dict = {
            p.stem: p for p in ground_truth_mask_patches_dir.rglob(f"*{OUTPUT_EXTENSION}")
        }

    def plot_patches_metrics(self):
        results = []
        for method in self.patch_level_dir.glob("*"):
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(
                        read_and_compute_metrics,
                        image_path,
                        self.patches_dict[image_path.stem],
                        AVAILABLE_METRICS,
                    )
                    for image_path in method.glob("*")
                ]

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    result["method"] = method.stem
                    results.append(result)

        df = results_to_dataframe(results)
        plot_metrics_boxplots(df, self.output_dir / (f"plot_{self.dataset_name}_patches"))

    def plot_images_metrics(self):
        results = []
        for method in self.image_level_dir.glob("*"):
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(
                        read_and_compute_metrics,
                        image_path,
                        self.ground_truth_masks_dir / image_path.name,
                        AVAILABLE_METRICS,
                    )
                    for image_path in method.glob("*")
                ]

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    result["method"] = method.stem
                    results.append(result)

        df = results_to_dataframe(results)
        plot_metrics_boxplots(df, self.output_dir / (f"plot_{self.dataset_name}_complete"))
