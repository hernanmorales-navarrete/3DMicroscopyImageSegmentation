import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger


class Visualizer:
    """Class for handling visualization and plotting."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

    def generate_plots(
        self,
        results_df: pd.DataFrame,
        dataset_name: str,
        metrics: List[str],
    ):
        """Generate all plots and save summary tables.

        Args:
            results_df: DataFrame with evaluation results
            dataset_name: Name of the dataset
            metrics: List of metrics to plot
        """
        # Create output directory
        dataset_output_dir = self.output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots for each evaluation type
        for eval_type in ["patch", "complete"]:
            logger.info(f"Generating plots for {eval_type}-level evaluations...")
            type_results = results_df[results_df["evaluation_type"] == eval_type]

            # Skip if no results for this type
            if type_results.empty:
                logger.warning(f"No results found for {eval_type}-level evaluation")
                continue

            # Use violin plots for patches and box plots for complete images
            if eval_type == "patch":
                self.plot_violin(type_results, metrics, f"{dataset_name}_patches")
                self.plot_radar_chart(type_results, f"{dataset_name}_patches")
                summary = self.create_summary_table(type_results)
                summary.to_csv(dataset_output_dir / f"metrics_summary_{dataset_name}_patches.csv")
            else:  # complete images
                self.plot_box(type_results, metrics, f"{dataset_name}_complete")
                self.plot_radar_chart(type_results, f"{dataset_name}_complete")
                summary = self.create_summary_table(type_results)
                summary.to_csv(dataset_output_dir / f"metrics_summary_{dataset_name}_complete.csv")

    def plot_violin(self, df: pd.DataFrame, metrics: List[str], dataset_name: str) -> None:
        """Create violin plots for metrics."""
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            sns.violinplot(data=df, x="method", y=metric, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")

        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"metrics_violin_plots_{dataset_name}.svg",
            bbox_inches="tight",
            format="svg",
        )
        plt.close()

    def plot_box(self, df: pd.DataFrame, metrics: List[str], dataset_name: str) -> None:
        """Create box plots for metrics."""
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            sns.boxplot(data=df, x="method", y=metric, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")

        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"metrics_box_plots_{dataset_name}.svg",
            bbox_inches="tight",
            format="svg",
        )
        plt.close()

    def plot_radar_chart(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Create radar chart comparing methods."""
        metrics = [
            "accuracy",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "f1",
            "dice",
            "iou",
            "volume_similarity",
        ]

        means = df.groupby("method")[metrics].mean()
        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        for idx, method in enumerate(means.index):
            values = means.loc[method].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle="solid", label=method)
            ax.fill(angles, values, alpha=0.1)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.title("Methods Comparison - Radar Chart")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"methods_radar_chart_{dataset_name}.svg",
            bbox_inches="tight",
            format="svg",
        )
        plt.close()

    @staticmethod
    def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
        """Create summary table with mean and std metrics for each method."""
        metrics = [
            "accuracy",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "f1",
            "dice",
            "iou",
            "volume_similarity",
        ]

        mean_df = df.groupby("method")[metrics].mean()
        std_df = df.groupby("method")[metrics].std()

        summary = pd.DataFrame(index=mean_df.index, columns=metrics)
        for metric in metrics:
            summary[metric] = (
                mean_df[metric].round(3).astype(str) + " ± " + std_df[metric].round(3).astype(str)
            )

        return summary
