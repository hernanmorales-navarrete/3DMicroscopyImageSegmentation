import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any


class Visualizer:
    """Class for handling visualization and plotting."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

    def plot_violin(self, df: pd.DataFrame, metrics: List[str]) -> None:
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
            self.output_dir / "metrics_violin_plots.svg", bbox_inches="tight", format="svg"
        )
        plt.close()

    def plot_radar_chart(self, df: pd.DataFrame) -> None:
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
        plt.savefig(self.output_dir / "methods_radar_chart.svg", bbox_inches="tight", format="svg")
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
