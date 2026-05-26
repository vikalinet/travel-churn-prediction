"""
Визуализация и сравнение моделей.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Визуализация и сравнение моделей."""

    @staticmethod
    def plot_comparison(
        results_df: pd.DataFrame, save_path: str = "reports/model_comparison.png"
    ):
        """
        Визуализация сравнения моделей.

        Args:
            results_df: DataFrame с результатами обучения
            save_path: Путь для сохранения графика
        """
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ["accuracy", "f1_score", "roc_auc", "precision"]
        metric_names = ["Accuracy", "F1-Score", "ROC AUC", "Precision"]

        for ax, metric, name in zip(axes.flat, metrics, metric_names):
            if metric in results_df.columns:
                sns.barplot(
                    data=results_df, x="model_name", y=metric, ax=ax, palette="viridis"
                )
                ax.set_title(name, fontsize=14, fontweight="bold")
                ax.set_ylabel("")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_ylim(0, 1)

                for i, v in enumerate(results_df[metric]):
                    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"График сохранён в {save_path}")
        plt.close()

    @staticmethod
    def save_results(
        results_df: pd.DataFrame, save_path: str = "reports/training_results.csv"
    ):
        """
        Сохранение результатов в CSV.

        Args:
            results_df: DataFrame с результатами
            save_path: Путь для сохранения
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=False)
        logger.info(f"Результаты сохранены в {save_path}")
