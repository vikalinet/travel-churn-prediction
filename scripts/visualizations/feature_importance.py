"""
Визуализация важности признаков.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_feature_importance(
    df: pd.DataFrame, save_path: str = "reports/feature_importance.png"
):
    """
    Визуализация важности признаков через корреляцию.

    Args:
        df: DataFrame с данными
        save_path: Путь для сохранения графика
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if "Target" not in numeric_df.columns:
        logger.warning("Target колонка не найдена в числовых данных")
        return

    # Корреляция с целевой переменной
    correlations = (
        numeric_df.corr()["Target"].drop("Target").abs().sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(
        x=correlations.values,
        y=correlations.index,
        ax=ax,
        palette="viridis",
        edgecolor="black",
    )

    ax.set_title(
        "Важность признаков (по корреляции с Target)", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Абсолютная корреляция")
    ax.set_ylabel("Признак")

    for i, v in enumerate(correlations.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")
