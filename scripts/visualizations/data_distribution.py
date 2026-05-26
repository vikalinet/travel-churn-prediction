"""
Визуализация распределения данных и корреляций.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_data_distribution(
    df: pd.DataFrame, save_path: str = "reports/data_distribution.png"
):
    """
    Визуализация распределения данных.

    Args:
        df: DataFrame с данными
        save_path: Путь для сохранения графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Распределение целевой переменной
    ax = axes[0, 0]
    if "Target" in df.columns:
        target_counts = df["Target"].value_counts()
        sns.barplot(
            x=target_counts.index, y=target_counts.values, ax=ax, palette="viridis"
        )
        ax.set_title(
            "Распределение целевой переменной (Churn)", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Churn")
        ax.set_ylabel("Количество")
        for i, v in enumerate(target_counts.values):
            ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Target колонка не найдена", ha="center", va="center")
        ax.set_title("Распределение целевой переменной")

    # 2. Распределение возраста
    ax = axes[0, 1]
    if "Age" in df.columns:
        sns.histplot(df["Age"], kde=True, ax=ax, color="skyblue")
        ax.set_title("Распределение возраста", fontsize=14, fontweight="bold")
        ax.set_xlabel("Возраст")
        ax.set_ylabel("Количество")
    else:
        ax.text(0.5, 0.5, "Age колонка не найдена", ha="center", va="center")

    # 3. Распределение услуг
    ax = axes[1, 0]
    if "ServicesOpted" in df.columns:
        sns.countplot(data=df, x="ServicesOpted", ax=ax, palette="coolwarm")
        ax.set_title("Распределение количества услуг", fontsize=14, fontweight="bold")
        ax.set_xlabel("Количество услуг")
        ax.set_ylabel("Количество клиентов")
    else:
        ax.text(0.5, 0.5, "ServicesOpted колонка не найдена", ha="center", va="center")

    # 4. Корреляционная матрица
    ax = axes[1, 1]
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
        ax.set_title("Матрица корреляций", fontsize=14, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Мало числовых колонок", ha="center", va="center")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")
