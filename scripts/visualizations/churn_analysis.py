"""
Анализ оттока клиентов.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_churn_analysis(
    df: pd.DataFrame, save_path: str = "reports/churn_analysis.png"
):
    """
    Анализ оттока по разным признакам.

    Args:
        df: DataFrame с данными
        save_path: Путь для сохранения графика
    """
    if "Target" not in df.columns:
        logger.warning("Target колонка не найдена")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Churn по возрасту
    ax = axes[0, 0]
    if "Age" in df.columns:
        df_age_churn = df.groupby(pd.cut(df["Age"], bins=[0, 30, 40, 50, 60, 100]))[
            "Target"
        ].mean()
        df_age_churn.plot(kind="bar", ax=ax, color="#e74c3c", edgecolor="black")
        ax.set_title(
            "Средний уровень оттока по возрасту", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Возраст")
        ax.set_ylabel("Доля оттока")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "Age колонка не найдена", ha="center", va="center")

    # 2. Churn по количеству услуг
    ax = axes[0, 1]
    if "ServicesOpted" in df.columns:
        df_services_churn = df.groupby("ServicesOpted")["Target"].mean()
        df_services_churn.plot(kind="bar", ax=ax, color="#3498db", edgecolor="black")
        ax.set_title(
            "Средний уровень оттока по количеству услуг", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Количество услуг")
        ax.set_ylabel("Доля оттока")
    else:
        ax.text(0.5, 0.5, "ServicesOpted колонка не найдена", ha="center", va="center")

    # 3. Распределение Churn
    ax = axes[1, 0]
    if "Target" in df.columns:
        churn_counts = df["Target"].value_counts()
        churn_percent = churn_counts / len(df) * 100
        ax.pie(
            churn_percent.values,
            labels=["Остался", "Ушёл"],
            autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"],
            startangle=90,
        )
        ax.set_title("Распределение оттока", fontsize=12, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Target колонка не найдена", ha="center", va="center")

    # 4. Box plot Age vs Churn
    ax = axes[1, 1]
    if "Age" in df.columns and "Target" in df.columns:
        sns.boxplot(data=df, x="Target", y="Age", ax=ax, palette="Set2")
        ax.set_title("Возраст по статусу оттока", fontsize=12, fontweight="bold")
        ax.set_xlabel("Churn (0 - остался, 1 - ушёл)")
        ax.set_ylabel("Возраст")
    else:
        ax.text(0.5, 0.5, "Недостаточно данных", ha="center", va="center")

    plt.suptitle("Анализ оттока клиентов", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")
