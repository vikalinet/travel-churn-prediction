"""
Сравнение моделей.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from scripts.visualizations.utils import load_training_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_model_comparison(save_path: str = "reports/model_comparison.png"):
    """
    Визуализация сравнения ML моделей.

    Args:
        save_path: Путь для сохранения графика
    """
    df_models = load_training_results()

    # Переименование колонок для удобства
    df_models = df_models.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "f1_score": "F1-Score",
            "roc_auc": "ROC AUC",
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Accuracy
    ax = axes[0]
    sns.barplot(
        data=df_models,
        x="Model",
        y="Accuracy",
        ax=ax,
        palette="viridis",
        edgecolor="black",
    )
    ax.set_title("Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    for i, v in enumerate(df_models["Accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)

    # F1-Score
    ax = axes[1]
    sns.barplot(
        data=df_models,
        x="Model",
        y="F1-Score",
        ax=ax,
        palette="plasma",
        edgecolor="black",
    )
    ax.set_title("F1-Score", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    for i, v in enumerate(df_models["F1-Score"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)

    # ROC AUC
    ax = axes[2]
    sns.barplot(
        data=df_models,
        x="Model",
        y="ROC AUC",
        ax=ax,
        palette="cividis",
        edgecolor="black",
    )
    ax.set_title("ROC AUC", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    for i, v in enumerate(df_models["ROC AUC"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)

    plt.suptitle("Сравнение ML моделей", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")


def plot_automl_comparison(save_path: str = "reports/model_comparison_with_automl.png"):
    """
    Сравнение лучших моделей с тюнингованными версиями.

    Args:
        save_path: Путь для сохранения графика
    """
    df_full = load_training_results()

    # Берём основные модели + tuned версии
    df_comparison = df_full[
        df_full["model"].isin(
            [
                "GradientBoosting",
                "XGBoost",
                "XGBoost_Tuned",
                "RandomForest",
                "RandomForest_Tuned",
            ]
        )
    ].copy()

    # Переименование колонок
    df_comparison = df_comparison.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "f1_score": "F1-Score",
            "roc_auc": "ROC AUC",
            "precision": "Precision",
            "recall": "Recall",
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    metrics = ["Accuracy", "F1-Score", "ROC AUC", "Precision", "Recall"]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#e74c3c"]

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx]
        if metric in df_comparison.columns:
            sns.barplot(
                data=df_comparison,
                x="Model",
                y=metric,
                ax=ax,
                palette=[color] * len(df_comparison),
                edgecolor="black",
            )
            ax.set_title(metric, fontsize=14, fontweight="bold")
            ax.set_ylabel("")
            ax.set_ylim(0, 1)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            for i, v in enumerate(df_comparison[metric]):
                ax.text(
                    i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9
                )

    # Пустая ось для выравнивания
    axes[5].axis("off")

    plt.suptitle(
        "Сравнение моделей с подобранными гиперпараметрами",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")


def create_automl_leaderboard(save_path: str = "reports/automl_leaderboard.png"):
    """
    Визуализация лидерборда моделей.

    Args:
        save_path: Путь для сохранения графика
    """
    df_leaderboard = load_training_results()

    # Сортируем по F1-score
    df_leaderboard = df_leaderboard.sort_values("f1_score", ascending=False)

    # Добавляем столбец с гиперпараметрами
    params_map = {
        "GradientBoosting": "n_estimators=100",
        "XGBoost": "default",
        "XGBoost_Tuned": "lr=0.2, max_depth=5, n_est=150",
        "RandomForest": "n_estimators=100",
        "RandomForest_Tuned": "max_depth=10, min_split=2, n_est=150",
        "KNeighbors": "n_neighbors=5",
        "LogisticRegression": "C=1.0, max_iter=1000",
        "SVC": "probability=True",
    }
    df_leaderboard["params"] = df_leaderboard["model"].map(params_map)

    # Выбираем нужные колонки
    df_leaderboard = df_leaderboard[["model", "f1_score", "params", "roc_auc"]]
    df_leaderboard.columns = ["Model", "F1-Score", "Parameters", "ROC AUC"]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Создание таблицы
    table = ax.table(
        cellText=df_leaderboard.values,
        colLabels=df_leaderboard.columns,
        cellLoc="center",
        loc="center",
        colColours=["#2c3e50"] * len(df_leaderboard.columns),
        cellColours=[
            [
                "#ecf0f1" if j == 0 else "#bdc3c7"
                for j in range(len(df_leaderboard.columns))
            ]
            for i in range(len(df_leaderboard))
        ],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Цвет заголовков
    for i in range(len(df_leaderboard.columns)):
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Leaderboard моделей (по F1-Score)", fontsize=16, fontweight="bold", pad=20
    )
    ax.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")
