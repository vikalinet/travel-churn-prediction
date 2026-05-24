"""
Генерация визуализаций для проекта.
Создаёт все необходимые графики и отчёты.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory():
    """Создание папки для отчётов."""
    Path("reports").mkdir(exist_ok=True)
    logger.info("Папка reports создана/проверена")


def load_data():
    """Загрузка обработанных данных."""
    data_path = "data/processed/processed_data.csv"
    if Path(data_path).exists():
        df = pd.read_csv(data_path)
        logger.info(f"Загружено {len(df)} строк из {data_path}")
        return df
    else:
        logger.error(f"Файл {data_path} не найден!")
        return None


def plot_data_distribution(df, save_path: str = "reports/data_distribution.png"):
    """Визуализация распределения данных."""
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")


def plot_model_comparison(save_path: str = "reports/model_comparison.png"):
    """Визуализация сравнения моделей."""
    # Данные из README (результаты обучения)
    models_data = {
        "Model": [
            "GradientBoosting",
            "XGBoost (Tuned)",
            "RandomForest (Tuned)",
            "Ensemble (AutoML)",
            "KNeighbors",
            "LogisticRegression",
        ],
        "Accuracy": [0.9110, 0.8953, 0.8848, 0.8950, 0.8690, 0.8320],
        "F1-Score": [0.7952, 0.7619, 0.7381, 0.7440, 0.6380, 0.5430],
        "ROC AUC": [0.9747, 0.9677, 0.9602, 0.9650, 0.9170, 0.8470],
    }

    df_models = pd.DataFrame(models_data)

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
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)

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
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)

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
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10)

    plt.suptitle("Сравнение ML моделей", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")


def plot_automl_comparison(save_path: str = "reports/model_comparison_with_automl.png"):
    """Сравнение AutoML с кастомными моделями."""
    # Данные для сравнения
    comparison_data = {
        "Model": ["GradientBoosting", "XGBoost", "RandomForest", "AutoML Ensemble"],
        "Accuracy": [0.9110, 0.8953, 0.8848, 0.8950],
        "F1-Score": [0.7952, 0.7619, 0.7381, 0.7440],
        "ROC AUC": [0.9747, 0.9677, 0.9602, 0.9650],
        "Precision": [0.8684, 0.8205, 0.7949, 0.8790],
        "Recall": [0.7333, 0.7111, 0.6889, 0.6440],
    }

    df_comparison = pd.DataFrame(comparison_data)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    metrics = ["Accuracy", "F1-Score", "ROC AUC", "Precision", "Recall"]
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#e74c3c"]

    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx]
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
                i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold", fontsize=10
            )

    # Пустая ось для выравнивания
    axes[5].axis("off")

    plt.suptitle(
        "Сравнение AutoML с кастомными моделями", fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")


def create_automl_leaderboard(save_path: str = "reports/automl_leaderboard.png"):
    """Визуализация лидерборда AutoML."""
    # Данные из обучения AutoML
    leaderboard_data = {
        "Model": [
            "GradientBoosting",
            "XGBoost",
            "RandomForest",
            "LogisticRegression",
            "KNN",
        ],
        "score": [0.9110, 0.8953, 0.8848, 0.8320, 0.8690],
        "params": [
            "n_estimators=100",
            "max_depth=6",
            "n_estimators=100",
            "C=1.0",
            "n_neighbors=5",
        ],
        "training_time": ["2.3s", "3.1s", "1.8s", "0.5s", "0.3s"],
    }

    df_leaderboard = pd.DataFrame(leaderboard_data)

    fig, ax = plt.subplots(figsize=(12, 6))

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
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax.set_title(
        "AutoML Leaderboard (по F1-Score)", fontsize=16, fontweight="bold", pad=20
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")


def create_feature_importance(df, save_path: str = "reports/feature_importance.png"):
    """Визуализация важности признаков."""
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")


def create_churn_analysis(df, save_path: str = "reports/churn_analysis.png"):
    """Анализ оттока по разным признакам."""
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {save_path}")


def main():
    """Основная функция генерации всех визуализаций."""
    logger.info("=== Запуск генерации визуализаций ===")

    # Создание папки
    create_directory()

    # Загрузка данных
    df = load_data()
    if df is None:
        logger.error("Не удалось загрузить данные. Пропуск генерации.")
        return

    # Генерация визуализаций
    logger.info("\n1. Распределение данных...")
    plot_data_distribution(df)

    logger.info("2. Сравнение моделей...")
    plot_model_comparison()

    logger.info("3. Сравнение с AutoML...")
    plot_automl_comparison()

    logger.info("4. Лидерборд AutoML...")
    create_automl_leaderboard()

    logger.info("5. Важность признаков...")
    create_feature_importance(df)

    logger.info("6. Анализ оттока...")
    create_churn_analysis(df)

    logger.info("\n=== Все визуализации сгенерированы ===")
    logger.info("Отчёты сохранены в папке reports/")


if __name__ == "__main__":
    main()
