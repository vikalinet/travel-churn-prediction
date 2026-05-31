"""
Генерация всех визуализаций для проекта.
Создаёт графики для README и отчётов.
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory():
    """Создание папки для отчётов."""
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    logger.info("Папка reports/ создана/проверена")


def load_data() -> pd.DataFrame:
    """Загрузка обработанных данных."""
    data_path = "data/processed/processed_data.csv"
    if not Path(data_path).exists():
        logger.warning(f"Файл {data_path} не найден!")
        return None
    return pd.read_csv(data_path)


def plot_model_comparison():
    """Сравнение всех моделей."""
    logger.info("Генерация сравнения моделей...")

    # Данные из результатов обучения
    data = {
        "Model": [
            "GradientBoosting",
            "XGBoost",
            "XGBoost (Tuned)",
            "RandomForest",
            "RandomForest (Tuned)",
            "KNeighbors",
            "LogisticRegression",
            "SVC",
        ],
        "Accuracy": [0.911, 0.895, 0.895, 0.885, 0.885, 0.869, 0.832, 0.764],
        "F1-Score": [0.795, 0.762, 0.762, 0.738, 0.738, 0.638, 0.543, 0.0],
        "ROC AUC": [0.975, 0.970, 0.968, 0.956, 0.960, 0.917, 0.847, 0.857],
    }

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Сравнение ML-моделей для прогнозирования оттока клиентов", fontsize=16
    )

    # Accuracy
    ax1 = axes[0, 0]
    colors = ["#27ae60" if m == "GradientBoosting" else "#3498db" for m in df["Model"]]
    bars1 = ax1.bar(df["Model"], df["Accuracy"], color=colors)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Точность (Accuracy)")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="x", rotation=45)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # F1-Score
    ax2 = axes[0, 1]
    colors = ["#27ae60" if m == "GradientBoosting" else "#3498db" for m in df["Model"]]
    bars2 = ax2.bar(df["Model"], df["F1-Score"], color=colors)
    ax2.set_ylabel("F1-Score")
    ax2.set_title("F1-Score (гармоническое среднее Precision и Recall)")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="x", rotation=45)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # ROC AUC
    ax3 = axes[1, 0]
    colors = ["#27ae60" if m == "GradientBoosting" else "#3498db" for m in df["Model"]]
    bars3 = ax3.bar(df["Model"], df["ROC AUC"], color=colors)
    ax3.set_ylabel("ROC AUC")
    ax3.set_title("Площадь под ROC-кривой")
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis="x", rotation=45)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Radar chart для лучшей модели
    ax4 = axes[1, 1]
    metrics = ["Accuracy", "F1-Score", "ROC AUC"]
    best_model = df[df["Model"] == "GradientBoosting"]
    values = best_model[metrics].values.flatten()
    values = np.append(values, values[0])

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.append(angles, angles[0])

    ax4 = plt.subplot(2, 2, 4, projection="polar")
    ax4.plot(angles, values, "o-", linewidth=2, color="#27ae60")
    ax4.fill(angles, values, alpha=0.25, color="#27ae60")
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics)
    ax4.set_ylim(0, 1)
    ax4.set_title("Лучшая модель: GradientBoosting", pad=20)

    plt.tight_layout()
    output_path = Path("reports/model_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {output_path}")


def plot_data_distribution(df: pd.DataFrame):
    """Распределение данных."""
    logger.info("Генерация распределения данных...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Анализ распределения данных", fontsize=16)

    # Распределение целевой переменной
    ax1 = axes[0, 0]
    churn_counts = df["Target"].value_counts()
    colors = ["#27ae60", "#e74c3c"]
    wedges, texts, autotexts = ax1.pie(
        churn_counts.values,
        labels=["Не ушёл (0)", "Ушёл (1)"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax1.set_title("Распределение целевой переменной (Target)")

    # Распределение по возрасту
    ax2 = axes[0, 1]
    ax2.hist(df["Age"], bins=20, color="#3498db", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Возраст")
    ax2.set_ylabel("Количество клиентов")
    ax2.set_title("Распределение по возрасту")
    ax2.axvline(
        df["Age"].mean(),
        color="red",
        linestyle="--",
        label=f"Средний: {df['Age'].mean():.1f}",
    )
    ax2.legend()

    # Распределение по услугам
    ax3 = axes[1, 0]
    services_counts = df["ServicesOpted"].value_counts().sort_index()
    ax3.bar(
        services_counts.index,
        services_counts.values,
        color="#9b59b6",
        edgecolor="black",
    )
    ax3.set_xlabel("Количество услуг")
    ax3.set_ylabel("Количество клиентов")
    ax3.set_title("Распределение по количеству услуг")

    # Корреляционная матрица
    ax4 = axes[1, 1]
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax4, fmt=".2f")
    ax4.set_title("Корреляционная матрица признаков")

    plt.tight_layout()
    output_path = Path("reports/data_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {output_path}")


def create_feature_importance(df: pd.DataFrame):
    """Важность признаков."""
    logger.info("Генерация важности признаков...")

    # Подготовка данных
    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Обучение модели
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Важность признаков
    importances = model.feature_importances_
    feature_names = X.columns

    # Сортировка
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(feature_names)))

    ax.barh(range(len(feature_names)), importances[indices], color=colors)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Важность признака")
    ax.set_title("Важность признаков для прогнозирования оттока")
    ax.invert_yaxis()

    # Добавление значений
    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.001, i, f"{v:.3f}", color="black", va="center", fontsize=10)

    plt.tight_layout()
    output_path = Path("reports/feature_importance.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {output_path}")


def create_churn_analysis(df: pd.DataFrame):
    """Анализ оттока."""
    logger.info("Генерация анализа оттока...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Анализ оттока клиентов", fontsize=16)

    # Отток по возрасту
    ax1 = axes[0, 0]
    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "55+"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=True)
    churn_by_age = df.groupby("AgeGroup")["Target"].mean() * 100
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(churn_by_age)))
    bars = ax1.bar(churn_by_age.index, churn_by_age.values, color=colors)
    ax1.set_ylabel("Процент оттока (%)")
    ax1.set_title("Отток по возрастным группам")
    for bar, val in zip(bars, churn_by_age.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
        )

    # Отток по количеству услуг
    ax2 = axes[0, 1]
    churn_by_services = df.groupby("ServicesOpted")["Target"].mean() * 100
    ax2.plot(
        churn_by_services.index,
        churn_by_services.values,
        "o-",
        linewidth=2,
        markersize=8,
        color="#e74c3c",
    )
    ax2.fill_between(
        churn_by_services.index, churn_by_services.values, alpha=0.3, color="#e74c3c"
    )
    ax2.set_xlabel("Количество услуг")
    ax2.set_ylabel("Процент оттока (%)")
    ax2.set_title("Отток по количеству использованных услуг")
    ax2.grid(True, alpha=0.3)

    # Распределение Churn по признакам
    ax3 = axes[1, 0]
    categorical_cols = [
        "FrequentFlyer",
        "AccountSyncedToSocialMedia",
        "BookedHotelOrNot",
    ]
    churn_data = []
    labels = []
    for col in categorical_cols:
        if col in df.columns:
            for val in df[col].unique():
                churn_rate = df[df[col] == val]["Target"].mean() * 100
                churn_data.append(churn_rate)
                labels.append(f"{col}: {val}")

    y_pos = np.arange(len(labels))
    colors = ["#e74c3c" if x > 40 else "#27ae60" for x in churn_data]
    ax3.barh(y_pos, churn_data, color=colors)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel("Процент оттока (%)")
    ax3.set_title("Отток по категориальным признакам")

    # Box plot Age по Target
    ax4 = axes[1, 1]
    target_labels = ["Не ушёл", "Ушёл"]
    data_for_box = [df[df["Target"] == 0]["Age"], df[df["Target"] == 1]["Age"]]
    bp = ax4.boxplot(data_for_box, labels=target_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#27ae60", "#e74c3c"]):
        patch.set_facecolor(color)
    ax4.set_ylabel("Возраст")
    ax4.set_title("Распределение возраста по оттоку")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path("reports/churn_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Сохранено: {output_path}")


def generate_all_visualizations():
    """Основная функция генерации всех визуализаций."""
    logger.info("=== Запуск генерации всех визуализаций ===")

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

    logger.info("3. Важность признаков...")
    create_feature_importance(df)

    logger.info("4. Анализ оттока...")
    create_churn_analysis(df)

    logger.info("\n=== Все визуализации сгенерированы ===")
    logger.info("Отчёты сохранены в папке reports/")

    # Вывод списка созданных файлов
    output_files = [
        "reports/model_comparison.png",
        "reports/data_distribution.png",
        "reports/feature_importance.png",
        "reports/churn_analysis.png",
    ]
    logger.info("\nСозданные файлы:")
    for f in output_files:
        if Path(f).exists():
            size_kb = Path(f).stat().st_size / 1024
            logger.info(f"  ✓ {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    generate_all_visualizations()
