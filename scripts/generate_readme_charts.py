"""
Генерация PNG-графиков для встраивания в README.md.
Запуск: python scripts/generate_readme_charts.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def plot_feature_importance(df: pd.DataFrame, save_path: Path):
    """Важность признаков по корреляции с Target."""
    numeric_df = df.select_dtypes(include=["number"])
    correlations = (
        numeric_df.corr()["Target"].drop("Target").abs().sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(correlations))
    correlations.plot(kind="barh", ax=ax, color=colors, edgecolor="black")
    ax.set_title(
        "Важность признаков (абс. корреляция с Target)", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Абсолютная корреляция")
    ax.set_xlim(0, 1)
    for i, v in enumerate(correlations.values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {save_path}")


def plot_confusion_and_roc(df: pd.DataFrame, save_cm: Path, save_roc: Path):
    """Confusion matrix и ROC-кривая для GradientBoosting."""
    X = df.drop(columns=["Target"])
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Остался", "Ушёл"],
        yticklabels=["Остался", "Ушёл"],
        ax=ax,
        linewidths=1,
        linecolor="black",
    )
    ax.set_title("Confusion Matrix (GradientBoosting)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Истинный класс")
    ax.set_xlabel("Предсказанный класс")
    plt.tight_layout()
    plt.savefig(save_cm, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {save_cm}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}", linewidth=2, color="#2ecc71")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2ecc71")
    ax.set_title("ROC-кривая (GradientBoosting)", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_roc, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {save_roc}")

    # Сохраним метрики drift-заглушки для демонстрации алерта
    drift_alert = {
        "drift_detected": False,
        "affected_columns": [],
        "timestamp": pd.Timestamp.now().isoformat(),
        "message": "Дрейф не обнаружен. Модель актуальна.",
    }
    with open(REPORTS_DIR / "drift_alert.json", "w", encoding="utf-8") as f:
        json.dump(drift_alert, f, ensure_ascii=False, indent=2)


def main():
    data_path = Path("data/processed/processed_data.csv")
    if not data_path.exists():
        print(f"Файл не найден: {data_path}")
        return

    df = pd.read_csv(data_path)
    plot_feature_importance(df, REPORTS_DIR / "feature_importance.png")
    plot_confusion_and_roc(
        df,
        save_cm=REPORTS_DIR / "confusion_matrix.png",
        save_roc=REPORTS_DIR / "roc_curve.png",
    )
    print("Готово! Все графики сохранены в reports/")


if __name__ == "__main__":
    main()
