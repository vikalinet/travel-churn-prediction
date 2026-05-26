"""
Обучение AutoML моделей (AutoGluon) для прогнозирования оттока.
"""

import logging
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoGluonTrainer:
    """Обучение AutoGluon модели."""

    def __init__(self, data_path: str, target_column: str = "Target"):
        self.data_path = data_path
        self.target_column = target_column
        self.trainer = None

    def load_data(self):
        """Загрузка данных."""
        logger.info(f"Загрузка данных из {self.data_path}")
        df = pd.read_csv(self.data_path)

        logger.info(f"Размер данных: {df.shape}")
        logger.info(
            f"Распределение целевой переменной:\n{df[self.target_column].value_counts()}"
        )

        return df

    def train_automl(self, df, time_limit: int = 180, presets: str = "medium_quality"):
        """
        Обучение AutoGluon модели.

        Args:
            df: DataFrame с данными
            time_limit: лимит времени в секундах
            presets: preset качества ('medium_quality', 'good_quality', 'best_quality')
        """
        try:
            logger.info("Инициализация AutoGluon...")

            # Разделение на train и test
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df[self.target_column]
            )

            logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")

            # Создание датасетов для AutoGluon
            train_dataset = TabularDataset(train_df.reset_index(drop=True))
            test_dataset = TabularDataset(test_df.reset_index(drop=True))

            # Создание папки для моделей
            Path("autogluon_models").mkdir(exist_ok=True)

            # Обучение
            logger.info(
                f"Начало обучения AutoGluon (время: {time_limit} сек, preset: {presets})..."
            )

            self.trainer = TabularPredictor(
                label=self.target_column,
                eval_metric="f1",
                path="autogluon_models/",
            )

            self.trainer.fit(
                train_data=train_dataset,
                time_limit=time_limit,
                presets=presets,
                verbosity=2,
            )

            # Предсказания на тестовой выборке
            logger.info("Генерация предсказаний...")
            y_pred = self.trainer.predict(test_dataset)
            y_proba = self.trainer.predict_proba(test_dataset).iloc[:, 1]
            y_true = test_df[self.target_column].values

            # Метрики
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "roc_auc": roc_auc_score(y_true, y_proba),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
            }

            logger.info("\nМетрики AutoGluon:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

            # Информация о моделях
            logger.info("\nЛидерборд моделей AutoGluon:")
            leaderboard = self.trainer.leaderboard(test_dataset, silent=True)
            logger.info(leaderboard.to_string())

            # Сохранение лидерборда
            leaderboard_path = "reports/automl_leaderboard.png"
            Path("reports").mkdir(exist_ok=True)

            plt.figure(figsize=(12, 8))
            plt.table(
                cellText=leaderboard.values,
                colLabels=leaderboard.columns,
                cellLoc="center",
                loc="center",
            )
            plt.axis("off")
            plt.title("AutoGluon Leaderboard", fontsize=16, fontweight="bold")
            plt.tight_layout()
            plt.savefig(leaderboard_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Лидерборд сохранён в {leaderboard_path}")

            return metrics, leaderboard

        except ImportError:
            logger.error(
                "AutoGluon не установлен. Установите: pip install autogluon.tabular"
            )
            return None, None
        except Exception as e:
            logger.error(f"Ошибка при обучении AutoGluon: {e}")
            logger.error(traceback.format_exc())
            return None, None

    def compare_with_custom_models(self, custom_results: list):
        """Сравнение AutoGluon с кастомными моделями."""
        if not self.trainer:
            logger.warning("AutoGluon модель не обучена")
            return None

        # Получаем метрики AutoGluon
        df = pd.read_csv(self.data_path)
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df[self.target_column]
        )

        test_dataset = TabularDataset(test_df.reset_index(drop=True))
        y_pred = self.trainer.predict(test_dataset)
        y_proba = self.trainer.predict_proba(test_dataset).iloc[:, 1]
        y_true = test_df[self.target_column].values

        automl_metrics = {
            "model_name": "AutoGluon",
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }

        # Создаём DataFrame для сравнения
        all_results = custom_results + [automl_metrics]
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("f1_score", ascending=False)

        logger.info("\n=== Сравнение всех моделей (включая AutoGluon) ===")
        logger.info(results_df.to_string(index=False))

        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ["accuracy", "f1_score", "roc_auc", "precision"]
        metric_names = ["Accuracy", "F1-Score", "ROC AUC", "Precision"]

        for ax, metric, name in zip(axes.flat, metrics, metric_names):
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
        comparison_path = "reports/model_comparison_with_automl.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"График сравнения сохранён в {comparison_path}")

        return results_df


def main():
    """Запуск обучения AutoGluon."""
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "data/processed/processed_data.csv"

    logger.info("=== Обучение AutoGluon модели ===")

    trainer = AutoGluonTrainer(data_file)
    df = trainer.load_data()

    # Обучение (time_limit=180 секунд = 3 минуты)
    metrics, leaderboard = trainer.train_automl(
        df, time_limit=180, presets="medium_quality"
    )

    if metrics:
        logger.info("\nAutoGluon обучение завершено успешно!")
        logger.info("Лучшая модель сохранена в autogluon_models/")

        # Сравнение с кастомными моделями
        custom_results = [
            {
                "model_name": "GradientBoosting",
                "accuracy": 0.9110,
                "f1_score": 0.7952,
                "roc_auc": 0.9747,
                "precision": 0.8684,
                "recall": 0.7333,
            },
            {
                "model_name": "XGBoost_Tuned",
                "accuracy": 0.8953,
                "f1_score": 0.7619,
                "roc_auc": 0.9677,
                "precision": 0.8205,
                "recall": 0.7111,
            },
            {
                "model_name": "RandomForest_Tuned",
                "accuracy": 0.8848,
                "f1_score": 0.7381,
                "roc_auc": 0.9602,
                "precision": 0.7949,
                "recall": 0.6889,
            },
        ]

        comparison_df = trainer.compare_with_custom_models(custom_results)

        if comparison_df is not None:
            comparison_df.to_csv("reports/model_comparison_full.csv", index=False)
            logger.info(
                "Полное сравнение сохранено в reports/model_comparison_full.csv"
            )

    else:
        logger.error("❌ AutoGluon обучение не удалось")


if __name__ == "__main__":
    main()
