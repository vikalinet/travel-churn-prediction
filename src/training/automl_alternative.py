"""
Обучение AutoML модели с использованием H2O AutoML.
Альтернатива AutoGluon.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class H2OAutoMLTrainer:
    """Обучение H2O AutoML модели."""

    def __init__(self, data_path: str, target_column: str = "Target"):
        self.data_path = data_path
        self.target_column = target_column
        self.automl = None

    def init_h2o(self):
        """Инициализация H2O."""
        try:
            import h2o

            h2o.init(max_mem_size="4G")
            logger.info("H2O успешно инициализирован")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации H2O: {e}")
            return False

    def load_data(self):
        """Загрузка данных."""
        logger.info(f"Загрузка данных из {self.data_path}")
        df = pd.read_csv(self.data_path)

        logger.info(f"Размер данных: {df.shape}")
        logger.info(
            f"Распределение целевой переменной:\n{df[self.target_column].value_counts()}"
        )

        return df

    def train_automl(self, df, max_runtime_secs: int = 120):
        """
        Обучение H2O AutoML модели.

        Args:
            df: DataFrame с данными
            max_runtime_secs: лимит времени в секундах
        """
        try:
            import h2o
            from h2o.automl import H2OAutoML

            # Преобразование в H2O DataFrame
            logger.info("Преобразование данных в H2O формат...")
            h2o_df = h2o.H2OFrame(df)

            # Разделение на train и test
            train, test = h2o_df.split_frame(ratios=[0.8], seed=42)

            logger.info(f"Train: {train.nrow}, Test: {test.nrow}")

            # Настройка AutoML
            logger.info(
                f"Начало обучения H2O AutoML (время: {max_runtime_secs} сек)..."
            )

            aml = H2OAutoML(
                max_runtime_secs=max_runtime_secs,
                seed=42,
                exclude_algos=["DeepLearning"],  # Исключаем нейросети для скорости
            )

            # Обучение
            x = [col for col in train.columns if col != self.target_column]
            y = self.target_column

            aml.train(x=x, y=y, training_frame=train)

            # Лидерборд
            logger.info("\nЛидерборд H2O AutoML:")
            lb = aml.leaderboard
            logger.info(lb.as_data_frame().head(10).to_string())

            # Лучшая модель
            best_model = aml.leader
            logger.info(f"\nЛучшая модель: {best_model.model_id}")

            # Метрики на тестовой выборке
            logger.info("\nМетрики на тестовой выборке:")
            perf = best_model.model_performance(test)

            metrics = {
                "accuracy": perf.accuracy(),
                "f1": perf.f1(),
                "auc": perf.auc(),
                "precision": perf.precision()[0],
                "recall": perf.recall()[0],
            }

            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

            # Сохранение лидерборда
            lb_df = lb.as_data_frame()
            lb_path = "reports/h2o_leaderboard.csv"
            lb_df.to_csv(lb_path, index=False)
            logger.info(f"Лидерборд сохранён в {lb_path}")

            # Сохранение модели
            h2o.save_model(best_model, path="h2o_models/", force=True)
            logger.info("Модель сохранена в h2o_models/")

            return metrics, lb_df, best_model

        except ImportError:
            logger.error("H2O не установлен. Попробуем альтернативный подход...")
            return self._train_ensemble_model(df, max_runtime_secs)
        except Exception as e:
            logger.error(f"Ошибка при обучении H2O AutoML: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return self._train_ensemble_model(df, max_runtime_secs)

    def _train_ensemble_model(self, df, time_limit: int = 120):
        """
        Альтернатива: обучение ансамбля моделей (VotingClassifier).
        Это упрощённая версия AutoML.
        """
        from sklearn.ensemble import (
            VotingClassifier,
            RandomForestClassifier,
            GradientBoostingClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            roc_auc_score,
            precision_score,
            recall_score,
        )
        import time

        logger.info("Обучение ансамбля моделей (VotingClassifier)...")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Создание ансамбля
        estimators = [
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]

        ensemble = VotingClassifier(estimators=estimators, voting="soft")

        logger.info("Обучение ансамбля...")
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Предсказания
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)[:, 1]

        # Метрики
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

        logger.info(f"\nВремя обучения: {training_time:.2f} сек")
        logger.info("\nМетрики ансамбля:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Кросс-валидация
        logger.info("\nКросс-валидация (5-fold):")
        cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring="f1")
        logger.info(f"  F1 CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Сохранение результатов
        results = {
            "model_name": "Ensemble (VotingClassifier)",
            **metrics,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
        }

        return metrics, pd.DataFrame([results]), ensemble

    def compare_with_custom_models(
        self, custom_results: list, automl_metrics: dict, model_name: str = "AutoML"
    ):
        """Сравнение AutoML с кастомными моделями."""
        # Добавляем метрики AutoML
        automl_result = {"model_name": model_name, **automl_metrics}

        all_results = custom_results + [automl_result]
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("f1_score", ascending=False)

        logger.info("\n=== Сравнение всех моделей ===")
        logger.info(results_df.to_string(index=False))

        # Визуализация
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics_to_plot = ["accuracy", "f1_score", "roc_auc", "precision"]
        metric_names = ["Accuracy", "F1-Score", "ROC AUC", "Precision"]

        for ax, metric, name in zip(axes.flat, metrics_to_plot, metric_names):
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
        Path("reports").mkdir(exist_ok=True)
        comparison_path = "reports/model_comparison_with_automl.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"График сравнения сохранён в {comparison_path}")

        return results_df


def main():
    """Запуск обучения AutoML."""
    import sys

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "data/processed/processed_data.csv"

    logger.info("=== Обучение AutoML модели ===")

    trainer = H2OAutoMLTrainer(data_file)

    # Попытка инициализации H2O
    if trainer.init_h2o():
        df = trainer.load_data()
        metrics, leaderboard, best_model = trainer.train_automl(
            df, max_runtime_secs=120
        )

        if metrics:
            logger.info("\n✅ H2O AutoML обучение завершено успешно!")

            # Сравнение
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

            comparison_df = trainer.compare_with_custom_models(
                custom_results, metrics, "H2O AutoML"
            )

            if comparison_df is not None:
                comparison_df.to_csv("reports/model_comparison_full.csv", index=False)
    else:
        # Фоллбэк на ансамбль
        logger.info("Использование альтернативного подхода (VotingClassifier)...")
        df = trainer.load_data()
        metrics, leaderboard, model = trainer._train_ensemble_model(df, time_limit=120)

        if metrics:
            logger.info("\n✅ Обучение ансамбля завершено успешно!")

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

            comparison_df = trainer.compare_with_custom_models(
                custom_results, metrics, "Ensemble"
            )

            if comparison_df is not None:
                comparison_df.to_csv("reports/model_comparison_full.csv", index=False)


if __name__ == "__main__":
    main()
