"""
Мониторинг дрейфа данных с использованием Evidently AI.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

try:
    from evidently.metrics import ClassificationClassificationMetrics, DataDriftTable
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftMonitor:
    """Мониторинг дрейфа данных с помощью Evidently."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = "Churn",
    ):
        """
        Инициализация монитора.

        Args:
            reference_data: Базовый датасет для сравнения (обучающая выборка)
            feature_columns: Список колонок признаков для мониторинга
            target_column: Имя целевой переменной
        """
        self.reference_data = reference_data[feature_columns + [target_column]]
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.current_data = None
        self.report_count = 0

    def update_current_data(self, new_data: pd.DataFrame):
        """
        Обновление текущих данных для мониторинга.

        Args:
            new_data: Новые данные для проверки
        """
        if self.current_data is None:
            self.current_data = new_data[self.feature_columns + [self.target_column]]
        else:
            self.current_data = pd.concat(
                [
                    self.current_data,
                    new_data[self.feature_columns + [self.target_column]],
                ],
                ignore_index=True,
            )

        logger.info(f"Обновлены текущие данные: {len(self.current_data)} записей")

    def generate_drift_report(self, output_path: Optional[str] = None) -> str:
        """
        Генерация отчёта о дрейфе данных.

        Args:
            output_path: Путь для сохранения HTML отчёта

        Returns:
            Путь к сохранённому отчёту
        """
        try:
            if not EVIDENTLY_AVAILABLE:
                logger.error(
                    "Evidently AI не установлен. Установите: pip install evidently"
                )
                return None

            logger.info("Генерация отчёта о дрейфе данных...")

            # Подготовка данных
            reference = self.reference_data.copy()
            current = self.current_data.copy()

            # Проверка на достаточный объём данных
            if len(reference) < 10 or len(current) < 10:
                logger.warning("Мало данных для генерации отчёта")
                return None

            # Создание отчёта
            report = Report(
                metrics=[
                    DataDriftTable(column_names=self.feature_columns),
                ]
            )

            report.run(reference_data=reference, current_data=current)

            # Сохранение отчёта
            self.report_count += 1
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = "evidently_reports/drift_report_" + timestamp + ".html"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            report.save_html(output_path)

            logger.info(f"Отчёт о дрейфе сохранён: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Ошибка при генерации отчёта: {e}")
            return None

    def calculate_drift_metrics(self) -> dict:
        """
        Расчёт метрик дрейфа для каждой колонки.

        Returns:
            Словарь с метриками дрейфа
        """
        drift_metrics = {}

        for column in self.feature_columns:
            if (
                column not in self.reference_data.columns
                or column not in self.current_data.columns
            ):
                continue

            ref_values = self.reference_data[column].dropna()
            curr_values = self.current_data[column].dropna()

            # KS-тест для числовых признаков
            if pd.api.types.is_numeric_dtype(ref_values):
                stat, p_value = stats.ks_2samp(ref_values, curr_values)
                drift_metrics[column] = {
                    "ks_statistic": float(stat),
                    "p_value": float(p_value),
                    "drift_detected": p_value < 0.05,
                }

            # Chi-square для категориальных признаков
            else:
                ref_counts = ref_values.value_counts()
                curr_counts = curr_values.value_counts()

                # Выравнивание индексов
                all_categories = ref_counts.index.union(curr_counts.index)
                ref_counts = ref_counts.reindex(all_categories, fill_value=0)
                curr_counts = curr_counts.reindex(all_categories, fill_value=0)

                try:
                    stat, p_value = stats.chisquare(ref_counts, curr_counts)
                    drift_metrics[column] = {
                        "chi_square_statistic": float(stat),
                        "p_value": float(p_value),
                        "drift_detected": p_value < 0.05,
                    }
                except Exception:
                    drift_metrics[column] = {
                        "error": "Could not calculate chi-square",
                        "drift_detected": False,
                    }

        return drift_metrics

    def check_drift_threshold(self, threshold: float = 0.05) -> bool:
        """
        Проверка наличия критического дрейфа.

        Args:
            threshold: Порог для определения дрейфа (по умолчанию 0.05)

        Returns:
            True если обнаружен критический дрейф
        """
        metrics = self.calculate_drift_metrics()

        drift_columns = [
            col for col, data in metrics.items() if data.get("drift_detected", False)
        ]

        if drift_columns:
            logger.warning(f"Обнаружен дрейф в колонках: {drift_columns}")
            return True

        logger.info("Критический дрейф не обнаружен")
        return False


class ModelPerformanceMonitor:
    """Мониторинг качества модели."""

    def __init__(
        self,
        reference_predictions: pd.DataFrame,
        prediction_column: str = "prediction",
        target_column: str = "Churn",
    ):
        """
        Инициализация монитора качества.

        Args:
            reference_predictions: Базовые предсказания с целевой переменной
            prediction_column: Имя колонки с предсказаниями
            target_column: Имя целевой переменной
        """
        self.reference_predictions = reference_predictions.copy()
        self.prediction_column = prediction_column
        self.target_column = target_column
        self.current_predictions = None

    def update_current_predictions(self, new_predictions: pd.DataFrame):
        """Обновление текущих предсказаний."""
        if self.current_predictions is None:
            self.current_predictions = new_predictions.copy()
        else:
            self.current_predictions = pd.concat(
                [self.current_predictions, new_predictions], ignore_index=True
            )

    def calculate_performance_metrics(self) -> dict:
        """Расчёт метрик качества на текущих данных."""
        try:
            y_true = self.current_predictions[self.target_column]
            y_pred = self.current_predictions[self.prediction_column]

            # ROC AUC требует вероятности, используем упрощённую версию
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
            }

            return metrics

        except Exception as e:
            logger.error(f"Ошибка при расчёте метрик: {e}")
            return {}

    def generate_performance_report(self, output_path: Optional[str] = None) -> str:
        """Генерация отчёта о качестве модели."""
        try:
            if not EVIDENTLY_AVAILABLE:
                logger.error("Evidently AI не установлен")
                return None

            logger.info("Генерация отчёта о качестве модели...")

            reference = self.reference_predictions.copy()
            current = self.current_predictions.copy()

            # Создание отчёта для классификации
            report = Report(
                metrics=[
                    ClassificationClassificationMetrics(
                        prediction_column=self.prediction_column,
                        target_column=self.target_column,
                    ),
                ]
            )

            report.run(reference_data=reference, current_data=current)

            # Сохранение
            self.report_count = getattr(self, "report_count", 0) + 1
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"evidently_reports/performance_report_{timestamp}.html"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            report.save_html(output_path)

            logger.info(f"Отчёт о качестве сохранён: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Ошибка при генерации отчёта: {e}")
            return None


def create_monitor_from_training_data(data_path: str, target_column: str = "Churn"):
    """
    Создание монитора из обработанных данных обучения.

    Args:
        data_path: Путь к обработанному датасету
        target_column: Имя целевой переменной

    Returns:
        DataDriftMonitor объект
    """
    df = pd.read_csv(data_path)

    # Разделение на признаки и таргет
    feature_columns = [col for col in df.columns if col != target_column]

    # Разделение на train и test (как reference и current)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    monitor = DataDriftMonitor(
        reference_data=train_df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    monitor.update_current_data(test_df)

    return monitor


if __name__ == "__main__":
    # Пример использования
    print("Модуль мониторинга дрейфа данных")
    print("Использование:")
    print("  from src.monitoring.drift_monitor import DataDriftMonitor")
    print("  monitor = DataDriftMonitor(reference_data, feature_columns)")
    print("  monitor.update_current_data(new_data)")
    print("  monitor.generate_drift_report()")
