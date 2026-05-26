"""
Мониторинг качества модели.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.monitoring.base_monitor import BaseMonitor

try:
    from evidently.metrics import ClassificationClassificationMetrics
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPerformanceMonitor(BaseMonitor):
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
        feature_columns = [prediction_column, target_column]
        super().__init__(feature_columns)

        self.reference_predictions = reference_predictions[feature_columns].copy()
        self.prediction_column = prediction_column
        self.target_column = target_column
        self.current_predictions = None
        self.report_count = 0

    def update_current_predictions(self, new_predictions: pd.DataFrame):
        """
        Обновление текущих предсказаний.

        Args:
            new_predictions: Новые предсказания для мониторинга
        """
        if self.current_predictions is None:
            self.current_predictions = new_predictions[
                [self.prediction_column, self.target_column]
            ].copy()
        else:
            self.current_predictions = pd.concat(
                [
                    self.current_predictions,
                    new_predictions[[self.prediction_column, self.target_column]],
                ],
                ignore_index=True,
            )

        logger.info(
            f"Обновлены текущие предсказания: {len(self.current_predictions)} записей"
        )

    def calculate_performance_metrics(self) -> dict:
        """
        Расчёт метрик качества на текущих данных.

        Returns:
            Словарь с метриками качества
        """
        if self.current_predictions is None:
            logger.error("Текущие предсказания не загружены")
            return {}

        try:
            y_true = self.current_predictions[self.target_column]
            y_pred = self.current_predictions[self.prediction_column]

            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
            }

            return metrics

        except Exception as e:
            logger.error(f"Ошибка при расчёте метрик: {e}")
            return {}

    def generate_performance_report(
        self, output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Генерация отчёта о качестве модели через Evidently.

        Args:
            output_path: Путь для сохранения HTML отчёта

        Returns:
            Путь к сохранённому отчёту или None
        """
        try:
            if not EVIDENTLY_AVAILABLE:
                logger.error("Evidently AI не установлен")
                return None

            if not self.check_data_size():
                logger.warning("Мало данных для генерации отчёта")
                return None

            logger.info("Генерация отчёта о качестве модели...")

            reference = self.reference_predictions.copy()
            current = self.current_predictions.copy()

            report = Report(
                metrics=[
                    ClassificationClassificationMetrics(
                        prediction_column=self.prediction_column,
                        target_column=self.target_column,
                    ),
                ]
            )

            report.run(reference_data=reference, current_data=current)

            self.report_count += 1
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
