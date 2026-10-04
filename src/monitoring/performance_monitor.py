"""
Мониторинг качества модели.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src import settings
from src.monitoring.base_monitor import BaseMonitor

# Evidently 0.7 перенёс прежний интерфейс отчётов (Report и метрики) в
# evidently.legacy. Код импортировал старый путь, ImportError молча
# проглатывался, и отчёты Evidently не строились вовсе — без единой ошибки.
# Произошло это потому, что версия не была закреплена (evidently>=0.4.0).
# Версия закреплена в poetry.lock, импорт — по пути этой версии.
# Класса ClassificationClassificationMetrics, который здесь импортировался,
# не было ни в одной версии Evidently: отчёт о качестве модели не строился
# никогда. Нужная метрика — ClassificationQualityMetric; какая колонка
# целевая, а какая предсказание, ей сообщает ColumnMapping.
try:
    from evidently.legacy.metrics import ClassificationQualityMetric
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report

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
        self.current_predictions: Optional[pd.DataFrame] = None
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

            if self.current_predictions is None:
                logger.warning(
                    "Нет текущих предсказаний — сначала update_current_predictions()"
                )
                return None

            logger.info("Генерация отчёта о качестве модели...")

            reference = self.reference_predictions.copy()
            current = self.current_predictions.copy()

            report = Report(metrics=[ClassificationQualityMetric()])
            report.run(
                reference_data=reference,
                current_data=current,
                column_mapping=ColumnMapping(
                    target=self.target_column,
                    prediction=self.prediction_column,
                ),
            )

            self.report_count += 1
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = str(
                    settings.runtime_reports_dir()
                    / f"performance_report_{timestamp}.html"
                )

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            report.save_html(output_path)

            logger.info(f"Отчёт о качестве сохранён: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Ошибка при генерации отчёта: {e}")
            return None
