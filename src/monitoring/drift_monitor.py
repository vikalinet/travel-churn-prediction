"""
Мониторинг дрейфа данных с использованием Evidently AI и статистических тестов.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from src.monitoring.base_monitor import BaseMonitor

try:
    from evidently.metrics import DataDriftTable
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftMonitor(BaseMonitor):
    """Мониторинг дрейфа данных с помощью Evidently и статистических тестов."""

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
        super().__init__(feature_columns)
        self.reference_data = reference_data[feature_columns + [target_column]].copy()
        self.target_column = target_column
        self.current_data = None
        self.report_count = 0

    def generate_drift_report(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Генерация отчёта о дрейфе данных через Evidently.

        Args:
            output_path: Путь для сохранения HTML отчёта

        Returns:
            Путь к сохранённому отчёту или None
        """
        try:
            if not EVIDENTLY_AVAILABLE:
                logger.error(
                    "Evidently AI не установлен. Установите: pip install evidently"
                )
                return None

            if not self.check_data_size():
                logger.warning("Мало данных для генерации отчёта")
                return None

            logger.info("Генерация отчёта о дрейфе данных...")

            reference = self.reference_data[self.feature_columns].copy()
            current = self.current_data[self.feature_columns].copy()

            # Создание отчёта
            report = Report(metrics=[DataDriftTable(column_names=self.feature_columns)])
            report.run(reference_data=reference, current_data=current)

            # Сохранение отчёта
            self.report_count += 1
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"evidently_reports/drift_report_{timestamp}.html"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            report.save_html(output_path)

            logger.info(f"Отчёт о дрейфе сохранён: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Ошибка при генерации отчёта: {e}")
            return None

    def calculate_drift_metrics(self) -> dict:
        """
        Расчёт метрик дрейфа для каждой колонки через статистические тесты.

        Returns:
            Словарь с метриками дрейфа
        """
        if self.current_data is None:
            logger.error("Текущие данные не загружены")
            return {}

        drift_metrics = {}

        for column in self.feature_columns:
            if (
                column not in self.reference_data.columns
                or column not in self.current_data.columns
            ):
                continue

            ref_values = self.reference_data[column].dropna()
            curr_values = self.current_data[column].dropna()

            if len(ref_values) == 0 or len(curr_values) == 0:
                continue

            # KS-тест для числовых признаков
            if pd.api.types.is_numeric_dtype(ref_values):
                stat, p_value = stats.ks_2samp(ref_values, curr_values)
                drift_metrics[column] = {
                    "type": "numeric",
                    "ks_statistic": float(stat),
                    "p_value": float(p_value),
                    "drift_detected": p_value < 0.05,
                    "ref_mean": float(ref_values.mean()),
                    "curr_mean": float(curr_values.mean()),
                    "ref_std": float(ref_values.std()),
                    "curr_std": float(curr_values.std()),
                }

            # Chi-square для категориальных признаков
            else:
                ref_counts = ref_values.value_counts()
                curr_counts = curr_values.value_counts()

                all_categories = ref_counts.index.union(curr_counts.index)
                ref_counts = ref_counts.reindex(all_categories, fill_value=0)
                curr_counts = curr_counts.reindex(all_categories, fill_value=0)

                try:
                    stat, p_value = stats.chisquare(ref_counts, curr_counts)
                    drift_metrics[column] = {
                        "type": "categorical",
                        "chi_square_statistic": float(stat),
                        "p_value": float(p_value),
                        "drift_detected": p_value < 0.05,
                    }
                except Exception:
                    drift_metrics[column] = {
                        "type": "categorical",
                        "error": "Could not calculate chi-square",
                        "drift_detected": False,
                    }

        return drift_metrics

    def send_alert(
        self, drift_columns: List[str], metrics: dict, webhook_url: Optional[str] = None
    ):
        """
        Отправка алерта при обнаружении дрейфа данных.

        Args:
            drift_columns: Список колонок с дрейфом
            metrics: Полный словарь метрик дрейфа
            webhook_url: URL для webhook-уведомления (опционально, берётся из env DRIFT_WEBHOOK_URL)
        """
        alert_data = {
            "alert_type": "data_drift",
            "timestamp": datetime.now().isoformat(),
            "drift_detected": True,
            "affected_columns": drift_columns,
            "details": {col: metrics[col] for col in drift_columns if col in metrics},
            "message": f"Обнаружен дрейф данных в колонках: {', '.join(drift_columns)}",
        }

        # Логирование
        logger.warning(alert_data["message"])

        # Сохранение JSON-алерта
        alert_path = Path("evidently_reports/drift_alert.json")
        alert_path.parent.mkdir(parents=True, exist_ok=True)
        with open(alert_path, "w", encoding="utf-8") as f:
            json.dump(alert_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Alert сохранён: {alert_path}")

        # Webhook (если настроен)
        webhook_url = webhook_url or os.getenv("DRIFT_WEBHOOK_URL")
        if webhook_url:
            try:
                import urllib.request

                req = urllib.request.Request(
                    webhook_url,
                    data=json.dumps(alert_data).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    logger.info(f"Webhook отправлен, статус: {resp.status}")
            except Exception as e:
                logger.error(f"Не удалось отправить webhook: {e}")

    def check_drift_threshold(self, threshold: float = 0.05) -> bool:
        """
        Проверка наличия критического дрейфа.

        Args:
            threshold: Порог p-value для определения дрейфа

        Returns:
            True если обнаружен критический дрейф
        """
        metrics = self.calculate_drift_metrics()

        drift_columns = [
            col for col, data in metrics.items() if data.get("drift_detected", False)
        ]

        if drift_columns:
            logger.warning(f"Обнаружен дрейф в колонках: {drift_columns}")
            self.send_alert(drift_columns, metrics)
            return True

        logger.info("Критический дрейф не обнаружен")
        return False


def create_monitor_from_training_data(
    data_path: str, target_column: str = "Churn", test_size: float = 0.2
) -> DataDriftMonitor:
    """
    Создание монитора из обработанных данных обучения.

    Args:
        data_path: Путь к обработанному датасету
        target_column: Имя целевой переменной
        test_size: Доля тестовой выборки

    Returns:
        DataDriftMonitor объект
    """
    df = pd.read_csv(data_path)

    feature_columns = [col for col in df.columns if col != target_column]

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    monitor = DataDriftMonitor(
        reference_data=train_df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    monitor.update_current_data(test_df)

    return monitor
