"""
Мониторинг дрейфа данных для датасета Customer Travel.
Использует общий DataDriftMonitor с настройками для этого датасета.
"""

import logging
import sys
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from src.monitoring.drift_monitor import DataDriftMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerTravelDriftMonitor(DataDriftMonitor):
    """
    Специализированный монитор для датасета Customer Travel.
    Наследуется от DataDriftMonitor с предопределёнными колонками.
    """

    DEFAULT_FEATURE_COLUMNS = [
        "Age",
        "FrequentFlyer",
        "AnnualIncomeClass",
        "ServicesOpted",
        "AccountSyncedToSocialMedia",
        "BookedHotelOrNot",
    ]

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: List[str] = None,
        target_column: str = "Target",
    ):
        if feature_columns is None:
            feature_columns = self.DEFAULT_FEATURE_COLUMNS

        super().__init__(reference_data, feature_columns, target_column)


def create_monitor_from_training_data(
    data_path: str, target_column: str = "Target", test_size: float = 0.2
) -> CustomerTravelDriftMonitor:
    df = pd.read_csv(data_path)
    feature_columns = [col for col in df.columns if col != target_column]
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    monitor = CustomerTravelDriftMonitor(
        reference_data=train_df,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    monitor.update_current_data(test_df)

    return monitor, feature_columns


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "data/processed/processed_data.csv"

    logger.info("=== Запуск мониторинга дрейфа данных (Customer Travel) ===")

    monitor, feature_columns = create_monitor_from_training_data(data_file)
    logger.info(f"Признаки для мониторинга: {feature_columns}")
    logger.info(f"Reference размер: {len(monitor.reference_data)}")
    logger.info(f"Current размер: {len(monitor.current_data)}")

    report_path = monitor.generate_drift_report()
    if report_path:
        logger.info(f"HTML отчёт: {report_path}")

    summary = monitor.calculate_drift_metrics()
    drift_count = sum(1 for f in summary.values() if f.get("drift_detected", False))

    logger.info(f"\nВсего признаков: {len(feature_columns)}")
    logger.info(f"Признаков с дрейфом: {drift_count}")

    if monitor.check_drift_threshold():
        logger.warning("⚠️ Обнаружен критический дрейф! Требуется переобучение модели.")
    else:
        logger.info("✅ Дрейф в пределах нормы")
