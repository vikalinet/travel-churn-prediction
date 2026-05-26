"""
Базовый класс для мониторов данных.
"""

import logging
from typing import List

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseMonitor:
    """Базовый класс для всех мониторов данных."""

    def __init__(self, feature_columns: List[str]):
        """
        Инициализация базового монитора.

        Args:
            feature_columns: Список колонок для мониторинга
        """
        self.feature_columns = feature_columns
        self.reference_data = None
        self.current_data = None
        self.report_count = 0

    def update_current_data(self, new_data: pd.DataFrame):
        """
        Обновление текущих данных.

        Args:
            new_data: Новые данные для мониторинга
        """
        if self.current_data is None:
            self.current_data = new_data[self.feature_columns].copy()
        else:
            self.current_data = pd.concat(
                [self.current_data, new_data[self.feature_columns].copy()],
                ignore_index=True,
            )

        logger.info(f"Обновлены текущие данные: {len(self.current_data)} записей")

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Проверка наличия всех необходимых колонок.

        Args:
            data: DataFrame для проверки

        Returns:
            True если все колонки присутствуют
        """
        missing = [col for col in self.feature_columns if col not in data.columns]
        if missing:
            logger.error(f"Отсутствуют колонки: {missing}")
            return False
        return True

    def check_data_size(self, min_size: int = 10) -> bool:
        """
        Проверка достаточного объёма данных.

        Args:
            min_size: Минимальное количество записей

        Returns:
            True если данных достаточно
        """
        if self.reference_data is not None and len(self.reference_data) < min_size:
            logger.warning(f"Мало данных в reference: {len(self.reference_data)}")
            return False
        if self.current_data is not None and len(self.current_data) < min_size:
            logger.warning(f"Мало данных в current: {len(self.current_data)}")
            return False
        return True
