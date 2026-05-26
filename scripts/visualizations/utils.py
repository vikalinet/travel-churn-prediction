"""
Утилиты для визуализаций.
Общие функции загрузки данных и создания директорий.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory(directory: str = "reports"):
    """Создание папки для отчётов."""
    Path(directory).mkdir(exist_ok=True)
    logger.info(f"Папка {directory} создана/проверена")


def load_data(data_path: str = "data/processed/processed_data.csv") -> pd.DataFrame:
    """
    Загрузка обработанных данных.

    Args:
        data_path: Путь к CSV файлу

    Returns:
        DataFrame или None если файл не найден
    """
    if Path(data_path).exists():
        df = pd.read_csv(data_path)
        logger.info(f"Загружено {len(df)} строк из {data_path}")
        return df
    else:
        logger.error(f"Файл {data_path} не найден!")
        return None


def load_training_results(
    csv_path: str = "reports/training_results.csv",
) -> pd.DataFrame:
    """
    Загрузка результатов обучения моделей.

    Args:
        csv_path: Путь к файлу с результатами

    Returns:
        DataFrame с результатами или дефолтные данные
    """
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        logger.info(f"Загружены результаты обучения из {csv_path}")
        return df
    else:
        logger.warning(f"Файл {csv_path} не найден, используем дефолтные данные")
        return _get_default_training_results()


def _get_default_training_results() -> pd.DataFrame:
    """Дефолтные данные для тестирования."""
    models_data = {
        "model": [
            "GradientBoosting",
            "XGBoost",
            "XGBoost_Tuned",
            "RandomForest",
            "RandomForest_Tuned",
            "KNeighbors",
            "LogisticRegression",
            "SVC",
        ],
        "accuracy": [
            0.9110,
            0.8953,
            0.8953,
            0.8848,
            0.8848,
            0.8691,
            0.8325,
            0.7644,
        ],
        "f1_score": [
            0.7952,
            0.7619,
            0.7619,
            0.7381,
            0.7381,
            0.6377,
            0.5429,
            0.0000,
        ],
        "roc_auc": [0.9747, 0.9699, 0.9677, 0.9556, 0.9602, 0.9168, 0.8467, 0.8572],
    }
    return pd.DataFrame(models_data)
