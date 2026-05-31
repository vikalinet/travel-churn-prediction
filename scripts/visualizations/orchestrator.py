"""
Оркестрация генерации всех визуализаций.
"""

import logging
import warnings
from scripts.visualizations.utils import create_directory, load_data
from scripts.visualizations.data_distribution import plot_data_distribution
from scripts.visualizations.model_comparison import (
    create_automl_leaderboard,
    plot_automl_comparison,
    plot_model_comparison,
)
from scripts.visualizations.feature_importance import create_feature_importance
from scripts.visualizations.churn_analysis import create_churn_analysis

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_all_visualizations():
    """Основная функция генерации всех визуализаций."""
    logger.info("=== Запуск генерации визуализаций ===")

    # Создание папки
    create_directory()

    # Загрузка данных
    df = load_data()
    if df is None:
        logger.error("Не удалось загрузить данные. Пропуск генерации.")
        return

    # Генерация визуализаций
    logger.info("\n1. Распределение данных...")
    plot_data_distribution(df)

    logger.info("2. Сравнение моделей...")
    plot_model_comparison()

    logger.info("3. Сравнение с AutoML...")
    plot_automl_comparison()

    logger.info("4. Лидерборд AutoML...")
    create_automl_leaderboard()

    logger.info("5. Важность признаков...")
    create_feature_importance(df)

    logger.info("6. Анализ оттока...")
    create_churn_analysis(df)

    logger.info("\n=== Все визуализации сгенерированы ===")
    logger.info("Отчёты сохранены в папке reports/")


if __name__ == "__main__":
    generate_all_visualizations()
