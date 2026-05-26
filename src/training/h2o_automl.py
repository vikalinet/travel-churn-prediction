"""
Обучение AutoML модели с использованием H2O AutoML.
Альтернатива AutoGluon.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.training.ensemble_models import EnsembleTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import h2o
    from h2o.automl import H2OAutoML

    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    h2o = None
    H2OAutoML = None


class H2OAutoMLTrainer:
    """Обучение H2O AutoML модели."""

    def __init__(self, data_path: str, target_column: str = "Target"):
        self.data_path = data_path
        self.target_column = target_column
        self.automl = None

    def init_h2o(self) -> bool:
        """Инициализация H2O."""
        if not H2O_AVAILABLE:
            logger.error("H2O не установлен. Установите: pip install h2o")
            return False
        try:
            h2o.init(max_mem_size="4G")
            logger.info("H2O успешно инициализирован")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации H2O: {e}")
            return False

    def load_data(self) -> pd.DataFrame:
        """Загрузка данных."""
        logger.info(f"Загрузка данных из {self.data_path}")
        df = pd.read_csv(self.data_path)

        logger.info(f"Размер данных: {df.shape}")
        logger.info(
            f"Распределение целевой переменной:\n{df[self.target_column].value_counts()}"
        )

        return df

    def train_automl(
        self, df: pd.DataFrame, max_runtime_secs: int = 120
    ) -> Optional[Tuple[Dict, Optional[object]]]:
        """
        Обучение H2O AutoML модели.

        Args:
            df: DataFrame с данными
            max_runtime_secs: лимит времени в секундах

        Returns:
            Кортеж (метрики, модель) или None
        """
        if not H2O_AVAILABLE:
            logger.error("H2O не установлен")
            return None

        try:
            logger.info("Преобразование данных в H2O формат...")
            h2o_df = h2o.H2OFrame(df)

            train, test = h2o_df.split_frame(ratios=[0.8], seed=42)
            logger.info(f"Train: {train.nrow}, Test: {test.nrow}")

            logger.info(
                f"Начало обучения H2O AutoML (время: {max_runtime_secs} сек)..."
            )

            aml = H2OAutoML(
                max_runtime_secs=max_runtime_secs,
                seed=42,
                exclude_algos=["DeepLearning"],
            )

            x = [col for col in train.columns if col != self.target_column]
            y = self.target_column

            aml.train(x=x, y=y, training_frame=train)

            logger.info("\nЛидерборд H2O AutoML:")
            lb = aml.leaderboard
            logger.info(lb.as_data_frame().head(10).to_string())

            best_model = aml.leader
            logger.info(f"\nЛучшая модель: {best_model.model_id}")

            logger.info("\nМетрики на тестовой выборке:")
            perf = best_model.model_performance(test)

            metrics = {
                "accuracy": perf.accuracy(),
                "f1_score": float(perf.f1()),
                "roc_auc": perf.auc(),
                "precision": float(perf.precision()[0]),
                "recall": float(perf.recall()[0]),
            }

            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

            lb_df = lb.as_data_frame()
            lb_df.to_csv("reports/h2o_leaderboard.csv", index=False)

            h2o.save_model(best_model, path="h2o_models/", force=True)
            logger.info("Модель сохранена в h2o_models/")

            return metrics, best_model

        except Exception as e:
            logger.error(f"Ошибка при обучении H2O AutoML: {e}")
            logger.error(traceback.format_exc())
            return None
