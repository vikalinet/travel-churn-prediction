"""
Интеграция с MLflow для логирования моделей и метрик.
"""

import logging
from typing import Dict, Optional

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowIntegration:
    """Логирование в MLflow."""

    @staticmethod
    def log_model(
        model_name: str,
        model,
        metrics: Dict,
        params: Optional[Dict] = None,
        experiment_name: str = "Travel Churn Prediction",
    ):
        """
        Логирование модели и метрик в MLflow.

        Args:
            model_name: Имя модели
            model: Обученная модель
            metrics: Словарь с метриками
            params: Словарь с параметрами (опционально)
            experiment_name: Имя эксперимента
        """
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=model_name):
            # Логирование метрик
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, value)

            # Логирование параметров
            if params:
                for param_name, value in params.items():
                    mlflow.log_param(param_name, value)

            # Логирование модели
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"Модель {model_name} залогирована в MLflow")

    @staticmethod
    def setup_tracking(uri: str = "sqlite:///mlflow.db"):
        """
        Настройка трекинга MLflow.

        Args:
            uri: URI для хранения метрик
        """
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow трекинг настроен: {uri}")
