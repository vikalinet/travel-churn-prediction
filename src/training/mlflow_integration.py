"""
Интеграция с MLflow для логирования моделей и метрик.
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn

from src import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS = ("accuracy", "f1_score", "roc_auc", "precision", "recall")


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
    def setup_tracking(uri: Optional[str] = None):
        """
        Настройка трекинга MLflow.

        Args:
            uri: URI для хранения метрик
        """
        # По умолчанию — mlflow.db в корне проекта, а не в папке запуска.
        uri = uri or settings.mlflow_tracking_uri()
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow трекинг настроен: {uri}")

    @staticmethod
    def log_training_results(
        results: List[Dict[str, Any]],
        thresholds: Dict[str, float],
        best_name: str,
        best_model: Any,
        params: Optional[Dict[str, Any]] = None,
        experiment_name: str = "Travel Churn Prediction",
        tracking_uri: Optional[str] = None,
    ) -> List[str]:
        """
        Записать результаты обучения в MLflow: по прогону на каждую модель.

        В прогон пишутся метрики, порог классификации и общие параметры
        обучения; лучшая модель помечается тегом best=true и сохраняется
        артефактом. Метрики и параметры попадают в базу (mlflow.db, хранится
        в git), а файл модели — в папку артефактов mlruns/ (в git не хранится).

        Returns:
            Идентификаторы созданных прогонов.
        """
        mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri())
        mlflow.set_experiment(experiment_name)
        run_ids = []
        for row in results:
            name = row["model_name"]
            with mlflow.start_run(run_name=name) as run:
                for metric in METRICS:
                    if isinstance(row.get(metric), (int, float)):
                        mlflow.log_metric(metric, float(row[metric]))
                mlflow.log_param("threshold", round(thresholds.get(name, 0.5), 3))
                for param_name, value in (params or {}).items():
                    mlflow.log_param(param_name, value)
                mlflow.set_tags(
                    {"pipeline": "improved", "best": str(name == best_name).lower()}
                )
                if name == best_name:
                    mlflow.sklearn.log_model(best_model, name="model")
                run_ids.append(run.info.run_id)
        logger.info(f"В MLflow записано прогонов: {len(run_ids)} (лучшая: {best_name})")
        return run_ids
