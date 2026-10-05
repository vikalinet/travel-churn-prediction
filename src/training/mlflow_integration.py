"""
Интеграция с MLflow для логирования моделей и метрик.
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS = ("accuracy", "f1_score", "roc_auc", "precision", "recall")

# Реестр моделей MLflow: лучшая модель каждого обучения — новая версия этой
# модели; alias указывает на версию, которую использует API.
REGISTERED_MODEL = "travel-churn-model"
CHAMPION_ALIAS = "champion"


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
    ) -> Dict[str, Any]:
        """
        Записать результаты обучения в MLflow: по прогону на каждую модель.

        В прогон пишутся метрики, порог классификации и общие параметры
        обучения; лучшая модель помечается тегом best=true, сохраняется
        артефактом и регистрируется новой версией REGISTERED_MODEL с alias
        CHAMPION_ALIAS. Метрики, параметры и реестр попадают в базу
        (mlflow.db, хранится в git), а файл модели — в папку артефактов
        mlruns/ (в git не хранится).

        Returns:
            run_ids — созданные прогоны, best_run_id — прогон лучшей модели,
            model_version — её номер версии в реестре.
        """
        mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri())
        mlflow.set_experiment(experiment_name)
        run_ids = []
        best_run_id = None
        model_version = None
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
                    info = mlflow.sklearn.log_model(
                        best_model,
                        name="model",
                        registered_model_name=REGISTERED_MODEL,
                    )
                    best_run_id = run.info.run_id
                    model_version = int(info.registered_model_version)
                run_ids.append(run.info.run_id)
        if model_version is not None:
            MlflowClient().set_registered_model_alias(
                REGISTERED_MODEL, CHAMPION_ALIAS, str(model_version)
            )
        logger.info(
            f"В MLflow записано прогонов: {len(run_ids)} (лучшая: {best_name}, "
            f"{REGISTERED_MODEL} v{model_version})"
        )
        return {
            "run_ids": run_ids,
            "best_run_id": best_run_id,
            "model_version": model_version,
        }
