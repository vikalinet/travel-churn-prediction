"""
FastAPI router для страницы мониторинга /monitoring.
Агрегирует данные из MLflow, Evidently drift-отчётов и системных метрик.
"""

import json
import logging
import os
from typing import Any, Dict, List

import mlflow
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from mlflow.tracking import MlflowClient

from src import settings
from src.monitoring.system_monitor import get_system_metrics

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitoring"])


def _get_mlflow_client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri())
    return MlflowClient()


def _load_drift_status() -> Dict[str, Any]:
    """Загрузка статуса дрейфа из JSON-отчётов."""
    # Сначала проверяем alert
    alert_path = settings.report_for_reading("drift_alert.json")
    if alert_path.exists():
        with open(alert_path, "r", encoding="utf-8") as f:
            alert: Dict[str, Any] = json.load(f)
            return alert

    # Затем проверяем summary
    summary_path = settings.report_for_reading("drift_summary.json")
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary: Dict[str, Any] = json.load(f)
            return summary

    return {"drift_detected": False, "message": "Нет данных о дрейфе"}


def _get_experiments_data() -> List[Dict[str, Any]]:
    """Получение списка экспериментов и run'ов из MLflow."""
    try:
        client = _get_mlflow_client()
        experiments = client.search_experiments()
        data = []
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["attribute.start_time DESC"],
                max_results=5,
            )
            runs_data = []
            for run in runs:
                metrics = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in run.data.metrics.items()
                }
                runs_data.append(
                    {
                        "run_id": run.info.run_id[:8],
                        "name": run.info.run_name or "Unnamed",
                        "status": run.info.status,
                        "metrics": metrics,
                        "start_time": run.info.start_time,
                    }
                )
            data.append(
                {
                    "id": exp.experiment_id,
                    "name": exp.name,
                    "runs": runs_data,
                }
            )
        return data
    except Exception as e:
        logger.error(f"Ошибка чтения MLflow: {e}")
        return []


def _get_model_registry() -> List[Dict[str, Any]]:
    """Получение зарегистрированных моделей из MLflow."""
    try:
        client = _get_mlflow_client()
        registered = client.search_registered_models(max_results=10)
        data = []
        for rm in registered:
            # Все версии, новые сверху. Стадии (Staging/Production) в MLflow 3
            # устарели — рабочую версию отмечает alias (champion).
            # search_model_versions не возвращает alias — берём их у модели.
            versions = client.search_model_versions(f"name='{rm.name}'")
            aliases = client.get_registered_model(rm.name).aliases or {}
            latest_versions = [
                {
                    "version": mv.version,
                    "aliases": [
                        a for a, v in aliases.items() if str(v) == str(mv.version)
                    ],
                    "run_id": mv.run_id[:8] if mv.run_id else None,
                }
                for mv in sorted(versions, key=lambda v: int(v.version), reverse=True)
            ]
            data.append(
                {
                    "name": rm.name,
                    "creation_time": rm.creation_timestamp,
                    "versions": latest_versions,
                }
            )
        return data
    except Exception as e:
        logger.error(f"Ошибка чтения Model Registry: {e}")
        return []


@router.get("/monitoring", response_class=HTMLResponse)
async def monitoring_dashboard(request: Request):
    """
    HTML-дашборд мониторинга ML-системы.
    Агрегирует MLflow, Evidently drift и системные метрики.
    """
    templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)

    experiments = _get_experiments_data()
    registry = _get_model_registry()
    drift = _load_drift_status()
    system = get_system_metrics()

    # Если нет экспериментов — показываем fallback с демо-данными
    demo_mode = len(experiments) == 0
    if demo_mode:
        experiments = [
            {
                "id": "0",
                "name": "Travel Churn Prediction",
                "runs": [
                    {
                        "run_id": "abc12345",
                        "name": "GradientBoosting",
                        "status": "FINISHED",
                        "metrics": {
                            "accuracy": 0.911,
                            "f1_score": 0.7952,
                            "roc_auc": 0.9747,
                            "precision": 0.8684,
                            "recall": 0.7333,
                        },
                        "start_time": 1700000000000,
                    }
                ],
            }
        ]

    return templates.TemplateResponse(
        request,
        "monitoring.html",
        {
            "experiments": experiments,
            "registry": registry,
            "drift": drift,
            "system": system,
            "demo_mode": demo_mode,
            "mlflow_ui_url": os.getenv("MLFLOW_UI_URL", "#"),
        },
    )


@router.get("/monitoring/status")
async def monitoring_api_status():
    """JSON endpoint для программного доступа к статусу мониторинга."""
    return {
        "experiments_count": len(_get_experiments_data()),
        "models_registered": len(_get_model_registry()),
        "drift": _load_drift_status(),
        "system": get_system_metrics(),
    }
