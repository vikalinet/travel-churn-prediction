"""Мониторинг: места, найденные проверкой типов (mypy).

1. Отчёт о дрейфе и отчёт о качестве модели без текущих данных. Проверка
   объёма данных пропускала случай «текущих данных нет вовсе», и код падал на
   обращении к None; общий except превращал это в невнятное
   «'NoneType' object is not subscriptable» в журнале.
2. HTML-страницы дрейфа и мониторинга. Шаблоны вызывались устаревшей формой
   TemplateResponse(имя, контекст); теперь — TemplateResponse(request, имя, ...).
"""

import logging

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.monitoring import drift_monitor, performance_monitor
from src.monitoring.drift_monitor import DataDriftMonitor
from src.monitoring.performance_monitor import ModelPerformanceMonitor


@pytest.fixture
def evidently_on(monkeypatch):
    """Пройти мимо проверки «Evidently установлен» к проверке данных.

    Сам Evidently здесь не вызывается: без текущих данных до него не доходит.
    """
    monkeypatch.setattr(drift_monitor, "EVIDENTLY_AVAILABLE", True)
    monkeypatch.setattr(performance_monitor, "EVIDENTLY_AVAILABLE", True)


def reference_frame(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": list(range(rows)),
            "ServicesOpted": [i % 6 for i in range(rows)],
            "Target": [i % 2 for i in range(rows)],
        }
    )


def test_drift_report_without_current_data_is_explained(caplog, evidently_on):
    monitor = DataDriftMonitor(reference_frame(), ["Age", "ServicesOpted"], "Target")
    with caplog.at_level(logging.WARNING):
        assert monitor.generate_drift_report() is None
    assert "NoneType" not in caplog.text
    assert "Нет текущих данных" in caplog.text
    assert monitor.calculate_drift_metrics() == {}


def test_performance_report_without_current_predictions_is_explained(
    caplog, evidently_on
):
    reference = pd.DataFrame({"prediction": [0, 1] * 10, "Churn": [0, 1] * 10})
    monitor = ModelPerformanceMonitor(reference)
    with caplog.at_level(logging.WARNING):
        assert monitor.generate_performance_report() is None
    assert "NoneType" not in caplog.text
    assert "Нет текущих предсказаний" in caplog.text


def test_drift_dashboard_template_renders():
    with TestClient(app) as client:
        response = client.get("/api/v1/drift")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_monitoring_page_template_renders():
    with TestClient(app) as client:
        response = client.get("/api/v1/monitoring")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# ---- Evidently -----------------------------------------------------------
# Раньше отчёты Evidently не строились вовсе: импорт по пути, которого нет в
# Evidently 0.7, падал, ImportError проглатывался, а монитор качества модели
# вдобавок импортировал несуществующий класс. Эти тесты строят настоящие отчёты.


def test_evidently_is_importable():
    """Сторож: обновление Evidently, ломающее импорт, должно ронять тесты,
    а не выключать отчёты молча."""
    assert drift_monitor.EVIDENTLY_AVAILABLE
    assert performance_monitor.EVIDENTLY_AVAILABLE


def test_drift_report_is_built(runtime_dir_in_tmp):
    monitor = DataDriftMonitor(reference_frame(40), ["Age", "ServicesOpted"], "Target")
    monitor.update_current_data(reference_frame(30))
    path = monitor.generate_drift_report()
    assert path is not None
    assert path.startswith(str(runtime_dir_in_tmp))
    assert open(path, encoding="utf-8").read().lstrip().lower().startswith("<!doctype")


def test_performance_report_is_built(runtime_dir_in_tmp):
    predictions = pd.DataFrame(
        {"prediction": [0, 1, 1, 0] * 10, "Churn": [0, 1, 0, 0] * 10}
    )
    monitor = ModelPerformanceMonitor(predictions)
    monitor.update_current_predictions(predictions)
    path = monitor.generate_performance_report()
    assert path is not None
    assert path.startswith(str(runtime_dir_in_tmp))
