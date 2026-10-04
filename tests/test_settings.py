"""Пути и папка выполнения (src/settings.py).

Главное, что проверяется: запуск приложения не меняет файлы, лежащие в git.
Раньше при каждом старте анализ дрейфа перезаписывал evidently_reports/, и
после любого прогона тестов в репозитории появлялись изменения.
"""

import hashlib

from fastapi.testclient import TestClient

from src import settings
from src.api.main import app


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_app_startup_does_not_touch_tracked_reports(runtime_dir_in_tmp):
    tracked = sorted(settings.SAMPLE_REPORTS_DIR.glob("*.json"))
    assert tracked, "образцы отчётов должны лежать в репозитории"
    before = {path.name: digest(path) for path in tracked}

    with TestClient(app):  # старт приложения запускает анализ дрейфа
        pass

    assert {path.name: digest(path) for path in tracked} == before
    # Свежий результат ушёл в папку выполнения.
    assert (runtime_dir_in_tmp / "evidently_reports" / "drift_summary.json").exists()


def test_fresh_report_is_preferred_over_sample(runtime_dir_in_tmp):
    assert (
        settings.report_for_reading("drift_summary.json")
        == settings.SAMPLE_REPORTS_DIR / "drift_summary.json"
    )
    fresh = runtime_dir_in_tmp / "evidently_reports" / "drift_summary.json"
    fresh.parent.mkdir(parents=True)
    fresh.write_text("{}", encoding="utf-8")
    assert settings.report_for_reading("drift_summary.json") == fresh


def test_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert settings.MODELS_DIR.is_absolute()
    assert (settings.MODELS_DIR / "best_model.pkl").exists()
    assert settings.PROCESSED_DATA_PATH.exists()


def test_mlflow_uri_from_environment(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert settings.mlflow_tracking_uri().endswith("/mlflow.db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert settings.mlflow_tracking_uri() == "http://mlflow:5000"
