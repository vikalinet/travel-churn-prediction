"""Общая настройка тестов."""

import os

import mlflow
import pytest

from src import settings


@pytest.fixture(autouse=True)
def runtime_dir_in_tmp(tmp_path, monkeypatch):
    """Всё, что программа пишет при работе, — во временную папку теста.

    Без этого тесты, поднимающие приложение, писали бы отчёты о дрейфе в
    рабочую папку runtime/ разработчика (а до src/settings.py — прямо поверх
    файлов в репозитории).
    """
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("CHURN_RUNTIME_DIR", str(runtime))
    return runtime


@pytest.fixture(autouse=True)
def restore_mlflow_tracking_uri():
    """Адрес MLflow после теста — как до него.

    mlflow.set_tracking_uri() в MLflow 3 записывает адрес и в переменную
    окружения MLFLOW_TRACKING_URI, а src/settings.py читает её. Тест,
    направивший MLflow во временную базу, иначе оставлял её следующим тестам.
    """
    before = os.environ.get("MLFLOW_TRACKING_URI")
    yield
    if before is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = before
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri())
