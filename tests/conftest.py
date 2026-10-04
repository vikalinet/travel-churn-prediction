"""Общая настройка тестов."""

import pytest


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
