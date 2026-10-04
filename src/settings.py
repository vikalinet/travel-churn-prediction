"""
Настройки проекта: пути и параметры из переменных окружения.

Два правила, ради которых модуль заведён:

1. **Пути считаются от корня проекта, а не от папки запуска.** Раньше они
   были относительными (``Path("evidently_reports/...")``), и результат
   зависел от того, из какой папки запущен процесс: из корня всё работало,
   из ``src/`` или из IDE — файлы искались и создавались не там.

2. **Файлы в git — образцы только для чтения; то, что программа пишет при
   работе, уходит в отдельную папку вне git.** Раньше при каждом старте
   приложения анализ дрейфа перезаписывал ``evidently_reports/drift_*.json``,
   лежащие в репозитории: после любого запуска, включая прогон тестов, в git
   появлялись изменения, которых никто не делал. Теперь свежий результат
   пишется в папку выполнения, а читается он, если есть, иначе — образец из
   репозитория, так что показ работает и на чистой копии.

Переменные окружения перечислены в ``.env.example``.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Образцы в репозитории — нужны для показа и развёртывания (Railway собирает
# прямо из git), поэтому хранятся в нём сознательно.
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_REPORTS_DIR = PROJECT_ROOT / "evidently_reports"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.csv"


def runtime_dir() -> Path:
    """Папка для того, что программа пишет при работе (вне git).

    Функция, а не константа: значение читается при каждом обращении, и тесты
    подменяют папку переменной окружения, не перезагружая модули.
    """
    return Path(os.environ.get("CHURN_RUNTIME_DIR", PROJECT_ROOT / "runtime"))


def runtime_reports_dir() -> Path:
    return runtime_dir() / "evidently_reports"


def report_for_reading(name: str) -> Path:
    """Свежий отчёт из папки выполнения, если он есть, иначе образец из репозитория."""
    fresh = runtime_reports_dir() / name
    return fresh if fresh.exists() else SAMPLE_REPORTS_DIR / name


def mlflow_tracking_uri() -> str:
    """База экспериментов MLflow. По умолчанию — mlflow.db в корне проекта."""
    return os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
