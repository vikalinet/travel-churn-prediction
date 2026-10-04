"""
Генерация отчёта о результатах обучения моделей.
Загружает актуальные результаты из reports/training_results.csv.

Пересобирает reports/training_report.html — он хранится в репозитории и
публикуется на GitHub Pages, поэтому скрипт сознательно пишет в репозиторий.
Разметка — в шаблоне templates/reports/training_report.html.

    python scripts/generate_training_report.py
"""

import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import settings  # noqa: E402

RESULTS_PATH = ROOT / "reports" / "training_results.csv"
METRICS = ["accuracy", "f1_score", "roc_auc", "precision", "recall"]

_templates = Environment(
    loader=FileSystemLoader(settings.TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
)


def build_summary(
    results_df: pd.DataFrame, data_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Данные для отчёта: модели по убыванию качества и лучшая из них.

    Модели ранжируются по F1, а если его нет в результатах — по accuracy, и
    одной и той же метрикой и выбирается лучшая модель, и сортируется таблица.
    Прежде лучшая выбиралась с запасным вариантом, а сортировка всегда шла по
    f1_score: без этой колонки скрипт падал, а подпись «по F1-Score» была бы
    неверной.
    """
    rank_metric = "f1_score" if "f1_score" in results_df.columns else "accuracy"
    ranked = results_df.sort_values(rank_metric, ascending=False).copy()
    # Отсутствующие метрики показываются нулями — как и прежде.
    for metric in METRICS:
        if metric not in ranked.columns:
            ranked[metric] = 0.0
    models = ranked.to_dict(orient="records")
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_info": data_info,
        "models": models,
        "best": models[0],
        "rank_metric": rank_metric,
        "rank_label": "F1-Score" if rank_metric == "f1_score" else "Accuracy",
    }


def render_training_report(summary: Dict[str, Any]) -> str:
    return _templates.get_template("reports/training_report.html").render(**summary)


def generate_training_report(
    results_path: Path = RESULTS_PATH, output_dir: Optional[Path] = None
) -> Optional[Path]:
    """Отчёт на основе актуальных результатов обучения."""
    if not results_path.exists():
        print(f"Файл {results_path} не найден! Запустите обучение модели сначала.")
        return None

    results_df = pd.read_csv(results_path)
    print(f"Данные загружены: {len(results_df)} моделей")
    print(results_df.to_string(index=False))

    if settings.PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(settings.PROCESSED_DATA_PATH)
        data_info: Dict[str, Any] = {
            "total_rows": len(df),
            "features": len(df.columns) - 1,
        }
    else:
        data_info = {"total_rows": "N/A", "features": "N/A"}

    target = output_dir or ROOT / "reports"
    target.mkdir(parents=True, exist_ok=True)
    html_path = target / "training_report.html"
    html_path.write_text(
        render_training_report(build_summary(results_df, data_info)), encoding="utf-8"
    )
    print(f"\n✅ HTML отчёт сохранён: {html_path}")
    return html_path


if __name__ == "__main__":
    # Установка UTF-8 кодировки для Windows
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=== Запуск генерации отчёта по обучению ===")
    generate_training_report()
    print("\n=== Отчёт сгенерирован ===")
