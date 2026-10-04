"""
Генерация HTML-отчёта по мониторингу дрейфа данных.

Пересобирает образцы evidently_reports/drift_report.html и
drift_summary.json, хранящиеся в репозитории (их показывает приложение,
пока анализ не запускался, и публикует GitHub Pages). Поэтому, в отличие от
приложения, скрипт сознательно пишет в репозиторий: это ручная пересборка
образцов, после которой изменения коммитятся.

Расчёт дрейфа — общий с API (src/monitoring/drift_stats.py), разметка — в
шаблоне templates/reports/drift_report.html.

    python scripts/generate_drift_report.py
"""

import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import settings  # noqa: E402
from src.monitoring.drift_stats import compute_drift  # noqa: E402

_templates = Environment(
    loader=FileSystemLoader(settings.TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
    # Сохранять перевод строки в конце: иначе хук end-of-file-fixer правит
    # каждый пересобранный отчёт.
    keep_trailing_newline=True,
)


def render_drift_report(summary: Dict[str, Any]) -> str:
    """HTML-отчёт по сводке compute_drift()."""
    return _templates.get_template("reports/drift_report.html").render(summary=summary)


def generate_drift_html_report(
    data_path: Path = settings.PROCESSED_DATA_PATH,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Посчитать дрейф и сохранить HTML-отчёт и JSON-сводку.

    По умолчанию — в evidently_reports/ (образцы в репозитории).
    """
    summary = compute_drift(pd.read_csv(data_path))
    target = output_dir or settings.SAMPLE_REPORTS_DIR
    target.mkdir(parents=True, exist_ok=True)

    html_path = target / "drift_report.html"
    html_path.write_text(render_drift_report(summary), encoding="utf-8")
    print(f"✅ HTML отчёт сохранён: {html_path}")

    json_path = target / "drift_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"JSON сводка сохранена: {json_path}")

    return summary


if __name__ == "__main__":
    # Установка UTF-8 кодировки для Windows
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=== Запуск генерации отчёта мониторинга ===")
    generate_drift_html_report()
    print("\n=== Отчёт сгенерирован ===")
