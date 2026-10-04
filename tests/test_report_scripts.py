"""Скрипты отчётов: отчёт об обучении и README.html (шаблоны вместо f-строк)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_readme_html import generate_readme_html  # noqa: E402
from generate_training_report import (  # noqa: E402
    build_summary,
    generate_training_report,
    render_training_report,
)

DATA_INFO = {"total_rows": 954, "features": 6}


def results(with_f1: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "model_name": ["Forest", "Boosting"],
            "accuracy": [0.95, 0.90],
            "f1_score": [0.70, 0.80],
            "roc_auc": [0.97, 0.96],
            "precision": [0.75, 0.84],
            "recall": [0.66, 0.77],
        }
    )
    return frame if with_f1 else frame.drop(columns="f1_score")


def test_models_ranked_by_f1():
    summary = build_summary(results(), DATA_INFO)
    assert [m["model_name"] for m in summary["models"]] == ["Boosting", "Forest"]
    assert summary["best"]["model_name"] == "Boosting"
    assert summary["rank_label"] == "F1-Score"


def test_without_f1_ranks_by_accuracy_consistently():
    """Прежде лучшая модель выбиралась по accuracy, а таблица всё равно
    сортировалась по f1_score — без этой колонки скрипт падал."""
    summary = build_summary(results(with_f1=False), DATA_INFO)
    assert summary["best"]["model_name"] == "Forest"
    assert summary["rank_label"] == "Accuracy"
    html = render_training_report(summary)
    assert "Лучшая модель по Accuracy:</strong> Forest" in html


def test_training_report_is_written(tmp_path):
    csv = tmp_path / "training_results.csv"
    results().to_csv(csv, index=False)
    path = generate_training_report(results_path=csv, output_dir=tmp_path)
    assert path is not None
    assert 'class="best"' in path.read_text(encoding="utf-8")


def test_missing_results_file(tmp_path):
    assert generate_training_report(results_path=tmp_path / "нет.csv") is None


def test_readme_page(tmp_path):
    source = tmp_path / "README.md"
    source.write_text("# Заголовок\n\nТекст с <тегом> и `кодом`.\n", encoding="utf-8")
    out = tmp_path / "README.html"
    generate_readme_html(source, out)
    html = out.read_text(encoding="utf-8")
    assert "<h1>Заголовок</h1>" in html
    assert "&lt;тегом&gt;" in html  # текст экранирован, а не вставлен как разметка
    assert "<code>кодом</code>" in html
