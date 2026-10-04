"""Расчёт дрейфа — одна реализация для API и скрипта отчёта (drift_stats.py)."""

import sys
from pathlib import Path

import pandas as pd

from src.monitoring.drift_stats import compute_drift

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_drift_report import (  # noqa: E402
    generate_drift_html_report,
    render_drift_report,
)


def mixed_frame(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [20 + i % 30 for i in range(rows)],
            "FrequentFlyer": ["Yes" if i % 3 else "No" for i in range(rows)],
            "Target": [i % 2 for i in range(rows)],
        }
    )


def test_numeric_and_categorical_features():
    summary = compute_drift(mixed_frame())
    by_feature = {r["feature"]: r for r in summary["results"]}
    assert summary["total_features"] == 2
    assert by_feature["Age"]["test"] == "Kolmogorov-Smirnov"
    assert by_feature["Age"]["p_value"] is not None
    assert by_feature["FrequentFlyer"]["test"] == "Jensen-Shannon divergence"
    assert by_feature["FrequentFlyer"]["p_value"] is None
    assert summary["reference_size"] + summary["current_size"] == 60


def test_report_with_categorical_feature_renders():
    """Старый скрипт печатал p_value у каждой строки и падал бы с KeyError
    на первом же категориальном признаке."""
    html = render_drift_report(compute_drift(mixed_frame()))
    assert "FrequentFlyer" in html
    assert "<td>-</td>" in html


def test_script_writes_report_and_summary(tmp_path):
    data = tmp_path / "data.csv"
    mixed_frame().to_csv(data, index=False)
    generate_drift_html_report(data_path=data, output_dir=tmp_path / "out")
    assert (tmp_path / "out" / "drift_report.html").exists()
    assert (tmp_path / "out" / "drift_summary.json").exists()
