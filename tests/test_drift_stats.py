"""Расчёт дрейфа — одна реализация для API и скрипта отчёта (drift_stats.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from src.api import drift_router
from src.api.main import app
from src.monitoring import prediction_log
from src.monitoring.drift_stats import (
    compute_drift,
    population_stability_index,
    psi_level,
)

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


# ---- PSI -------------------------------------------------------------------


def test_psi_is_zero_for_identical_distributions():
    values = pd.Series(["Yes", "No", "No", "Yes"] * 10)
    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_psi_known_value_for_categories():
    # 50/50 → 80/20: 0.3·ln(1.6) + (−0.3)·ln(0.4) ≈ 0.4159
    ref = pd.Series(["Yes"] * 50 + ["No"] * 50)
    curr = pd.Series(["Yes"] * 80 + ["No"] * 20)
    assert population_stability_index(ref, curr) == pytest.approx(0.4159, abs=1e-4)


def test_psi_numeric_shift_uses_quantile_bins():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 2000))
    same = pd.Series(rng.normal(0, 1, 2000))
    shifted = pd.Series(rng.normal(1.5, 1, 2000))
    assert population_stability_index(ref, same) < 0.1
    # Значения вне диапазона эталона попадают в крайние интервалы, а не теряются.
    assert population_stability_index(ref, shifted) > 0.25


def test_psi_handles_category_missing_in_reference():
    ref = pd.Series(["A"] * 100)
    curr = pd.Series(["A"] * 90 + ["B"] * 10)
    psi = population_stability_index(ref, curr)
    assert np.isfinite(psi) and psi > 0.25


@pytest.mark.parametrize(
    "psi, level",
    [(0.05, "стабильно"), (0.1, "умеренный сдвиг"), (0.3, "значимый сдвиг")],
)
def test_psi_levels(psi, level):
    assert psi_level(psi) == level


def test_summary_has_psi_for_every_feature():
    summary = compute_drift(mixed_frame())
    assert all("psi" in r and "psi_level" in r for r in summary["results"])
    assert summary["max_psi"] == max(r["psi"] for r in summary["results"])
    assert summary["psi_threshold"] == 0.25


def test_large_psi_marks_drift():
    """Признак, сдвинутый между эталонной и текущей частью, — дрейф по PSI."""
    df = mixed_frame(200)
    # train_test_split перемешивает строки: сдвигаем именно текущую часть.
    _, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    df.loc[test_idx, "FrequentFlyer"] = "No"
    row = next(
        r for r in compute_drift(df)["results"] if r["feature"] == "FrequentFlyer"
    )
    assert row["psi"] > 0.25
    assert row["drift_detected"]


def test_report_and_dashboard_show_psi(monkeypatch):
    summary = compute_drift(mixed_frame())
    assert "PSI" in render_drift_report(summary)

    monkeypatch.setattr(drift_router, "_load_drift_summary", lambda: summary)
    # Без `with`: lifespan загрузил бы настоящую модель в глобальное состояние
    # src.api.main, а тесты test_integration подменяют только model.
    html = TestClient(app).get("/drift").text
    assert "<th>PSI</th>" in html
    assert "Макс. PSI" in html
    # Строки признаков действительно выводятся (прежде таблица была пустой).
    assert "<strong>Age</strong>" in html
    assert "<strong>FrequentFlyer</strong>" in html


# ---- дрейф на реальных запросах -------------------------------------------


def test_compute_drift_against_logged_requests():
    df = mixed_frame(100)
    current = df.drop(columns=["Target"]).head(30)
    summary = compute_drift(df, current=current)
    assert summary["source"] == "live"
    assert summary["reference_size"] == 100
    assert summary["current_size"] == 30


def test_split_mode_is_reported():
    assert compute_drift(mixed_frame())["source"] == "split"


def test_analyze_uses_logged_requests_once_enough(tmp_path):
    data = tmp_path / "data.csv"
    df = mixed_frame(200)
    df.to_csv(data, index=False)

    # Мало запросов — по-прежнему части датасета.
    shifted = df.drop(columns=["Target"]).assign(Age=70, FrequentFlyer="No")
    prediction_log.log_inputs(shifted.head(prediction_log.MIN_LIVE_ROWS - 1))
    assert drift_router._analyze_drift(data_path=data)["source"] == "split"

    # Набралось — сравниваются запросы, и сдвиг виден.
    prediction_log.log_inputs(shifted.head(1))
    summary = drift_router._analyze_drift(data_path=data)
    assert summary["source"] == "live"
    assert summary["current_size"] == prediction_log.MIN_LIVE_ROWS
    assert summary["alert"]
    assert {"Age", "FrequentFlyer"} <= {
        r["feature"] for r in summary["results"] if r["drift_detected"]
    }
