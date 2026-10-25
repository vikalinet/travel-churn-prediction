"""SHAP-объяснение прогноза — поверх уже обученной модели, без переобучения."""

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.api.main as main_module
from src import settings
from src.models.explain import PredictionExplainer


@pytest.fixture(scope="module")
def package():
    return joblib.load(settings.MODELS_DIR / "best_model_improved.pkl")


def test_top_factors_shape_and_direction(package):
    explainer = PredictionExplainer(package["model"], package["feature_engineer"])
    raw = pd.read_csv(settings.PROCESSED_DATA_PATH).drop(columns=["Target"])
    X = package["feature_engineer"].transform(raw)[package["feature_names"]].iloc[[0]]

    factors = explainer.top_factors(X, top_n=5)

    assert len(factors) == 5
    for factor in factors:
        assert set(factor) == {"feature", "contribution", "direction"}
        assert isinstance(factor["contribution"], float)
        assert factor["direction"] in {"к оттоку", "удерживает"}
        if factor["direction"] == "к оттоку":
            assert factor["contribution"] > 0
        else:
            assert factor["contribution"] < 0
    # Отсортированы по модулю вклада, не по знаку.
    magnitudes = [abs(f["contribution"]) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_poly_feature_names_are_readable(package):
    explainer = PredictionExplainer(package["model"], package["feature_engineer"])
    # Ни одно читаемое имя не должно остаться обобщённым poly_N.
    assert all(not name.startswith("poly_") for name in explainer._poly_rename.values())


def test_predict_endpoint_returns_top_factors(package, monkeypatch):
    monkeypatch.setattr(main_module, "model", package["model"])
    monkeypatch.setattr(main_module, "model_package", package)
    monkeypatch.setattr(main_module, "model_threshold", package["threshold"])
    monkeypatch.setattr(main_module, "model_metrics", package["metrics"])
    monkeypatch.setattr(
        main_module,
        "explainer",
        PredictionExplainer(package["model"], package["feature_engineer"]),
    )

    customer = {
        "age": 34,
        "frequent_flyer": "Yes",
        "annual_income_class": "Low Income",
        "services_opted": 5,
        "account_synced_to_social_media": "Yes",
        "booked_hotel_or_not": "No",
    }
    response = TestClient(main_module.app).post("/api/v1/predict", json=customer)

    assert response.status_code == 200
    top_factors = response.json()["top_factors"]
    assert top_factors is not None
    assert len(top_factors) == 5
    assert all({"feature", "contribution", "direction"} == set(f) for f in top_factors)
