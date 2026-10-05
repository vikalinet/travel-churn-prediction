"""API кодирует клиента так же, как данные, на которых обучалась модель.

Прежде API масштабировал Age и ServicesOpted и кодировал BookedHotelOrNot
наоборот: на тех же клиентах ROC AUC через API был 0.68, а напрямую — 0.98.
"""

import glob

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

import src.api.main as main_module
from src import settings
from src.api.preprocessing import preprocess_single_customer
from src.monitoring import prediction_log

API_FIELDS = {
    "Age": "age",
    "FrequentFlyer": "frequent_flyer",
    "AnnualIncomeClass": "annual_income_class",
    "ServicesOpted": "services_opted",
    "AccountSyncedToSocialMedia": "account_synced_to_social_media",
    "BookedHotelOrNot": "booked_hotel_or_not",
}


@pytest.fixture(scope="module")
def raw_and_processed():
    raw = pd.read_csv(glob.glob(str(settings.RAW_DATA_DIR / "*.csv"))[0])
    processed = pd.read_csv(settings.PROCESSED_DATA_PATH)
    assert len(raw) == len(processed)
    customers = [
        {API_FIELDS[k]: v for k, v in row.items() if k in API_FIELDS}
        for row in raw.to_dict("records")
    ]
    return customers, processed


def test_api_encoding_matches_processed_data(raw_and_processed):
    customers, processed = raw_and_processed
    encoded = pd.concat(
        [preprocess_single_customer(c) for c in customers], ignore_index=True
    )
    pd.testing.assert_frame_equal(
        encoded, processed[encoded.columns], check_dtype=False
    )


def test_api_predictions_match_model_on_training_data(raw_and_processed, monkeypatch):
    customers, processed = raw_and_processed
    package = joblib.load(settings.MODELS_DIR / "best_model_improved.pkl")
    # Модель — в глобальные переменные API на время теста (без lifespan,
    # чтобы не оставлять её другим тестам).
    monkeypatch.setattr(main_module, "model", package["model"])
    monkeypatch.setattr(main_module, "model_package", package)
    monkeypatch.setattr(main_module, "model_threshold", package["threshold"])
    monkeypatch.setattr(main_module, "model_metrics", package["metrics"])

    response = TestClient(main_module.app).post("/api/v1/predict_batch", json=customers)
    api_proba = [p["probability"] for p in response.json()["predictions"]]

    X = package["feature_engineer"].transform(processed.drop(columns=["Target"]))
    direct = package["model"].predict_proba(X[package["feature_names"]])[:, 1]
    assert api_proba == pytest.approx(list(direct), abs=1e-9)
    assert roc_auc_score(processed["Target"], api_proba) > 0.9


def test_predictions_are_logged_for_drift(raw_and_processed, monkeypatch):
    customers, processed = raw_and_processed
    package = joblib.load(settings.MODELS_DIR / "best_model_improved.pkl")
    monkeypatch.setattr(main_module, "model", package["model"])
    monkeypatch.setattr(main_module, "model_package", package)

    client = TestClient(main_module.app)
    client.post("/api/v1/predict", json=customers[0])
    client.post("/api/v1/predict_batch", json=customers[1:3])

    logged = prediction_log.load_logged()
    expected = processed.drop(columns=["Target"]).head(3)
    pd.testing.assert_frame_equal(logged, expected, check_dtype=False)


def test_models_endpoint_reports_registry_version(monkeypatch):
    package = joblib.load(settings.MODELS_DIR / "best_model_improved.pkl")
    monkeypatch.setattr(main_module, "model", package["model"])
    monkeypatch.setattr(main_module, "model_package", package)

    info = TestClient(main_module.app).get("/api/v1/models").json()
    assert info["registered_model"] == "travel-churn-model"
    assert info["model_name"] == package["model_name"]
    assert info["model_version"] == package["model_version"]
    assert info["mlflow_run_id"] == package["mlflow_run_id"]
