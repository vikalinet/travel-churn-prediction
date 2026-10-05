"""Обучение моделей (src/training): базовые модели, улучшенный пайплайн,
подбор гиперпараметров Optuna, сравнение и логирование в MLflow.

Модели обучаются на выборке из 300 клиентов реального датасета — быстро, но
на настоящих признаках. Всё, что обучение записывает (модель, CSV, графики,
база MLflow), уходит во временную папку теста, а не в репозиторий.
"""

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src import settings
from src.training.base_trainer import BaseTrainer
from src.training.hyperparameter_tuning import HyperparameterTuner
from src.training.improved_training import ImprovedModelTrainer
from src.training.mlflow_integration import MLflowIntegration
from src.training.model_comparison import ModelComparator
from src.training.model_training import ModelTrainer, train_base_models

METRICS = {"accuracy", "f1_score", "roc_auc", "precision", "recall"}


@pytest.fixture(scope="module")
def data_csv(tmp_path_factory):
    """300 клиентов из обработанного датасета с сохранением доли оттока."""
    df = pd.read_csv(settings.PROCESSED_DATA_PATH)
    sample = (
        df.groupby("Target", group_keys=False)
        .apply(lambda part: part.sample(frac=300 / len(df), random_state=42))
        .reset_index(drop=True)
    )
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    sample.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def split(data_csv):
    trainer = BaseTrainer(str(data_csv))
    return trainer.prepare_data(*trainer.load_data())


def assert_metrics(row: dict) -> None:
    assert METRICS <= set(row)
    assert all(0.0 <= row[m] <= 1.0 for m in METRICS)


# ---- BaseTrainer -----------------------------------------------------------


def test_split_is_stratified(data_csv):
    trainer = BaseTrainer(str(data_csv))
    X, y = trainer.load_data()
    assert "Target" not in X.columns
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
    # stratify=y: доля оттока в частях совпадает с общей
    assert abs(y_train.mean() - y.mean()) < 0.02
    assert abs(y_test.mean() - y.mean()) < 0.02


def test_calculate_metrics_on_known_values():
    metrics = BaseTrainer("unused").calculate_metrics(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0, 1, 1, 1]),
        pd.Series([0.1, 0.6, 0.8, 0.9]),
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == 1.0


def test_best_model_requires_trained_models():
    with pytest.raises(ValueError, match="Нет обученных моделей"):
        BaseTrainer("unused").get_best_model()


# ---- базовые модели --------------------------------------------------------


def test_base_models_train_and_rank(split):
    trainer = ModelTrainer("unused")
    X_train, X_test, y_train, y_test = split
    trainer.train_models(X_train, y_train, X_test, y_test)
    assert set(trainer.models) == {
        "LogisticRegression",
        "RandomForest",
        "KNeighbors",
        "XGBoost",
        "GradientBoosting",
        "SVC",
    }
    for row in trainer.results:
        assert_metrics(row)

    ranked = trainer.compare_models()
    assert list(ranked["f1_score"]) == sorted(ranked["f1_score"], reverse=True)
    best_name, best_model = trainer.get_best_model()
    # При равном F1 у нескольких моделей лучшей может оказаться любая из них.
    best_f1 = next(
        r["f1_score"] for r in trainer.results if r["model_name"] == best_name
    )
    assert best_f1 == ranked.iloc[0]["f1_score"]
    assert best_model is trainer.models[best_name]


def test_train_base_models_from_file(data_csv):
    trainer = train_base_models(str(data_csv))
    assert len(trainer.results) == 6


# ---- улучшенный пайплайн ---------------------------------------------------


def test_optimal_threshold_is_searched_in_range():
    trainer = ImprovedModelTrainer("unused")
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.35, 0.4, 0.7, 0.9])
    threshold, score = trainer.find_optimal_threshold(y_true, y_proba)
    assert 0.1 <= threshold < 0.9
    assert isinstance(threshold, float) and isinstance(score, float)
    # Порог между 0.35 и 0.4 разделяет классы без ошибок.
    assert score == 1.0
    assert 0.35 < threshold <= 0.4


def test_improved_pipeline_saves_model_package(data_csv, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    (tmp_path / "reports").mkdir()
    output = tmp_path / "models" / "best_model_improved.pkl"

    trainer = ImprovedModelTrainer(str(data_csv))
    best_name, _, results = trainer.run_improved_pipeline(output_path=str(output))

    package = joblib.load(output)
    assert set(package) >= {"model", "threshold", "feature_names", "metrics"}
    assert package["model_name"] == best_name
    assert 0.0 < package["threshold"] < 1.0
    assert len(package["feature_names"]) == 45  # 6 признаков → 45 после FE
    assert_metrics(package["metrics"])
    assert "Stacking" in set(results["model_name"])
    assert (tmp_path / "reports" / "training_results_improved.csv").exists()


# ---- Optuna ----------------------------------------------------------------


@pytest.mark.parametrize("method", ["tune_xgboost", "tune_random_forest"])
def test_hyperparameter_tuning(split, method):
    X_train, X_test, y_train, y_test = split
    model, result = getattr(
        HyperparameterTuner(X_train, y_train, X_test, y_test), method
    )(n_trials=2)
    assert hasattr(model, "predict_proba")
    assert result["best_params"]
    assert_metrics(result)


# ---- сравнение и MLflow ----------------------------------------------------


def test_comparator_writes_plot_and_csv(tmp_path):
    results = pd.DataFrame(
        [
            {
                "model_name": "A",
                "accuracy": 0.9,
                "f1_score": 0.8,
                "roc_auc": 0.95,
                "precision": 0.85,
            },
            {
                "model_name": "B",
                "accuracy": 0.8,
                "f1_score": 0.7,
                "roc_auc": 0.90,
                "precision": 0.75,
            },
        ]
    )
    ModelComparator.plot_comparison(results, save_path=str(tmp_path / "plot.png"))
    ModelComparator.save_results(results, save_path=str(tmp_path / "results.csv"))
    assert (tmp_path / "plot.png").stat().st_size > 0
    assert list(pd.read_csv(tmp_path / "results.csv")["model_name"]) == ["A", "B"]


def test_mlflow_logging_goes_to_given_store(split, tmp_path, monkeypatch):
    # Артефакты MLflow по умолчанию пишутся в ./mlruns — уводим в tmp.
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    MLflowIntegration.setup_tracking(uri)
    try:
        model = ModelTrainer("unused")
        model.train_models(split[0], split[2], split[1], split[3])
        name, fitted = model.get_best_model()
        metrics = next(r for r in model.results if r["model_name"] == name)
        MLflowIntegration.log_model(
            name, fitted, metrics, params={"source": "test"}, experiment_name="test"
        )
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name("test")
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        assert runs[0].data.metrics["f1_score"] == pytest.approx(metrics["f1_score"])
    finally:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri())


def test_mlflow_default_store_is_project_root(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    try:
        MLflowIntegration.setup_tracking()
        assert mlflow.get_tracking_uri() == settings.mlflow_tracking_uri()
        assert mlflow.get_tracking_uri().endswith("/mlflow.db")
    finally:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri())
