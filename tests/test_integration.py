"""
Интеграционные тесты для всего пайплайна.
"""

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import src.api.main as main_module

from src.api.main import app
from src.etl.etl_pipeline import run_etl
from src.monitoring.drift_monitor_customer import CustomerTravelDriftMonitor
from src.monitoring.system_monitor import generate_system_report, get_system_metrics
from src.training.model_training import ModelTrainer


class TestFullPipeline:
    """Тесты полного пайплайна от данных до предсказания."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Создание тестового датасета."""
        # Генерация синтетических данных
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame(
            {
                "age": np.random.randint(18, 70, n_samples),
                "annual_income": np.random.uniform(20000, 150000, n_samples),
                "flight_count": np.random.randint(0, 20, n_samples),
                "gender": np.random.choice(["M", "F"], n_samples),
                "marital_status": np.random.choice(
                    ["Single", "Married", "Divorced"], n_samples
                ),
                "education": np.random.choice(
                    ["High School", "Bachelor", "Master", "PhD"], n_samples
                ),
                "occupation": np.random.choice(
                    ["Student", "Employed", "Self-Employed"], n_samples
                ),
                "city_tier": np.random.choice(["Tier1", "Tier2", "Tier3"], n_samples),
                "number_of_dependents": np.random.randint(0, 6, n_samples),
                "total_spending": np.random.uniform(1000, 50000, n_samples),
                "walk_in_count": np.random.randint(0, 10, n_samples),
                "web_login_count": np.random.randint(0, 50, n_samples),
                "mobile_app_login_count": np.random.randint(0, 100, n_samples),
                "last_visit_date_days": np.random.randint(1, 365, n_samples),
                "complaint_count": np.random.randint(0, 5, n_samples),
                "is_member": np.random.choice([0, 1], n_samples),
                "Target": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            }
        )

        # Сохранение в временный файл
        csv_path = tmp_path / "sample_data.csv"
        data.to_csv(csv_path, index=False)

        return str(csv_path)

    def test_etl_pipeline(self, sample_dataset, tmp_path):
        """Тест ETL пайплайна."""
        output_path = tmp_path / "processed_data.csv"

        # Запуск ETL
        df_processed = run_etl(sample_dataset, str(output_path))

        # Проверки
        assert df_processed is not None
        assert len(df_processed) > 0
        assert output_path.exists()
        assert "Target" in df_processed.columns

    def test_model_training(self, sample_dataset, tmp_path):
        """Тест обучения модели."""
        # Загрузка и предобработка
        df = pd.read_csv(sample_dataset)

        # Удаление строк с пропусками для теста
        df = df.dropna()

        # Кодирование категориальных признаков
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        processed_path = tmp_path / "processed_data.csv"
        df.to_csv(processed_path, index=False)

        # Инициализация и обучение
        trainer = ModelTrainer(str(processed_path), target_column="Target")
        X, y = trainer.load_data()
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)

        # Обучение одной модели для быстрого теста
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)

        # Проверка предсказаний
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_end_to_end_prediction(self, sample_dataset, tmp_path):
        """Тест сквозного предсказания."""
        # 1. Загрузка данных
        df = pd.read_csv(sample_dataset)
        y = df["Target"].copy()
        X = df.drop(columns=["Target"]).copy()

        # 2. Кодирование ВСЕХ категориальных признаков (включая неявные)
        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            else:
                # На всякий случай конвертируем в числовой тип
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        # 3. Обучение простой модели
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        # 4. Предсказание для новых данных
        new_customer = X.iloc[:1]
        prediction = model.predict(new_customer)
        probability = model.predict_proba(new_customer)

        assert prediction[0] in [0, 1]
        assert 0 <= probability[0][1] <= 1


class TestAPIIntegration:
    """Интеграционные тесты API."""

    def test_api_health_check(self):
        """Тест проверки здоровья API."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_api_root(self):
        """Тест главной страницы API."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        assert "Travel Churn Prediction API" in response.json()["message"]

    def test_api_predict_endpoint(self, tmp_path):
        """Тест эндпоинта предсказания."""
        # Создание тестовой модели
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X_dummy = pd.DataFrame(
            {
                "Age": [25, 30, 35],
                "FrequentFlyer": [1, 0, 1],
                "AnnualIncomeClass": [1, 2, 0],
                "ServicesOpted": [3, 5, 2],
                "AccountSyncedToSocialMedia": [1, 0, 1],
                "BookedHotelOrNot": [0, 1, 0],
            }
        )
        y_dummy = [0, 1, 0]
        model.fit(X_dummy, y_dummy)

        # Сохранение модели во временную папку models/
        models_dir = tmp_path / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "best_model.pkl"
        joblib.dump(model, model_path)

        # Мокаем путь к модели в API
        main_module.model = model
        main_module.model_mapping = {
            "FrequentFlyer": {"Yes": 1, "No": 0},
            "AnnualIncomeClass": {
                "Low Income": 0,
                "Middle Income": 1,
                "High Income": 2,
            },
            "AccountSyncedToSocialMedia": {"Yes": 1, "No": 0},
            "BookedHotelOrNot": {"Yes": 1, "No": 0},
        }

        client = TestClient(app)

        # Тестовые данные (соответствуют CustomerInput схеме)
        test_data = {
            "age": 30,
            "frequent_flyer": "No",
            "annual_income_class": "Middle Income",
            "services_opted": 5,
            "account_synced_to_social_media": "Yes",
            "booked_hotel_or_not": "No",
        }

        response = client.post("/predict", json=test_data)

        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result
        assert "probability" in result
        assert "risk_level" in result
        assert result["prediction"] in [0, 1]
        assert 0 <= result["probability"] <= 1
        assert result["risk_level"] in ["Low", "Medium", "High"]

    def test_api_predict_batch_endpoint(self, tmp_path):
        """Тест пакетного предсказания."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X_dummy = pd.DataFrame(
            {
                "Age": [25, 30, 35],
                "FrequentFlyer": [1, 0, 1],
                "AnnualIncomeClass": [1, 2, 0],
                "ServicesOpted": [3, 5, 2],
                "AccountSyncedToSocialMedia": [1, 0, 1],
                "BookedHotelOrNot": [0, 1, 0],
            }
        )
        y_dummy = [0, 1, 0]
        model.fit(X_dummy, y_dummy)

        main_module.model = model
        main_module.model_mapping = {
            "FrequentFlyer": {"Yes": 1, "No": 0},
            "AnnualIncomeClass": {
                "Low Income": 0,
                "Middle Income": 1,
                "High Income": 2,
            },
            "AccountSyncedToSocialMedia": {"Yes": 1, "No": 0},
            "BookedHotelOrNot": {"Yes": 1, "No": 0},
        }

        client = TestClient(app)

        batch_data = [
            {
                "age": 30,
                "frequent_flyer": "No",
                "annual_income_class": "Middle Income",
                "services_opted": 5,
                "account_synced_to_social_media": "Yes",
                "booked_hotel_or_not": "No",
            },
            {
                "age": 42,
                "frequent_flyer": "Yes",
                "annual_income_class": "High Income",
                "services_opted": 4,
                "account_synced_to_social_media": "No",
                "booked_hotel_or_not": "Yes",
            },
        ]

        response = client.post("/predict_batch", json=batch_data)

        assert response.status_code == 200
        result = response.json()
        assert "predictions" in result
        assert len(result["predictions"]) == 2
        for pred in result["predictions"]:
            assert "prediction" in pred
            assert "probability" in pred
            assert "risk_level" in pred
            assert pred["prediction"] in [0, 1]

    def test_api_models_info(self):
        """Тест эндпоинта информации о модели."""
        client = TestClient(app)
        response = client.get("/models")

        assert response.status_code == 200
        result = response.json()
        assert "model_loaded" in result


class TestModelPersistence:
    """Тесты сохранения и загрузки моделей."""

    def test_save_and_load_model(self, tmp_path):
        """Тест сохранения и загрузки модели."""
        # Создание и обучение модели
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X_train = np.random.rand(20, 5)
        y_train = np.random.randint(0, 2, 20)
        model.fit(X_train, y_train)

        # Сохранение
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        assert model_path.exists()

        # Загрузка
        loaded_model = joblib.load(model_path)

        # Проверка, что модель работает
        test_input = np.random.rand(1, 5)
        original_pred = model.predict(test_input)
        loaded_pred = loaded_model.predict(test_input)

        assert np.array_equal(original_pred, loaded_pred)


class TestAPIEdgeCases:
    """Тесты граничных случаев API."""

    def test_predict_without_model(self):
        """Тест предсказания при отсутствии модели."""
        # Сохраняем текущее состояние
        old_model = main_module.model
        main_module.model = None

        try:
            client = TestClient(app)
            test_data = {
                "age": 30,
                "frequent_flyer": "No",
                "annual_income_class": "Middle Income",
                "services_opted": 5,
                "account_synced_to_social_media": "Yes",
                "booked_hotel_or_not": "No",
            }
            response = client.post("/predict", json=test_data)
            assert response.status_code == 500
            assert "Модель не загружена" in response.json()["detail"]
        finally:
            # Восстанавливаем состояние
            main_module.model = old_model


class TestMonitoring:
    """Тесты мониторинга дрейфа данных."""

    def test_drift_monitor_creation(self, tmp_path):
        """Тест создания монитора дрейфа."""
        df = pd.DataFrame(
            {
                "Age": [25, 30, 35, 40, 45],
                "ServicesOpted": [1, 2, 3, 4, 5],
                "Target": [0, 1, 0, 1, 0],
            }
        )

        monitor = CustomerTravelDriftMonitor(
            reference_data=df, feature_columns=["Age", "ServicesOpted"]
        )

        # Добавляем текущие данные
        monitor.update_current_data(df)

        # Проверяем сводку
        summary = monitor.calculate_drift_metrics()
        assert "features" in summary

    def test_drift_no_critical_drift(self):
        """Тест отсутствия дрейфа на идентичных данных."""
        df = pd.DataFrame(
            {
                "Age": [25, 30, 35, 40, 45],
                "ServicesOpted": [1, 2, 3, 4, 5],
                "Target": [0, 1, 0, 1, 0],
            }
        )

        monitor = CustomerTravelDriftMonitor(
            reference_data=df, feature_columns=["Age", "ServicesOpted"]
        )
        monitor.update_current_data(df)

        assert monitor.check_drift_threshold() is False


class TestSystemMonitor:
    """Тесты мониторинга инфраструктуры."""

    def test_system_metrics_format(self):
        """Тест формата системных метрик."""
        metrics = get_system_metrics()
        assert "timestamp" in metrics
        assert "platform" in metrics
        assert "python_version" in metrics

    def test_system_report_generation(self, tmp_path):
        """Тест генерации HTML-отчета."""
        output_dir = tmp_path / "reports"
        html_path = generate_system_report(str(output_dir))

        assert Path(html_path).exists()
        assert (output_dir / "system_metrics.json").exists()


class TestMLflowIntegration:
    """Тесты интеграции с MLflow."""

    def test_mlflow_logging(self, tmp_path):
        """Тест логирования в MLflow."""
        # Настройка MLflow на SQLite базу данных (кроссплатформенное решение)
        db_path = tmp_path / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Test Experiment")

        with mlflow.start_run():
            # Обучение модели
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            X_train = np.random.rand(20, 5)
            y_train = np.random.randint(0, 2, 20)
            model.fit(X_train, y_train)

            # Метрики
            metrics = {"accuracy": 0.85, "f1_score": 0.82, "roc_auc": 0.90}

            # Логирование
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            mlflow.sklearn.log_model(model, "model")

        # Проверка, что данные записаны
        assert db_path.exists()
