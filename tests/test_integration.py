"""
Интеграционные тесты для всего пайплайна.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib


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
                "Churn": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            }
        )

        # Сохранение в временный файл
        csv_path = tmp_path / "sample_data.csv"
        data.to_csv(csv_path, index=False)

        return str(csv_path)

    def test_etl_pipeline(self, sample_dataset, tmp_path):
        """Тест ETL пайплайна."""
        from src.etl.etl_pipeline import run_etl

        output_path = tmp_path / "processed_data.csv"

        # Запуск ETL
        df_processed = run_etl(sample_dataset, str(output_path))

        # Проверки
        assert df_processed is not None
        assert len(df_processed) > 0
        assert output_path.exists()
        assert "Churn" in df_processed.columns

    def test_model_training(self, sample_dataset, tmp_path):
        """Тест обучения модели."""
        from src.training.model_training import ModelTrainer

        processed_path = tmp_path / "processed_data.csv"
        # Упрощённая версия для быстрого теста
        df = pd.read_csv(sample_dataset)

        # Удаление строк с пропусками для теста
        df = df.dropna()
        df.to_csv(processed_path, index=False)

        # Инициализация и обучение
        trainer = ModelTrainer(str(processed_path))
        X, y = trainer.load_data()
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)

        # Обучение одной модели для быстрого теста
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)

        # Проверка предсказаний
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_end_to_end_prediction(self, sample_dataset, tmp_path):
        """Тест сквозного предсказания."""
        from src.etl.etl_pipeline import DataTransformer
        from sklearn.ensemble import RandomForestClassifier

        # 1. Загрузка данных
        df = pd.read_csv(sample_dataset)

        # 2. Предобработка
        transformer = DataTransformer(df.drop(columns=["Churn"]))
        df_transformed = transformer.transform()

        # 3. Подготовка для обучения
        y = df["Churn"].iloc[: len(df_transformed)]
        X = df_transformed

        # 4. Обучение простой модели
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        # 5. Предсказание для новых данных
        new_customer = X.iloc[:1]
        prediction = model.predict(new_customer)
        probability = model.predict_proba(new_customer)

        assert prediction[0] in [0, 1]
        assert 0 <= probability[0][1] <= 1


class TestAPIIntegration:
    """Интеграционные тесты API."""

    def test_api_health_check(self):
        """Тест проверки здоровья API."""
        from src.api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_api_root(self):
        """Тест главной страницы API."""
        from src.api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        assert "Travel Churn Prediction API" in response.json()["message"]

    def test_api_predict_endpoint(self):
        """Тест эндпоинта предсказания."""
        from src.api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Тестовые данные
        test_data = {
            "age": 30,
            "annual_income": 60000.0,
            "flight_count": 5,
            "gender": "M",
            "marital_status": "Married",
            "education": "Bachelor",
            "occupation": "Employed",
            "city_tier": "Tier1",
            "number_of_dependents": 2,
            "total_spending": 15000.0,
            "walk_in_count": 3,
            "web_login_count": 20,
            "mobile_app_login_count": 45,
            "last_visit_date_days": 30,
            "complaint_count": 0,
            "is_member": True,
        }

        response = client.post("/predict", json=test_data)

        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result
        assert "probability" in result
        assert result["prediction"] in [0, 1]
        assert 0 <= result["probability"] <= 1


class TestModelPersistence:
    """Тесты сохранения и загрузки моделей."""

    def test_save_and_load_model(self, tmp_path):
        """Тест сохранения и загрузки модели."""
        from sklearn.ensemble import RandomForestClassifier

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


class TestMLflowIntegration:
    """Тесты интеграции с MLflow."""

    def test_mlflow_logging(self, tmp_path):
        """Тест логирования в MLflow."""
        import mlflow
        from sklearn.ensemble import RandomForestClassifier

        # Настройка MLflow на временную директорию
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
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
        assert (tmp_path / "mlruns").exists()
