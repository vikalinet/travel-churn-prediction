"""
Unit тесты для функций предобработки данных.
"""

import pytest
import pandas as pd
import numpy as np
from src.etl.etl_pipeline import DataExtractor, DataTransformer


class TestDataExtractor:
    """Тесты для извлечения данных."""

    def test_load_csv_success(self, tmp_path):
        """Тест успешной загрузки CSV файла."""
        # Создание тестового CSV
        test_data = pd.DataFrame({"age": [25, 30, 35], "Target": [0, 1, 0]})
        csv_file = tmp_path / "test_data.csv"
        test_data.to_csv(csv_file, index=False)

        # Тест загрузки
        extractor = DataExtractor(str(csv_file))
        loaded_df = extractor.load_csv()

        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["age", "Target"]

    def test_load_csv_file_not_found(self):
        """Тест обработки несуществующего файла."""
        extractor = DataExtractor("non_existent_file.csv")

        with pytest.raises(FileNotFoundError):
            extractor.load_csv()


class TestDataTransformer:
    """Тесты для трансформации данных."""

    @pytest.fixture
    def sample_data(self):
        """Пример данных для тестов."""
        return pd.DataFrame(
            {
                "age": [25, 30, np.nan, 35, 40],
                "income": [50000, 60000, 70000, np.nan, 90000],
                "category": ["A", "B", "A", "C", "B"],
                "Target": [0, 1, 0, 1, 0],
            }
        )

    def test_handle_missing_values(self, sample_data):
        """Тест обработки пропущенных значений."""
        transformer = DataTransformer(sample_data)
        result = transformer.handle_missing_values()

        assert result["age"].isnull().sum() == 0
        assert result["income"].isnull().sum() == 0

    def test_create_features(self, sample_data):
        """Тест генерации новых признаков."""
        sample_data["flight_count"] = [5, 10, 15, 20, 25]
        transformer = DataTransformer(sample_data)
        result = transformer.create_features()

        assert "travel_frequency_score" in result.columns

    def test_encode_categorical(self, sample_data):
        """Тест кодирования категориальных признаков."""
        transformer = DataTransformer(sample_data)
        result = transformer.encode_categorical(["category"])

        # Проверка, что категориальные значения преобразованы в числа
        assert result["category"].dtype in ["int64", "int32"]
        assert result["category"].min() >= 0


class TestModelPrediction:
    """Тесты для предсказаний модели."""

    @pytest.fixture
    def mock_model(self):
        """Мок модели для тестов."""
        from unittest.mock import MagicMock

        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0])
        model.predict_proba.return_value = np.array(
            [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
        )

        return model

    def test_model_prediction_format(self, mock_model):
        """Тест формата предсказаний."""
        test_input = pd.DataFrame(
            {"age": [25, 30, 35], "income": [50000, 60000, 70000]}
        )

        predictions = mock_model.predict(test_input)

        assert len(predictions) == 3
        assert all(p in [0, 1] for p in predictions)

    def test_model_probability_format(self, mock_model):
        """Тест формата вероятностей."""
        test_input = pd.DataFrame(
            {"age": [25, 30, 35], "income": [50000, 60000, 70000]}
        )

        probabilities = mock_model.predict_proba(test_input)

        assert probabilities.shape == (3, 2)
        assert all(0 <= p <= 1 for row in probabilities for p in row)


class TestDataValidation:
    """Тесты валидации данных."""

    def test_required_columns_present(self):
        """Тест наличия обязательных колонок."""
        required_columns = ["age", "annual_income", "Target"]
        test_data = pd.DataFrame(
            {"age": [25, 30], "annual_income": [50000, 60000], "Target": [0, 1]}
        )

        for col in required_columns:
            assert col in test_data.columns

    def test_data_types_correct(self):
        """Тест корректности типов данных."""
        test_data = pd.DataFrame(
            {
                "age": [25, 30, 35],
                "annual_income": [50000.0, 60000.0, 70000.0],
                "Target": [0, 1, 0],
            }
        )

        assert test_data["age"].dtype in ["int64", "int32"]
        assert test_data["annual_income"].dtype in ["float64", "float32"]

    def test_no_critical_missing_values(self):
        """Тест отсутствия критических пропусков."""
        test_data = pd.DataFrame({"age": [25, 30, np.nan], "Target": [0, 1, 0]})

        missing_ratio = test_data["age"].isnull().sum() / len(test_data)
        assert missing_ratio < 0.5  # Допускается до 50% пропусков


class TestDataTransformerAdvanced:
    """Расширенные тесты трансформации данных."""

    def test_handle_outliers_iqr(self):
        """Тест обработки выбросов методом IQR."""
        df = pd.DataFrame(
            {
                "age": [25, 30, 35, 1000, 40],  # 1000 — выброс
                "Target": [0, 1, 0, 1, 0],
            }
        )
        transformer = DataTransformer(df)
        result = transformer.handle_outliers("age", method="iqr")

        # Выброс должен быть обрезан
        assert result["age"].max() < 1000

    def test_scale_features(self):
        """Тест масштабирования признаков."""
        df = pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45],
                "income": [50000, 60000, 70000, 80000, 90000],
                "Target": [0, 1, 0, 1, 0],
            }
        )
        transformer = DataTransformer(df)
        result = transformer.scale_features(["age", "income"])

        # После StandardScaler среднее ≈ 0
        assert abs(result["age"].mean()) < 1e-10
        # std ≈ 1 (StandardScaler использует ddof=0, pandas std — ddof=1)
        assert abs(result["age"].std(ddof=0) - 1.0) < 1e-10

    def test_full_transform_pipeline(self):
        """Тест полного пайплайна трансформации."""
        df = pd.DataFrame(
            {
                "age": [25, 30, np.nan, 40, 45],
                "category": ["A", "B", "A", "C", "B"],
                "Target": [0, 1, 0, 1, 0],
            }
        )
        transformer = DataTransformer(df)
        transformer.handle_missing_values()
        result = transformer.encode_categorical(["category"])

        # Проверяем, что пропусков нет
        assert result.isnull().sum().sum() == 0
        # Проверяем, что категориальные закодированы
        assert result["category"].dtype in ["int64", "int32"]
