"""
ETL пайплайн для датасета Customer Travel Churn.
Адаптирован под фактическую структуру данных.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerTravelETL:
    """ETL для датасета Customer Travel."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Загрузка данных."""
        logger.info(f"Загрузка данных из {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка данных."""
        logger.info("Очистка данных...")

        df_clean = df.copy()

        # Проверка пропусков
        missing = df_clean.isnull().sum()
        if missing.sum() > 0:
            logger.info(f"Найдено пропусков: {missing.to_dict()}")
            # Заполнение пропусков
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    if df_clean[col].dtype == "object":
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                    else:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        return df_clean

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование категориальных признаков."""
        logger.info("Кодирование категориальных признаков...")

        df_encoded = df.copy()

        # Категориальные колонки
        categorical_cols = df_encoded.select_dtypes(include=["object"]).columns.tolist()

        for col in categorical_cols:
            unique_vals = df_encoded[col].dropna().unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            df_encoded[col] = df_encoded[col].map(mapping).fillna(0).astype(int)
            logger.info(
                f"  Закодирована колонка {col}: {unique_vals} -> {list(mapping.values())}"
            )

        return df_encoded

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание новых признаков."""
        logger.info("Создание новых признаков...")

        df_features = df.copy()

        # Соотношение частоты перелётов к возрасту
        if "Age" in df_features.columns and "FrequentFlyer" in df_features.columns:
            # Преобразуем Yes/No в числа
            fly_mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
            if df_features["FrequentFlyer"].dtype == "object":
                df_features["FrequentFlyer"] = (
                    df_features["FrequentFlyer"].map(fly_mapping).fillna(0).astype(int)
                )

        # Соотношение услуг к доходу
        if "AnnualIncomeClass" in df_features.columns:
            # Упорядочим уровни дохода
            income_order = {
                "Low Income": 0,
                "low income": 0,
                "Middle Income": 1,
                "middle income": 1,
                "High Income": 2,
                "high income": 2,
            }
            if df_features["AnnualIncomeClass"].dtype == "object":
                df_features["AnnualIncomeClass"] = (
                    df_features["AnnualIncomeClass"]
                    .map(income_order)
                    .fillna(0)
                    .astype(int)
                )

        # ServicesOpted уже числовой
        if (
            "ServicesOpted" in df_features.columns
            and df_features["ServicesOpted"].dtype == "object"
        ):
            df_features["ServicesOpted"] = (
                pd.to_numeric(df_features["ServicesOpted"], errors="coerce")
                .fillna(0)
                .astype(int)
            )

        return df_features

    def prepare_for_training(
        self, df: pd.DataFrame, target_column: str = "Target"
    ) -> tuple:
        """Подготовка данных для обучения."""
        logger.info("Подготовка данных для обучения...")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        logger.info(f"Признаки: {X.shape}, Целевая переменная: {y.shape}")

        return X, y

    def full_pipeline(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """Полный ETL пайплайн."""
        logger.info("=== Запуск полного ETL пайплайна ===")

        # Загрузка
        df = self.load_data()

        # Очистка
        df_clean = self.clean_data(df)

        # Создание признаков
        df_features = self.create_features(df_clean)

        # Кодирование
        df_encoded = self.encode_categorical(df_features)

        # Сохранение
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df_encoded.to_csv(output_path, index=False)
            logger.info(f"Обработанные данные сохранены в {output_path}")

        logger.info("=== ETL пайплайн завершён ===")

        return df_encoded


def main():
    """Запуск ETL."""
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = (
            sys.argv[2] if len(sys.argv) > 2 else "data/processed/processed_data.csv"
        )

        etl = CustomerTravelETL(input_file)
        df_processed = etl.full_pipeline(output_file)

        print(f"\nРезультаты:")
        print(f"  Строки: {len(df_processed)}")
        print(f"  Столбцы: {len(df_processed.columns)}")
        print(f"  Колонки: {df_processed.columns.tolist()}")
    else:
        print("Использование: python customer_travel_etl.py <input_csv> [output_csv]")


if __name__ == "__main__":
    main()
