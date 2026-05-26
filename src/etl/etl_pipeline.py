"""
Полный ETL пайплайн для данных о путешествиях.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractor:
    """Извлечение данных из различных источников."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_csv(self) -> pd.DataFrame:
        """Загрузка данных из CSV файла."""
        logger.info(f"Загрузка данных из {self.file_path}")
        try:
            df = pd.read_csv(self.file_path)
            logger.info(
                f"Успешно загружено {len(df)} строк и {len(df.columns)} столбцов"
            )
            return df
        except FileNotFoundError:
            logger.error(f"Файл {self.file_path} не найден")
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            raise


class DataTransformer:
    """Трансформация и очистка данных."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def handle_missing_values(self) -> pd.DataFrame:
        """Обработка пропущенных значений."""
        logger.info("Обработка пропущенных значений...")

        # Числовые колонки — заполняем медианой
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                median = self.df[col].median()
                self.df[col] = self.df[col].fillna(median)
                logger.info(
                    f"  Заполнено {self.df[col].isnull().sum()} пропусков в {col} медианой"
                )

        # Категориальные колонки — заполняем модой
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode)
                logger.info(
                    f"  Заполнено {self.df[col].isnull().sum()} пропусков в {col} модой"
                )

        return self.df

    def handle_outliers(
        self, column: str, method: str = "iqr", threshold: float = 1.5
    ) -> pd.DataFrame:
        """Обработка выбросов."""
        logger.info(f"Обработка выбросов в колонке {column}...")

        if column not in self.df.columns:
            logger.warning(f"Колонка {column} не найдена")
            return self.df

        if method == "iqr":
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = (
                (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
            ).sum()
            logger.info(f"  Найдено выбросов: {outliers}")

            # Обрезаем выбросы
            self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)

        return self.df

    def create_features(self) -> pd.DataFrame:
        """Генерация новых признаков."""
        logger.info("Генерация новых признаков...")

        # Активность по путешествиям
        if "flight_count" in self.df.columns:
            self.df["travel_frequency_score"] = pd.qcut(
                self.df["flight_count"],
                q=3,
                labels=["low", "medium", "high"],
                duplicates="drop",
            )

        # Общий уровень использования услуг
        service_cols = ["walk_in_count", "web_login_count", "mobile_app_login_count"]
        available_cols = [col for col in service_cols if col in self.df.columns]
        if available_cols:
            self.df["service_utilization_rate"] = self.df[available_cols].sum(axis=1)

        # Соотношение расходов к доходу
        if "annual_income" in self.df.columns and "total_spending" in self.df.columns:
            self.df["spending_income_ratio"] = self.df["total_spending"] / (
                self.df["annual_income"] + 1
            )

        logger.info("  Создано новых признаков")

        return self.df

    def encode_categorical(self, columns: Optional[list] = None) -> pd.DataFrame:
        """Кодирование категориальных признаков."""
        logger.info("Кодирование категориальных признаков...")

        if columns is None:
            columns = self.df.select_dtypes(include=["object"]).columns.tolist()

        self.label_encoders = {}

        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"  Закодирована колонка {col}")

        return self.df

    def scale_features(self, columns: Optional[list] = None) -> pd.DataFrame:
        """Масштабирование числовых признаков."""
        logger.info("Масштабирование признаков...")

        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Исключаем целевую переменную
            if "Churn" in numeric_cols:
                numeric_cols.remove("Churn")
            columns = numeric_cols

        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])

        logger.info(f"  Масштабировано {len(columns)} признаков")

        return self.df

    def transform(self) -> pd.DataFrame:
        """Полный процесс трансформации."""
        logger.info("Запуск полного процесса трансформации...")

        self.handle_missing_values()
        self.create_features()

        # Обработка выбросов для ключевых колонок
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != "Churn":
                self.handle_outliers(col)

        logger.info("Трансформация завершена")
        return self.df


class DataLoader:
    """Загрузка обработанных данных."""

    def __init__(self, df: pd.DataFrame, output_path: str):
        self.df = df
        self.output_path = output_path

    def save_csv(self) -> None:
        """Сохранение данных в CSV."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        logger.info(f"Данные сохранены в {self.output_path}")

    def save_parquet(self) -> None:
        """Сохранение данных в Parquet."""
        Path(self.output_path.replace(".csv", ".parquet")).parent.mkdir(
            parents=True, exist_ok=True
        )
        self.df.to_parquet(self.output_path.replace(".csv", ".parquet"), index=False)
        logger.info(
            f"Данные сохранены в {self.output_path.replace('.csv', '.parquet')}"
        )


def run_etl(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Запуск полного ETL пайплайна.

    Args:
        input_path: Путь к входному CSV файлу
        output_path: Путь для сохранения обработанных данных

    Returns:
        Обработанный DataFrame
    """
    logger.info("=== Запуск ETL пайплайна ===")

    # Extract
    extractor = DataExtractor(input_path)
    df = extractor.load_csv()

    # Transform
    transformer = DataTransformer(df)
    df_transformed = transformer.transform()

    # Load
    loader = DataLoader(df_transformed, output_path)
    loader.save_csv()
    loader.save_parquet()

    logger.info("=== ETL пайплайн завершён ===")

    return df_transformed


if __name__ == "__main__":
    # Пример использования
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = (
            sys.argv[2] if len(sys.argv) > 2 else "data/processed/processed_data.csv"
        )

        df = run_etl(input_file, output_file)
        print("\nРезультаты:")
        print(f"  Строки: {len(df)}")
        print(f"  Столбцы: {len(df.columns)}")
        print("  Пропуски:", df.isnull().sum().sum())
    else:
        print("Использование: python etl_pipeline.py <input_csv> [output_csv]")
