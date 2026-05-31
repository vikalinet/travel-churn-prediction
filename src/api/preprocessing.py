"""
Единый модуль предобработки данных для обучения и инференса.
Сохраняет параметры трансформации для консистентности.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Единый класс предобработки данных.

    Сохраняет и применяет те же трансформации, что и при обучении:
    - Кодирование категориальных признаков (LabelEncoder)
    - Масштабирование числовых признаков (StandardScaler)
    - Обработка выбросов (IQR)
    """

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = None
        self.is_fitted = False
        self.numerical_cols = []
        self.categorical_cols = []

    def fit(
        self, df: pd.DataFrame, target_col: Optional[str] = "Target"
    ) -> "DataPreprocessor":
        """
        Подбор параметров трансформации на обучающих данных.

        Args:
            df: DataFrame с данными
            target_col: имя целевой переменной

        Returns:
            self
        """
        logger.info("Подбор параметров предобработки...")

        df = df.copy()

        # Разделение на числовые и категориальные
        self.categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Исключаем целевую переменную из числовых
        if target_col and target_col in self.numerical_cols:
            self.numerical_cols.remove(target_col)

        # Кодирование категориальных признаков
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"  Закодирован признак: {col}")

        # Масштабирование числовых признаков
        if self.numerical_cols:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])
            logger.info(
                f"  Масштабировано {len(self.numerical_cols)} числовых признаков"
            )

        self.is_fitted = True
        logger.info("Параметры предобработки подобраны")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применение трансформации к новым данным.

        Args:
            df: DataFrame для трансформации

        Returns:
            Трансформированный DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        df = df.copy()

        # Обработка пропущенных значений
        df = self._handle_missing_values(df)

        # Кодирование категориальных признаков
        df = self._encode_categorical(df)

        # Масштабирование числовых признаков
        if self.scaler is not None:
            numeric_to_scale = [c for c in self.numerical_cols if c in df.columns]
            if numeric_to_scale:
                df[numeric_to_scale] = self.scaler.transform(df[numeric_to_scale])

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений."""
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.int64, np.float64]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кодирование категориальных признаков."""
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    logger.warning(
                        f"Новая категория в {col}, используем дефолтное кодирование"
                    )
                    df[col] = 0
        return df

    def fit_transform(
        self, df: pd.DataFrame, target_col: Optional[str] = "Target"
    ) -> pd.DataFrame:
        """Подбор параметров и применение трансформации."""
        return self.fit(df, target_col).transform(df)

    def save(self, path: str):
        """Сохранение параметров предобработки."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "label_encoders": {
                col: {"classes": le.classes_.tolist()}
                for col, le in self.label_encoders.items()
            },
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
        }

        # Сохранение scaler отдельно (pickle)
        import joblib

        with open(output_path.with_suffix(".scaler"), "wb") as f:
            joblib.dump(self.scaler, f)

        # Сохранение остальных параметров (JSON)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Параметры предобработки сохранены: {path}")

    def load(self, path: str):
        """Загрузка параметров предобработки."""
        input_path = Path(path)

        import joblib

        # Загрузка scaler
        scaler_path = input_path.with_suffix(".scaler")
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = joblib.load(f)

        # Загрузка остальных параметров
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.numerical_cols = data["numerical_cols"]
        self.categorical_cols = data["categorical_cols"]

        # Восстановление LabelEncoders
        from sklearn.preprocessing import LabelEncoder

        self.label_encoders = {}
        for col, col_data in data["label_encoders"].items():
            le = LabelEncoder()
            le.classes_ = np.array(col_data["classes"])
            self.label_encoders[col] = le

        self.is_fitted = True
        logger.info(f"Параметры предобработки загружены: {path}")
        return self


# Дефолтный маппинг для быстрого предсказания без preprocessor
DEFAULT_MAPPING = {
    "FrequentFlyer": {"Yes": 1, "No": 0},
    "AnnualIncomeClass": {"Low Income": 0, "Middle Income": 1, "High Income": 2},
    "AccountSyncedToSocialMedia": {"Yes": 1, "No": 0},
    "BookedHotelOrNot": {"Yes": 1, "No": 0},
}


def preprocess_single_customer(
    customer_data: Dict, mapping: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Быстрая предобработка для одного клиента (для совместимости с текущим API).

    Args:
        customer_data: словарь с данными клиента
        mapping: маппинг категорий (по умолчанию DEFAULT_MAPPING)

    Returns:
        DataFrame для предсказания
    """
    if mapping is None:
        mapping = DEFAULT_MAPPING

    data = {
        "Age": customer_data.get("age"),
        "FrequentFlyer": mapping["FrequentFlyer"].get(
            customer_data.get("frequent_flyer", "No"), 0
        ),
        "AnnualIncomeClass": mapping["AnnualIncomeClass"].get(
            customer_data.get("annual_income_class", "Low Income"), 0
        ),
        "ServicesOpted": customer_data.get("services_opted"),
        "AccountSyncedToSocialMedia": mapping["AccountSyncedToSocialMedia"].get(
            customer_data.get("account_synced_to_social_media", "No"), 0
        ),
        "BookedHotelOrNot": mapping["BookedHotelOrNot"].get(
            customer_data.get("booked_hotel_or_not", "No"), 0
        ),
    }

    return pd.DataFrame([data])
