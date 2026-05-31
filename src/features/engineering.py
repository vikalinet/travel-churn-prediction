"""
Единый модуль feature engineering для обучения и инференса.
Гарантирует воспроизводимость полиномиальных признаков.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Класс для создания новых признаков.

    Сохраняет обученный PolynomialFeatures для консистентности
    между обучением и инференсом.
    """

    def __init__(self, poly_degree: int = 2, interaction_only: bool = True):
        self.poly_degree = poly_degree
        self.interaction_only = interaction_only
        self.poly: Optional[PolynomialFeatures] = None
        self.poly_feature_names: Optional[List[str]] = None
        self._numeric_cols: Optional[List[str]] = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Подбор параметров и применение трансформации."""
        X_new = self.create_base_features(X.copy())

        numeric_cols = X_new.select_dtypes(include=[np.number]).columns.tolist()
        self._numeric_cols = numeric_cols

        if len(numeric_cols) >= 2:
            self.poly = PolynomialFeatures(
                degree=self.poly_degree,
                interaction_only=self.interaction_only,
                include_bias=False,
            )
            poly_features = self.poly.fit_transform(X_new[numeric_cols])
            self.poly_feature_names = [
                f"poly_{i}" for i in range(poly_features.shape[1] - len(numeric_cols))
            ]
            poly_df = pd.DataFrame(
                poly_features[:, len(numeric_cols) :],
                columns=self.poly_feature_names,
                index=X_new.index,
            )
            X_new = pd.concat([X_new, poly_df], axis=1)

        logger.info(f"Feature engineering: {X.shape[1]} → {X_new.shape[1]} признаков")
        return X_new

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Применение трансформации к новым данным."""
        if self.poly is None:
            raise ValueError("FeatureEngineer not fitted. Call fit_transform() first.")

        X_new = self.create_base_features(X.copy())

        # Убедимся, что все числовые колонки из обучения присутствуют
        for col in self._numeric_cols or []:
            if col not in X_new.columns:
                X_new[col] = 0

        numeric_cols = [c for c in self._numeric_cols if c in X_new.columns]
        if numeric_cols and self.poly is not None:
            poly_features = self.poly.transform(X_new[numeric_cols])
            poly_df = pd.DataFrame(
                poly_features[:, len(numeric_cols) :],
                columns=self.poly_feature_names or [],
                index=X_new.index,
            )
            X_new = pd.concat([X_new, poly_df], axis=1)

        return X_new

    @staticmethod
    def create_base_features(X: pd.DataFrame) -> pd.DataFrame:
        """Создание базовых признаков-взаимодействий."""
        if "ServicesOpted" in X.columns and "Age" in X.columns:
            X["services_per_age"] = X["ServicesOpted"] / (X["Age"] + 1)

        if "FrequentFlyer" in X.columns and "BookedHotelOrNot" in X.columns:
            X["flyer_and_hotel"] = X["FrequentFlyer"] * X["BookedHotelOrNot"]

        if "AnnualIncomeClass" in X.columns and "ServicesOpted" in X.columns:
            X["income_x_services"] = X["AnnualIncomeClass"] * X["ServicesOpted"]

        return X
