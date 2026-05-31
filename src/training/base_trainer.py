"""
Базовый класс для обучения моделей.
Содержит общую логику загрузки данных и подготовки.
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTrainer:
    """Базовый класс для всех тренажёров моделей."""

    def __init__(self, data_path: str, target_column: str = "Target"):
        self.data_path = data_path
        self.target_column = target_column
        self.models: Dict[str, object] = {}
        self.results: List[Dict] = []

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Загрузка и разделение данных."""
        logger.info(f"Загрузка данных из {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Разделение на признаки и целевую переменную
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        logger.info(f"Размер выборки: {X.shape}")
        logger.info(f"Распределение целевой переменной:\n{y.value_counts()}")

        return X, y

    def prepare_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Подготовка данных для обучения."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(f"Обучающая выборка: {X_train.shape}")
        logger.info(f"Тестовая выборка: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def calculate_metrics(
        self, y_test: pd.Series, y_pred: pd.Series, y_proba: pd.Series
    ) -> Dict[str, float]:
        """Расчёт метрик качества."""
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
        }

    def get_best_model(self) -> Tuple[str, object]:
        """Получение лучшей модели по F1-score."""
        if not self.results:
            raise ValueError("Нет обученных моделей")

        best_result = max(self.results, key=lambda x: x["f1_score"])
        best_model_name = best_result["model_name"]
        best_model = self.models[best_model_name]

        logger.info(f"\nЛучшая модель: {best_model_name}")
        logger.info(f"F1-score: {best_result['f1_score']:.4f}")

        return best_model_name, best_model

    def compare_models(self) -> pd.DataFrame:
        """Сравнение всех моделей."""
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values("f1_score", ascending=False)

        logger.info("\n=== Сравнение моделей ===")
        logger.info(results_df.to_string(index=False))

        return results_df
