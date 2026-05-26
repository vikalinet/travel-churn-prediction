"""
Ансамблевые модели для AutoML альтернативы.
VotingClassifier и другие ансамбли.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """Обучение ансамблевых моделей."""

    def __init__(self, data_path: str, target_column: str = "Target"):
        self.data_path = data_path
        self.target_column = target_column
        self.ensemble_model = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Загрузка данных."""
        logger.info(f"Загрузка данных из {self.data_path}")
        df = pd.read_csv(self.data_path)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        logger.info(f"Размер данных: {df.shape}")
        return X, y

    def train_ensemble(
        self, X: pd.DataFrame, y: pd.Series, time_limit: int = 120
    ) -> Tuple[VotingClassifier, Dict]:
        """
        Обучение ансамбля моделей (VotingClassifier).

        Args:
            X: Признаки
            y: Целевая переменная
            time_limit: Лимит времени в секундах

        Returns:
            Кортеж (модель, метрики)
        """
        logger.info("Обучение ансамбля моделей (VotingClassifier)...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Создание ансамбля
        estimators = [
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ]

        ensemble = VotingClassifier(estimators=estimators, voting="soft")

        logger.info("Обучение ансамбля...")
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Предсказания
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)[:, 1]

        # Метрики
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
        }

        logger.info(f"Время обучения: {training_time:.2f} сек")
        logger.info("Метрики ансамбля:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Кросс-валидация
        logger.info("Кросс-валидация (5-fold):")
        cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring="f1")
        logger.info(f"  F1 CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        self.ensemble_model = ensemble

        results = {
            "model_name": "Ensemble (VotingClassifier)",
            "training_time_sec": training_time,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            **metrics,
        }

        return ensemble, results

    def compare_with_custom_models(
        self,
        ensemble_metrics: Dict,
        custom_results: List[Dict] = None,
    ) -> pd.DataFrame:
        """
        Сравнение ансамбля с кастомными моделями.

        Args:
            ensemble_metrics: Метрики ансамбля
            custom_results: Метрики других моделей

        Returns:
            DataFrame с результатами сравнения
        """
        if custom_results is None:
            custom_results = [
                {
                    "model_name": "GradientBoosting",
                    "accuracy": 0.9110,
                    "f1_score": 0.7952,
                    "roc_auc": 0.9747,
                    "precision": 0.8684,
                    "recall": 0.7333,
                },
                {
                    "model_name": "XGBoost_Tuned",
                    "accuracy": 0.8953,
                    "f1_score": 0.7619,
                    "roc_auc": 0.9677,
                    "precision": 0.8205,
                    "recall": 0.7111,
                },
                {
                    "model_name": "RandomForest_Tuned",
                    "accuracy": 0.8848,
                    "f1_score": 0.7381,
                    "roc_auc": 0.9602,
                    "precision": 0.7949,
                    "recall": 0.6889,
                },
            ]

        ensemble_result = {
            "model_name": "Ensemble (VotingClassifier)",
            **ensemble_metrics,
        }

        all_results = custom_results + [ensemble_result]
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("f1_score", ascending=False)

        logger.info("\n=== Сравнение всех моделей ===")
        logger.info(results_df.to_string(index=False))

        return results_df
