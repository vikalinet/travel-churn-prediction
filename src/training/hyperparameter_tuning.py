"""
Подбор гиперпараметров с помощью Optuna.
"""

import logging
from typing import Dict, Tuple

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Подбор гиперпараметров для моделей через Optuna."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def tune_xgboost(self, n_trials: int = 30) -> Tuple[XGBClassifier, Dict]:
        """
        Подбор гиперпараметров для XGBoost.

        Args:
            n_trials: Количество trials в Optuna

        Returns:
            Кортеж (обученная модель, лучшие параметры)
        """
        logger.info(
            f"Подбор гиперпараметров для XGBoost (Optuna, {n_trials} trials)..."
        )

        # Отключаем логи Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "use_label_encoder": False,
                "eval_metric": "logloss",
            }
            model = XGBClassifier(**params)

            # Кросс-валидация с F1-score
            scores = cross_val_score(
                model, self.X_train, self.y_train, cv=3, scoring="f1", n_jobs=-1
            )
            return scores.mean()

        # Создание study
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Лучшие параметры: {study.best_params}")
        logger.info(f"Лучший F1-score (CV): {study.best_value:.4f}")

        # Обучение финальной модели с лучшими параметрами
        best_model = XGBClassifier(
            **study.best_params,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        best_model.fit(self.X_train, self.y_train)

        # Оценка на тестовой выборке
        y_pred = best_model.predict(self.X_test)
        y_proba = best_model.predict_proba(self.X_test)[:, 1]

        tuned_metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1_score": f1_score(self.y_test, y_pred),
            "roc_auc": roc_auc_score(self.y_test, y_proba),
            "precision": precision_score(self.y_test, y_pred, zero_division=0),
            "recall": recall_score(self.y_test, y_pred, zero_division=0),
        }

        logger.info("Тестовые метрики после настройки:")
        for metric, value in tuned_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return best_model, {
            "model_name": "XGBoost_Tuned",
            "best_params": study.best_params,
            **tuned_metrics,
        }

    def tune_random_forest(
        self, n_trials: int = 30
    ) -> Tuple[RandomForestClassifier, Dict]:
        """
        Подбор гиперпараметров для RandomForest.

        Args:
            n_trials: Количество trials в Optuna

        Returns:
            Кортеж (обученная модель, лучшие параметры)
        """
        logger.info(
            f"Подбор гиперпараметров для RandomForest (Optuna, {n_trials} trials)..."
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42,
            }
            model = RandomForestClassifier(**params)

            scores = cross_val_score(
                model, self.X_train, self.y_train, cv=3, scoring="f1", n_jobs=-1
            )
            return scores.mean()

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Лучшие параметры: {study.best_params}")
        logger.info(f"Лучший F1-score (CV): {study.best_value:.4f}")

        best_model = RandomForestClassifier(**study.best_params, random_state=42)
        best_model.fit(self.X_train, self.y_train)

        y_pred = best_model.predict(self.X_test)
        y_proba = best_model.predict_proba(self.X_test)[:, 1]

        tuned_metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1_score": f1_score(self.y_test, y_pred),
            "roc_auc": roc_auc_score(self.y_test, y_proba),
            "precision": precision_score(self.y_test, y_pred, zero_division=0),
            "recall": recall_score(self.y_test, y_pred, zero_division=0),
        }

        logger.info("Тестовые метрики после настройки:")
        for metric, value in tuned_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return best_model, {
            "model_name": "RandomForest_Tuned",
            "best_params": study.best_params,
            **tuned_metrics,
        }
