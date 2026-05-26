"""
Обучение базовых моделей для прогнозирования оттока.
"""

import logging
from typing import Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.training.base_trainer import BaseTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer(BaseTrainer):
    """Класс для обучения базовых моделей ML."""

    def train_models(
        self,
        X_train: Dict,
        y_train: Dict,
        X_test: Dict,
        y_test: Dict,
    ) -> Dict[str, object]:
        """Обучение нескольких базовых моделей."""
        models = {
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
            "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100),
            "KNeighbors": KNeighborsClassifier(),
            "XGBoost": XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric="logloss"
            ),
            "GradientBoosting": GradientBoostingClassifier(
                random_state=42, n_estimators=100
            ),
        }

        logger.info("Начало обучения моделей...")

        for name, model in models.items():
            logger.info(f"\nОбучение модели: {name}")
            model.fit(X_train, y_train)
            self.models[name] = model

            # Предсказания
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Метрики
            metrics = self.calculate_metrics(y_test, y_pred, y_proba)
            self.results.append({"model_name": name, **metrics})

            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1-score: {metrics['f1_score']:.4f}")
            logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")

        return self.models


def train_base_models(data_path: str) -> ModelTrainer:
    """
    Быстрое обучение базовых моделей.

    Args:
        data_path: Путь к обработанным данным

    Returns:
        Обученный тренажёр
    """
    logger.info("=== Запуск обучения базовых моделей ===")

    trainer = ModelTrainer(data_path)
    X, y = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    trainer.train_models(X_train, y_train, X_test, y_test)

    return trainer


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        trainer = train_base_models(data_file)
        results = trainer.compare_models()
        print("\nРезультаты:")
        print(results.to_string(index=False))
    else:
        print("Использование: python model_training.py <processed_data_csv>")
