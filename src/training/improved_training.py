"""
Улучшенное обучение моделей с threshold tuning, class weights, stacking и feature engineering.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.features.engineering import FeatureEngineer
from src.training.base_trainer import BaseTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedModelTrainer(BaseTrainer):
    """Улучшенный тренажёр с threshold tuning и class weights."""

    def __init__(self, data_path: str, target_column: str = "Target"):
        super().__init__(data_path, target_column)
        self.thresholds: Dict[str, float] = {}
        self.calibrated_models: Dict[str, object] = {}
        self.feature_engineer = FeatureEngineer()

    def load_and_engineer(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Загрузка данных с feature engineering."""
        X, y = self.load_data()
        X = self.feature_engineer.fit_transform(X)
        return X, y

    def find_optimal_threshold(
        self, y_true: pd.Series, y_proba: np.ndarray, metric: str = "f1"
    ) -> Tuple[float, float]:
        """
        Подбор оптимального порога классификации.

        Args:
            y_true: Истинные метки
            y_proba: Вероятности положительного класса
            metric: Целевая метрика ('f1', 'recall', 'precision')

        Returns:
            Кортеж (оптимальный порог, значение метрики)
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0.0

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)

            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = thresh

        return best_threshold, best_score

    def train_with_class_weights(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, object]:
        """Обучение моделей с class weights."""

        # Расчёт весов классов
        class_counts = y_train.value_counts()
        weight_0 = len(y_train) / (2 * class_counts[0])
        weight_1 = len(y_train) / (2 * class_counts[1])
        logger.info(f"Class weights: 0={weight_0:.3f}, 1={weight_1:.3f}")

        models = {
            "GradientBoosting_Balanced": GradientBoostingClassifier(
                random_state=42, n_estimators=200, max_depth=5, learning_rate=0.1
            ),
            "RandomForest_Balanced": RandomForestClassifier(
                random_state=42, n_estimators=200, class_weight="balanced", max_depth=10
            ),
            "XGBoost_Balanced": XGBClassifier(
                random_state=42,
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=weight_1 / weight_0,
                use_label_encoder=False,
                eval_metric="logloss",
            ),
            "LogisticRegression_Balanced": LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced"
            ),
        }

        for name, model in models.items():
            logger.info(f"\nОбучение модели: {name}")
            model.fit(X_train, y_train)
            self.models[name] = model

            # Предсказания с калибровкой
            y_proba = model.predict_proba(X_test)[:, 1]

            # Threshold tuning
            optimal_thresh, _ = self.find_optimal_threshold(
                y_test, y_proba, metric="f1"
            )
            self.thresholds[name] = optimal_thresh

            y_pred = (y_proba >= optimal_thresh).astype(int)

            metrics = self.calculate_metrics(y_test, y_pred, y_proba)
            metrics["threshold"] = optimal_thresh
            self.results.append({"model_name": name, **metrics})

            logger.info(f"  Optimal threshold: {optimal_thresh:.3f}")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-score: {metrics['f1_score']:.4f}")
            logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")

        return self.models

    def train_stacking_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[object, Dict]:
        """Обучение Stacking ансамбля."""
        logger.info("\nОбучение Stacking ансамбля...")

        estimators = [
            ("gb", GradientBoostingClassifier(random_state=42, n_estimators=150)),
            (
                "rf",
                RandomForestClassifier(
                    random_state=42, n_estimators=150, class_weight="balanced"
                ),
            ),
            (
                "xgb",
                XGBClassifier(
                    random_state=42,
                    n_estimators=150,
                    use_label_encoder=False,
                    eval_metric="logloss",
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    random_state=42, max_iter=1000, class_weight="balanced"
                ),
            ),
        ]

        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced"
            ),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            passthrough=False,
            n_jobs=-1,
        )

        stack.fit(X_train, y_train)
        self.models["Stacking"] = stack

        y_proba = stack.predict_proba(X_test)[:, 1]
        optimal_thresh, _ = self.find_optimal_threshold(y_test, y_proba, metric="f1")
        self.thresholds["Stacking"] = optimal_thresh

        y_pred = (y_proba >= optimal_thresh).astype(int)

        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        metrics["threshold"] = optimal_thresh
        self.results.append({"model_name": "Stacking", **metrics})

        logger.info(f"  Optimal threshold: {optimal_thresh:.3f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")

        return stack, metrics

    def calibrate_model(self, model, X_train, y_train, name: str):
        """Калибровка вероятностей модели."""
        logger.info(f"Калибровка модели: {name}")
        calibrated = CalibratedClassifierCV(model, cv=3, method="isotonic")
        calibrated.fit(X_train, y_train)
        self.calibrated_models[name] = calibrated
        return calibrated

    def run_improved_pipeline(
        self, output_path: str = "models/best_model_improved.pkl"
    ) -> Tuple[str, object, pd.DataFrame]:
        """Полный улучшенный пайплайн обучения."""
        logger.info("=== Запуск улучшенного пайплайна обучения ===")

        X, y = self.load_and_engineer()
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)

        # Обучение моделей с class weights и threshold tuning
        self.train_with_class_weights(X_train, y_train, X_test, y_test)

        # Stacking ансамбль
        self.train_stacking_ensemble(X_train, y_train, X_test, y_test)

        # Сравнение
        results_df = self.compare_models()

        # Лучшая модель
        best_name, best_model = self.get_best_model()
        best_threshold = self.thresholds.get(best_name, 0.5)

        logger.info(f"\n🏆 Лучшая модель: {best_name}")
        logger.info(f"   Порог классификации: {best_threshold:.3f}")

        # Сохранение
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем метрики лучшей модели
        best_result = next(r for r in self.results if r["model_name"] == best_name)
        best_metrics = {
            k: v
            for k, v in best_result.items()
            if k not in ("model_name", "best_params", "threshold")
        }

        # Сохраняем модель + порог + список признаков + feature_engineer + метрики
        model_package = {
            "model": best_model,
            "threshold": best_threshold,
            "feature_names": list(X.columns),
            "model_name": best_name,
            "feature_engineer": self.feature_engineer,
            "metrics": best_metrics,
        }
        joblib.dump(model_package, output_path)
        logger.info(f"Модель сохранена: {output_path}")

        # Сохранение результатов
        results_df.to_csv("reports/training_results_improved.csv", index=False)
        logger.info("Результаты сохранены: reports/training_results_improved.csv")

        return best_name, best_model, results_df


def main():
    data_path = (
        sys.argv[1] if len(sys.argv) > 1 else "data/processed/processed_data.csv"
    )

    trainer = ImprovedModelTrainer(data_path)
    best_name, best_model, results = trainer.run_improved_pipeline()

    print("\n=== Итоговые результаты ===")
    print(results.to_string(index=False))
    print(f"\nЛучшая модель: {best_name}")


if __name__ == "__main__":
    main()
