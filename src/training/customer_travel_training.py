"""
Обучение моделей для датасета Customer Travel Churn.
Переиспользует общую логику из ModelTrainer.
"""

import logging
import sys
from typing import Dict

from sklearn.svm import SVC

from src.training.model_training import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerTravelModelTrainer(ModelTrainer):
    """
    Специализированный тренажёр для Customer Travel датасета.
    Наследуется от ModelTrainer с добавлением SVC модели.
    """

    def train_models(self, X_train, y_train, X_test, y_test) -> Dict[str, object]:
        """Обучение моделей с добавлением SVC."""
        # Базовые модели из родителя
        super().train_models(X_train, y_train, X_test, y_test)

        # Добавляем SVC специфично для Customer Travel
        logger.info("\nОбучение модели: SVC")
        model = SVC(random_state=42, probability=True)
        model.fit(X_train, y_train)
        self.models["SVC"] = model

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        self.results.append({"model_name": "SVC", **metrics})

        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")

        return self.models


def train_full_pipeline(data_path: str, output_path: str = "models/best_model.pkl"):
    """Полный пайплайн обучения для Customer Travel."""
    import joblib
    from pathlib import Path

    import mlflow

    from src.training.hyperparameter_tuning import HyperparameterTuner
    from src.training.model_comparison import ModelComparator
    from src.training.mlflow_integration import MLflowIntegration

    logger.info("=== Запуск полного пайплайна обучения (Customer Travel) ===")

    MLflowIntegration.setup_tracking("sqlite:///mlflow.db")
    mlflow.set_experiment("Customer Travel Churn Prediction")

    trainer = CustomerTravelModelTrainer(data_path)
    X, y = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)

    trainer.train_models(X_train, y_train, X_test, y_test)

    tuner = HyperparameterTuner(X_train, y_train, X_test, y_test)

    xgb_model, xgb_results = tuner.tune_xgboost(n_trials=30)
    trainer.models["XGBoost_Tuned"] = xgb_model
    trainer.results.append(xgb_results)

    rf_model, rf_results = tuner.tune_random_forest(n_trials=30)
    trainer.models["RandomForest_Tuned"] = rf_model
    trainer.results.append(rf_results)

    results_df = trainer.compare_models()
    ModelComparator.plot_comparison(results_df)
    ModelComparator.save_results(results_df)

    best_model_name, best_model = trainer.get_best_model()

    best_result = next(r for r in trainer.results if r["model_name"] == best_model_name)
    MLflowIntegration.log_model(
        best_model_name,
        best_model,
        {
            k: v
            for k, v in best_result.items()
            if k not in ["model_name", "best_params"]
        },
        best_result.get("best_params", {}),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_path)
    logger.info(f"Лучшая модель сохранена в {output_path}")

    logger.info("=== Пайплайн обучения завершён ===")

    return best_model_name, best_model, results_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        best_model_name, model_obj, results = train_full_pipeline(data_file)
        print(f"\nЛучшая модель: {best_model_name}")
    else:
        print("Использование: python customer_travel_training.py <processed_data_csv>")
