"""
Полный пайплайн обучения моделей.
Оркестрация всех компонентов обучения.
"""

import logging
import sys
from pathlib import Path

import joblib

from src.training.model_training import ModelTrainer
from src.training.hyperparameter_tuning import HyperparameterTuner
from src.training.model_comparison import ModelComparator
from src.training.mlflow_integration import MLflowIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_full_pipeline(data_path: str, output_path: str = "models/best_model.pkl"):
    """
    Полный пайплайн обучения моделей.

    Args:
        data_path: Путь к обработанным данным
        output_path: Путь для сохранения лучшей модели

    Returns:
        Кортеж (имя лучшей модели, модель, результаты)
    """
    logger.info("=== Запуск полного пайплайна обучения ===")

    # Настройка MLflow
    MLflowIntegration.setup_tracking("sqlite:///mlflow.db")

    # Инициализация тренажёра
    trainer = ModelTrainer(data_path)

    # Загрузка и подготовка данных
    X, y = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)

    # Обучение базовых моделей
    trainer.train_models(X_train, y_train, X_test, y_test)

    # Подбор гиперпараметров для лучших моделей
    tuner = HyperparameterTuner(X_train, y_train, X_test, y_test)

    xgb_model, xgb_results = tuner.tune_xgboost(n_trials=30)
    trainer.models["XGBoost_Tuned"] = xgb_model
    trainer.results.append(xgb_results)

    rf_model, rf_results = tuner.tune_random_forest(n_trials=30)
    trainer.models["RandomForest_Tuned"] = rf_model
    trainer.results.append(rf_results)

    # Сравнение моделей
    results_df = trainer.compare_models()
    ModelComparator.plot_comparison(results_df)
    ModelComparator.save_results(results_df)

    # Получение лучшей модели
    best_model_name, best_model = trainer.get_best_model()

    # Логирование в MLflow
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

    # Сохранение модели
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
        print("Использование: python train_full_pipeline.py <processed_data_csv>")
