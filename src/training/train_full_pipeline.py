"""
Полный пайплайн обучения моделей.
Оркестрация всех компонентов обучения.
"""

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.training.model_training import ModelTrainer
from src.training.hyperparameter_tuning import HyperparameterTuner
from src.training.model_comparison import ModelComparator
from src.training.mlflow_integration import MLflowIntegration
from src.api.preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_full_pipeline(
    data_path: str,
    output_path: str = "models/best_model.pkl",
    preprocessor_path: str = "models/preprocessor.json",
):
    """
    Полный пайплайн обучения моделей.

    Args:
        data_path: Путь к обработанным данным
        output_path: Путь для сохранения лучшей модели
        preprocessor_path: Путь для сохранения preprocessor

    Returns:
        Кортеж (имя лучшей модели, модель, результаты)
    """
    logger.info("=== Запуск полного пайплайна обучения ===")

    # Настройка MLflow
    MLflowIntegration.setup_tracking("sqlite:///mlflow.db")

    # Загрузка данных для обучения preprocessor
    df = pd.read_csv(data_path)
    y = df["Target"]
    X = df.drop(columns=["Target"])

    # Создание и обучение preprocessor
    logger.info("Обучение preprocessor...")
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.fit_transform(X, target_col=None)
    preprocessor.save(preprocessor_path)
    logger.info(f"Preprocessor сохранён: {preprocessor_path}")

    # Разделение на train/test
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # Инициализация тренажёра
    trainer = ModelTrainer("")  # Пустой путь, так как данные уже загружены
    trainer.models = {}
    trainer.results = []

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

    # Сохранение preprocessor (если не сохранён выше)
    preprocessor_path_obj = Path(preprocessor_path)
    if not preprocessor_path_obj.exists():
        preprocessor.save(preprocessor_path)
        logger.info(f"Preprocessor сохранён: {preprocessor_path}")

    logger.info("=== Пайплайн обучения завершён ===")

    return best_model_name, best_model, results_df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        best_model_name, model_obj, results = train_full_pipeline(data_file)
        print(f"\nЛучшая модель: {best_model_name}")
    else:
        print("Использование: python train_full_pipeline.py <processed_data_csv>")
