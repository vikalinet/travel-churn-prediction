"""
Обучение и сравнение моделей для прогнозирования оттока.
"""

import joblib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
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
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Класс для обучения и сравнения моделей."""

    def __init__(self, data_path: str, target_column: str = "Target"):
        self.data_path = data_path
        self.target_column = target_column
        self.models = {}
        self.results = []

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Загрузка и разделение данных."""
        logger.info(f"Загрузка данных из {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Разделение на признаки и целевую переменную
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        logger.info(f"Размер выборки: {X.shape}")
        logger.info(f"Распределение целевой переменной:\n{y.value_counts()}")

        return X, y

    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Подготовка данных для обучения."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Обучающая выборка: {X_train.shape}")
        logger.info(f"Тестовая выборка: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        """Обучение нескольких моделей."""
        models = {
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
            "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100),
            "KNeighbors": KNeighborsClassifier(),
            "XGBoost": XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric="logloss"
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
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
            }

            self.results.append({"model_name": name, **metrics})

            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1-score: {metrics['f1_score']:.4f}")
            logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")

        return self.models

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "XGBoost",
        n_trials: int = 30,
    ) -> Dict:
        """Подбор гиперпараметров с помощью Optuna."""
        logger.info(
            f"Подбор гиперпараметров для {model_name} (Optuna, {n_trials} trials)..."
        )

        # Отключаем логи Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            if model_name == "XGBoost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    ),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.6, 1.0
                    ),
                    "random_state": 42,
                    "use_label_encoder": False,
                    "eval_metric": "logloss",
                }
                model = XGBClassifier(**params)

            elif model_name == "RandomForest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "random_state": 42,
                }
                model = RandomForestClassifier(**params)

            else:
                logger.warning(f"Подбор гиперпараметров не реализован для {model_name}")
                return 0.0

            # Кросс-валидация с F1-score
            scores = cross_val_score(
                model, X_train, y_train, cv=3, scoring="f1", n_jobs=-1
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
        if model_name == "XGBoost":
            best_model = XGBClassifier(
                **study.best_params,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        elif model_name == "RandomForest":
            best_model = RandomForestClassifier(**study.best_params, random_state=42)

        best_model.fit(X_train, y_train)

        # Оценка на тестовой выборке
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        tuned_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

        logger.info("Тестовые метрики после настройки:")
        for metric, value in tuned_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        self.models[f"{model_name}_Tuned"] = best_model
        self.results.append(
            {
                "model_name": f"{model_name}_Tuned",
                "best_params": study.best_params,
                **tuned_metrics,
            }
        )

        return self.models

    def train_automl(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Optional[object]:
        """Обучение AutoML модели (AutoGluon)."""
        try:
            from autogluon.tabular import TabularDataset, TabularPredictor

            logger.info("Обучение AutoGluon модели...")

            # Подготовка данных для AutoGluon
            train_data = X_train.copy()
            train_data[self.target_column] = y_train

            test_data = X_test.copy()
            test_data[self.target_column] = y_test

            train_dataset = TabularDataset(train_data)
            test_dataset = TabularDataset(test_data)

            # Обучение
            predictor = TabularPredictor(
                label=self.target_column,
                eval_metric="f1",
                path="autogluon_models/",
            )

            predictor.fit(
                train_data=train_dataset,
                time_limit=120,  # 2 минуты
                presets="medium_quality",
            )

            # Предсказания
            y_pred = predictor.predict(test_dataset)
            y_proba = predictor.predict_proba(test_dataset).iloc[:, 1]

            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                roc_auc_score,
                precision_score,
                recall_score,
            )

            automl_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
            }

            logger.info("Метрики AutoGluon:")
            for metric, value in automl_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

            self.results.append({"model_name": "AutoGluon", **automl_metrics})

            return predictor

        except ImportError:
            logger.warning("AutoGluon не установлен. Пропускаем AutoML обучение.")
            return None
        except Exception as e:
            logger.error(f"Ошибка при обучении AutoGluon: {e}")
            return None

    def log_to_mlflow(self, model_name: str, model, metrics: Dict, params: Dict = None):
        """Логирование модели и метрик в MLflow."""
        with mlflow.start_run(run_name=model_name):
            # Логирование метрик
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, value)

            # Логирование параметров
            if params:
                for param_name, value in params.items():
                    mlflow.log_param(param_name, value)

            # Логирование модели
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"Модель {model_name} залогирована в MLflow")

    def compare_models(self) -> pd.DataFrame:
        """Сравнение всех моделей."""
        results_df = pd.DataFrame(self.results)

        # Сортировка по F1-score
        results_df = results_df.sort_values("f1_score", ascending=False)

        logger.info("\n=== Сравнение моделей ===")
        logger.info(results_df.to_string(index=False))

        return results_df

    def get_best_model(self) -> Tuple[str, object]:
        """Получение лучшей модели."""
        best_result = max(self.results, key=lambda x: x["f1_score"])
        best_model_name = best_result["model_name"]
        best_model = self.models[best_model_name]

        logger.info(f"\nЛучшая модель: {best_model_name}")
        logger.info(f"F1-score: {best_result['f1_score']:.4f}")

        return best_model_name, best_model

    def plot_comparison(
        self, results_df: pd.DataFrame, save_path: str = "reports/model_comparison.png"
    ):
        """Визуализация сравнения моделей."""
        # Настройка стиля
        plt.style.use("seaborn-v0_8")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ["accuracy", "f1_score", "roc_auc", "precision"]
        metric_names = ["Accuracy", "F1-Score", "ROC AUC", "Precision"]

        for ax, metric, name in zip(axes.flat, metrics, metric_names):
            sns.barplot(
                data=results_df, x="model_name", y=metric, ax=ax, palette="viridis"
            )
            ax.set_title(name, fontsize=14, fontweight="bold")
            ax.set_ylabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_ylim(0, 1)

            # Добавление значений на столбцы
            for i, v in enumerate(results_df[metric]):
                ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"График сохранён в {save_path}")
        plt.close()


def train_full_pipeline(data_path: str, output_path: str = "models/best_model.pkl"):
    """Полный пайплайн обучения моделей."""
    logger.info("=== Запуск полного пайплайна обучения ===")

    # Настройка MLflow трекинга
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Travel Churn Prediction")

    # Инициализация
    trainer = ModelTrainer(data_path)

    # Загрузка данных
    X, y = trainer.load_data()
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)

    # Обучение базовых моделей
    trainer.train_models(X_train, y_train, X_test, y_test)

    # Подбор гиперпараметров для лучших моделей
    trainer.tune_hyperparameters(X_train, y_train, X_test, y_test, model_name="XGBoost")
    trainer.tune_hyperparameters(
        X_train, y_train, X_test, y_test, model_name="RandomForest"
    )

    # AutoML (опционально)
    trainer.train_automl(X_train, y_train, X_test, y_test)

    # Сравнение моделей
    results_df = trainer.compare_models()
    trainer.plot_comparison(results_df)

    # Получение лучшей модели
    best_model_name, best_model = trainer.get_best_model()

    # Логирование лучшей модели
    best_result = next(r for r in trainer.results if r["model_name"] == best_model_name)
    trainer.log_to_mlflow(
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
    import sys

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        best_model, model_obj, results = train_full_pipeline(data_file)
        print(f"\nЛучшая модель: {best_model}")
    else:
        print("Использование: python model_training.py <processed_data_csv>")
