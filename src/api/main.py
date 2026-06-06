from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, List
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import uvicorn

from src.api.config import API_V1_PREFIX
from src.api.monitoring_router import router as monitoring_router
from src.api.drift_router import router as drift_router, _analyze_drift
from src.api.preprocessing import DataPreprocessor, preprocess_single_customer
from src.features.engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerInput(BaseModel):
    """Входные данные для предсказания оттока."""

    age: int
    frequent_flyer: str  # Yes/No
    annual_income_class: str  # Low/Middle/High Income
    services_opted: int
    account_synced_to_social_media: str  # Yes/No
    booked_hotel_or_not: str  # Yes/No


class PredictionResult(BaseModel):
    """Результат предсказания."""

    prediction: int
    probability: float
    risk_level: str
    customer_data: Dict[str, Any]
    metrics: Dict[str, float]  # Добавлены метрики модели


# Глобальные переменные
model = None  # Модель (или dict с package)
preprocessor = None  # Единый preprocessor для обучения и инференса
model_metrics = None  # Метрики модели
model_threshold = 0.5  # Порог классификации
model_package = None  # Полный package модели (для improved моделей)


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Применение feature engineering через сохранённый FeatureEngineer."""
    if model_package is not None and "feature_engineer" in model_package:
        feature_engineer: FeatureEngineer = model_package["feature_engineer"]
        return feature_engineer.transform(df)

    # Fallback: базовые признаки без полиномиальных
    return FeatureEngineer.create_base_features(df.copy())


def load_model():
    """Загрузка модели и preprocessor."""
    global model, preprocessor, model_metrics, model_threshold, model_package

    if model is not None:
        return model

    model_paths = [
        "models/best_model.pkl",
        "models/GradientBoosting_model.pkl",
        "models/model.pkl",
    ]

    for path in model_paths:
        if Path(path).exists():
            try:
                loaded = joblib.load(path)

                # Проверяем, это package (dict) или просто модель
                if isinstance(loaded, dict) and "model" in loaded:
                    model_package = loaded
                    model = loaded["model"]
                    model_threshold = loaded.get("threshold", 0.5)
                    logger.info(f"Модель загружена из {path} (package format)")
                    logger.info(f"Порог классификации: {model_threshold:.3f}")
                else:
                    model = loaded
                    model_threshold = 0.5
                    logger.info(f"Модель загружена из {path}")

                # Загрузка preprocessor (если есть)
                preprocessor_path = Path("models/preprocessor.json")
                if preprocessor_path.exists():
                    preprocessor = DataPreprocessor()
                    preprocessor.load(str(preprocessor_path))
                    logger.info("Preprocessor загружен")
                else:
                    logger.warning(
                        "Preprocessor не найден, используется дефолтный маппинг"
                    )

                # Загрузка метрик из model_package (если есть)
                if isinstance(loaded, dict) and "metrics" in loaded:
                    model_metrics = loaded["metrics"]
                else:
                    model_metrics = {}

                return model
            except Exception as e:
                logger.error(f"Ошибка загрузки модели из {path}: {e}")

    logger.warning("Модель не найдена! Будет использована заглушка.")
    return None


def preprocess_input(customer: CustomerInput) -> pd.DataFrame:
    """
    Предобработка входных данных.

    Использует единый preprocessor, если он загружен,
    иначе fallback на дефолтный маппинг.
    """
    customer_dict = customer.model_dump()

    # Если есть preprocessor - используем его
    if preprocessor is not None:
        df = pd.DataFrame([customer_dict])
        df = df.rename(
            columns={
                "frequent_flyer": "FrequentFlyer",
                "annual_income_class": "AnnualIncomeClass",
                "account_synced_to_social_media": "AccountSyncedToSocialMedia",
                "booked_hotel_or_not": "BookedHotelOrNot",
                "age": "Age",
                "services_opted": "ServicesOpted",
            }
        )
        df = preprocessor.transform(df)
    else:
        # Fallback на дефолтный маппинг
        df = preprocess_single_customer(customer_dict)

    # Feature engineering (для improved моделей)
    if model_package is not None and "feature_names" in model_package:
        df = _apply_feature_engineering(df)
        # Убедимся, что колонки совпадают с обучением
        expected_cols = model_package["feature_names"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

    return df


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекстный менеджер для управления жизненным циклом приложения."""
    # Startup: загрузка модели
    logger.info("Загрузка модели...")
    load_model()

    # Startup: автоматический анализ дрейфа (если данные есть)
    try:
        logger.info("Автоматический анализ дрейфа данных...")
        _analyze_drift()
        logger.info("Анализ дрейфа завершён успешно")
    except FileNotFoundError:
        logger.warning("Датасет не найден — пропускаем автоматический анализ дрейфа")
    except Exception as e:
        logger.error(f"Ошибка при автоматическом анализе дрейфа: {e}")

    yield
    # Shutdown: можно добавить очистку ресурсов при необходимости
    logger.info("Закрытие приложения...")


# API router v1 — все ML-эндпоинты с префиксом /api/v1
api_router = APIRouter(prefix=API_V1_PREFIX)

app = FastAPI(
    title="Travel Churn Prediction API",
    description="API для прогнозирования оттока клиентов туристического агентства",
    version="1.0.0",
    lifespan=lifespan,
)

# Подключение роутеров
app.include_router(api_router)
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(drift_router, prefix="/api/v1")


# HTML страницы мониторинга и дрейфа без префикса
@app.get("/monitoring", response_class=HTMLResponse, include_in_schema=False)
async def monitoring_page(request: Request):
    templates = Jinja2Templates(directory="templates")
    context = {
        "request": request,
        "experiments_count": 0,
        "experiments": [],
        "models_registered": 0,
        "registry": [],
        "drift": {"drift_detected": False, "affected_columns": [], "timestamp": None},
        "system": {},
        "demo_mode": True,
        "mlflow_ui_url": "#",
    }
    return templates.TemplateResponse("monitoring.html", context)


@app.get("/drift", response_class=HTMLResponse, include_in_schema=False)
async def drift_page(request: Request):
    templates = Jinja2Templates(directory="templates")
    context = {
        "request": request,
        "timestamp": None,
        "total_features": 0,
        "drift_features": 0,
        "reference_size": 0,
        "current_size": 0,
        "p_threshold": 0.05,
        "results": [],
        "message": "Анализ дрейфа ещё не проводился.",
        "alert": None,
    }
    return templates.TemplateResponse("drift_dashboard.html", context)


# Подключение статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Главная страница с UI для предсказания."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "model_loaded": model is not None}
    )


@api_router.get("/health")
async def health_check():
    """Проверка здоровья сервиса."""
    return {"status": "healthy", "model_loaded": model is not None}


@api_router.post("/predict", response_model=PredictionResult)
async def predict_churn(customer: CustomerInput):
    """
    Предсказание оттока для клиента.

    Принимает данные клиента и возвращает вероятность оттока.
    Использует оптимальный порог классификации (threshold tuning).
    В данном датасете Target=1 означает churn (клиент ушёл).
    """
    if model is None:
        raise HTTPException(
            status_code=500, detail="Модель не загружена! Проверьте путь к модели."
        )

    try:
        # Предобработка
        df_processed = preprocess_input(customer)

        # Предсказание вероятности
        probability = float(model.predict_proba(df_processed)[0][1])

        # Классификация с учётом оптимального порога
        prediction = 1 if probability >= model_threshold else 0

        # Определение уровня риска
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return PredictionResult(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            customer_data=customer.model_dump(),
            metrics=model_metrics if model_metrics else {},
        )
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/predict_batch")
async def predict_churn_batch(customers: List[CustomerInput]):
    """
    Пакетное предсказание оттока для нескольких клиентов.
    Использует оптимальный порог классификации.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена!")

    try:
        results = []
        for customer in customers:
            df_processed = preprocess_input(customer)
            probability = float(model.predict_proba(df_processed)[0][1])
            prediction = 1 if probability >= model_threshold else 0

            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"

            results.append(
                {
                    "prediction": int(prediction),
                    "probability": probability,
                    "risk_level": risk_level,
                    "customer_data": customer.model_dump(),
                }
            )

        return {"predictions": results}
    except Exception as e:
        logger.error(f"Ошибка при пакетном предсказании: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/models")
async def get_model_info():
    """Информация о загруженной модели."""
    return {
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "threshold": model_threshold,
        "metrics": model_metrics if model_metrics else {},
    }


@app.get("/test", response_class=HTMLResponse)
async def test_ui(request: Request):
    """Страница тестирования UI с готовыми сценариями."""
    return templates.TemplateResponse("test_ui.html", {"request": request})


@api_router.get("/test-data")
async def get_test_data():
    """Получение тестовых данных для UI тестирования."""
    test_data_path = Path("data/test_scenarios/test_data.json")
    if test_data_path.exists():
        with open(test_data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Fallback - возвращаем встроенные данные
        return {
            "positive_scenarios": [],
            "negative_scenarios": [],
            "drift_scenarios": [],
            "edge_scenarios": [],
        }


@api_router.get("/test-data/{scenario_id}")
async def get_scenario(scenario_id: str):
    """Получение конкретного тестового сценария."""
    test_data_path = Path("data/test_scenarios/test_data.json")
    if test_data_path.exists():
        with open(test_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Поиск сценария
        for category in [
            "positive_scenarios",
            "negative_scenarios",
            "drift_scenarios",
            "edge_scenarios",
        ]:
            for scenario in data.get(category, []):
                if scenario["id"] == scenario_id:
                    return scenario

        raise HTTPException(status_code=404, detail=f"Сценарий {scenario_id} не найден")
    else:
        raise HTTPException(status_code=404, detail="Тестовые данные не найдены")


# Подключение API router v1 — после всех определений маршрутов
app.include_router(api_router)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
