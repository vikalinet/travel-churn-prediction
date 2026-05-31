from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any, List
from contextlib import asynccontextmanager
import os
import uvicorn
import joblib
import pandas as pd
from pathlib import Path
import logging
import json

from src.api.config import API_V1_PREFIX
from src.api.monitoring_router import router as monitoring_router
from src.api.drift_router import router as drift_router, _analyze_drift

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


# Глобальная переменная для модели
model = None
model_mapping = None
model_metrics = None  # Метрики модели (precision, recall и др.)


def load_model():
    """Загрузка модели."""
    global model, model_mapping, model_metrics

    if model is not None:
        return model

    # Поиск модели
    model_paths = [
        "models/best_model.pkl",
        "models/GradientBoosting_model.pkl",
        "models/model.pkl",
    ]

    for path in model_paths:
        if Path(path).exists():
            try:
                model = joblib.load(path)
                logger.info(f"Модель загружена из {path}")

                # Маппинг для кодирования признаков
                model_mapping = {
                    "FrequentFlyer": {"Yes": 1, "No": 0},
                    "AnnualIncomeClass": {
                        "Low Income": 0,
                        "Middle Income": 1,
                        "High Income": 2,
                    },
                    "AccountSyncedToSocialMedia": {"Yes": 1, "No": 0},
                    "BookedHotelOrNot": {"Yes": 1, "No": 0},
                }

                # Метрики модели (сохранены при обучении)
                model_metrics = {
                    "accuracy": 0.911,
                    "precision": 0.868,
                    "recall": 0.733,
                    "f1_score": 0.795,
                    "roc_auc": 0.975,
                }

                return model
            except Exception as e:
                logger.error(f"Ошибка загрузки модели из {path}: {e}")

    logger.warning("Модель не найдена! Будет использована заглушка.")
    return None


def preprocess_input(customer: CustomerInput) -> pd.DataFrame:
    """Предобработка входных данных."""
    # Создание DataFrame
    data = {
        "Age": [customer.age],
        "FrequentFlyer": [customer.frequent_flyer],
        "AnnualIncomeClass": [customer.annual_income_class],
        "ServicesOpted": [customer.services_opted],
        "AccountSyncedToSocialMedia": [customer.account_synced_to_social_media],
        "BookedHotelOrNot": [customer.booked_hotel_or_not],
    }

    df = pd.DataFrame(data)

    # Кодирование
    if model_mapping:
        df["FrequentFlyer"] = (
            df["FrequentFlyer"]
            .map(model_mapping["FrequentFlyer"])
            .fillna(0)
            .astype(int)
        )

        df["AnnualIncomeClass"] = (
            df["AnnualIncomeClass"]
            .map(model_mapping["AnnualIncomeClass"])
            .fillna(0)
            .astype(int)
        )

        df["AccountSyncedToSocialMedia"] = (
            df["AccountSyncedToSocialMedia"]
            .map(model_mapping["AccountSyncedToSocialMedia"])
            .fillna(0)
            .astype(int)
        )

        df["BookedHotelOrNot"] = (
            df["BookedHotelOrNot"]
            .map(model_mapping["BookedHotelOrNot"])
            .fillna(0)
            .astype(int)
        )

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
    docs_url=None,  # отключаем встроенный Swagger (используем кастомный /docs)
    redoc_url=None,  # отключаем встроенный ReDoc
)

# Подключение роутеров
app.include_router(api_router)
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(drift_router, prefix="/api/v1")


# HTML страницы мониторинга и дрейфа без префикса
@app.get("/monitoring", response_class=HTMLResponse, include_in_schema=False)
async def monitoring_page(request: Request):
    templates = Jinja2Templates(directory="templates")
    data = {
        "experiments_count": 0,
        "models_registered": 0,
        "drift": {"drift_detected": False},
        "system": {},
        "demo_mode": True,
        "mlflow_ui_url": "#",
    }
    return templates.TemplateResponse("monitoring.html", {"request": request, **data})


@app.get("/drift", response_class=HTMLResponse, include_in_schema=False)
async def drift_page(request: Request):
    templates = Jinja2Templates(directory="templates")
    data = {
        "timestamp": None,
        "total_features": 0,
        "drift_features": 0,
        "results": [],
        "message": "Анализ дрейфа ещё не проводился.",
    }
    return templates.TemplateResponse(
        "drift_dashboard.html", {"request": request, "data": data}
    )


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
    """
    if model is None:
        raise HTTPException(
            status_code=500, detail="Модель не загружена! Проверьте путь к модели."
        )

    try:
        # Предобработка
        df_processed = preprocess_input(customer)

        # Предсказание
        prediction = model.predict(df_processed)[0]
        probability = model.predict_proba(df_processed)[0][1]

        # Определение уровня риска
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return PredictionResult(
            prediction=prediction,
            probability=float(probability),
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
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена!")

    try:
        results = []
        for customer in customers:
            df_processed = preprocess_input(customer)
            prediction = model.predict(df_processed)[0]
            probability = model.predict_proba(df_processed)[0][1]

            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"

            results.append(
                {
                    "prediction": int(prediction),
                    "probability": float(probability),
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
    }


@app.get("/docs", response_class=HTMLResponse)
async def get_docs(request: Request):
    """Swagger UI документация API."""
    return templates.TemplateResponse("api_docs.html", {"request": request})


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
