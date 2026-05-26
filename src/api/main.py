from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from contextlib import asynccontextmanager
import uvicorn
import joblib
import pandas as pd
from pathlib import Path
import logging

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


# Глобальная переменная для модели
model = None
model_mapping = None


def load_model():
    """Загрузка модели."""
    global model, model_mapping

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
    yield
    # Shutdown: можно добавить очистку ресурсов при необходимости
    logger.info("Закрытие приложения...")


app = FastAPI(
    title="Travel Churn Prediction API",
    description="API для прогнозирования оттока клиентов туристического агентства",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Главная страница API."""
    return {
        "message": "Travel Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "model_loaded": model is not None,
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResult)
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
        )
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
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


@app.get("/models")
async def get_model_info():
    """Информация о загруженной модели."""
    return {
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
