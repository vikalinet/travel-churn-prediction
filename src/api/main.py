from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

app = FastAPI(
    title="Travel Churn Prediction API",
    description="API для прогнозирования оттока клиентов туристического агентства",
    version="1.0.0",
)


class CustomerInput(BaseModel):
    """Входные данные для предсказания оттока."""

    age: int
    annual_income: float
    flight_count: int
    gender: str
    marital_status: str
    education: str
    occupation: str
    city_tier: str
    number_of_dependents: int
    total_spending: float
    walk_in_count: int
    web_login_count: int
    mobile_app_login_count: int
    last_visit_date_days: int
    complaint_count: int
    is_member: bool


class PredictionResult(BaseModel):
    """Результат предсказания."""

    prediction: int
    probability: float
    customer_data: Dict[str, Any]


@app.get("/")
async def root():
    """Главная страница API."""
    return {
        "message": "Travel Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResult)
async def predict_churn(customer: CustomerInput):
    """
    Предсказание оттока для клиента.

    Принимает данные клиента и возвращает вероятность оттока.
    """
    try:
        # TODO: Загрузить модель и сделать предсказание
        # model = load_model()
        # prediction = model.predict(customer.dict())
        # probability = model.predict_proba(customer.dict())

        # Временная заглушка
        prediction = 0  # 0 - не уйдёт, 1 - уйдёт
        probability = 0.15

        return PredictionResult(
            prediction=prediction,
            probability=probability,
            customer_data=customer.dict(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_churn_batch(customers: List[CustomerInput]):
    """
    Пакетное предсказание оттока для нескольких клиентов.
    """
    try:
        results = []
        for customer in customers:
            # TODO: Реализовать пакетное предсказание
            results.append(
                {"prediction": 0, "probability": 0.15, "customer_data": customer.dict()}
            )
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
