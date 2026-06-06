from fastapi import FastAPI, HTTPException, Request, APIRouter, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
import json
import logging
import os
import signal
import asyncio
import uuid
from pathlib import Path
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor
import time

import joblib
import pandas as pd
import uvicorn

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Structured logging
from pythonjsonlogger import jsonlogger

# Prometheus metrics
from src.api.metrics import (
    HTTP_REQUEST_COUNT,
    HTTP_REQUEST_DURATION,
    HTTP_ERROR_COUNT,
    PREDICTION_COUNT,
    PREDICTION_PROBABILITY,
    PREDICTION_DURATION,
    get_prometheus_metrics,
    get_metrics_content_type,
    update_system_metrics,
)

from src.api.config import settings, API_V1_PREFIX
from src.api.monitoring_router import router as monitoring_router
from src.api.drift_router import router as drift_router
from src.api.preprocessing import DataPreprocessor, FeatureEngineer

# Игнорируем специфичные warnings от сторонних библиотек
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", category=UserWarning, module="requests.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="requests.*")

# ============================================================
# Настройка структурированного логирования
# ============================================================
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, settings.LOG_LEVEL))

# Создаём обработчик для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# JSON формат для продакшена
json_formatter = jsonlogger.JsonFormatter(fmt=settings.LOG_FORMAT)
console_handler.setFormatter(json_formatter)

# Очищаем старые обработчики и добавляем новый
logger.handlers.clear()
logger.addHandler(console_handler)


# Добавляем фильтр для request_id
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, "request_id", "no-request-id")
        return True


logger.addFilter(RequestIdFilter())

# ============================================================
# Rate Limiting
# ============================================================
limiter = Limiter(key_func=get_remote_address)
app_state_limiter = Limiter(key_func=get_remote_address)

# ThreadPoolExecutor для асинхронных ML-операций
executor = ThreadPoolExecutor(max_workers=4)


class CustomerInput(BaseModel):
    """Входные данные для предсказания оттока с валидацией."""

    age: int = Field(..., ge=18, le=100, description="Возраст от 18 до 100")
    frequent_flyer: str = Field(
        ...,
        pattern="^(Yes|No)$",
        description="Yes/No — является ли постоянным клиентом",
    )
    annual_income_class: str = Field(
        ...,
        pattern="^(Low|Middle|High) Income$",
        description="Low Income / Middle Income / High Income",
    )
    services_opted: int = Field(
        ..., ge=0, le=10, description="Количество услуг от 0 до 10"
    )
    account_synced_to_social_media: str = Field(
        ...,
        pattern="^(Yes|No)$",
        description="Yes/No — синхронизирован ли аккаунт с соцсетями",
    )
    booked_hotel_or_not: str = Field(
        ..., pattern="^(Yes|No)$", description="Yes/No — бронировал ли отель"
    )

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        """Валидация возраста."""
        if v < 18 or v > 100:
            raise ValueError("Возраст должен быть от 18 до 100 лет")
        return v

    @field_validator("services_opted")
    @classmethod
    def validate_services_opted(cls, v: int) -> int:
        """Валидация количества услуг."""
        if v < 0 or v > 10:
            raise ValueError("Количество услуг должно быть от 0 до 10")
        return v


class PredictionResult(BaseModel):
    """Результат предсказания."""

    prediction: int
    probability: float
    risk_level: str
    customer_data: Dict[str, Any]
    metrics: Dict[str, float]


class ModelContainer:
    """
    Контейнер для модели и связанных объектов.
    Заменяет глобальные переменные, позволяет Dependency Injection.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_metrics = {}
        self.model_threshold = 0.5
        self.model_package = None
        self._loaded = False

    def is_loaded(self) -> bool:
        """Проверка, что модель загружена."""
        return self._loaded and self.model is not None

    def get_model_type(self) -> Optional[str]:
        """Получение типа модели."""
        return type(self.model).__name__ if self.model else None


# Глобальный экземпляр контейнера (синглтон)
model_container = ModelContainer()


async def get_model_container() -> ModelContainer:
    """Dependency injection для получения контейнера модели."""
    return model_container


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Применение feature engineering через сохранённый FeatureEngineer."""
    if (
        model_container.model_package is not None
        and "feature_engineer" in model_container.model_package
    ):
        feature_engineer: FeatureEngineer = model_container.model_package[
            "feature_engineer"
        ]
        # Проверяем, обучен ли FeatureEngineer (есть ли poly)
        if feature_engineer.poly is not None:
            return feature_engineer.transform(df)
        else:
            # Если не обучен, используем static метод
            return FeatureEngineer.create_base_features(df.copy())

    # Fallback: базовые признаки без полиномиальных
    return FeatureEngineer.create_base_features(df.copy())


def load_model() -> bool:
    """
    Загрузка модели и preprocessor в контейнер.

    Returns:
        True если модель успешно загружена, False иначе.
    """
    # Проверка кэша
    if model_container.is_loaded():
        return True

    # Использование путей из конфигурации
    model_paths = settings.MODEL_PATHS

    for path in model_paths:
        if Path(path).exists():
            try:
                loaded = joblib.load(path)

                # Проверяем, это package (dict) или просто модель
                if isinstance(loaded, dict) and "model" in loaded:
                    model_container.model_package = loaded
                    model_container.model = loaded["model"]
                    model_container.model_threshold = loaded.get("threshold", 0.5)
                    logger.info(f"Модель загружена из {path} (package format)")
                    logger.info(
                        f"Порог классификации: {model_container.model_threshold:.3f}"
                    )
                else:
                    model_container.model = loaded
                    model_container.model_threshold = 0.5
                    logger.info(f"Модель загружена из {path}")

                # Загрузка preprocessor (если есть)
                preprocessor_path = Path("models/preprocessor.json")
                if preprocessor_path.exists():
                    model_container.preprocessor = DataPreprocessor()
                    model_container.preprocessor.load(str(preprocessor_path))
                    logger.info("Preprocessor загружен")
                else:
                    logger.warning(
                        "Preprocessor не найден, используется дефолтный маппинг"
                    )

                # Загрузка метрик из model_package (если есть)
                if isinstance(loaded, dict) and "metrics" in loaded:
                    model_container.model_metrics = loaded["metrics"]
                else:
                    model_container.model_metrics = {}

                model_container._loaded = True
                return True

            except Exception as e:
                logger.error(f"Ошибка загрузки модели из {path}: {e}")

    logger.warning("Модель не найдена!")
    model_container._loaded = False
    return False


def preprocess_input(customer: CustomerInput) -> pd.DataFrame:
    """
    Предобработка входных данных с использованием конфигурации.

    Модель обучалась на данных из processed_data.csv (уже закодированных),
    с применённым feature_engineer.
    """
    # Использование маппингов из конфигурации
    mapping = settings.CATEGORICAL_MAPPINGS

    df = pd.DataFrame(
        [
            {
                "Age": int(customer.age),
                "FrequentFlyer": int(
                    mapping["FrequentFlyer"].get(customer.frequent_flyer, 0)
                ),
                "AnnualIncomeClass": int(
                    mapping["AnnualIncomeClass"].get(customer.annual_income_class, 0)
                ),
                "ServicesOpted": int(customer.services_opted),
                "AccountSyncedToSocialMedia": int(
                    mapping["AccountSyncedToSocialMedia"].get(
                        customer.account_synced_to_social_media, 0
                    )
                ),
                "BookedHotelOrNot": int(
                    mapping["BookedHotelOrNot"].get(customer.booked_hotel_or_not, 0)
                ),
            }
        ]
    )

    # Применяем feature engineering (полиномиальные признаки)
    df = _apply_feature_engineering(df)

    # Убедимся, что все нужные колонки есть
    if (
        model_container.model_package is not None
        and "feature_names" in model_container.model_package
    ):
        expected_cols = model_container.model_package["feature_names"]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

    logger.info(f"Final preprocess_input: {df.shape[1]} признаков")
    return df


shutdown_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: logger.info(
                f"Получен сигнал {s.name}, начинается graceful shutdown..."
            ),
        )

    # Загрузка модели при старте
    logger.info("Загрузка модели...")
    try:
        model_container.load_model()
        logger.info(f"Модель загружена: {model_container.get_model_type()}")
    except Exception as e:
        logger.warning(f"Модель не загружена: {e}")

    yield

    logger.info("Закрытие приложения...")
    shutdown_event.set()
    await asyncio.sleep(2)  # Дать время завершиться запросам
    logger.info("Shutdown complete")


# API router v1 — все ML-эндпоинты с префиксом /api/v1
api_router = APIRouter(prefix=API_V1_PREFIX)

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Кастомная обработка ошибок валидации."""
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
        )

    return JSONResponse(
        status_code=422, content={"detail": "Validation error", "errors": errors}
    )


# Обработчик rate limit exceeded
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Обработчик превышения лимита запросов."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too many requests",
            "message": "Слишком много запросов. Попробуйте позже.",
            "retry_after": 60,
        },
    )


# Middleware для request tracing
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Добавление request_id для трассировки запросов."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    # Логирование начала запроса
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={"request_id": request_id},
    )

    # Prometheus metrics
    start_time = time.time()

    try:
        response = await call_next(request)

        # Prometheus metrics
        duration = time.time() - start_time
        HTTP_REQUEST_DURATION.labels(
            method=request.method, endpoint=request.url.path
        ).observe(duration)

        HTTP_REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        # Логирование завершения запроса
        logger.info(
            f"Request completed: {request.method} {request.url.path} - Status: {response.status_code}",
            extra={"request_id": request_id},
        )

        # Добавляем request_id в заголовки ответа
        response.headers["X-Request-ID"] = request_id

        return response
    except Exception as e:
        # Prometheus metrics для ошибок
        HTTP_ERROR_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            error_type=type(e).__name__,
        ).inc()

        logger.error(
            f"Request failed: {request.method} {request.url.path} - Error: {str(e)}",
            extra={"request_id": request_id},
        )
        raise


# Подключение роутеров
app.include_router(api_router)
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(drift_router, prefix="/api/v1")


# Эндпоинт для Prometheus
@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Экспорт метрик для Prometheus."""
    update_system_metrics()
    return PlainTextResponse(
        get_prometheus_metrics(), media_type=get_metrics_content_type()
    )


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
        "mlflow_ui_url": os.getenv("MLFLOW_UI_URL", "#"),
        "model_loaded": model_container.is_loaded(),
        "model_type": model_container.get_model_type(),
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
        "index.html", {"request": request, "model_loaded": model_container.is_loaded()}
    )


@api_router.get("/health")
async def health_check():
    """Расширенная проверка здоровья с детальной информацией."""
    health_status = {
        "status": "healthy",
        "model_loaded": model_container.is_loaded(),
        "model_type": model_container.get_model_type(),
        "timestamp": datetime.now().isoformat(),
    }

    if not model_container.is_loaded():
        health_status["status"] = "unhealthy"
        health_status["detail"] = "Model not loaded"

    return health_status


@api_router.post("/predict", response_model=PredictionResult)
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_PERIOD}")
async def predict_churn(
    request: Request,
    customer: CustomerInput,
    container: ModelContainer = Depends(get_model_container),
):
    """
    Предсказание оттока для клиента.

    Принимает данные клиента и возвращает вероятность оттока.
    Использует оптимальный порог классификации (threshold tuning).
    В данном датасете Target=1 означает churn (клиент ушёл).
    """
    if not container.is_loaded():
        raise HTTPException(
            status_code=500, detail="Модель не загружена! Проверьте путь к модели."
        )

    request_id = getattr(request.state, "request_id", "no-request-id")
    start_time = time.time()

    try:
        # Предобработка (синхронная, быстрая операция)
        df_processed = preprocess_input(customer)

        # Circuit breaker для предсказания
        try:
            # Асинхронное предсказание (ML операция может блокировать event loop)
            loop = asyncio.get_event_loop()
            probability = await loop.run_in_executor(
                executor,
                lambda: float(container.model.predict_proba(df_processed)[0][1]),
            )
        except Exception as circuit_error:
            logger.error(
                f"Circuit breaker triggered: {circuit_error}",
                extra={"request_id": request_id},
            )
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Please try again later.",
            )

        # Классификация с учётом оптимального порога
        prediction = 1 if probability >= container.model_threshold else 0

        # Определение уровня риска с использованием конфигурации
        if probability < settings.RISK_LOW_THRESHOLD:
            risk_level = "Low"
        elif probability < settings.RISK_HIGH_THRESHOLD:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Prometheus metrics
        duration = time.time() - start_time
        PREDICTION_DURATION.labels(
            model_type=container.get_model_type() or "unknown"
        ).observe(duration)

        PREDICTION_COUNT.labels(
            model_type=container.get_model_type() or "unknown",
            prediction=str(prediction),
        ).inc()

        PREDICTION_PROBABILITY.labels(risk_level=risk_level).observe(probability)

        logger.info(
            f"Prediction completed: probability={probability:.4f}, risk={risk_level}, duration={duration:.3f}s",
            extra={"request_id": request_id},
        )

        return PredictionResult(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            customer_data=customer.model_dump(),
            metrics=container.model_metrics if container.model_metrics else {},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/predict_batch")
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_PERIOD}")
async def predict_churn_batch(
    request: Request,
    customers: List[CustomerInput],
    container: ModelContainer = Depends(get_model_container),
):
    """
    Пакетное предсказание оттока для нескольких клиентов.
    Использует оптимальный порог классификации.
    """
    if not container.is_loaded():
        raise HTTPException(status_code=500, detail="Модель не загружена!")

    request_id = getattr(request.state, "request_id", "no-request-id")

    try:
        results = []
        loop = asyncio.get_event_loop()

        for customer in customers:
            # Предобработка
            preprocess_input(customer)

            # Асинхронное предсказание
            probability = await loop.run_in_executor(
                executor,
                lambda c=customer: float(
                    container.model.predict_proba(preprocess_input(c))[0][1]
                ),
            )

            prediction = 1 if probability >= container.model_threshold else 0

            if probability < settings.RISK_LOW_THRESHOLD:
                risk_level = "Low"
            elif probability < settings.RISK_HIGH_THRESHOLD:
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

        logger.info(
            f"Batch prediction completed: {len(results)} customers",
            extra={"request_id": request_id},
        )

        return {"predictions": results}
    except Exception as e:
        logger.error(
            f"Ошибка при пакетном предсказании: {e}", extra={"request_id": request_id}
        )
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/models")
async def get_model_info(container: ModelContainer = Depends(get_model_container)):
    """Информация о загруженной модели."""
    return {
        "model_loaded": container.is_loaded(),
        "model_type": container.get_model_type(),
        "threshold": container.model_threshold,
        "metrics": container.model_metrics if container.model_metrics else {},
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
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
