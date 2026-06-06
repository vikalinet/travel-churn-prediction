"""
Конфигурация API: префикс версии, настройки, маппинги.
Использует pydantic-settings для валидации и управления окружением.
"""

from typing import Dict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Конфигурация приложения с валидацией через Pydantic."""

    # API настройки
    API_V1_PREFIX: str = "/api/v1"
    APP_TITLE: str = "Travel Churn Prediction API"
    APP_DESCRIPTION: str = (
        "API для прогнозирования оттока клиентов туристического агентства"
    )
    APP_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Путь к модели
    MODEL_PATH: str = "models/best_model.pkl"
    MODEL_PATHS: list = [
        "models/best_model.pkl",
        "models/GradientBoosting_model.pkl",
        "models/model.pkl",
    ]

    # MLflow
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"

    # Маппинги категориальных признаков (единый источник правды)
    CATEGORICAL_MAPPINGS: Dict[str, Dict[str, int]] = {
        "FrequentFlyer": {"Yes": 1, "No": 0},
        "AnnualIncomeClass": {"Low Income": 0, "Middle Income": 1, "High Income": 2},
        "AccountSyncedToSocialMedia": {"Yes": 1, "No": 0},
        "BookedHotelOrNot": {"Yes": 1, "No": 0},
    }

    # Пороги для уровней риска
    RISK_LOW_THRESHOLD: float = 0.3
    RISK_HIGH_THRESHOLD: float = 0.7

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: str = "minute"  # minute, hour, day

    # Cache
    CACHE_TTL: int = 300
    CACHE_MAX_SIZE: int = 1000

    # Prometheus metrics
    PROMETHEUS_ENABLED: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = (
        "%(asctime)s %(levelname)s %(message)s %(name)s %(filename)s %(lineno)s"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Глобальный экземпляр настроек
settings = Settings()

# Префикс API (для обратной совместимости)
API_V1_PREFIX = settings.API_V1_PREFIX
