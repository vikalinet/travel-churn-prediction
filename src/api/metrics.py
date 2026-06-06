"""
Prometheus метрики для мониторинга API.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import psutil
import os

# ============================================================
# HTTP метрики
# ============================================================

# Счётчик запросов по эндпоинтам и методам
HTTP_REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

# Гистограмма длительности запросов
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Счётчик ошибок по типу
HTTP_ERROR_COUNT = Counter(
    "http_errors_total", "Total HTTP errors", ["method", "endpoint", "error_type"]
)

# ============================================================
# ML метрики
# ============================================================

# Счётчик предсказаний
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions made", ["model_type", "prediction"]
)

# Гистограмма вероятностей предсказаний
PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of prediction probabilities",
    ["risk_level"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Время предсказания
PREDICTION_DURATION = Histogram(
    "prediction_duration_seconds",
    "Prediction duration in seconds",
    ["model_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# ============================================================
# Системные метрики (Gauges)
# ============================================================

# CPU usage
CPU_USAGE = Gauge("process_cpu_percent", "Process CPU usage percentage")

# Memory usage
MEMORY_USAGE = Gauge("process_memory_bytes", "Process memory usage in bytes")

# Memory percent
MEMORY_PERCENT = Gauge("process_memory_percent", "Process memory usage percentage")

# Disk usage
DISK_USAGE = Gauge("disk_usage_bytes", "Disk usage in bytes")

# Disk free
DISK_FREE = Gauge("disk_free_bytes", "Disk free space in bytes")

# Active connections
ACTIVE_CONNECTIONS = Gauge("active_connections", "Number of active connections")

# ============================================================
# Бизнес метрики
# ============================================================

# Количество пользователей в системе (для примера)
ACTIVE_USERS = Gauge("active_users", "Number of active users")

# Среднее время сессии
SESSION_DURATION = Histogram(
    "session_duration_seconds",
    "Session duration in seconds",
    buckets=[60, 300, 600, 1800, 3600, 7200, 10800],
)

# ============================================================
# Функции для обновления метрик
# ============================================================


def update_system_metrics():
    """Обновление системных метрик."""
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        CPU_USAGE.set(cpu_percent)

        # Memory
        memory = psutil.Process(os.getpid()).memory_info()
        MEMORY_USAGE.set(memory.rss)
        MEMORY_PERCENT.set(psutil.Process(os.getpid()).memory_percent())

        # Disk
        disk = psutil.disk_usage("/")
        DISK_USAGE.set(disk.used)
        DISK_FREE.set(disk.free)

    except Exception as e:
        # Логирование ошибки, но не падение
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Ошибка обновления системных метрик: {e}")


def get_prometheus_metrics():
    """Получение всех метрик в формате Prometheus."""
    return generate_latest()


def get_metrics_content_type():
    """Получение Content-Type для Prometheus."""
    return CONTENT_TYPE_LATEST
