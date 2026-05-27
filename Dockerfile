# ============================================================
# Multi-stage Dockerfile для ML API сервиса
# ============================================================
# Этап 1: Сборка зависимостей (builder)
# Цель: Компиляция зависимостей и установка Python-пакетов
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Установка системных зависимостей для сборки Python-пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    pkg-config \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements для кэширования
COPY requirements.txt .

# Установка Python-зависимостей в локальный каталог
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================
# Этап 2: Финальный минимальный образ (production)
# Цель: Легковесный образ для продакшена
# ============================================================
FROM python:3.11-slim AS production

WORKDIR /app

# Импортируем установленные пакеты из builder
COPY --from=builder /root/.local /root/.local

# Добавляем PATH для user-installed packages
ENV PATH=/root/.local/bin:$PATH

# Переменная окружения для Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Копирование только необходимых файлов проекта
# Код приложения
COPY src/ ./src/

# Модели (если есть)
COPY --optional models/ ./models/

# Данные (если есть)
COPY --optional data/ ./data/

# Отчёты (если есть)
COPY --optional reports/ ./reports/

# Evidently отчёты (если есть)
COPY --optional evidently_reports/ ./evidently_reports/

# Скрипты
COPY scripts/ ./scripts/

# Презентация и статика
COPY presentation.html .
COPY static/ ./static/

# README и документация
COPY README.md .
COPY DOCKER_GUIDE.md .
COPY UI_GUIDE.md .
COPY QUICKSTART.md .
COPY DOCKER_QUICKSTART.md .

# ============================================================
# Безопасность: создание непривилегированного пользователя
# ============================================================
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# ============================================================
# Health Check: проверка здоровья сервиса
# ============================================================
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=5s \
            --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ============================================================
# Проброс порта для FastAPI
# ============================================================
EXPOSE 8000

# ============================================================
# Команда запуска приложения
# ============================================================
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================
# Альтернативные команды для запуска:
# - Обучение модели:     python src/training/model_training.py
# - ETL пайплайн:        python src/etl/etl_pipeline.py
# - Генерация отчётов:   python scripts/generate_visualizations.py
# - Тесты:               pytest tests/ -v
# ============================================================
