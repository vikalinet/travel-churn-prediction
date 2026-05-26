# Этап 1: Сборка зависимостей
FROM python:3.11-slim AS builder

WORKDIR /app

# Установка системных зависимостей для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка Python-зависимостей
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Этап 2: Финальный минимальный образ
FROM python:3.11-slim

WORKDIR /app

# Копирование установленных пакетов из builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Копирование только необходимых файлов
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY reports/ ./reports/
COPY evidently_reports/ ./evidently_reports/

# Создание непривилегированного пользователя
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
