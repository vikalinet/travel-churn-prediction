FROM python:3.11-slim

# Установка рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода проекта
COPY . .

# Создание нерута пользователя для безопасности
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Экспонирование порта для FastAPI
EXPOSE 8000

# Команда запуска приложения
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
