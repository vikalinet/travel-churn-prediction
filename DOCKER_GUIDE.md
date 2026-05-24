# Docker Deployment Guide

## Предварительные требования

- Docker Desktop установлен и запущен
- Docker Component (обычно входит в Docker Desktop)

## Запуск проекта

### 1. Сборка и запуск всех сервисов

```bash
docker-compose up --build
```

Это запустит:
- **web** — FastAPI сервис на порту 8000
- **mlflow** — MLflow сервер на порту 5000

### 2. Запуск только FastAPI сервиса

```bash
docker-compose up web
```

### 3. Проверка работы

Откройте браузер:
- FastAPI: http://localhost:8000
- FastAPI Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000

### 4. Остановка сервисов

```bash
docker-compose down
```

### 5. Остановка и удаление всех данных

```bash
docker-compose down -v
```

## API Endpoints

### Проверка здоровья
```bash
curl http://localhost:8000/health
```

### Предсказание для одного клиента
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "frequent_flyer": "No",
    "annual_income_class": "Middle Income",
    "services_opted": 5,
    "account_synced_to_social_media": "Yes",
    "booked_hotel_or_not": "No"
  }'
```

### Пакетное предсказание
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "age": 35,
        "frequent_flyer": "No",
        "annual_income_class": "Middle Income",
        "services_opted": 5,
        "account_synced_to_social_media": "Yes",
        "booked_hotel_or_not": "No"
      },
      {
        "age": 42,
        "frequent_flyer": "Yes",
        "annual_income_class": "High Income",
        "services_opted": 4,
        "account_synced_to_social_media": "No",
        "booked_hotel_or_not": "Yes"
      }
    ]
  }'
```

## Тестирование через Swagger UI

Откройте http://localhost:8000/docs для интерактивного тестирования API.

## Логи

### Просмотр логов всех сервисов
```bash
docker-compose logs -f
```

### Логи конкретного сервиса
```bash
docker-compose logs -f web
docker-compose logs -f mlflow
```

## Ограничения ресурсов

В `docker-compose.yml` настроены ограничения:
- **CPU:** 0.5-1 ядро
- **Memory:** 1-2 GB

Для изменения отредактируйте секцию `deploy.resources`.

## Безопасность

- Контейнер запускается от нерута пользователя (`appuser`)
- Сетевые порты ограничены
- Нет доступа к хостовой файловой системе

## Устранение проблем

### Проблемы с сборкой

```bash
# Очистка кэша Docker
docker system prune -a

# Пересборка без кэша
docker-compose build --no-cache
```

### Проблемы с портами

```bash
# Проверка занятости портов
netstat -ano | findstr :8000
netstat -ano | findstr :5000
```

### Проблемы с памятью

Увеличьте лимит памяти в Docker Desktop:
- Settings → Resources → Memory

## Локальная разработка

### Запуск без Docker
```bash
# Виртуальное окружение
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Установка зависимостей
pip install -r requirements.txt

# Запуск FastAPI
uvicorn src.api.main:app --reload
```

## Развёртывание в production

Для production используйте:
- Переменные окружения для конфигурации
- Отдельную базу данных для MLflow
- Reverse proxy (nginx)
- SSL/TLS сертификаты

Пример `.env` файла:
```env
MLFLOW_TRACKING_URI=http://mlflow:5000
UVICORN_HOST=0.0.0.0
UVICORN_PORT=8000
```
