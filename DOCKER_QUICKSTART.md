# 🐳 Docker — Полное руководство по запуску

---

## 📋 Содержание

1. [Первый запуск с Docker](#первый-запуск-с-docker)
2. [Перезапуск Docker](#перезапуск-docker)
3. [Полезные команды](#полезные-команды)
4. [Мониторинг](#мониторинг)
5. [Обновление](#обновление)
6. [Устранение проблем](#устранение-проблем)

---

## 🚀 Первый запуск с Docker

### Шаг 1: Проверка Docker

```bash
# Проверить версию Docker
docker --version

# Проверить Docker Compose
docker-compose --version
```

### Шаг 2: Клонирование репозитория

```bash
git clone https://github.com/vikalinet/travel-churn-prediction.git
cd travel-churn-prediction
```

### Шаг 3: Запуск

```bash
# Сборка и запуск всех сервисов
docker-compose up --build

# Или запуск в фоне
docker-compose up -d --build
```

### Шаг 4: Открыть в браузере

```bash
# Основной сервис
http://localhost:8000/

# Тестирование UI
http://localhost:8000/test

# MLflow
http://localhost:5000
```

---

## 🔄 Перезапуск Docker

### Быстрый перезапуск (без пересборки)

```bash
# Остановить и запустить
docker-compose stop
docker-compose start

# Или одной командой
docker-compose restart
```

### Перезапуск с пересборкой

```bash
# Остановить все
docker-compose down

# Очистить volumes (данные MLflow)
docker-compose down -v

# Пересобрать и запустить
docker-compose up --build
```

### Перезапуск конкретного сервиса

```bash
# Перезапустить только API
docker-compose restart web

# Перезапустить только MLflow
docker-compose restart mlflow
```

---

## 🎮 Полезные команды

### Статус сервисов

```bash
# Список запущенных контейнеров
docker-compose ps

# Статистика ресурсов
docker stats

# Подробный статус
docker-compose ps -a
```

### Просмотр логов

```bash
# Все логи в реальном времени
docker-compose logs -f

# Логи конкретного сервиса
docker-compose logs -f web
docker-compose logs -f mlflow

# Последние 100 строк
docker-compose logs --tail=100 web
```

### Доступ в контейнер

```bash
# Bash внутрь контейнера
docker exec -it travel-churn-prediction-web-1 bash

# Внутрь MLflow
docker exec -it travel-churn-prediction-mlflow-1 sh
```

### Работа с volumes

```bash
# Посмотреть volumes
docker volume ls

# Очистить неиспользуемые volumes
docker volume prune

# Очистить все
docker system prune -a
```

---

## 📊 Мониторинг

### Проверка здоровья

```bash
# Health check API
curl http://localhost:8000/api/v1/health

# Ответ должен быть:
# {"status": "healthy", "model_loaded": true}
```

### Проверка модели

```bash
# Тестовое предсказание
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "frequent_flyer": "Yes",
    "annual_income_class": "High Income",
    "services_opted": 5,
    "account_synced_to_social_media": "Yes",
    "booked_hotel_or_not": "Yes"
  }'
```

### Мониторинг MLflow

```bash
# Проверить, что MLflow запущен
curl http://localhost:5000/

# Открыть UI в браузере
# http://localhost:5000
```

---

## 🔧 Обновление

### Обновление кода

```bash
# 1. Остановить
docker-compose down

# 2. Обновить код (git pull)
git pull origin main

# 3. Пересобрать
docker-compose up --build
```

### Обновление модели

```bash
# Скопировать новую модель
# 1. Обучить модель локально
python -m src.training.model_training

# 2. Скопировать в models/
cp models/best_model.pkl models/

# 3. Перезапустить (модель загрузится автоматически)
docker-compose restart web
```

---

## ❓ Устранение проблем

### Проблема: `docker: command not found`

```bash
# Windows: Установить Docker Desktop с docker.com
# Linux: sudo apt install docker.io docker-compose
```

### Проблема: Порт занят

```bash
# Найти процесс на порту
# Windows
netstat -ano | findstr ":8000"

# Linux
lsof -i :8000

# Убить процесс или изменить порт в docker-compose.yml
```

### Проблема: Контейнер не запускается

```bash
# Посмотреть логи
docker-compose logs web

# Пересобрать с очисткой
docker-compose down
docker system prune -f
docker-compose up --build
```

### Проблема: Не загружается модель

```bash
# Проверить, что модель есть в контейнере
docker exec -it travel-churn-prediction-web-1 ls -la models/

# Если нет — скопировать
docker cp models/best_model.pkl travel-churn-prediction-web-1:/app/models/
```

### Проблема: MLflow не работает

```bash
# Проверить logs
docker-compose logs mlflow

# Пересоздать volume
docker-compose down -v
docker-compose up -d
```

### Проблема: Permission denied

```bash
# Изменить владельца папки проекта
# Linux/macOS
sudo chown -R $USER:$USER .

# Или создать volume заново
docker volume rm travel-churn-prediction_mlflow_data
```

---

## 🧹 Полная очистка

```bash
# Остановить все
docker-compose down

# Удалить volumes
docker-compose down -v

# Удалить образы
docker rmi $(docker images -q -f "reference=travel-churn-prediction*")

# Очистить всё
docker system prune -a -f --volumes

# Пересобрать с нуля
docker-compose up --build
```

---

## 📁 Структура Docker

```
travel-churn-prediction/
├── docker-compose.yml      # Конфигурация сервисов
├── Dockerfile              # Образ для web-сервиса
├── models/                 # Модели (монтируются в контейнер)
├── data/                   # Данные (опционально)
└── mlflow_data/            # Данные MLflow (volume)
```

### Сервисы

| Сервис | Порт | Описание |
|--------|------|----------|
| web | 8000 | FastAPI приложение |
| mlflow | 5000 | MLflow сервер |

### Volumes

| Volume | Описание |
|--------|----------|
| mlflow_data | Хранение экспериментов MLflow |

---

## 🎯 Чеклист

После запуска проверьте:

- [ ] `docker-compose ps` показывает 2 сервиса
- [ ] http://localhost:8000/ открывается
- [ ] http://localhost:5000/ открывается
- [ ] `/health` возвращает healthy
- [ ] Тестовое предсказание работает

---

## 🔗 Быстрые команды

```bash
# Запуск
docker-compose up -d --build

# Остановка
docker-compose down

# Перезапуск
docker-compose restart

# Логи
docker-compose logs -f

# Войти в контейнер
docker exec -it travel-churn-prediction-web-1 bash
```
