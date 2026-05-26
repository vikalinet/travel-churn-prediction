# 🐳 Docker Guide для ML Пайплайна

Полное руководство по использованию Docker в проекте прогнозирования оттока клиентов.

## 📋 Содержание

1. [Описание Dockerfile](#-описание-dockerfile)
2. [Многоступенчатая сборка](#-многоступенчатая-сборка)
3. [Оптимизация ресурсов](#-оптимизация-ресурсов)
4. [Безопасность](#-безопасность)
5. [Запуск контейнеров](#-запуск-контейнеров)
6. [Команды Docker](#-команды-docker)
7. [Troubleshooting](#-troubleshooting)

---

## 📄 Описание Dockerfile

### Базовый образ

```dockerfile
FROM python:3.11-slim
```

**Функция:** Использование минимизированного образа Python 3.11 `slim` версии.

**Преимущества:**
- ⚖️ **Меньший размер** (~150MB vs ~1GB у full образа)
- 🔒 **Меньше уязвимостей** — меньше установленных пакетов
- ⚡ **Быстрее загрузка** — оптимизация для production

### Рабочая директория

```dockerfile
WORKDIR /app
```

**Функция:** Создание и установка рабочей директории внутри контейнера.

### Многоступенчатая сборка

#### Этап 1: Builder

```dockerfile
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    pkg-config \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
```

**Функция:** Компиляция системных зависимостей для Python-пакетов.

**Комментарии:**
- `gcc`, `g++`, `make` — компиляторы для пакетов с C-расширениями
- `libpq-dev` — зависимости для PostgreSQL (если используется)
- `--no-install-recommends` — установка только обязательных пакетов
- `rm -rf /var/lib/apt/lists/*` — очистка кэша apt для уменьшения размера

#### Этап 2: Production

```dockerfile
FROM python:3.11-slim AS production
COPY --from=builder /root/.local /root/.local
```

**Функция:** Копирование установленных пакетов из builder в финальный образ.

**Преимущества:**
- 🔧 Зависимости скомпилированы в builder
- 📦 Финальный образ не содержит инструментов компиляции
- 🚀 Размер образа ~400MB вместо ~1GB

### Переменные окружения

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1
```

**Функции:**
- `PYTHONDONTWRITEBYTECODE=1` — не создавать `.pyc` файлы (чистая файловая система)
- `PYTHONUNBUFFERED=1` — вывод логов в реальном времени (без буферизации)
- `PYTHONFAULTHANDLER=1` — корректная обработка фатальных ошибок

### Безопасность: непривилегированный пользователь

```dockerfile
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser
```

**Функции:**
- 🔒 Запуск от непривилегированного пользователя (не root)
- 👤 UID 1000 — стандартный пользователь в Linux
- 📁 Правильные права доступа к файлам

### Health Check

```dockerfile
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=5s \
            --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
```

**Параметры:**
- `--interval=30s` — проверка каждые 30 секунд
- `--timeout=10s` — таймаут проверки 10 секунд
- `--start-period=5s` — стартовая задержка (запуск приложения)
- `--retries=3` — количество попыток перед помечанием как unhealthy

**Функция:** Автоматический мониторинг здоровья контейнера.

### Проброс порта

```dockerfile
EXPOSE 8000
```

**Функция:** Документирование порта для FastAPI (не пробрасывает автоматически).

### Команда запуска

```dockerfile
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Функция:** Запуск ASGI сервера uvicorn для FastAPI приложения.

**Параметры:**
- `--host 0.0.0.0` — слушать все сетевые интерфейсы
- `--port 8000` — порт для входящих запросов

---

## 🏗️ Многоступенчатая сборка

### Схема процесса

```
┌─────────────────────────────────────┐
│   Этап 1: Builder                   │
│   - Установка gcc, g++              │
│   - Компиляция зависимостей         │
│   - pip install --user              │
└──────────────┬──────────────────────┘
               │
               ▼ COPY --from=builder
┌─────────────────────────────────────┐
│   Этап 2: Production                │
│   - Минимальный образ               │
│   - Только runtime зависимости      │
│   - Непривилегированный пользователь│
└─────────────────────────────────────┘
```

### Преимущества многоступенчатой сборки

| Аспект | Одноступенчатая | Многоступенчатая |
|--------|-----------------|------------------|
| Размер образа | ~1GB | ~400MB |
| Уязвимости | Высокие | Минимальные |
| Скорость загрузки | Медленно | Быстро |
| Безопасность | Низкая | Высокая |

---

## ⚙️ Оптимизация ресурсов

### Ограничения в docker-compose.yml

```yaml
services:
  web:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
```

**Параметры:**
- `cpus: '1'` — максимум 1 CPU ядро
- `memory: 2G` — максимум 2 GB RAM
- `cpus: '0.5'` — гарантировано 0.5 ядра
- `memory: 1G` — гарантировано 1 GB RAM

### Мониторинг использования ресурсов

```bash
# Статистика в реальном времени
docker stats

# Конкретный контейнер
docker stats <container_id>

# Пример вывода:
# CONTAINER ID   NAME    CPU %     MEM USAGE / LIMIT   MEM %     NET I/O
# abc123def      web     0.15%     150MB / 2GB         7.50%     100kB / 50kB
```

### Оптимизации в Dockerfile

| Оптимизация | Функция |
|-------------|---------|
| `--no-cache-dir` в pip | Уменьшение размера образа |
| `slim` версия Python | Минимизация уязвимостей |
| `--no-install-recommends` в apt | Только обязательные пакеты |
| Многоступенчатая сборка | Разделение build и runtime |
| `.dockerignore` | Исключение ненужных файлов |

---

## 🔒 Безопасность

### 1. Непривилегированный пользователь

```dockerfile
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
```

**Защита:**
- ❌ Ограничение прав доступа к системе
- 🛡️ Защита от эксплуатации уязвимостей
- ✅ Соответствие best practices

### 2. Health Check

```dockerfile
HEALTHCHECK --interval=30s ...
```

**Защита:**
- 🔍 Автоматическое обнаружение сбоев
- 🔄 Автоматический перезапуск unhealthy контейнеров
- 📊 Мониторинг состояния сервиса

### 3. Секреты не встраиваются в образ

```bash
# ❌ ПЛОХО:
ENV API_KEY=secret123

# ✅ ХОРОШО:
# Передавать через docker-compose или runtime
docker run -e API_KEY=secret123 ...
```

### 4. Регулярное обновление базовых образов

```bash
# Проверка обновлений
docker scan <image_name>

# Обновление образа
docker pull python:3.11-slim
```

---

## 🚀 Запуск контейнеров

### Локальный запуск

```bash
# Сборка и запуск всех сервисов
docker-compose up --build

# Запуск в фоновом режиме
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка сервисов
docker-compose down
```

### Запуск одного сервиса

```bash
# Только API сервис
docker-compose up -d web

# Только MLflow
docker-compose up -d mlflow
```

### Пересборка без кэша

```bash
# Полная пересборка
docker-compose build --no-cache

# Пересборка конкретного сервиса
docker-compose build --no-cache web
```

### Доступ к контейнеру

```bash
# Вход в bash контейнера
docker-compose exec web bash

# Выполнение команды
docker-compose exec web python --version
docker-compose exec web pytest tests/ -v
```

---

## 💻 Команды Docker

### Управление контейнерами

```bash
# Список всех контейнеров
docker ps -a

# Старт контейнера
docker start <container_id>

# Остановка контейнера
docker stop <container_id>

# Перезапуск контейнера
docker restart <container_id>

# Удаление контейнера
docker rm <container_id>

# Удаление всех остановленных контейнеров
docker container prune
```

### Управление образами

```bash
# Список образов
docker images

# Удаление образа
docker rmi <image_id>

# Удаление неиспользуемых образов
docker image prune

# Сохранение образа в файл
docker save -o myimage.tar <image_name>

# Загрузка образа из файла
docker load -i myimage.tar
```

### Логи и отладка

```bash
# Логи контейнера
docker logs <container_id>

# Логи с временными метками
docker logs -t <container_id>

# Последние N строк логов
docker logs --tail 100 <container_id>

# Логи в реальном времени
docker logs -f <container_id>
```

### Мониторинг

```bash
# Статистика контейнеров
docker stats

# Информация о системе Docker
docker system info

# Удаление неиспользуемых данных
docker system prune -a
```

---

## 🐛 Troubleshooting

### Проблема: Контейнер не запускается

```bash
# Просмотр логов
docker-compose logs web

# Проверка состояния
docker inspect <container_id>

# Запуск с интерактивным терминалом
docker-compose run --rm web bash
```

### Проблема: Ошибка подключения к базе данных

```bash
# Проверка сети Docker
docker network ls

# Проверка подключения между контейнерами
docker-compose exec web ping mlflow

# Пересоздание сети
docker-compose down
docker-compose up -d
```

### Проблема: Не хватает памяти

```bash
# Увеличение лимита в docker-compose.yml
services:
  web:
    deploy:
      resources:
        limits:
          memory: 4G  # Увеличить до 4GB

# Пересборка с новыми настройками
docker-compose down
docker-compose up -d --build
```

### Проблема: Ошибка сборки образа

```bash
# Очистка кэша Docker
docker builder prune -a

# Пересборка без кэша
docker-compose build --no-cache

# Проверка Dockerfile
docker build --progress=plain .
```

### Проблема: Модель не загружается

```bash
# Проверка наличия модели в контейнере
docker-compose exec web ls -la models/

# Копирование модели в контейнер
docker cp models/best_model.pkl web:/app/models/best_model.pkl

# Перезапуск контейнера
docker-compose restart web
```

---

## 📚 Дополнительные ресурсы

- [Официальная документация Docker](https://docs.docker.com/)
- [Best Practices для Dockerfile](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Compose referencia](https://docs.docker.com/compose/compose-file/)
- [FastAPI в Docker](https://fastapi.tiangolo.com/deployment/docker/)

---

**Версия:** 1.0.0
**Последнее обновление:** 2024
