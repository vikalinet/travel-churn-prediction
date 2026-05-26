# 🚀 Быстрый старт проекта

Полное руководство по запуску проекта с нуля и перезапуску.

---

## 📋 Содержание

1. [Первый запуск с нуля](#первый-запуск-с-нуля)
2. [Перезапуск проекта](#перезапуск-проекта)
3. [Обновление модели](#обновление-модели)
4. [Полезные команды](#полезные-команды)
5. [Устранение проблем](#устранение-проблем)

---

## 🖥️ Первый запуск с нуля

### Шаг 1: Клонирование репозитория

```bash
# Клонировать репозиторий
git clone https://github.com/vikalinet/travel-churn-prediction.git

# Перейти в папку проекта
cd travel-churn-prediction
```

### Шаг 2: Проверка Python

```bash
# Проверить версию Python (нужен 3.11+)
python --version

# Если Python < 3.11, установить новую версию:
# Windows: скачать с python.org
# macOS: brew install python@3.11
# Linux: sudo apt install python3.11
```

### Шаг 3: Создание виртуального окружения

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Шаг 4: Установка зависимостей

```bash
# Установка всех пакетов
pip install -r requirements.txt

# Проверка установки
pip list | grep -E "(fastapi|sklearn|mlflow)"
```

### Шаг 5: Проверка наличия данных

```bash
# Проверить наличие данных
ls data/raw/

# Если папка пуста, скачать датасет:
# 1. Скачать с Kaggle: https://www.kaggle.com/datasets/sagnik1511/tour-travel-customer-churn-prediction
# 2. Или использовать готовый файл из data/raw/
```

### Шаг 6: Первый запуск API

```bash
# Запуск FastAPI сервера
uvicorn src.api.main:app --reload

# Сервер запустится на http://localhost:8000
```

### Шаг 7: Открыть в браузере

```bash
# Основной интерфейс предсказаний
http://localhost:8000/

# Тестирование UI (18 готовых сценариев)
http://localhost:8000/test

# Swagger документация
http://localhost:8000/docs
```

---

## 🔄 Перезапуск проекта

### Быстрый перезапуск (без изменений)

```bash
# Активировать виртуальное окружение
# Windows
.\venv\Scripts\Activate.ps1
# Linux/macOS
source venv/bin/activate

# Запустить сервер
uvicorn src.api.main:app --reload
```

### Перезапуск после изменений в коде

```bash
# Сервер автоматически перезагрузится при изменении файлов
# (флаг --reload)

# Для полной перезагрузки:
# 1. Остановить сервер (Ctrl+C)
# 2. Запустить заново
uvicorn src.api.main:app --reload
```

### Перезапуск Docker

```bash
# Остановить все контейнеры
docker-compose down

# Удалить volumes (опционально)
docker-compose down -v

# Запустить заново
docker-compose up --build
```

---

## 🔧 Обновление модели

### Вариант 1: Новая модель уже есть в `/models/`

```bash
# Просто перезапустите сервер — модель загрузится автоматически
# Сервер найдёт файл models/best_model.pkl
```

### Вариант 2: Обучение новой модели

```bash
# Запустить обучение
python -m src.training.model_training

# Или полный пайплайн
python -m src.training.customer_travel_training

# После обучения модель сохранится в models/best_model.pkl
```

### Вариант 3: Обновление через MLflow

```bash
# Запустить MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Открыть http://localhost:5000
# Зарегистрировать новую версию модели
```

---

## 🎮 Полезные команды

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# С покрытием
pytest tests/ -v --cov=src --cov-report=html

# Конкретный файл
pytest tests/test_preprocessing.py -v
```

### Генерация отчётов

```bash
# Визуализации
python scripts/generate_visualizations.py

# Отчёт о дрейфе
python scripts/generate_drift_report.py

# Отчёт об обучении
python scripts/generate_training_report.py
```

### Работа с Docker

```bash
# Запуск в фоне
docker-compose up -d --build

# Просмотр логов
docker-compose logs -f web
docker-compose logs -f mlflow

# Статус контейнеров
docker-compose ps

# Статистика ресурсов
docker stats

# Остановка
docker-compose down

# Полная очистка (с удалением volumes)
docker-compose down -v
docker system prune -f
```

### Работа с MLflow

```bash
# Локальный запуск MLflow
mlflow ui --host 0.0.0.0 --port 5000

# Очистка экспериментов
rm -rf mlruns/
```

---

## ❓ Устранение проблем

### Проблема: `ModuleNotFoundError`

```bash
# Переустановить зависимости
pip install -r requirements.txt
```

### Проблема: Модель не загружается

```bash
# Проверить наличие файла модели
ls -la models/

# Если файла нет — обучить модель
python -m src.training.model_training
```

### Проблема: Порты заняты

```bash
# Найти процесс на порту 8000
# Windows
netstat -ano | findstr :8000
# Linux/macOS
lsof -i :8000

# Убить процесс
# Windows
taskkill /PID <номер> /F
# Linux
kill -9 <номер>
```

### Проблема: Docker не запускается

```bash
# Проверить Docker
docker --version

# Очистить кэш Docker
docker builder prune -f

# Пересобрать образ
docker-compose build --no-cache
```

### Проблема: Нет данных

```bash
# Проверить структуру папок
ls -la data/raw/

# Если нужно, скачать данные:
# 1. Создать папку data/raw/
# 2. Поместить туда CSV файл с данными
```

### Проблема: Тесты падают

```bash
# Очистить кэш pytest
rm -rf .pytest_cache __pycache__ */__pycache__

# Запустить тесты заново
pytest tests/ -v --tb=short
```

---

## 📊 Структура проекта после запуска

```
travel-churn-prediction/
├── data/
│   ├── raw/                    # Исходные данные
│   ├── processed/              # Обработанные данные
│   └── test_scenarios/         # Тестовые данные
├── models/                     # Сохранённые модели (.pkl)
├── reports/                    # Графики и отчёты
├── evidently_reports/          # Отчёты о дрейфе
├── mlruns/                     # MLflow эксперименты
├── src/
│   ├── api/                    # FastAPI приложение
│   ├── etl/                    # ETL пайплайн
│   ├── training/               # Обучение моделей
│   └── monitoring/             # Мониторинг
├── tests/                      # Pytest тесты
├── templates/                   # HTML шаблоны
├── static/                     # CSS, JS
└── venv/                       # Виртуальное окружение
```

---

## 🎯 Чеклист для проверки

После первого запуска проверьте:

- [ ] Сервер запущен на `http://localhost:8000`
- [ ] Главная страница открывается
- [ ] Тестовая страница `/test` доступна
- [ ] `/health` возвращает `{"status": "healthy"}`
- [ ] `/api/test-data` возвращает JSON с тестовыми данными
- [ ] Модель загружена (можно сделать тестовое предсказание)

---

## 🧪 Тестирование UI

Проект включает 18 готовых тестовых сценариев для проверки модели.

### Доступные сценарии

| Категория | Кол-во | Описание |
|-----------|--------|----------|
| 🟢 Положительные | 5 | Клиенты с низким риском оттока |
| 🔴 Отрицательные | 5 | Клиенты с высоким риском оттока |
| 🟡 Дрифт данных | 5 | Аномалии и редкие паттерны |
| 🟣 Граничные | 3 | На границе классов |

### Запуск тестирования

```bash
# Открыть страницу тестирования
http://localhost:8000/test
```

### Что можно делать на странице `/test`

1. **Кликнуть на карточку** — данные автоматически заполнятся в форме
2. **Запустить категорию целиком** — кнопки "Запустить все положительные" и т.д.
3. **Увидеть результаты** — PASS/FAIL для каждого сценария с сравнением ожидаемого и полученного

### Тестовые данные

```bash
# Получить все тестовые данные в JSON
GET /api/test-data

# Получить конкретный сценарий
GET /api/test-data/pos_001
GET /api/test-data/neg_003
GET /api/test-data/drift_002
```

### Файлы тестовых данных

- `data/test_scenarios/test_data.json` — полные данные в JSON
- `data/test_scenarios/test_data.csv` — таблица для просмотра

---

## 🔗 Быстрые ссылки

| URL | Описание |
|-----|----------|
| http://localhost:8000/ | Главная страница предсказаний |
| http://localhost:8000/test | Тестирование UI (18 сценариев) |
| http://localhost:8000/docs | Swagger документация API |
| http://localhost:5000 | MLflow UI (если запущен) |

---

## 📞 Дополнительная помощь

- **README.md** — полная документация проекта
- **UI_GUIDE.md** — руководство по веб-интерфейсу
- **DOCKER_GUIDE.md** — работа с Docker
- **DOCKER_QUICKSTART.md** — быстрый старт с Docker
