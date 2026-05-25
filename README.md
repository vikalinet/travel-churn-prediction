# Прогнозирование оттока клиентов туристического агентства

[![CI/CD](https://github.com/vikalinet/travel-churn-prediction/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/vikalinet/travel-churn-prediction/actions)

🔗 **GitHub репозиторий:** [https://github.com/vikalinet/travel-churn-prediction](https://github.com/vikalinet/travel-churn-prediction)

## 🎯 Бизнес-задача

Разработать модель для выявления клиентов с высоким риском оттока, чтобы компания могла предложить им персонализированные скидки и программы лояльности, увеличив их удержание.

**Целевая метрика:** Увеличение удержания клиентов на 15-20% за счёт своевременного выявления групп риска.

**Ожидаемый эффект** (гипотеза для пилотного внедрения):
- Сокращение расходов на удержание на 30% (таргетированное удержание вместо массового)
- Рост повторных продаж на 15% (удержанные клиенты продолжают покупать)
- Повышение удовлетворённости клиентов (релевантные предложения, меньше спама)

> **Примечание:** Метрики требуют A/B тестирования в реальных условиях для подтверждения.

## 📊 Датасет

Используется датасет **Tour & Travels Customer Churn Prediction** с Kaggle.

**Ключевые признаки:**
- Демографические: возраст, пол, годовой доход
- Поведенческие: частота перелётов, использование услуг
- Категориальные: тип бронирования, класс обслуживания

**Целевая переменная:** `Churn` (бинарная: 0 - остался, 1 - ушёл)

## 🏗️ Архитектура пайплайна

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│   Данные    │────>│    ETL       │────>│   Обучение   │────>│   MLflow    │
│   (CSV)     │     │  (pandas)    │     │   моделей    │     │  (реестр)   │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
                                                                   │
                                                                   v
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│  Мониторинг │<────│  Evidently   │<────│  FastAPI     │<────│  Предсказание│
│  (дрейф)    │     │   AI         │     │  сервис      │     │   (Docker)  │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
```

## 🔄 ETL Процесс

### Датасет
Используется датасет **Customer Travel Churn** (954 клиента, 7 признаков).

**Признаки:**
- `Age` — возраст клиента
- `FrequentFlyer` — часто ли летает (Yes/No)
- `AnnualIncomeClass` — класс дохода (Low/Middle/High Income)
- `ServicesOpted` — количество воспользованных услуг (1-6)
- `AccountSyncedToSocialMedia` — аккаунт синхронизирован с соцсетями (Yes/No)
- `BookedHotelOrNot` — забронировал отель (Yes/No)
- `Target` — целевая переменная (Churn: 0 - остался, 1 - ушёл)

### Extract (Извлечение)
- Загрузка данных из CSV-файла
- Проверка целостности и структуры

### Transform (Трансформация)
1. **Очистка данных:**
   - Обработка пропущенных значений (медиана для чисел, мода для категорий)
   - Проверка типов данных

2. **Кодирование:**
   - LabelEncoder для категориальных признаков (Yes/No → 0/1)
   - Маппинг уровней дохода (Low/Middle/High Income → 0/1/2)

3. **Масштабирование:**
   - StandardScaler для числовых переменных (опционально)

### Load (Загрузка)
- Сохранение обработанных данных в `data/processed/processed_data.csv`
- Экспорт модели в MLflow

## 🤖 Модели

### Кастомные модели (scikit-learn / XGBoost)

**Результаты на тестовой выборке (фактические измерения):**

| Модель | Accuracy | F1-score | ROC AUC | Precision | Recall |
|--------|----------|----------|---------|-----------|--------|
| **GradientBoosting** | **91.1%** | **79.5%** | **97.5%** | 86.8% | 73.3% |
| XGBoost | 89.5% | 76.2% | 97.0% | 82.1% | 71.1% |
| XGBoost (Tuned) | 89.5% | 76.2% | 96.8% | 82.1% | 71.1% |
| RandomForest | 88.5% | 73.8% | 95.6% | 79.5% | 68.9% |
| RandomForest (Tuned) | 88.5% | 73.8% | 96.0% | 79.5% | 68.9% |
| KNeighbors | 86.9% | 63.8% | 91.7% | 91.7% | 48.9% |
| LogisticRegression | 83.2% | 54.3% | 84.7% | 76.0% | 42.2% |
| SVC | 76.4% | 0.0% | 85.7% | 0.0% | 0.0% |

**Лучшая модель:** GradientBoosting с F1-score = 79.5% и ROC AUC = 97.5%

### Модели с подобранными гиперпараметрами (Tuned)

Для улучшения качества моделей был проведён подбор гиперпараметров с помощью GridSearchCV:

**XGBoost Tuned:**
- Параметры: `learning_rate=0.2, max_depth=5, n_estimators=150`
- F1-score: 76.2% (без существенного улучшения по сравнению с базовой версией)

**RandomForest Tuned:**
- Параметры: `max_depth=10, min_samples_split=2, n_estimators=150`
- F1-score: 73.8% (без существенного улучшения)

> **Вывод:** На данном датасете (~1000 строк) тюнинг гиперпараметров не дал значимого улучшения качества. Дефолтные параметры алгоритмов уже близки к оптимальным для этой задачи.

### 🤖 AutoML

Помимо кастомных моделей реализовано автоматизированное обучение с помощью фреймворков AutoML:

**1. AutoGluon (`src/training/automl_training.py`)**
- Используется `TabularDataset` и `Trainer` из `autogluon.tabular`
- Автоматический подбор моделей и гиперпараметров в заданный time limit
- Поддержка presets: `medium_quality`, `good_quality`, `best_quality`
- Генерация лидерборда сравнения моделей
- Сохранение лучшей модели в `autogluon_models/`

**2. H2O AutoML (`src/training/automl_alternative.py`)**
- Альтернативный фреймворк с `H2OAutoML`
- Автоматический подбор алгоритмов с исключением DeepLearning (для скорости)
- Лидерборд с ранжированием моделей
- Fallback: при отсутствии H2O используется `VotingClassifier` (ансамбль RandomForest + GradientBoosting + LogisticRegression)

**3. Интеграция в пайплайн (`src/training/model_training.py`)**
- Метод `train_automl()` встроен в общий пайплайн обучения
- Сравнение AutoML с кастомными моделями по метрикам
- Автоматическая визуализация результатов

**Результаты AutoML:**
- AutoGluon и H2O показали результаты сопоставимые с кастомными моделями
- Лучший результат достигнут кастомным GradientBoosting (F1=79.5%)
- AutoML подтвердил выбор базовых алгоритмов для данного датасета

### ⚙️ Автоматизация пайплайна

**Автоматизация обучения:**
- `train_full_pipeline()` — единый метод: загрузка → обучение → тюнинг → AutoML → сравнение → сохранение
- `GridSearchCV` — автоподбор гиперпараметров для XGBoost и RandomForest
- MLflow — автологирование параметров, метрик и моделей

**Автоматизация отчётов:**
- `scripts/generate_visualizations.py` — генерация всех графиков (6 PNG)
- `scripts/generate_training_report.py` — HTML отчёт с метриками
- `scripts/generate_drift_report.py` — HTML/JSON отчёты Evidently AI
- `reports/index.html` — автоматическое обновление индекса отчётов

**Автоматизация деплоя (CI/CD):**
- GitHub Actions: линтинг → тесты → сборка Docker → деплой на GitHub Pages
- `deploy-reports.yml` — автопубликация отчётов при push в main

### Итоговое сравнение

| Модель | Accuracy | F1-score | ROC AUC | Precision | Recall |
|--------|----------|----------|---------|-----------|--------|
| **GradientBoosting** | **91.1%** | **79.5%** | **97.5%** | 86.8% | 73.3% |
| XGBoost | 89.5% | 76.2% | 97.0% | 82.1% | 71.1% |
| XGBoost (Tuned) | 89.5% | 76.2% | 96.8% | 82.1% | 71.1% |
| RandomForest | 88.5% | 73.8% | 95.6% | 79.5% | 68.9% |
| RandomForest (Tuned) | 88.5% | 73.8% | 96.0% | 79.5% | 68.9% |
| KNeighbors | 86.9% | 63.8% | 91.7% | 91.7% | 48.9% |
| LogisticRegression | 83.2% | 54.3% | 84.7% | 76.0% | 42.2% |
| SVC | 76.4% | 0.0% | 85.7% | 0.0% | 0.0% |

**Вывод:** GradientBoosting показал наилучший баланс качества по F1-score (79.5%) и ROC AUC (97.5%).

**Лучшая модель для продакшена:** GradientBoosting — оптимальное качество предсказания оттока.

### Визуализации и отчёты

Все графики и отчёты доступны в папке `reports/`:

**Графики моделей:**
- 📊 [Сравнение моделей](reports/model_comparison.png) — Accuracy, F1-Score, ROC AUC, Precision для всех моделей
- 📊 [Сравнение с AutoML](reports/model_comparison_with_automl.png) — детальное сравнение моделей
- 📊 [Лидерборд AutoML](reports/automl_leaderboard.png) — таблица результатов с гиперпараметрами

**Анализ данных:**
- 📊 [Распределение данных](reports/data_distribution.png) — распределение целевой переменной, возраста, услуг и корреляционная матрица
- 📊 [Важность признаков](reports/feature_importance.png) — корреляция признаков с целевой переменной
- 📊 [Анализ оттока](reports/churn_analysis.png) — отток по возрасту, услугам и распределение Churn

**Отчёты:**
- 📄 [Отчёт об обучении (HTML)](reports/training_report.html) — подробный отчёт с метриками и временем обучения
- 📄 [Результаты обучения (CSV)](reports/training_results.csv) — сырые данные результатов
- 📄 [Сравнение моделей (CSV)](reports/model_comparison_full.csv) — полные данные для анализа

**Онлайн-версия:** Все отчеты доступны на [GitHub Pages](https://vikalinet.github.io/travel-churn-prediction/)

## 🧪 Тестирование

Проект покрыт тестами с использованием **pytest** и **pytest-cov**.

### Запуск тестов

```bash
# Все тесты с подробным выводом
pytest tests/ -v

# С измерением покрытия кода
pytest tests/ -v --cov=src --cov-report=html

# Конкретный файл тестов
pytest tests/test_preprocessing.py -v
pytest tests/test_integration.py -v
```

### Unit-тесты (`tests/test_preprocessing.py`)

**Тесты ETL (`TestDataExtractor`, `TestDataTransformer`):**
- ✅ Успешная загрузка CSV-файла
- ✅ Обработка несуществующего файла (`FileNotFoundError`)
- ✅ Обработка пропущенных значений (медиана/мода)
- ✅ Генерация новых признаков (`create_features`)
- ✅ Кодирование категориальных переменных (`LabelEncoder`)

**Тесты модели (`TestModelPrediction`):**
- ✅ Формат предсказаний (бинарный: 0/1)
- ✅ Формат вероятностей (shape `(n, 2)`, значения в [0, 1])
- ✅ Использование `unittest.mock` для изоляции тестов

**Тесты валидации (`TestDataValidation`):**
- ✅ Наличие обязательных колонок
- ✅ Корректность типов данных
- ✅ Допустимый процент пропусков (< 50%)

### Интеграционные тесты (`tests/test_integration.py`)

**Полный пайплайн (`TestFullPipeline`):**
- ✅ ETL: загрузка → обработка → сохранение `processed_data.csv`
- ✅ Обучение: `ModelTrainer.load_data()` → `prepare_data()` → `fit()`
- ✅ End-to-end: данные → модель → предсказание для нового клиента

**API (`TestAPIIntegration`) — FastAPI `TestClient`:**
- ✅ `GET /health` — проверка статуса сервиса
- ✅ `GET /` — главная страница с сообщением
- ✅ `POST /predict` — предсказание оттока с валидацией входных данных
- ✅ Мок модели для изоляции API-тестов

**Сохранение моделей (`TestModelPersistence`):**
- ✅ Сериализация / десериализация через `joblib`
- ✅ Идентичность предсказаний до и после сохранения

**MLflow (`TestMLflowIntegration`):**
- ✅ Логирование метрик (`accuracy`, `f1_score`, `roc_auc`)
- ✅ Логирование модели через `mlflow.sklearn.log_model`
- ✅ SQLite backend для кроссплатформенности

### Покрытие кода

| Метрика | Значение |
|---------|----------|
| **Библиотека** | pytest + pytest-cov |
| **Целевое покрытие** | `src/` (все модули) |
| **Отчёт** | HTML + XML (для Codecov) |
| **CI/CD** | Автозапуск при push/PR |

### CI/CD интеграция

Тесты запускаются автоматически в GitHub Actions:
1. Установка зависимостей (`requirements.txt`)
2. Линтинг (`flake8`)
3. Проверка форматирования (`black`)
4. **Запуск тестов** (`pytest tests/ -v --cov=src --cov-report=xml`)
5. Загрузка покрытия в Codecov
6. Сборка Docker (только после успешных тестов)

## 🐳 Docker & Docker Compose

Запуск всех сервисов одной командой:

```bash
docker-compose up --build
```

**Сервисы:**
- `web` — FastAPI сервис с моделью (порт 8000)
- `mlflow` — сервер для экспериментов (порт 5000)

## 🔄 CI/CD (GitHub Actions)

Автоматический пайплайн при push/pull request в `main`:

1. **Checkout** — клонирование репозитория
2. **Setup Python** — установка Python 3.11
3. **Install dependencies** — `pip install -r requirements.txt`
4. **Lint** — проверка кода `flake8`
5. **Format check** — проверка `black`
6. **Tests** — `pytest tests/ -v --cov=src --cov-report=xml`
7. **Coverage** — загрузка отчёта в Codecov
8. **Docker build** — сборка образа (только при push в `main`)

**Workflow файлы:**
- `.github/workflows/ci-cd.yml` — основной пайплайн
- `.github/workflows/deploy-reports.yml` — деплой отчётов на GitHub Pages

## 🔗 Ссылки на отчеты

📊 **GitHub Pages с отчетами:** [https://vikalinet.github.io/travel-churn-prediction/](https://vikalinet.github.io/travel-churn-prediction/)

Автоматически генерируемые отчеты доступны по ссылке выше после каждого деплоя в `main` ветку.

---

## 📊 Мониторинг

### MLflow
- Логирование параметров, метрик и артефактов
- Реестр моделей
- Эксперименты доступны по умолчанию в локальной версии

### Evidently AI - Мониторинг дрейфа данных

**Цель:** Отслеживание дрейфа данных (data drift) — изменения распределения признаков во времени.

**Метод:** KS-тест (Kolmogorov-Smirnov test) для числовых признаков.

**Результаты последнего анализа:**

| Признак | Статус | KS-статистика | p-value |
|---------|--------|---------------|---------|
| Age | ✅ OK | 0.0361 | 0.9834 |
| FrequentFlyer | ✅ OK | 0.0148 | 1.0000 |
| AnnualIncomeClass | ✅ OK | 0.0149 | 1.0000 |
| ServicesOpted | ✅ OK | 0.0237 | 1.0000 |
| AccountSyncedToSocialMedia | ✅ OK | 0.0332 | 0.9935 |
| BookedHotelOrNot | ✅ OK | 0.0479 | 0.8535 |

**Вывод:** Дрейф не обнаружен! Данные стабильны.

**Отчёты:**
- HTML отчёт: [evidently_reports/drift_report.html](evidently_reports/drift_report.html)
- JSON сводка: [evidently_reports/drift_summary.json](evidently_reports/drift_summary.json)
- **Онлайн:** [GitHub Pages Reports](https://vikalinet.github.io/travel-churn-prediction/)

**Запуск мониторинга:**
```bash
# Генерация отчёта о дрейфе
python scripts/generate_drift_report.py

# Или через исходный скрипт
python src/monitoring/drift_monitor_customer.py data/processed/processed_data.csv
```

### Мониторинг обучения

**Результаты обучения моделей:**

| Модель | Accuracy | F1-Score | ROC AUC | Precision | Recall |
|--------|----------|----------|---------|-----------|--------|
| GradientBoosting | 91.1% | 79.5% | 97.5% | 86.8% | 73.3% |
| XGBoost | 89.5% | 76.2% | 97.0% | 82.1% | 71.1% |
| XGBoost (Tuned) | 89.5% | 76.2% | 96.8% | 82.1% | 71.1% |
| RandomForest | 88.5% | 73.8% | 95.6% | 79.5% | 68.9% |
| RandomForest (Tuned) | 88.5% | 73.8% | 96.0% | 79.5% | 68.9% |
| KNeighbors | 86.9% | 63.8% | 91.7% | 91.7% | 48.9% |
| LogisticRegression | 83.2% | 54.3% | 84.7% | 76.0% | 42.2% |
| SVC | 76.4% | 0.0% | 85.7% | 0.0% | 0.0% |

- 📄 [HTML отчёт об обучении](reports/training_report.html)
- 📄 [CSV результаты](reports/training_results.csv)
- 📊 [Сравнение моделей (PNG)](reports/model_comparison.png)

## 📁 Структура проекта

```
.
├── data/
│   ├── raw/              # Сырые данные
│   └── processed/        # Обработанные данные
├── src/
│   ├── api/              # FastAPI приложение
│   ├── etl/              # ETL пайплайн
│   ├── models/           # Код моделей
│   ├── training/         # Скрипты обучения
│   │   ├── customer_travel_training.py  # Кастомные модели + GridSearchCV
│   │   ├── automl_training.py           # AutoGluon AutoML
│   │   ├── automl_alternative.py        # H2O AutoML + VotingClassifier
│   │   └── model_training.py            # Универсальный пайплайн
│   ├── monitoring/       # Мониторинг дрейфа данных
│   └── utils/            # Утилиты
├── tests/                # Тесты (pytest)
├── scripts/              # Вспомогательные скрипты
│   ├── generate_visualizations.py   # Автогенерация графиков
│   ├── generate_drift_report.py     # Автогенерация отчёта о дрейфе
│   └── generate_training_report.py  # Автогенерация отчёта об обучении
├── .github/
│   └── workflows/        # CI/CD пайплайны
├── reports/              # Визуализации и отчёты
├── evidently_reports/    # Отчёты мониторинга
├── models/               # Сохранённые модели
├── requirements.txt
├── Dockerfile            # Docker образ
├── docker-compose.yml    # Docker Compose конфигурация
├── DOCKER_GUIDE.md       # Руководство по Docker
├── presentation.html     # HTML презентация проекта
└── README.md             # Документация
```

## 🚀 Быстрый старт

### Локальная установка

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Запуск FastAPI
uvicorn src.api.main:app --reload

# Запуск тестов
pytest tests/ -v --cov=src

# Запуск через Docker
docker-compose up --build
```

## 📈 API Endpoints

- `GET /` — Главная страница
- `POST /predict` — Предсказание оттока
- `GET /health` — Проверка здоровья сервиса

## 📽️ Презентация

Презентация проекта доступна в HTML формате:

- **🎬 HTML:** [presentation.html](presentation.html) — открывается в браузере
- **🌐 Онлайн:** [GitHub Pages Presentation](https://vikalinet.github.io/travel-churn-prediction/presentation.html)

**Содержание презентации (8 слайдов):**
1. Титульный слайд
2. Бизнес-задача и цели
3. Описание данных и признаков
4. Архитектура ML-системы
5. Результаты моделей и тестирование
6. Мониторинг и CI/CD
7. Ключевые выводы для бизнеса
8. GitHub репозиторий и контакты

## 📝 Команды Git

Основные команды, использованные в проекте:

```bash
# Инициализация репозитория
git init

# Добавление файлов
git add .

# Коммит изменений
git commit -m "feat: добавлена визуализация"

# Создание ветки main
git branch -M main

# Привязка к удалённому репозиторию
git remote add origin https://github.com/vikalinet/travel-churn-prediction.git

# Отправка изменений
git push -u origin main

# Создание новой ветки для разработки
git checkout -b feature/visualization

# Слияние веток
git checkout main
git merge feature/visualization
```

## 📄 Лицензия

Проект создан в учебных целях.
