# 🎯 Прогнозирование оттока клиентов туристического агентства

[![CI/CD](https://github.com/vikalinet/travel-churn-prediction/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/vikalinet/travel-churn-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🔗 **GitHub репозиторий:** [https://github.com/vikalinet/travel-churn-prediction](https://github.com/vikalinet/travel-churn-prediction)

📊 **Отчёты и визуализации:** [https://vikalinet.github.io/travel-churn-prediction](https://vikalinet.github.io/travel-churn-prediction)

🎬 **Презентация проекта:** [Смотреть презентацию](https://vikalinet.github.io/travel-churn-prediction/presentation.html)

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
- `ServicesOpted` — количество выбранных услуг (1-6)
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

Для улучшения качества моделей был проведён подбор гиперпараметров с помощью **Optuna** (байесовская оптимизация):

**XGBoost Tuned (Optuna, 30 trials):**
- Подбираемые параметры: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- Оптимизатор: TPE (Tree-structured Parzen Estimator) — адаптивный поиск
- Ранняя остановка: отсечение неперспективных trials
- Результат: F1=76.2% (сопоставимо с базовой версией — ограничение малого датасета)

**RandomForest Tuned (Optuna, 30 trials):**
- Подбираемые параметры: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- Результат: F1=73.8% (сопоставимо с базовой версией)

> **Преимущества Optuna над GridSearchCV:**
> - Адаптивный поиск (TPE) вместо полного перебора
> - Ранняя остановка — отсечение неудачных конфигураций
> - Больше параметров и диапазонов без роста времени
> - На больших датасетах даёт значительный прирост качества

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
- `Optuna` — байесовская оптимизация гиперпараметров для XGBoost и RandomForest
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
| AutoGluon AutoML | 89.5% | 76.2% | 96.8% | 82.1% | 71.1% |
| H2O AutoML | 89.0% | 74.5% | 96.2% | 80.5% | 69.5% |
| KNeighbors | 86.9% | 63.8% | 91.7% | 91.7% | 48.9% |
| LogisticRegression | 83.2% | 54.3% | 84.7% | 76.0% | 42.2% |
| SVC | 76.4% | 0.0% | 85.7% | 0.0% | 0.0% |

**Вывод:** GradientBoosting показал наилучший баланс качества по F1-score (79.5%) и ROC AUC (97.5%). AutoGluon AutoML достиг сопоставимого качества с XGBoost (F1=76.2%), подтвердив эффективность ансамблевых методов для данной задачи.

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

### Pre-commit hooks

Для автоматической проверки кода перед коммитом настроены pre-commit hooks:

```bash
# Установка pre-commit
pip install pre-commit
pre-commit install

# Проверка всех файлов
pre-commit run --all-files
```

**Настроенные hooks:**
- `black` — автоматическое форматирование кода
- `flake8` — проверка стиля и ошибок
- `trailing-whitespace` — удаление лишних пробелов
- `end-of-file-fixer` — добавление пустой строки в конце файлов

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

### Dockerfile

**Описание команд:**

| Команда | Функция |
|---------|---------|
| `FROM python:3.11-slim` | Базовый образ — минимизированный Python 3.11 для уменьшения размера контейнера |
| `WORKDIR /app` | Создание рабочей директории внутри контейнера |
| `COPY requirements.txt .` | Копирование списка зависимостей для кэширования |
| `RUN pip install --no-cache-dir` | Установка Python-пакетов без кэша (оптимизация размера) |
| `COPY . .` | Копирование кода проекта в контейнер |
| `RUN useradd -m -u 1000 appuser` | Создание нерута пользователя для безопасности |
| `USER appuser` | Запуск приложения от непривилегированного пользователя |
| `EXPOSE 8000` | Документирование порта для FastAPI |
| `CMD ["uvicorn", ...]` | Команда запуска веб-сервера |

**Оптимизации:**
- `--no-cache-dir` в pip — уменьшение размера образа
- `slim` версия Python — минимизация уязвимостей
- **Multi-stage build** — сборка зависимостей в отдельном `builder` этапе, финальный образ содержит только установленные пакеты и код приложения (экономия ~300-500 МБ)

### docker-compose.yml

**Ограничения ресурсов:**

| Сервис | CPU | Memory |
|--------|-----|--------|
| web | 0.5-1 ядро | 1-2 GB |
| mlflow | 0.5-1 ядро | 1-2 GB |

**Функции контейнеризации:**

| Аспект | Реализация |
|--------|------------|
| **Безопасность** | Запуск от нерута пользователя (`appuser`) |
| **Изоляция** | Отдельные контейнеры для API и MLflow |
| **Ресурсы** | Лимиты CPU и памяти через `deploy.resources` |
| **Хранение данных** | Persistent volume для MLflow (`mlflow_data`) |
| **Сеть** | Внутренняя сеть Docker, проброс портов на хост |
| **Зависимости** | `depends_on` для порядка запуска сервисов |
| **Масштабирование** | Легкое развёртывание на новых серверах |

**Полное описание:** [DOCKER_GUIDE.md](DOCKER_GUIDE.md)

## 🔄 CI/CD (GitHub Actions)

Полная реализация непрерывной интеграции и доставки (CI/CD) с автоматизацией всех этапов разработки.

### Архитектура пайплайна

**Триггеры:**
- `push` в ветки `main`, `develop`
- `pull_request` в ветку `main`
- Ручной запуск (`workflow_dispatch`)

### CI/CD Пайплайн 1: `ci-cd.yml` (Тесты и сборка Docker)

| Шаг | Описание | Команды/Actions |
|-----|----------|-----------------|
| 1. Checkout | Клонирование репозитория | `actions/checkout@v4` |
| 2. Setup Python | Установка Python 3.11 | `actions/setup-python@v5` |
| 3. Install deps | Установка зависимостей | `pip install -r requirements.txt` |
| 4. Linting | Проверка кода | `flake8 src tests --count` |
| 5. Format check | Проверка форматирования | `black --check src tests` |
| 6. Type check | Проверка типов | `mypy src` |
| 7. Tests | Запуск тестов | `pytest tests/ -v --cov=src --cov-report=xml` |
| 8. Coverage | Загрузка покрытия | `codecov/codecov-action@v3` |
| 9. Docker build | Сборка образа | `docker/build-push-action@v5` |
| 10. Docker push | Публикация в Docker Hub | `docker/build-push-action@v5` (при `release:` коммите) |

**Условия:**
- Docker собирается только при `push` в `main`
- Job `docker` зависит от `test` (неудачные тесты → блокировка сборки)
- Push в Docker Hub выполняется только при коммите с префиксом `release:` (например, `release: v1.2.0`)

### CI/CD Пайплайн 2: `deploy-reports.yml` (GitHub Pages)

| Шаг | Описание | Команды/Actions |
|-----|----------|-----------------|
| 1. Checkout | Клонирование репозитория | `actions/checkout@v4` |
| 2. Setup Python | Установка Python 3.11 | `actions/setup-python@v5` |
| 3. Install deps | Установка ML библиотек | `pip install pandas numpy ...` |
| 4. Generate visualizations | Создание графиков | `python scripts/generate_visualizations.py` |
| 5. Generate training report | Отчёт об обучении | `python scripts/generate_training_report.py` |
| 6. Generate drift report | Отчёт о дрейфе | `python scripts/generate_drift_report.py` |
| 7. Copy reports | Копирование отчётов | `cp -r evidently_reports/* reports/` |
| 8. Update index | Обновление даты | `sed -i 's/25.05.2026/$(date +%d.%m.%Y)/g'` |
| 9. Copy presentation | Копирование презентации | `cp presentation.html reports/` |
| 10. Setup Pages | Настройка GitHub Pages | `actions/configure-pages@v5` |
| 11. Upload artifact | Загрузка артефактов | `actions/upload-pages-artifact@v3` |
| 12. Deploy | Публикация | `actions/deploy-pages@v4` |

**Триггеры:**
- `push` в `main` при изменении `reports/**`, `evidently_reports/**`
- Ручной запуск через GitHub UI

### Git Flow: Изменения с локальной машины на удалённый сервис

**Рабочий процесс:**

```bash
# 1. Инициализация репозитория (первый запуск)
git init

# 2. Создание основной ветки
git branch -M main

# 3. Привязка к удалённому репозиторию
git remote add origin https://github.com/vikalinet/travel-churn-prediction.git

# 4. Добавление файлов в индекс
git add .                              # Все файлы
git add src/api/main.py               # Конкретный файл
git add -A                             # Все изменения (включая удалённые)

# 5. Коммит с описанием
git commit -m "feat: добавлена визуализация моделей"
git commit -m "fix: исправлена ошибка в API"
git commit -m "docs: обновлён README"
git commit -m "test: добавлены unit-тесты"

# 6. Отправка в удалённый репозиторий
git push -u origin main                # Первый пуш с установкой upstream
git push                               # Последующие push

# 7. Работа с ветками (feature development)
git checkout -b feature/new-model      # Создание и переход в ветку
git checkout develop                   # Переход в develop
git branch -a                          # Показать все ветки

# 8. Слияние веток
git checkout main
git merge feature/new-model            # Слияние с main
git branch -d feature/new-model        # Удаление локальной ветки
git push origin main                   # Пуш слияния

# 9. Работа с pull request
git checkout -b feature/visualization
# ... разработка ...
git add .
git commit -m "feat: добавлены графики"
git push origin feature/visualization  # Пуш feature-ветки
# Создаётся PR через GitHub UI
# После мержа в main запускается CI/CD

# 10. Откат изменений (при необходимости)
git reset --hard HEAD~1                # Откат последнего коммита
git revert <commit-hash>               # Создание коммита-отмены
git checkout <commit-hash>             # Просмотр состояния на коммите

# 11. Работа с тегами (версионирование)
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin --tags

# 12. Просмотр истории
git log --oneline                      # Краткая история
git log --graph --oneline --all        # Графическая история
git diff HEAD~1 HEAD                   # Разница между коммитами
git status                             # Статус рабочей директории
```

**Типы коммитов (Conventional Commits):**
- `feat:` — новая функция
- `fix:` — исправление ошибки
- `docs:` — изменения документации
- `test:` — добавление/изменение тестов
- `refactor:` — рефакторинг кода
- `chore:` — изменения сборки/инструментов
- `perf:` — улучшения производительности

**Локальный цикл разработки:**
```
1. git pull origin main           # Синхронизация с удалённой версией
2. git checkout -b feature/xxx    # Создание ветки для задачи
3. ... кодирование ...
4. git add . && git commit -m "..."
5. pytest tests/ -v               # Локальный запуск тестов
6. git push origin feature/xxx    # Пуш в удалённый репозиторий
7. Создать Pull Request в GitHub
8. CI/CD автоматически запускает тесты
9. После мержа → деплой в production
```

**Результаты CI/CD:**
- **GitHub Pages:** https://vikalinet.github.io/travel-churn-prediction/
- **Автоматическое обновление** отчётов при каждом push в `main`
- **Docker образ** собирается при успешном прохождении всех тестов

## 🔗 Ссылки на отчеты

📊 **GitHub Pages с отчетами:** [https://vikalinet.github.io/travel-churn-prediction/](https://vikalinet.github.io/travel-churn-prediction/)

Автоматически генерируемые отчеты доступны по ссылке выше после каждого деплоя в `main` ветку.

---

## 📊 Мониторинг

Полная система мониторинга ML-системы включает контроль качества данных, дрейф признаков, метрики моделей и инфраструктуру.

### 📈 Мониторинг качества модели

#### 1. MLflow — Версионирование и логирование

**Функции:**
- Логирование параметров модели (гиперпараметры, алгоритмы)
- Логирование метрик (Accuracy, F1-Score, ROC AUC, Precision, Recall)
- Сохранение артефактов (файлы моделей `.pkl`)
- Реестр моделей с версионированием
- Сравнение экспериментов в UI

**Локальный запуск:**
```bash
# Запуск MLflow сервера
mlflow ui --host 0.0.0.0 --port 5000

# Открыть в браузере: http://localhost:5000
```

**Логирование в коде:**
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("model", "GradientBoosting")
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", 0.911)
    mlflow.log_metric("f1_score", 0.795)
    mlflow.sklearn.log_model(model, "model")
```

**Результаты:**
- Эксперименты доступны по умолчанию в локальной версии
- Все метрики и параметры записываются автоматически

#### 2. Evidently AI — Мониторинг дрейфа данных

**Цель:** Обнаружение data drift — изменения распределения признаков во времени.

**Метод:** KS-тест (Kolmogorov-Smirnov test) для числовых признаков.

**Порог значимости:** p-value < 0.05 → дрейф обнаружен ⚠️

**Результаты последнего анализа:**

| Признак | Статус | KS-статистика | p-value | Интерпретация |
|---------|--------|---------------|---------|---------------|
| Age | ✅ OK | 0.0361 | 0.9834 | Распределение стабильно |
| FrequentFlyer | ✅ OK | 0.0148 | 1.0000 | Распределение стабильно |
| AnnualIncomeClass | ✅ OK | 0.0149 | 1.0000 | Распределение стабильно |
| ServicesOpted | ✅ OK | 0.0237 | 1.0000 | Распределение стабильно |
| AccountSyncedToSocialMedia | ✅ OK | 0.0332 | 0.9935 | Распределение стабильно |
| BookedHotelOrNot | ✅ OK | 0.0479 | 0.8535 | Распределение стабильно |

**Вывод:** Дрейф **не обнаружен**! Данные стабильны, модель актуальна.

**Отчёты:**
- 📄 HTML отчёт: [evidently_reports/drift_report.html](evidently_reports/drift_report.html)
- 📄 JSON сводка: [evidently_reports/drift_summary.json](evidently_reports/drift_summary.json)
- 🌐 **Онлайн:** [GitHub Pages Reports](https://vikalinet.github.io/travel-churn-prediction/)

**Запуск мониторинга:**
```bash
# Генерация отчёта о дрейфе
python scripts/generate_drift_report.py

# Или через исходный скрипт
python src/monitoring/drift_monitor_customer.py data/processed/processed_data.csv
```

**Дополнительный анализ:**
- Корреляционная матрица признаков
- Распределение целевой переменной (Churn: ~40%)
- Визуализация гистограмм для каждого признака
- Сравнение reference и current датасетов

#### 3. Контроль качества данных

**Валидация входных данных:**
- Проверка наличия обязательных колонок
- Типы данных (числовые, категориальные)
- Отсутствие критических пропусков (< 50%)
- Диапазоны значений (age: 18-70, income: 20000-150000)

**Предобработка:**
- Заполнение пропусков: медиана (числовые), мода (категориальные)
- Кодирование категорий: LabelEncoder
- Генерация новых признаков: travel_frequency_score

#### 4. Анализ метрик модели

**Результаты обучения 8 моделей:**

| Модель | Accuracy | F1-Score | ROC AUC | Precision | Recall | Время обучения |
|--------|----------|----------|---------|-----------|--------|----------------|
| **GradientBoosting** | **91.1%** | **79.5%** | **97.5%** | 86.8% | 73.3% | ~2 сек |
| XGBoost | 89.5% | 76.2% | 97.0% | 82.1% | 71.1% | ~1 сек |
| XGBoost (Tuned) | 89.5% | 76.2% | 96.8% | 82.1% | 71.1% | ~30 сек |
| RandomForest | 88.5% | 73.8% | 95.6% | 79.5% | 68.9% | ~1 сек |
| RandomForest (Tuned) | 88.5% | 73.8% | 96.0% | 79.5% | 68.9% | ~45 сек |
| KNeighbors | 86.9% | 63.8% | 91.7% | 91.7% | 48.9% | < 1 сек |
| LogisticRegression | 83.2% | 54.3% | 84.7% | 76.0% | 42.2% | < 1 сек |
| SVC | 76.4% | 0.0% | 85.7% | 0.0% | 0.0% | ~3 сек |

**Графики:**
- 📊 [Сравнение моделей (PNG)](reports/model_comparison.png)
- 📄 [HTML отчёт об обучении](reports/training_report.html)
- 📄 [CSV результаты](reports/training_results.csv)

**Вывод:** GradientBoosting выбран как лучшая модель — оптимальный баланс F1-score (79.5%) и ROC AUC (97.5%).

---

### 🖥️ Мониторинг инфраструктуры

#### 1. Docker — Ограничение ресурсов

**Конфигурация в `docker-compose.yml`:**

| Сервис | CPU (min-max) | Memory (min-max) |
|--------|---------------|------------------|
| web (FastAPI) | 0.5 - 1 ядро | 1 - 2 GB |
| mlflow (MLflow Server) | 0.5 - 1 ядро | 1 - 2 GB |

**Преимущества:**
- Защита от исчерпания ресурсов на хосте
- Гарантированная минимальная производительность
- Изоляция сервисов друг от друга

**Просмотр использования ресурсов:**
```bash
# Статистика контейнеров в реальном времени
docker stats

# Пример вывода:
# CONTAINER ID   NAME              CPU %     MEM USAGE / LIMIT
# abc123         web               0.15%     150MB / 2GB
# def456         mlflow            0.08%     120MB / 2GB
```

#### 2. Производительность API

**Скорость предсказания:**
- Среднее время ответа: < 100 мс
- Поддержка пакетных предсказаний (batch)
- Асинхронная обработка запросов (uvicorn)

**Загрузка:**
- Одиночные запросы: `POST /predict`
- Пакетные запросы: `POST /predict_batch` (до 100 клиентов за раз)

#### 3. Версионирование и реестр моделей

**MLflow Model Registry:**
- Версии моделей с метаданными
- Статусы: `None` → `Staging` → `Production`
- Откат к предыдущей версии при ухудшении метрик

**Сохранение моделей:**
```bash
# Локальное сохранение
models/best_model.pkl
models/xgboost_model.pkl
```

#### 4. Логирование и аудит

**Логи Docker:**
```bash
# Просмотр логов всех сервисов
docker-compose logs -f

# Логи конкретного сервиса
docker-compose logs -f web
docker-compose logs -f mlflow
```

**Формат логов:**
- Timestamp
- Уровень (INFO, WARNING, ERROR)
- Сообщение
- Request ID (для отслеживания запросов)

#### 5. CI/CD мониторинг

**GitHub Actions:**
- Успешность каждого шага пайплайна
- Время выполнения тестов
- Покрытие кода (codecov.io)
- Артефакты сборки (Docker образы)

**Метрики:**
- Среднее время выполнения пайплайна: ~3-5 минут
- Покрытие кода тестами: > 80%
- Количество успешных деплоев: автоматическое при каждом push в `main`

---

### 🔄 Цикл мониторинга

**Автоматический процесс:**

```
1. Запуск обучения → MLflow логирование
2. Сохранение модели → Версионирование
3. Генерация отчётов → Evidently AI
4. Деплой на GitHub Pages → Автоматическое обновление
5. Мониторинг дрейфа → Еженедельная проверка
6. При обнаружении дрейфа → Переобучение модели
```

**Регулярность:**
- **Дрейф данных:** Еженедельно или при поступлении новых данных
- **Переобучение модели:** Ежемесячно или при ухудшении метрик
- **Аудит инфраструктуры:** При каждом деплое

## 📑 Содержание

1. [Бизнес-задача](#-бизнес-задача)
2. [Описание пайплайна](#-описание-пайплайна)
3. [ETL Процесс](#-etl-процесс)
4. [Архитектура ML-модели](#-архитектура-ml-модели)
5. [AutoML](#-automl-автоматизированное-обучение)
6. [Метрики модели](#-метрики-модели)
7. [Тестирование](#-тестирование)
8. [Docker & Контейнеризация](#-docker--контейнеризация)
9. [CI/CD](#-cicd)
10. [Мониторинг](#-мониторинг)
11. [GitHub-репозиторий](#-github-репозиторий)
12. [Презентация](#-презентация)
13. [Быстрый старт](#-быстрый-старт)

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
│   │   ├── customer_travel_training.py  # Кастомные модели + Optuna тюнинг
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

# Запуск FastAPI с веб-интерфейсом
uvicorn src.api.main:app --reload

# Открыть в браузере
http://localhost:8000/

# Запуск тестов
pytest tests/ -v --cov=src

# Запуск через Docker
docker-compose up --build
```

### 📖 Документация по UI

Подробное руководство по использованию веб-интерфейса:
- **[UI_GUIDE.md](UI_GUIDE.md)** — полное руководство
- **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** — работа с Docker

## 📈 API Endpoints

- `GET /` — **Веб-интерфейс для предсказания** (новая!) 🎯
- `GET /docs` — Документация API
- `POST /predict` — Предсказание оттока (возвращает Precision, Recall и другие метрики)
- `POST /predict_batch` — Пакетное предсказание
- `GET /health` — Проверка здоровья сервиса
- `GET /models` — Информация о модели

### 🔍 Метрики модели в ответе API

При вызове `/predict` возвращается полная информация о качестве модели:

```json
{
  "prediction": 0,
  "probability": 0.25,
  "risk_level": "Low",
  "customer_data": {...},
  "metrics": {
    "accuracy": 0.911,    // Точность модели
    "precision": 0.868,   // Точность положительного класса
    "recall": 0.733,      // Полнота
    "f1_score": 0.795,    // F1-мера
    "roc_auc": 0.975      // Площадь под ROC-кривой
  }
}
```

**Precision (86.8%):** Из всех предсказанных как "уйдёт", 86.8% действительно ушли.
**Recall (73.3%):** Из всех реально ушедших, модель нашла 73.3%.

## 🌐 Веб-интерфейс

Создан удобный веб-интерфейс для тестирования API без использования curl или Postman:

### Функции интерфейса:

✅ **Форма ввода данных клиента**
- Все необходимые поля с валидацией
- Выпадающие списки для категориальных признаков
- Числовые поля с ограничением диапазона

✅ **Визуализация результатов**
- Цветовая индикация уровня риска (🟢 низкий, 🟡 средний, 🔴 высокий)
- Прогресс-бар вероятности оттока
- Персонализированные рекомендации для каждого уровня риска

✅ **Отображение данных клиента**
- Все введенные параметры в результатах
- Удобное форматирование значений

### Запуск:

```bash
# Запуск API сервера
uvicorn src.api.main:app --reload

# Открыть в браузере
http://localhost:8000/
```

### Скриншот интерфейса:

1. **Главная страница** — форма ввода данных клиента
2. **Результаты** — визуализация вероятности и рекомендации
3. **Документация** — страница `/docs` с описанием всех endpoints

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

## 🌐 Деплой на внешний хостинг

Полное руководство по бесплатному деплою проекта.

### 📚 Документация по деплою

| Документ | Описание |
|----------|----------|
| **[DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)** | Обзор всех бесплатных платформ |
| **[RENDER_DEPLOY.md](RENDER_DEPLOY.md)** | Пошаговая инструкция для Render.com |
| **[DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md)** | Запуск через Docker |

### 🚀 Рекомендуемый способ: Render.com

**Преимущества:**
- ✅ Полностью бесплатно (750 часов/мес)
- ✅ Автоматический деплой из GitHub
- ✅ HTTPS автоматически

**Быстрый старт:**

```bash
# 1. Добавить модель в репозиторий
git add models/best_model.pkl
git commit -m "add model for deployment"
git push origin main

# 2. Создать аккаунт на https://render.com
# 3. Новый сервис → Web Service → Выбрать репозиторий
# 4. Настройка:
#    Build: pip install -r requirements.txt
#    Start: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
# 5. Получить URL: https://travel-churn-prediction.onrender.com
```

**Полная инструкция:** [RENDER_DEPLOY.md](RENDER_DEPLOY.md)

---

## 📄 Лицензия

Проект создан в учебных целях.
