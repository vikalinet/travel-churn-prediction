# Прогнозирование оттока клиентов туристического агентства

[![CI/CD](https://github.com/vikalinet/travel-churn-prediction/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/vikalinet/travel-churn-prediction/actions)

🔗 **GitHub репозиторий:** [https://github.com/vikalinet/travel-churn-prediction](https://github.com/vikalinet/travel-churn-prediction)

## 🎯 Бизнес-задача

Разработать модель для выявления клиентов с высоким риском оттока, чтобы компания могла предложить им персонализированные скидки и программы лояльности, увеличив их удержание.

**Целевая метрика:** Увеличение удержания клиентов на 15-20% за счёт своевременного выявления групп риска.

**Ожидаемый эффект:**
- Сокращение расходов на удержание на 30%
- Рост повторных продаж на 15%
- Повышение удовлетворённости клиентов

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

| Модель | Accuracy | F1-score | ROC AUC | Время обучения |
|--------|----------|----------|---------|----------------|
| **GradientBoosting** | **91.1%** | **79.5%** | **97.5%** | 1.18 сек |
| XGBoost (Tuned) | 89.5% | 76.2% | 96.8% | 0.28 сек |
| RandomForest (Tuned) | 88.5% | 73.8% | 96.0% | 0.10 сек |
| Ensemble (AutoML) | 89.5% | 74.4% | 96.5% | 2.5 сек |
| KNeighbors | 86.9% | 63.8% | 91.7% | 0.05 сек |
| LogisticRegression | 83.2% | 54.3% | 84.7% | 0.03 сек |

**Лучшая модель:** GradientBoosting с F1-score = 79.5% и ROC AUC = 97.5%

**Кросс-валидация F1-score:** 72.4% (+/- 14.0%)

### AutoML модель (Ensemble)

Используется автоматизированный подбор моделей через:
- **AutoGluon** — автоматический подбор гиперпараметров
- **H2O AutoML** — альтернативный фреймворк
- **VotingClassifier** — ансамбль моделей (RandomForest + GradientBoosting + LogisticRegression)

**Результаты AutoML Ensemble:**

| Модель | Accuracy | F1-score | ROC AUC | Precision | Recall |
|--------|----------|----------|---------|-----------|--------|
| Ensemble (VotingClassifier) | 89.5% | 74.4% | 96.5% | 87.9% | 64.4% |

### Итоговое сравнение

| Модель | Accuracy | F1-score | ROC AUC | Время обучения |
|--------|----------|----------|---------|----------------|
| **GradientBoosting** | **91.1%** | **79.5%** | **97.5%** | 1.18 сек |
| XGBoost (Tuned) | 89.5% | 76.2% | 96.8% | 0.28 сек |
| Ensemble (AutoML) | 89.5% | 74.4% | 96.5% | 2.5 сек |
| RandomForest (Tuned) | 88.5% | 73.8% | 96.0% | 0.10 сек |
| KNeighbors | 86.9% | 63.8% | 91.7% | 0.05 сек |
| LogisticRegression | 83.2% | 54.3% | 84.7% | 0.03 сек |

**Вывод:** GradientBoosting показал наилучший баланс качества по F1-score (79.5%) и ROC AUC (97.5%).

**Лучшая модель для продакшена:** GradientBoosting — оптимальное качество предсказания оттока.

### Визуализации и отчёты

Все графики и отчёты доступны в папке `reports/`:

**Графики моделей:**
- 📊 [Сравнение моделей](reports/model_comparison.png) — Accuracy, F1-Score, ROC AUC для всех моделей
- 📊 [Сравнение с AutoML](reports/model_comparison_with_automl.png) — детальное сравнение кастомных моделей и AutoML Ensemble
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

### Unit-тесты
- Проверка функций предобработки
- Валидация входных данных

### Интеграционные тесты
- Полный пайплайн от данных до предсказания

### Model & Data tests
- Проверка загрузки модели
- Валидация формата предсказаний

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
1. Установка зависимостей
2. Запуск линтеров
3. Запуск тестов (pytest)
4. Сборка и пуш Docker-образа

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
- **Онлайн:** [GitHub Pages](https://vikalinet.github.io/travel-churn-prediction/)

**Запуск мониторинга:**
```bash
# Генерация отчёта о дрейфе
python scripts/generate_drift_report.py

# Или через исходный скрипт
python src/monitoring/drift_monitor_customer.py data/processed/processed_data.csv
```

### Мониторинг обучения

**Результаты измерения времени обучения:**

| Модель | Время обучения | Accuracy | F1-Score | ROC AUC |
|--------|----------------|----------|----------|---------|
| GradientBoosting | 1.18 сек | 91.1% | 79.5% | 97.5% |
| XGBoost (Tuned) | 0.28 сек | 89.5% | 76.2% | 96.8% |
| RandomForest (Tuned) | 0.10 сек | 88.5% | 73.8% | 96.0% |
| Ensemble (AutoML) | 2.5 сек | 89.5% | 74.4% | 96.5% |
| KNeighbors | 0.05 сек | 86.9% | 63.8% | 91.7% |
| LogisticRegression | 0.03 сек | 83.2% | 54.3% | 84.7% |

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
│   ├── training/         # Скрипты обучения (включая AutoML)
│   ├── monitoring/       # Мониторинг дрейфа данных
│   └── utils/            # Утилиты
├── tests/                # Тесты (pytest)
├── scripts/              # Вспомогательные скрипты
│   ├── generate_visualizations.py   # Генерация графиков
│   ├── generate_drift_report.py     # Отчёт о дрейфе
│   └── generate_training_report.py  # Отчёт об обучении
├── .github/
│   └── workflows/        # CI/CD пайплайны
├── reports/              # Визуализации и отчёты
├── evidently_reports/    # Отчёты мониторинга
├── models/               # Сохранённые модели
├── requirements.txt
├── Dockerfile            # Docker образ
├── docker-compose.yml    # Docker Compose конфигурация
├── DOCKER_GUIDE.md       # Руководство по Docker
├── PRESENTATION.md       # Презентация проекта
├── presentation.html     # HTML версия презентации
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

Презентация проекта доступна в двух форматах:

- **Markdown:** [PRESENTATION.md](PRESENTATION.md) — текстовая версия для редактирования
- **HTML:** [presentation.html](presentation.html) — открывается в браузере

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
