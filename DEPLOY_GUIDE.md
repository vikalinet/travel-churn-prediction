# 🌐 Деплой проекта на Railway

Руководство по деплою ML-проекта на платформу Railway.

---

## 📋 Содержание

1. [Обзор платформы Railway](#обзор-платформы-railway)
2. [Подготовка репозитория](#подготовка-репозитория)
3. [Создание аккаунта](#создание-аккаунта)
4. [Деплой](#деплой)
5. [Получение URL](#получение-url)
6. [Проверка](#проверка)
7. [Устранение проблем](#устранение-проблем)

---

## 🎯 Обзор платформы Railway

| Параметр | Значение |
|----------|----------|
| **Бесплатный tier** | ✅ $5/мес кредит |
| **Ограничения** | 500 часов CPU |
| **Sleep-режим** | ❌ Нет |
| **HTTPS** | ✅ Автоматически |
| **Сложность** | ⭐ Легко |

### Преимущества
- ✅ Простая настройка
- ✅ Автоматический деплой из GitHub
- ✅ Нет sleep-режима
- ✅ HTTPS автоматически

### Недостатки
- ⚠️ Ограничение 500 часов CPU
- ⚠️ После $5 — платно

---

## 🔧 Подготовка репозитория

### 1. Добавьте файл `railway.json`

Создайте файл в корне проекта (уже создан):

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### 2. Проверьте структуру репозитория

```
travel-churn-prediction/
├── models/
│   └── best_model.pkl       # Обученная модель
├── requirements.txt         # Зависимости
├── railway.json             # Конфиг для Railway
└── src/api/main.py          # Точка входа
```

### 3. Добавьте модель в репозиторий

```bash
git add models/best_model.pkl
git commit -m "chore: add model for railway deploy"
git push origin main
```

---

## 🌐 Создание аккаунта

1. Перейдите на [https://railway.app](https://railway.app)
2. Нажмите **Start a New Project**
3. Авторизуйтесь через GitHub

---

## 🚀 Деплой

### Шаг 1: Создание проекта

1. Нажмите **New Project** → **Deploy from GitHub repo**
2. Выберите репозиторий `travel-churn-prediction`
3. Railway автоматически определит Python проект

### Шаг 2: Настройка переменных окружения (если нужны)

1. Откройте сервис → вкладка **Variables**
2. Добавьте при необходимости:
   ```
   MLFLOW_TRACKING_URI: sqlite:///./mlflow.db
   ```

### Шаг 3: Деплой

Railway автоматически запустит деплой после подключения репозитория.

**Статусы:**
- 🟡 Building — сборка образа
- 🟡 Deploying — развёртывание
- 🟢 Live — сервис готов

Время деплоя: 3–7 минут.

---

## 🔗 Получение URL

1. Откройте **Settings** → **Domains**
2. Скопируйте публичный URL вида:
   ```
   https://travel-churn-prediction.up.railway.app
   ```

---

## ✅ Проверка

### Проверка здоровья сервиса

```bash
curl https://travel-churn-prediction.up.railway.app/api/v1/health
```

Ожидаемый ответ:
```json
{"status": "healthy", "model_loaded": true}
```

### Тестовое предсказание

```bash
curl -X POST https://travel-churn-prediction.up.railway.app/api/v1/predict \
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

### Веб-интерфейс

Откройте в браузере:
```
https://travel-churn-prediction.up.railway.app/
```

---

## 🔄 Автоматический деплой

Railway автоматически делает деплой при каждом push в ветку `main`:

```bash
git add .
git commit -m "feat: update API"
git push origin main
# Railway автоматически пересоберёт и задеплоит
```

### Ручной деплой

1. Откройте проект на Railway
2. Выберите сервис
3. Нажмите **Deploy** → **Redeploy**

---

## ❓ Устранение проблем

### Проблема: Ошибка сборки

```bash
# Проверьте логи на Railway: вкладка Deploy Logs

# Частые ошибки:
# - requirements.txt не найден → должен быть в корне
# - models/best_model.pkl не найден → добавьте в git
# - Ошибка импорта → проверьте PYTHONPATH
```

### Проблема: Сервис не отвечает

```bash
# Проверьте Health Check
curl https://travel-churn-prediction.up.railway.app/health

# Проверьте логи на Railway: вкладка Logs
```

### Проблема: Модель не загружена

```bash
# Проверьте, что models/best_model.pkl в репозитории
git ls-files | grep best_model

# Если файла нет:
git add models/best_model.pkl
git commit -m "chore: add model for deploy"
git push origin main
```

### Проблема: Исчерпан лимит ($5 кредит)

**Решение:** Railway остановит сервис при исчерпании кредита. Для продолжения работы нужно пополнить баланс или перейти на платный тариф.

---

## 📞 Полезные ссылки

- **Railway Docs:** [https://docs.railway.app](https://docs.railway.app)
- **Railway GitHub:** [https://github.com/railwayapp](https://github.com/railwayapp)
- **Python Deployment:** [https://docs.railway.app/guides/languages/python](https://docs.railway.app/guides/languages/python)
- **Environment Variables:** [https://docs.railway.app/develop/variables](https://docs.railway.app/develop/variables)

---

## 📝 Чеклист деплоя

- [ ] Файл `railway.json` создан в корне
- [ ] `requirements.txt` в корне репозитория
- [ ] Модель `models/best_model.pkl` добавлена в git
- [ ] Аккаунт Railway создан и привязан к GitHub
- [ ] Проект создан и подключён к репозиторию
- [ ] Сервис запущен (статус 🟢 Live)
- [ ] Health Check возвращает `healthy`
- [ ] Тестовое предсказание работает

---

*Руководство актуально для Railway на момент создания проекта.*
