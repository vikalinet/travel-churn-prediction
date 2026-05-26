# 🌐 Деплой проекта на внешний хостинг (бесплатно)

Полное руководство по деплою ML-проекта на бесплатные платформы.

---

## 📋 Содержание

1. [Обзор бесплатных платформ](#обзор-бесплатных-платформ)
2. [Render.com](#rendercom)
3. [Railway.app](#railwayapp)
4. [Fly.io](#flyio)
5. [Hugging Face Spaces](#hugging-face-spaces)
6. [Google Cloud Run](#google-cloud-run)
7. [Oracle Cloud Free Tier](#oracle-cloud-free-tier)
8. [Сравнение платформ](#сравнение-платформ)

---

## 🎯 Обзор бесплатных платформ

| Платформа | Бесплатный tier | Ограничения | Сложность |
|-----------|-----------------|-------------|-----------|
| **Render.com** | ✅ 750 часов/мес | Sleep после 15 мин бездействия | ⭐ Легко |
| **Railway.app** | ✅ $5/мес кредит | 500 часов CPU | ⭐ Легко |
| **Fly.io** | ✅ $5/мес кредит | 3 VM 256MB | ⭐⭐ Средне |
| **Hugging Face** | ✅ Всегда бесплатно | Только CPU, 4GB RAM | ⭐ Легко |
| **Google Cloud Run** | ✅ $300 кредит 90 дней | После кредита платно | ⭐⭐⭐ Сложно |
| **Oracle Cloud** | ✅ Всегда бесплатно | 4 OCPU, 24GB RAM | ⭐⭐⭐ Сложно |

---

## 🟢 Render.com (Рекомендуется)

### Преимущества
- ✅ Полностью бесплатный tier
- ✅ Автоматический деплой из GitHub
- ✅ HTTPS автоматически
- ✅ Поддержка Python 3.11
- ✅ Простая настройка

### Недостатки
- ⚠️ Sleep после 15 мин бездействия (первый запрос ~30 сек)
- ⚠️ 750 часов/мес (хватает на 1 сервис)

### Пошаговая инструкция

#### Шаг 1: Подготовка репозитория

1. Создайте новый файл `render.yaml`:

```yaml
services:
  - type: web
    name: travel-churn-prediction
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: MLFLOW_TRACKING_URI
        sync: false
    disk:
      name: ml-models
      mountPath: /opt/render/project/src/models
      sizeGB: 1
```

2. Добавьте `requirements.txt` в корень репозитория
3. Убедитесь, что модель `models/best_model.pkl` добавлена в репозиторий

#### Шаг 2: Создание аккаунта

1. Зайдите на [https://render.com](https://render.com)
2. Нажмите **Get Started for Free**
3. Авторизуйтесь через GitHub

#### Шаг 3: Создание сервиса

1. Нажмите **New** → **Web Service**
2. Выберите **Connect a repository**
3. Выберите ваш репозиторий `travel-churn-prediction`
4. Настройте:
   ```
   Name: travel-churn-prediction
   Region: Frankfurt (ближе к Европе)
   Branch: main
   Root Directory: (оставить пустым)
   Runtime: Python
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
   ```

#### Шаг 4: Настройка бесплатного тарифа

1. Scroll down до **Instance Type**
2. Выберите **Free** (0$ / месяц)

#### Шаг 5: Переменные окружения (если нужны)

```
MLFLOW_TRACKING_URI: sqlite:///./mlflow.db
```

#### Шаг 6: Деплой

1. Нажмите **Create Web Service**
2. Дождитесь завершения деплоя (5-10 минут)
3. После деплоя получите URL: `https://travel-churn-prediction.onrender.com`

#### Шаг 7: Проверка

```bash
# Проверка здоровья
curl https://travel-churn-prediction.onrender.com/health

# Тестовое предсказание
curl -X POST https://travel-churn-prediction.onrender.com/predict \
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

---

## 🟡 Railway.app

### Преимущества
- ✅ Простая настройка
- ✅ Автоматический деплой
- ✅ $5 кредит каждый месяц
- ✅ Нет sleep режима

### Недостатки
- ⚠️ Ограничение 500 часов CPU
- ⚠️ После $5 — платно

### Пошаговая инструкция

#### Шаг 1: Подготовка

1. Создайте файл `railway.json` в корне:

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

#### Шаг 2: Создание аккаунта

1. Зайдите на [https://railway.app](https://railway.app)
2. Авторизуйтесь через GitHub

#### Шаг 3: Деплой

1. Нажмите **New Project** → **Deploy from GitHub repo**
2. Выберите репозиторий
3. Railway автоматически определит Python проект
4. После деплоя нажмите на сервис → **Variables**
5. Добавьте переменные (если нужны)

#### Шаг 4: Настройка PORT

Railway автоматически передаёт переменную `PORT` в контейнер.

#### Шаг 5: Получение URL

1. После деплоя откройте **Settings** → **Domains**
2. Скопируйте публичный URL

---

## 🔵 Fly.io

### Преимущества
- ✅ $5 кредит каждый месяц
- ✅ 3 VM 256MB бесплатно
- ✅ Глобальное распределение

### Недостатки
- ⚠️ Требует привязку карты
- ⚠️ Сложнее в настройке

### Пошаговая инструкция

#### Шаг 1: Установка CLI

```bash
# Windows (PowerShell)
winget install flyctl

# Linux
curl -L https://fly.io/install.sh | sh

# macOS
brew install flyctl
```

#### Шаг 2: Регистрация

```bash
fly auth signup
```

#### Шаг 3: Инициализация проекта

```bash
cd travel-churn-prediction
fly launch
```

#### Шаг 4: Настройка `fly.toml`

```toml
app = "travel-churn-prediction"
primary_region = "fra"

[build]

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = false
  auto_start_machines = true

[[vm]]
  size = "shared-cpu-1x"
  memory_mb = 256
```

#### Шаг 5: Деплой

```bash
fly deploy
```

#### Шаг 6: Получение URL

```bash
fly apps list
```

URL будет: `https://travel-churn-prediction.fly.dev`

---

## 🟣 Hugging Face Spaces (для ML моделей)

### Преимущества
- ✅ Полностью бесплатно
- ✅ Специализация на ML
- ✅ Простая интеграция с Gradio/Streamlit

### Недостатки
- ⚠️ Ограничение 4GB RAM
- ⚠️ Лучше подходит для Gradio/Streamlit UI

### Инструкция

1. Создайте аккаунт на [Hugging Face](https://huggingface.co/)
2. Создайте Space: **New Space**
3. Выберите **Gradio** или **Streamlit**
4. Создайте `app.py`:

```python
import gradio as gr
import requests

API_URL = "https://travel-churn-prediction.onrender.com/predict"

def predict_churn(age, frequent_flyer, income, services, sync, hotel):
    response = requests.post(API_URL, json={
        "age": age,
        "frequent_flyer": frequent_flyer,
        "annual_income_class": income,
        "services_opted": services,
        "account_synced_to_social_media": sync,
        "booked_hotel_or_not": hotel
    })
    return response.json()

gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Возраст"),
        gr.Dropdown(["Yes", "No"], label="Часто летает"),
        gr.Dropdown(["Low Income", "Middle Income", "High Income"], label="Доход"),
        gr.Slider(1, 6, label="Услуги"),
        gr.Dropdown(["Yes", "No"], label="Синхронизация"),
        gr.Dropdown(["Yes", "No"], label="Отели")
    ],
    outputs=gr.JSON(),
    title="Прогнозирование оттока клиентов"
).launch()
```

---

## 🟠 Google Cloud Run

### Преимущества
- ✅ $300 кредит на 90 дней
- ✅ Автоматическое масштабирование
- ✅ Глобальное распределение

### Недостатки
- ⚠️ После кредита платно
- ⚠️ Сложная настройка

### Инструкция

1. Создайте аккаунт на [Google Cloud](https://cloud.google.com/)
2. Создайте проект
3. Соберите Docker образ:

```bash
gcloud auth configure-docker
docker build -t gcr.io/PROJECT-ID/travel-churn .
docker push gcr.io/PROJECT-ID/travel-churn
```

4. Разверните на Cloud Run:

```bash
gcloud run deploy travel-churn \
  --image gcr.io/PROJECT-ID/travel-churn \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated
```

---

## 🟤 Oracle Cloud Free Tier

### Преимущества
- ✅ **Самый щедрый бесплатный tier**
- ✅ 4 OCPU, 24GB RAM навсегда
- ✅ 200GB дискового пространства

### Недостатки
- ⚠️ Сложная регистрация
- ⚠️ Требует привязку карты

### Инструкция

1. Создайте аккаунт на [Oracle Cloud](https://www.oracle.com/cloud/)
2. Создайте VM (Ubuntu 22.04)
3. Подключитесь по SSH:

```bash
ssh -i key ubuntu@<IP>
```

4. Установите Docker и запустите:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu
sudo docker-compose up -d
```

---

## ⚖️ Сравнение платформ

| Критерий | Render | Railway | Fly.io | Hugging Face | Oracle |
|----------|--------|---------|--------|--------------|--------|
| **Сложность** | ⭐ | ⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐ |
| **Бесплатно** | ✅ | ✅$5 | ✅$5 | ✅ | ✅ навсегда |
| **Sleep** | ⚠️ 15 мин | ❌ | ❌ | ❌ | ❌ |
| **HTTPS** | ✅ | ✅ | ✅ | ✅ | ⚠️ вручную |
| **GPU** | ❌ | ❌ | ❌ | ✅ (платно) | ✅ (огр.) |
| **RAM** | 512MB | 512MB | 256MB | 4GB | 24GB |

---

## 🚀 Рекомендация

### Для быстрой демонстрации
**Render.com** — самый простой вариант

### Для продакшена без sleep
**Railway.app** — $5 кредит хватает на тесты

### Для полного контроля
**Oracle Cloud** — 24GB RAM навсегда

---

## 📞 Дополнительная информация

- **Render:** [https://render.com/docs](https://render.com/docs)
- **Railway:** [https://docs.railway.app](https://docs.railway.app)
- **Fly.io:** [https://fly.io/docs](https://fly.io/docs)
- **Hugging Face:** [https://huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
