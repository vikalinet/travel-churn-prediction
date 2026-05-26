# 🚀 Деплой на Render.com — Пошаговая инструкция

Самый простой способ бесплатно разместить ML-приложение.

---

## 📋 Что понадобится

- Аккаунт GitHub
- Аккаунт на [Render.com](https://render.com)
- Модель, сохранённая в `models/best_model.pkl`

---

## 🔧 Шаг 1: Подготовка репозитория

### 1.1. Добавьте файл `render.yaml`

Создайте файл в корне проекта:

```yaml
services:
  - type: web
    name: travel-churn-prediction
    env: python
    region: frankfurt
    branch: main
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
    disk:
      name: model-storage
      mountPath: /opt/render/project/src/models
      sizeGB: 1
```

### 1.2. Проверьте структуру проекта

```
travel-churn-prediction/
├── src/
│   └── api/
│       └── main.py          # FastAPI приложение
├── models/
│   └── best_model.pkl       # Обученная модель
├── requirements.txt         # Зависимости
├── render.yaml             # Конфиг для Render
└── .gitignore
```

### 1.3. Обновите `.gitignore`

Убедитесь, что модель НЕ игнорируется:

```gitignore
# Исключите из игнорирования:
# models/*.pkl  <- УДАЛИТЕ ЭТУ СТРОКУ

# Или добавьте исключение:
!models/best_model.pkl
```

### 1.4. Добавьте файл `runtime.txt`

Создайте файл для указания версии Python:

```
python-3.11
```

---

## 🌐 Шаг 2: Создание аккаунта на Render

1. Перейдите на [https://render.com](https://render.com)
2. Нажмите **Get Started for Free**
3. Авторизуйтесь через GitHub
4. Разрешите доступ к репозиториям

---

## 📦 Шаг 3: Создание Web Service

### 3.1. Нажмите **New** → **Web Service**

### 3.2. Подключите репозиторий

```
Connect a repository
↓
Выберите travel-churn-prediction
```

### 3.3. Настройте параметры

| Параметр | Значение |
|----------|----------|
| **Name** | `travel-churn-prediction` |
| **Region** | Frankfurt (ближе к Европе) |
| **Branch** | `main` |
| **Root Directory** | (оставить пустым) |
| **Runtime** | Python |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT` |

### 3.4. Выберите бесплатный тариф

Scroll down до **Instance Type** → **Free**

### 3.5. Нажмите **Create Web Service**

---

## ⚙️ Шаг 4: Настройка переменных окружения

После создания сервиса:

1. Перейдите на вкладку **Environment**
2. Добавьте переменные (если нужны):

```
MLFLOW_TRACKING_URI=sqlite:///./mlflow.db
```

3. Нажмите **Save Changes**

---

## 🚀 Шаг 5: Деплой

Render автоматически запустит деплой после создания сервиса.

**Статусы:**
- 🟡 Building — сборка образа
- 🟡 Deploying — развёртывание
- 🟢 Live — сервис готов

**Время:** 5-10 минут

---

## ✅ Шаг 6: Проверка работы

### 6.1. Получите URL

После деплоя получите URL вида:
```
https://travel-churn-prediction.onrender.com
```

### 6.2. Проверка здоровья

```bash
curl https://travel-churn-prediction.onrender.com/health
```

**Ожидаемый ответ:**
```json
{"status": "healthy", "model_loaded": true}
```

### 6.3. Тестовое предсказание

```bash
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

### 6.4. Открыть веб-интерфейс

```
https://travel-churn-prediction.onrender.com/
```

---

## 🔧 Шаг 7: Обновление кода

### Автоматический деплой

Render автоматически делает деплой при каждом push в ветку `main`:

```bash
git add .
git commit -m "feat: обновление модели"
git push origin main
```

### Ручной деплой

1. Откройте сервис на Render
2. Нажмите **Manual Deploy** → **Deploy latest commit**

---

## 🐛 Устранение проблем

### Проблема: Модель не загружается

```bash
# Проверьте логи
# На Render: вкладка Logs

# Ошибка: FileNotFound
# Решение: Убедитесь, что models/best_model.pkl добавлен в репозиторий
git add models/best_model.pkl
git commit -m "add model"
git push origin main
```

### Проблема: Сервис sleep-ит

**Причина:** Бесплатный тариф Render sleep-ит после 15 минут бездействия.

**Решение 1:** Используйте сторонний сервис для keep-alive:
- [UptimeRobot](https://uptimerobot.com/) — бесплатный мониторинг
- [Cron-Job.org](https://cron-job.org/) — вызов каждые 15 минут

**Решение 2:** Перейти на платный тариф ($7/месяц)

### Проблема: Ошибка сборки

```bash
# Посмотрите логи сборки
# На Render: вкладка Logs → Build Logs

# Частые ошибки:
# 1. Missing requirements.txt
# 2. Ошибка установки зависимостей

# Решение:
# Проверьте requirements.txt
pip freeze > requirements.txt
git add requirements.txt
git commit -m "update requirements"
git push origin main
```

### Проблема: 502 Bad Gateway

```bash
# Проверьте, что PORT передаётся правильно
# В uvicorn: --port $PORT

# Проверьте Health Check
curl https://travel-churn-prediction.onrender.com/health
```

---

## 📊 Мониторинг

### Логи в реальном времени

```bash
# На Render: вкладка Logs
# Или через CLI (требуется установка Render CLI)
render logs --follow
```

### Проверка статуса

```bash
# Health Check каждые 30 секунд
curl https://travel-churn-prediction.onrender.com/health
```

---

## 🎯 Чеклист после деплоя

- [ ] Сервис имеет статус **Live**
- [ ] `/health` возвращает `{"status": "healthy"}`
- [ ] `/predict` возвращает предсказание
- [ ] Веб-интерфейс открывается в браузере
- [ ] Модель загружается (`model_loaded: true`)
- [ ] Тестовые сценарии работают (`/test`)

---

## 🔗 Полезные ссылки

- **Render Docs:** [https://render.com/docs](https://render.com/docs)
- **Python Deployment:** [https://render.com/docs/deploy-fastapi](https://render.com/docs/deploy-fastapi)
- **Environment Variables:** [https://render.com/docs/environment-variables](https://render.com/docs/environment-variables)

---

## 💡 Советы

1. **Не храните секреты в коде** — используйте Environment Variables
2. **Добавьте health check** — Render использует его для мониторинга
3. **Используйте `.gitignore` правильно** — не игнорируйте модель
4. **Тестируйте перед деплоем** — запустите локально перед пушем
5. **Мониторьте логи** — регулярно проверяйте логи на ошибки

---

## 📞 Нужна помощь?

Если что-то не работает:

1. Проверьте логи на Render
2. Протестируйте локально (`uvicorn src.api.main:app --reload`)
3. Убедитесь, что все файлы добавлены в git
4. Проверьте `requirements.txt` на наличие ошибок
