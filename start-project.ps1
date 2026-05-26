# Travel Churn Prediction - PowerShell скрипт для запуска
# Для Windows 10/11

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Travel Churn Prediction - Настройка и запуск" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Функция для вывода цветного текста
function Write-Step {
    param([string]$Text)
    Write-Host "  ➜ $Text" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Text)
    Write-Host "  ✅ $Text" -ForegroundColor Green
}

function Write-Error {
    param([string]$Text)
    Write-Host "  ❌ $Text" -ForegroundColor Red
}

# Проверка Python
Write-Host "`nПроверка Python..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Success "Установлен: $pythonVersion"
} else {
    Write-Error "Python не найден. Установите Python 3.8+ с python.org"
    exit 1
}

# Проверка виртуального окружения
Write-Host "`nПроверка виртуального окружения..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Success "Виртуальное окружение существует"

    Write-Host "`nАктивация виртуального окружения..." -ForegroundColor Cyan
    & ".\venv\Scripts\Activate.ps1"

    if ($null -eq $env:VIRTUAL_ENV) {
        Write-Host "Введите вручную: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    }
} else {
    Write-Step "Создание виртуального окружения..."
    python -m venv venv

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Виртуальное окружение создано"
        Write-Step "Активация..."
        & ".\venv\Scripts\Activate.ps1"
    }
}

# Установка зависимостей
Write-Host "`nУстановка зависимостей..." -ForegroundColor Cyan
if (Test-Path "requirements.txt") {
    Write-Step "Это может занять 2-3 минуты..."
    pip install -r requirements.txt

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Все зависимости установлены"
    } else {
        Write-Error "Ошибка установки зависимостей"
        exit 1
    }
} else {
    Write-Error "Файл requirements.txt не найден"
    exit 1
}

# Проверка данных
Write-Host "`nПроверка данных..." -ForegroundColor Cyan
if (Test-Path "data/raw") {
    $files = Get-ChildItem "data/raw" -Filter "*.csv" -ErrorAction SilentlyContinue
    if ($files) {
        Write-Success "Данные найдены: $($files.Count) файлов"
    } else {
        Write-Error "Папка data/raw пуста. Добавьте CSV файл."
    }
} else {
    Write-Error "Папка data/raw не найдена"
}

# Проверка модели
Write-Host "`nПроверка модели..." -ForegroundColor Cyan
$modelPaths = @("models/best_model.pkl", "models/GradientBoosting_model.pkl", "models/model.pkl")
$modelFound = $false

foreach ($path in $modelPaths) {
    if (Test-Path $path) {
        $size = (Get-Item $path).Length / 1KB
        Write-Success "Модель: $path ($('{0:N1}' -f $size) KB)"
        $modelFound = $true
        break
    }
}

if (-not $modelFound) {
    Write-Error "Модель не найдена"
    Write-Host ""
    Write-Host "  Для обучения модели запустите:" -ForegroundColor Yellow
    Write-Host "    python -m src.training.model_training" -ForegroundColor White
    Write-Host ""
}

# Итог
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Проверка завершена" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Запуск сервера
Write-Host "Запуск сервера..." -ForegroundColor Cyan
Write-Host ""
Write-Host "  После запуска откройте в браузере:" -ForegroundColor White
Write-Host "    http://localhost:8000/      — Главная страница" -ForegroundColor Gray
Write-Host "    http://localhost:8000/test — Тестирование UI" -ForegroundColor Gray
Write-Host "    http://localhost:8000/docs — Swagger документация" -ForegroundColor Gray
Write-Host ""

# Запуск uvicorn
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
