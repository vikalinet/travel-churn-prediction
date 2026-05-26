#!/bin/bash
# Travel Churn Prediction - Bash скрипт для запуска (Linux/macOS)

echo ""
echo "============================================================"
echo "  Travel Churn Prediction - Настройка и запуск"
echo "============================================================"
echo ""

# Функции для вывода
step() { echo "  ➜ $1"; }
success() { echo "  ✅ $1"; }
error() { echo "  ❌ $1"; }

# Проверка Python
echo ""
echo "Проверка Python..."
PYTHON_VERSION=$(python3 --version 2>&1)
if [ $? -eq 0 ]; then
    success "Установлен: $PYTHON_VERSION"
else
    error "Python не найден. Установите Python 3.8+"
    exit 1
fi

# Проверка виртуального окружения
echo ""
echo "Проверка виртуального окружения..."
if [ -d "venv" ]; then
    success "Виртуальное окружение существует"
    echo ""
    step "Активация..."
    source venv/bin/activate
else
    step "Создание виртуального окружения..."
    python3 -m venv venv

    if [ $? -eq 0 ]; then
        success "Виртуальное окружение создано"
        step "Активация..."
        source venv/bin/activate
    fi
fi

# Установка зависимостей
echo ""
echo "Установка зависимостей..."
if [ -f "requirements.txt" ]; then
    step "Это может занять 2-3 минуты..."
    pip install -r requirements.txt

    if [ $? -eq 0 ]; then
        success "Все зависимости установлены"
    else
        error "Ошибка установки зависимостей"
        exit 1
    fi
else
    error "Файл requirements.txt не найден"
    exit 1
fi

# Проверка данных
echo ""
echo "Проверка данных..."
if [ -d "data/raw" ]; then
    FILE_COUNT=$(find data/raw -name "*.csv" 2>/dev/null | wc -l)
    if [ "$FILE_COUNT" -gt 0 ]; then
        success "Данные найдены: $FILE_COUNT файлов"
    else
        error "Папка data/raw пуста. Добавьте CSV файл."
    fi
else
    error "Папка data/raw не найдена"
fi

# Проверка модели
echo ""
echo "Проверка модели..."
MODEL_FOUND=false

for path in "models/best_model.pkl" "models/GradientBoosting_model.pkl" "models/model.pkl"; do
    if [ -f "$path" ]; then
        SIZE=$(du -h "$path" | cut -f1)
        success "Модель: $path ($SIZE)"
        MODEL_FOUND=true
        break
    fi
done

if [ "$MODEL_FOUND" = false ]; then
    error "Модель не найдена"
    echo ""
    echo "  Для обучения модели запустите:"
    echo "    python -m src.training.model_training"
    echo ""
fi

# Итог
echo ""
echo "============================================================"
echo "  Проверка завершена"
echo "============================================================"
echo ""

# Запуск сервера
echo "Запуск сервера..."
echo ""
echo "  После запуска откройте в браузере:"
echo "    http://localhost:8000/      — Главная страница"
echo "    http://localhost:8000/test — Тестирование UI"
echo "    http://localhost:8000/docs — Swagger документация"
echo ""

# Запуск uvicorn
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
