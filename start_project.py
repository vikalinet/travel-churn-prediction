#!/usr/bin/env python3
"""
🚀 Скрипт быстрого старта проекта
Автоматическая проверка и запуск сервиса
"""

import subprocess
import sys
from pathlib import Path
import platform


def print_header(text):
    """Красивый заголовок"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_status(ok, text):
    """Статус с эмодзи"""
    symbol = "✅" if ok else "❌"
    print(f"  {symbol} {text}")


def check_python():
    """Проверка версии Python"""
    print_header("Проверка Python")
    version = sys.version_info
    print(f"  Текущая версия: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print_status(True, "Python версия подходит (3.8+)")
        return True
    else:
        print_status(False, "Нужна версия Python 3.8 или выше")
        return False


def check_venv():
    """Проверка виртуального окружения"""
    print_header("Проверка виртуального окружения")

    venv_path = Path("venv")
    if venv_path.exists():
        print_status(True, "Виртуальное окружение существует")

        # Проверяем, активировано ли оно
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            print_status(True, "Виртуальное окружение активировано")
        else:
            print("\n  💡 Для активации выполните:")
            if platform.system() == "Windows":
                print("     venv\\Scripts\\Activate.ps1")
            else:
                print("     source venv/bin/activate")
        return True
    else:
        print_status(False, "Виртуальное окружение не найдено")
        print("\n  💡 Создайте виртуальное окружение:")
        print("     python -m venv venv")
        return False


def check_dependencies():
    """Проверка установленных зависимостей"""
    print_header("Проверка зависимостей")

    required = [
        "fastapi",
        "uvicorn",
        "scikit-learn",
        "pandas",
        "numpy",
        "joblib",
    ]

    all_ok = True
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print_status(True, f"{package}")
        except ImportError:
            print_status(False, f"{package} — НЕ установлен!")
            all_ok = False

    return all_ok


def check_data():
    """Проверка наличия данных"""
    print_header("Проверка данных")

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    if raw_dir.exists() and any(raw_dir.iterdir()):
        files = list(raw_dir.glob("*.csv"))
        print_status(True, f"Даные в data/raw/: {len(files)} файлов")
    else:
        print_status(False, "Папка data/raw/ пуста или отсутствует")

    if processed_dir.exists() and any(processed_dir.iterdir()):
        print_status(True, "Обработанные данные найдены")
    else:
        print_status(False, "Обработанные данные не найдены")


def check_model():
    """Проверка наличия модели"""
    print_header("Проверка модели")

    model_paths = [
        "models/best_model.pkl",
        "models/GradientBoosting_model.pkl",
        "models/model.pkl",
    ]

    for path in model_paths:
        if Path(path).exists():
            size = Path(path).stat().st_size / 1024
            print_status(True, f"Модель: {path} ({size:.1f} KB)")
            return True

    print_status(False, "Модель не найдена!")
    print("\n  💡 Для обучения модели выполните:")
    print("     python -m src.training.model_training")
    return False


def install_dependencies():
    """Установка зависимостей"""
    print_header("Установка зависимостей")

    if not Path("requirements.txt").exists():
        print_status(False, "Файл requirements.txt не найден")
        return False

    print("  ⏳ Установка... (это может занять 2-3 минуты)\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
        print_status(True, "Все зависимости установлены")
        return True
    except subprocess.CalledProcessError:
        print_status(False, "Ошибка при установке")
        return False


def run_server():
    """Запуск сервера"""
    print_header("🚀 Запуск сервера")

    print("  Запускаю uvicorn...\n")
    print("  📍 После запуска откройте в браузере:")
    print("     http://localhost:8000/      — Главная страница")
    print("     http://localhost:8000/test — Тестирование UI")
    print("     http://localhost:8000/docs — Swagger документация\n")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api.main:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ]
        )
    except KeyboardInterrupt:
        print("\n\n  👋 Сервер остановлен")
    except Exception as e:
        print(f"\n  ❌ Ошибка: {e}")


def main():
    """Главная функция"""
    print("\n" + "=" * 60)
    print("  🚀 Travel Churn Prediction — Быстрый старт")
    print("=" * 60)

    # Проверки
    python_ok = check_python()
    venv_ok = check_venv()
    deps_ok = check_dependencies()

    if not deps_ok:
        print("\n  ⚠️ Зависимости не установлены. Хотите установить? (y/n)")
        response = input("  > ").strip().lower()
        if response == "y":
            if install_dependencies():
                deps_ok = True

    check_data()
    model_ok = check_model()

    # Итоги
    print_header("📊 Итоговая проверка")

    checks = [
        ("Python", python_ok),
        ("Виртуальное окружение", venv_ok),
        ("Зависимости", deps_ok),
        ("Модель", model_ok),
    ]

    all_ok = all(ok for _, ok in checks)

    print()
    for name, ok in checks:
        print_status(ok, name)

    print()

    if all_ok:
        print("  🎉 Все проверки пройдены!")
        print("\n  💡 Запустить сервер? (y/n)")
        response = input("  > ").strip().lower()
        if response == "y":
            run_server()
    else:
        print("  ⚠️ Есть проблемы. Исправьте их перед запуском.")
        print("\n  💡 Для автоматической установки зависимостей:")
        print("     pip install -r requirements.txt")

        print("\n  💡 Для обучения модели:")
        print("     python -m src.training.model_training")


if __name__ == "__main__":
    main()
