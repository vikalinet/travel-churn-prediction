#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script для установки проекта и подготовки окружения.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Выполнение команды и вывод статуса."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")

    try:
        subprocess.run(command, check=True)
        print(f"OK: {description} - успешно!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR при {description}: {e}")
        return False


def main():
    """Основная функция установки."""
    print("\n" + "=" * 60)
    print(" Установка проекта: Прогнозирование оттока клиентов")
    print("=" * 60)

    # Проверка Python версии
    if sys.version_info < (3, 9):
        print("ERROR: Требуется Python 3.9 или выше")
        sys.exit(1)

    print(f"OK: Python версия: {sys.version}")

    # Создание структуры директорий
    directories = [
        "data/raw",
        "data/processed",
        "src/api",
        "src/etl",
        "src/models",
        "src/training",
        "src/utils",
        "tests",
        "reports",
        "evidently_reports",
    ]

    print("\nСоздание структуры директорий...")
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   OK: {dir_path}")

    # Установка зависимостей
    print("\nУстановка Python зависимостей...")
    run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "установку зависимостей",
    )

    # Установка pre-commit (опционально)
    print("\nНастройка pre-commit...")
    run_command(
        [sys.executable, "-m", "pip", "install", "pre-commit"], "установку pre-commit"
    )
    run_command(["pre-commit", "install"], "настройку pre-commit hooks")

    print("\n" + "=" * 60)
    print("OK: Установка завершена!")
    print("=" * 60)
    print("\nСледующие шаги:")
    print("   1. Скачайте датасет с Kaggle и поместите в data/raw/")
    print("   2. Запустите тесты: pytest tests/ -v")
    print("   3. Запустите FastAPI: uvicorn src.api.main:app --reload")
    print("   4. Или используйте Docker: docker-compose up --build")
    print()


if __name__ == "__main__":
    main()
