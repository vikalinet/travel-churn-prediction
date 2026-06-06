#!/usr/bin/env python3
"""
🚀 Скрипт быстрого старта проекта
Автоматическая проверка и запуск сервиса
"""

import subprocess
import sys
from pathlib import Path
from typing import Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ProjectChecker:
    """Класс для проверки состояния проекта"""

    REQUIRED_PACKAGES = [
        "fastapi",
        "uvicorn",
        "scikit-learn",
        "pandas",
        "numpy",
        "joblib",
    ]

    MODEL_PATHS = [
        "models/best_model.pkl",
        "models/GradientBoosting_model.pkl",
        "models/model.pkl",
    ]

    def __init__(self):
        self.results: dict[str, bool] = {}

    def check_python_version(self, min_version: Tuple[int, int] = (3, 8)) -> bool:
        """Проверка версии Python"""
        version = sys.version_info
        ok = version.major == min_version[0] and version.minor >= min_version[1]
        self.results["Python"] = ok
        return ok

    def check_venv(self, venv_path: str = "venv") -> Tuple[bool, bool]:
        """Проверка виртуального окружения. Возвращает (существует, активировано)"""
        path = Path(venv_path)
        exists = path.exists()
        activated = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )
        self.results["Venv"] = exists and activated
        return exists, activated

    def check_dependencies(self) -> bool:
        """Проверка зависимостей через importlib.metadata"""
        all_ok = True
        for package in self.REQUIRED_PACKAGES:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                all_ok = False
        self.results["Dependencies"] = all_ok
        return all_ok

    def check_data_dirs(self) -> bool:
        """Проверка наличия данных"""
        raw_dir = Path("data/raw")
        processed_dir = Path("data/processed")

        raw_ok = raw_dir.exists() and any(raw_dir.iterdir())
        processed_ok = processed_dir.exists() and any(processed_dir.iterdir())

        self.results["Data"] = raw_ok and processed_ok
        return self.results["Data"]

    def check_model(self) -> bool:
        """Проверка наличия модели"""
        for path_str in self.MODEL_PATHS:
            if Path(path_str).exists():
                self.results["Model"] = True
                return True

        self.results["Model"] = False
        return False

    def check_all(self) -> bool:
        """Выполнить все проверки"""
        self.check_python_version()
        self.check_venv()
        self.check_dependencies()
        self.check_data_dirs()
        self.check_model()
        return all(self.results.values())


class ProjectRunner:
    """Класс для запуска проекта"""

    def __init__(self):
        self.checker = ProjectChecker()

    def install_dependencies(self, requirements_path: str = "requirements.txt") -> bool:
        """Установка зависимостей"""
        if not Path(requirements_path).exists():
            logger.error(f"Файл {requirements_path} не найден")
            return False

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", requirements_path],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка установки: {e.stderr}")
            return False

    def run_server(
        self,
        module: str = "src.api.main:app",
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = True,
    ) -> None:
        """Запуск uvicorn сервера"""
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            module,
            "--host",
            host,
            "--port",
            str(port),
        ]
        if reload:
            cmd.append("--reload")

        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            logger.info("Сервер остановлен")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка запуска сервера: {e}")
            sys.exit(1)

    def interactive_menu(self) -> None:
        """Интерактивное меню для пользователя"""
        print("\n" + "=" * 60)
        print("  🚀 Travel Churn Prediction — Быстрый старт")
        print("=" * 60 + "\n")

        # Запускаем проверки
        if not self.checker.check_all():
            self._print_failures()
            self._offer_fixes()

        # Предложение запустить сервер
        print("\n  💡 Запустить сервер? (y/n)")
        response = input("  > ").strip().lower()
        if response == "y":
            self.run_server()

    def _print_failures(self) -> None:
        """Вывод неудачных проверок"""
        print("\n  ⚠️ Не пройдены следующие проверки:")
        for name, ok in self.checker.results.items():
            if not ok:
                print(f"     ❌ {name}")

    def _offer_fixes(self) -> None:
        """Предложения по исправлению"""
        if not self.checker.results.get("Dependencies", True):
            print("\n  💡 Установить зависимости? (y/n)")
            if input("  > ").strip().lower() == "y":
                self.install_dependencies()

        if not self.checker.results.get("Model", True):
            print("\n  💡 Модель не найдена. Обучить? (y/n)")
            if input("  > ").strip().lower() == "y":
                subprocess.run([sys.executable, "-m", "src.training.model_training"])


def main():
    """Главная функция"""
    runner = ProjectRunner()
    runner.interactive_menu()


if __name__ == "__main__":
    main()
