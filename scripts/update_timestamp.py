"""
Обновление timestamp в index.html при деплое.
"""

from pathlib import Path
from datetime import datetime


def update_index_timestamp():
    """Обновление timestamp в index.html."""
    index_path = Path("reports/index.html")

    if not index_path.exists():
        print("index.html не найден!")
        return

    # Текущее время
    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    # Чтение файла
    content = index_path.read_text(encoding="utf-8")

    # Замена {{timestamp}} на реальное время
    content = content.replace("{{timestamp}}", timestamp)

    # Запись обратно
    index_path.write_text(content, encoding="utf-8")

    print(f"✅ Timestamp обновлён: {timestamp}")


if __name__ == "__main__":
    update_index_timestamp()
