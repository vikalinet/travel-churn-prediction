"""
Мониторинг инфраструктуры: CPU, RAM, диск, сеть.
Генерирует JSON-отчет и HTML-дашборд.
"""

import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src import settings

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_system_metrics() -> Dict:
    """Сбор системных метрик."""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    if not PSUTIL_AVAILABLE:
        logger.warning("psutil не установлен. Установите: pip install psutil")
        metrics["error"] = "psutil not installed"
        return metrics

    # CPU
    metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
    metrics["cpu_count_logical"] = psutil.cpu_count(logical=True)
    metrics["cpu_count_physical"] = psutil.cpu_count(logical=False)

    # RAM
    mem = psutil.virtual_memory()
    metrics["ram_total_gb"] = round(mem.total / (1024**3), 2)
    metrics["ram_used_gb"] = round(mem.used / (1024**3), 2)
    metrics["ram_percent"] = mem.percent

    # Диск
    disk = psutil.disk_usage("/")
    metrics["disk_total_gb"] = round(disk.total / (1024**3), 2)
    metrics["disk_used_gb"] = round(disk.used / (1024**3), 2)
    metrics["disk_percent"] = disk.percent

    # Сеть
    net = psutil.net_io_counters()
    metrics["net_sent_mb"] = round(net.bytes_sent / (1024**2), 2)
    metrics["net_recv_mb"] = round(net.bytes_recv / (1024**2), 2)

    return metrics


# Загрузка ресурса выше этого порога выделяется на странице как предупреждение.
WARNING_PERCENT = 80

_templates = Environment(
    loader=FileSystemLoader(settings.TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
)


def _usage(percent, title: str = "", used_gb=None, total_gb=None) -> Dict:
    """Карточка загрузки ресурса: значение, уровень и ширина полосы.

    Решения принимаются здесь, а не в шаблоне: шаблон только показывает.
    Если метрика недоступна (нет psutil), показывается «N/A» и пустая полоса.
    """
    known = isinstance(percent, (int, float))
    return {
        "title": title,
        "percent": percent if known else "N/A",
        "level": "warning" if known and percent > WARNING_PERCENT else "ok",
        "bar": min(percent, 100) if known else 0,
        "used_gb": "N/A" if used_gb is None else used_gb,
        "total_gb": "N/A" if total_gb is None else total_gb,
    }


def render_system_report(metrics: Dict) -> str:
    """HTML-страница по собранным метрикам (шаблон templates/reports/)."""
    return _templates.get_template("reports/system_monitor.html").render(
        timestamp=metrics.get("timestamp", "N/A"),
        platform=metrics.get("platform", "N/A")[:30],
        python_version=metrics.get("python_version", "N/A"),
        cpu_count_logical=metrics.get("cpu_count_logical", "N/A"),
        cpu=_usage(metrics.get("cpu_percent")),
        ram=_usage(
            metrics.get("ram_percent"),
            "🧠 RAM",
            metrics.get("ram_used_gb"),
            metrics.get("ram_total_gb"),
        ),
        disk=_usage(
            metrics.get("disk_percent"),
            "💾 Диск",
            metrics.get("disk_used_gb"),
            metrics.get("disk_total_gb"),
        ),
    )


def generate_system_report(output_dir: Optional[str] = None) -> str:
    """Сохранение JSON с метриками и HTML-дашборда.

    По умолчанию — в папку выполнения (src/settings.py), а не в reports/:
    reports/ хранится в git и публикуется на GitHub Pages, а прежний
    относительный путь к тому же зависел от папки запуска.
    """
    metrics = get_system_metrics()

    target = Path(output_dir) if output_dir else settings.runtime_dir() / "reports"
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "system_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON отчет сохранен: {json_path}")

    html_path = target / "system_monitor.html"
    html_path.write_text(render_system_report(metrics), encoding="utf-8")
    logger.info(f"HTML отчет сохранен: {html_path}")

    return str(html_path)


if __name__ == "__main__":
    generate_system_report()
