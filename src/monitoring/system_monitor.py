"""
Мониторинг инфраструктуры: CPU, RAM, диск, сеть.
Генерирует JSON-отчет и HTML-дашборд.
"""

import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict

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


def generate_system_report(output_dir: str = "reports") -> str:
    """Генерация HTML-отчета по инфраструктуре."""
    metrics = get_system_metrics()

    Path(output_dir).mkdir(exist_ok=True)

    # JSON
    json_path = Path(output_dir) / "system_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON отчет сохранен: {json_path}")

    # HTML
    html_path = Path(output_dir) / "system_monitor.html"
    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Мониторинг инфраструктуры</title>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
        }}
        .warning {{ color: #e74c3c; }}
        .ok {{ color: #27ae60; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🖥️ Мониторинг инфраструктуры</h1>
        <p>Сгенерировано: {metrics.get('timestamp', 'N/A')}</p>
    </div>

    <div class="card">
        <h3>💻 Система</h3>
        <div class="metric">
            <div class="metric-value">
                {metrics.get('platform', 'N/A')[:30]}
            </div>
            <div class="metric-label">Платформа</div>
        </div>
        <div class="metric">
            <div class="metric-value">
                {metrics.get('python_version', 'N/A')}
            </div>
            <div class="metric-label">Python</div>
        </div>
    </div>

    <div class="card">
        <h3>⚡ CPU</h3>
        <div class="metric">
            <div class="metric-value {'warning' if metrics.get('cpu_percent', 0) > 80 else 'ok'}">
                {metrics.get('cpu_percent', 'N/A')}%
            </div>
            <div class="metric-label">Загрузка</div>
        </div>
        <div class="metric">
            <div class="metric-value">
                {metrics.get('cpu_count_logical', 'N/A')}
            </div>
            <div class="metric-label">Логических ядер</div>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {min(metrics.get('cpu_percent', 0), 100)}%"></div>
        </div>
    </div>

    <div class="card">
        <h3>🧠 RAM</h3>
        <div class="metric">
            <div class="metric-value">
                {metrics.get('ram_used_gb', 'N/A')} / {metrics.get('ram_total_gb', 'N/A')} GB
            </div>
            <div class="metric-label">Использование</div>
        </div>
        <div class="metric">
            <div class="metric-value {'warning' if metrics.get('ram_percent', 0) > 80 else 'ok'}">
                {metrics.get('ram_percent', 'N/A')}%
            </div>
            <div class="metric-label">Загрузка</div>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {min(metrics.get('ram_percent', 0), 100)}%"></div>
        </div>
    </div>

    <div class="card">
        <h3>💾 Диск</h3>
        <div class="metric">
            <div class="metric-value">
                {metrics.get('disk_used_gb', 'N/A')} / {metrics.get('disk_total_gb', 'N/A')} GB
            </div>
            <div class="metric-label">Использование</div>
        </div>
        <div class="metric">
            <div class="metric-value {'warning' if metrics.get('disk_percent', 0) > 80 else 'ok'}">
                {metrics.get('disk_percent', 'N/A')}%
            </div>
            <div class="metric-label">Загрузка</div>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {min(metrics.get('disk_percent', 0), 100)}%"></div>
        </div>
    </div>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"HTML отчет сохранен: {html_path}")

    return str(html_path)


if __name__ == "__main__":
    generate_system_report()
