"""Страница мониторинга инфраструктуры: шаблон вместо HTML в f-строке.

Прежде вся страница со стилями была одной f-строкой в Python. Теперь разметка
в templates/reports/system_monitor.html, а решения (уровень загрузки, ширина
полосы) — в коде, и их можно проверить отдельно от разметки.
"""

from src.monitoring import system_monitor
from src.monitoring.system_monitor import WARNING_PERCENT, _usage, render_system_report

METRICS = {
    "timestamp": "2026-10-04T10:00",
    "platform": "Windows-11",
    "python_version": "3.13.7",
    "cpu_percent": 85.5,
    "cpu_count_logical": 8,
    "ram_total_gb": 16.0,
    "ram_used_gb": 9.1,
    "ram_percent": 57.0,
    "disk_total_gb": 476.0,
    "disk_used_gb": 450.0,
    "disk_percent": 120.0,
}


def test_usage_levels_and_bar():
    assert _usage(WARNING_PERCENT + 1)["level"] == "warning"
    assert _usage(WARNING_PERCENT)["level"] == "ok"
    assert _usage(120)["bar"] == 100
    missing = _usage(None)
    assert (missing["percent"], missing["bar"], missing["level"]) == ("N/A", 0, "ok")


def test_report_shows_metrics():
    html = render_system_report(METRICS)
    assert "Сгенерировано: 2026-10-04T10:00" in html
    assert 'class="metric-value warning">85.5%' in html  # CPU выше порога
    assert 'class="metric-value ok">57.0%' in html  # RAM в норме
    assert "9.1 / 16.0 GB" in html
    assert "width: 100%" in html  # диск 120 % — полоса не шире 100 %


def test_report_without_psutil_metrics():
    html = render_system_report({"timestamp": "t", "error": "psutil not installed"})
    assert "N/A%" in html
    assert "N/A / N/A GB" in html


def test_report_is_written_outside_git_by_default(runtime_dir_in_tmp, monkeypatch):
    monkeypatch.setattr(system_monitor, "get_system_metrics", lambda: METRICS)
    path = system_monitor.generate_system_report()
    assert path.startswith(str(runtime_dir_in_tmp))
    assert (runtime_dir_in_tmp / "reports" / "system_metrics.json").exists()
