"""
FastAPI router для мониторинга дрейфа данных /drift.
Позволяет просматривать статус дрейфа и запускать пересчёт.
"""

import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src import settings
from src.monitoring.drift_stats import compute_drift

logger = logging.getLogger(__name__)
router = APIRouter(tags=["drift"])

DATA_PATH = settings.PROCESSED_DATA_PATH
DRIFT_SUMMARY = "drift_summary.json"
DRIFT_ALERT = "drift_alert.json"


def _load_drift_summary() -> Dict[str, Any]:
    """Загрузка сохранённого отчёта о дрейфа с нормализацией формата."""
    # Свежий результат из папки выполнения, если анализ уже запускался,
    # иначе образец из репозитория (см. src/settings.py).
    summary_path = settings.report_for_reading(DRIFT_SUMMARY)
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        # Нормализация: миграция старого формата (ks_statistic) в новый (statistic)
        for row in data.get("results", []):
            if "statistic" not in row and "ks_statistic" in row:
                row["statistic"] = row["ks_statistic"]
            if "statistic" not in row and "js_divergence" in row:
                row["statistic"] = row["js_divergence"]
            if "test" not in row:
                row["test"] = (
                    "Kolmogorov-Smirnov"
                    if row.get("type") == "numeric"
                    else "Jensen-Shannon divergence"
                )
            if "threshold" not in row:
                row["threshold"] = (
                    data.get("p_threshold", 0.05)
                    if row.get("type") == "numeric"
                    else 0.2
                )
        return data
    return {
        "timestamp": None,
        "total_features": 0,
        "drift_features": 0,
        "results": [],
        "message": "Анализ дрейфа ещё не проводился. Нажмите 'Обновить анализ'.",
    }


def _analyze_drift(
    data_path: Path = DATA_PATH,
    test_size: float = 0.2,
    p_threshold: float = 0.05,
    js_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Пересчёт метрик дрейфа на актуальных данных.

    Args:
        data_path: Путь к обработанному датасету
        test_size: Доля тестовой выборки (current)
        p_threshold: Порог p-value для KS-теста
        js_threshold: Порог JS-divergence для категориальных

    Returns:
        Словарь с результатами анализа
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Датасет не найден: {data_path}")

    # Сам расчёт — общий с scripts/generate_drift_report.py (drift_stats.py).
    summary = compute_drift(
        pd.read_csv(data_path),
        test_size=test_size,
        p_threshold=p_threshold,
        js_threshold=js_threshold,
    )
    results = summary["results"]
    drift_count = summary["drift_features"]
    timestamp = summary["timestamp"]
    feature_columns = [r["feature"] for r in results]

    # Сохранение JSON — в папку выполнения, а не поверх образца в репозитории.
    output_dir = settings.runtime_reports_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / DRIFT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Сохранение alert для совместимости с /monitoring
    alert = {
        "alert_type": "data_drift",
        "timestamp": timestamp,
        "drift_detected": drift_count > 0,
        "affected_columns": [r["feature"] for r in results if r["drift_detected"]],
        "message": summary["message"],
    }
    with open(output_dir / DRIFT_ALERT, "w", encoding="utf-8") as f:
        json.dump(alert, f, ensure_ascii=False, indent=2)

    # Telegram alert при дрейфе
    if drift_count > 0:
        _send_telegram_alert(
            affected=[r["feature"] for r in results if r["drift_detected"]],
            total=len(feature_columns),
            timestamp=timestamp,
        )

    logger.info(f"Анализ дрейфа завершён: {summary['message']}")
    return summary


def _send_telegram_alert(affected: List[str], total: int, timestamp: str):
    """Отправка алерта в Telegram при обнаружении дрейфа."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        return

    msg = (
        f"⚠️ <b>Data Drift Alert</b>\n\n"
        f"🔴 Обнаружен дрейф в признаках: {', '.join(affected)}\n"
        f"📊 Всего признаков: {total}\n"
        f"🕐 {timestamp}\n\n"
        f"🔗 <a href=\"{os.getenv('APP_URL', '')}/drift\">Открыть дашборд</a>"
    )

    try:
        url = (
            f"https://api.telegram.org/bot{bot_token}/sendMessage"
            f"?chat_id={chat_id}&parse_mode=HTML&text={urllib.parse.quote(msg)}"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            logger.info(f"Telegram alert отправлен, статус: {resp.status}")
    except Exception as e:
        logger.error(f"Не удалось отправить Telegram alert: {e}")


@router.get("/drift", response_class=HTMLResponse)
async def drift_dashboard(request: Request):
    """HTML-дашборд мониторинга дрейфа данных."""
    templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)
    data = _load_drift_summary()
    # Шаблон читает поля сводки напрямую (results, drift_features, ...);
    # обёртка {"data": data} оставляла таблицу и сводку пустыми.
    return templates.TemplateResponse(request, "drift_dashboard.html", data)


@router.post("/drift/analyze")
async def drift_analyze():
    """
    Ручной запуск анализа дрейфа данных.
    Пересчитывает метрики на актуальном датасете и сохраняет результаты.
    """
    try:
        summary = _analyze_drift()
        return JSONResponse(
            content={
                "status": "success",
                "timestamp": summary["timestamp"],
                "drift_features": summary["drift_features"],
                "total_features": summary["total_features"],
                "message": summary["message"],
            }
        )
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=404, content={"status": "error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Ошибка анализа дрейфа: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Внутренняя ошибка: {str(e)}"},
        )


@router.get("/drift/status")
async def drift_api_status():
    """JSON endpoint со статусом дрейфа."""
    return _load_drift_summary()


@router.get("/drift/history")
async def drift_history(limit: int = 10):
    """
    История анализов дрейфа (если ведётся логирование).
    Пока возвращает текущий результат.
    """
    current = _load_drift_summary()
    return {"history": [current], "limit": limit}
