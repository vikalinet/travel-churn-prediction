"""
FastAPI router для мониторинга дрейфа данных /drift.
Позволяет просматривать статус дрейфа и запускать пересчёт.
"""

import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from scipy import stats
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
router = APIRouter(tags=["drift"])

DATA_PATH = Path("data/processed/processed_data.csv")
DRIFT_JSON = Path("evidently_reports/drift_summary.json")
DRIFT_HTML = Path("evidently_reports/drift_report.html")


def _load_drift_summary() -> Dict[str, Any]:
    """Загрузка сохранённого отчёта о дрейфа с нормализацией формата."""
    if DRIFT_JSON.exists():
        with open(DRIFT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
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

    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    feature_columns = [col for col in df.columns if col != "Target"]

    results: List[Dict[str, Any]] = []

    for column in feature_columns:
        ref_values = train_df[column].dropna()
        curr_values = test_df[column].dropna()

        if pd.api.types.is_numeric_dtype(ref_values):
            stat, p_value = stats.ks_2samp(ref_values, curr_values)
            drift_detected = p_value < p_threshold
            results.append(
                {
                    "feature": column,
                    "type": "numeric",
                    "test": "Kolmogorov-Smirnov",
                    "statistic": round(float(stat), 4),
                    "p_value": round(float(p_value), 4),
                    "drift_detected": bool(drift_detected),
                    "ref_mean": round(float(ref_values.mean()), 4),
                    "curr_mean": round(float(curr_values.mean()), 4),
                    "ref_std": round(float(ref_values.std()), 4),
                    "curr_std": round(float(curr_values.std()), 4),
                    "threshold": p_threshold,
                }
            )
        else:
            ref_counts = ref_values.value_counts(normalize=True)
            curr_counts = curr_values.value_counts(normalize=True)
            all_categories = ref_counts.index.union(curr_counts.index)
            ref_norm = ref_counts.reindex(all_categories, fill_value=0)
            curr_norm = curr_counts.reindex(all_categories, fill_value=0)
            js_div = 0.5 * np.sum(np.abs(ref_norm - curr_norm))
            drift_detected = js_div > js_threshold
            results.append(
                {
                    "feature": column,
                    "type": "categorical",
                    "test": "Jensen-Shannon divergence",
                    "statistic": round(float(js_div), 4),
                    "p_value": None,
                    "drift_detected": bool(drift_detected),
                    "ref_mean": None,
                    "curr_mean": None,
                    "ref_std": None,
                    "curr_std": None,
                    "threshold": js_threshold,
                }
            )

    drift_count = sum(1 for r in results if r["drift_detected"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "timestamp": timestamp,
        "total_features": len(feature_columns),
        "drift_features": drift_count,
        "p_threshold": p_threshold,
        "test_size": test_size,
        "reference_size": len(train_df),
        "current_size": len(test_df),
        "results": results,
        "alert": drift_count > 0,
        "message": (
            f"Обнаружен дрейф в {drift_count} признаках! Требуется внимание."
            if drift_count > 0
            else "Дрейф не обнаружен. Данные стабильны."
        ),
    }

    # Сохранение JSON
    DRIFT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Сохранение alert для совместимости с /monitoring
    alert = {
        "alert_type": "data_drift",
        "timestamp": timestamp,
        "drift_detected": drift_count > 0,
        "affected_columns": [r["feature"] for r in results if r["drift_detected"]],
        "message": summary["message"],
    }
    with open(DRIFT_JSON.parent / "drift_alert.json", "w", encoding="utf-8") as f:
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
        f"🔗 <a href=\"{os.getenv('RAILWAY_APP_URL', '')}/drift\">Открыть дашборд</a>"
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
    templates = Jinja2Templates(directory="templates")
    data = _load_drift_summary()
    return templates.TemplateResponse(
        "drift_dashboard.html", {"request": request, "data": data}
    )


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
