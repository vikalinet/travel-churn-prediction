"""
Расчёт дрейфа данных — одна реализация для API и для скрипта отчёта.

Прежде один и тот же расчёт (KS-тест для числовых признаков,
JS-расхождение для категориальных) был написан дважды: в
src/api/drift_router.py и в scripts/generate_drift_report.py. Копии
разошлись — разные имена полей (statistic / ks_statistic) и разный набор
данных в результате, — и API пришлось читать «старый формат». Теперь расчёт
здесь, без записи файлов и побочных действий; что делать с результатом,
решает вызывающий: API сохраняет его в папку выполнения и шлёт уведомление,
скрипт строит HTML-отчёт.
"""

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Target"


def compute_drift(
    df: pd.DataFrame,
    test_size: float = 0.2,
    p_threshold: float = 0.05,
    js_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Дрейф между эталонной (train) и текущей (test) частями датасета.

    Args:
        df: Обработанный датасет
        test_size: Доля текущей выборки
        p_threshold: Порог p-value для KS-теста (числовые признаки)
        js_threshold: Порог JS-расхождения (категориальные признаки)

    Returns:
        Сводка: по признаку — тест, статистика, p-value, признак дрейфа,
        средние; итог — число признаков с дрейфом и сообщение.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]

    results: List[Dict[str, Any]] = []
    for column in feature_columns:
        ref_values = train_df[column].dropna()
        curr_values = test_df[column].dropna()

        if pd.api.types.is_numeric_dtype(ref_values):
            stat, p_value = stats.ks_2samp(ref_values, curr_values)
            results.append(
                {
                    "feature": column,
                    "type": "numeric",
                    "test": "Kolmogorov-Smirnov",
                    "statistic": round(float(stat), 4),
                    "p_value": round(float(p_value), 4),
                    "drift_detected": bool(p_value < p_threshold),
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
            results.append(
                {
                    "feature": column,
                    "type": "categorical",
                    "test": "Jensen-Shannon divergence",
                    "statistic": round(float(js_div), 4),
                    # У категориальных признаков p-value нет — None, а не
                    # отсутствие ключа: отчёт показывает «-» вместо падения.
                    "p_value": None,
                    "drift_detected": bool(js_div > js_threshold),
                    "ref_mean": None,
                    "curr_mean": None,
                    "ref_std": None,
                    "curr_std": None,
                    "threshold": js_threshold,
                }
            )

    drift_count = sum(1 for r in results if r["drift_detected"])
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
