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

# Пороги PSI (Population Stability Index) — общепринятая шкала:
# < 0.1 — распределение стабильно, 0.1–0.25 — умеренный сдвиг,
# > 0.25 — значимый сдвиг (считаем дрейфом).
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25
# Признак с таким числом значений и меньше сравниваем по значениям,
# а не по квантильным интервалам.
PSI_MAX_CATEGORIES = 10
PSI_BINS = 10
# Доля для пустого интервала, чтобы не делить на ноль и не брать log(0).
PSI_EPS = 1e-4


def _psi_shares(ref: pd.Series, curr: pd.Series) -> tuple:
    """Доли эталонной и текущей выборок по общим интервалам/значениям."""
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > PSI_MAX_CATEGORIES:
        # Интервалы — по квантилям эталона; повторяющиеся границы убираем,
        # крайние расширяем, чтобы текущие значения вне диапазона не терялись.
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, PSI_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts = np.histogram(ref, bins=edges)[0]
        curr_counts = np.histogram(curr, bins=edges)[0]
    else:
        values = ref.value_counts().index.union(curr.value_counts().index)
        ref_counts = ref.value_counts().reindex(values, fill_value=0).to_numpy()
        curr_counts = curr.value_counts().reindex(values, fill_value=0).to_numpy()
    ref_share = np.clip(ref_counts / max(len(ref), 1), PSI_EPS, None)
    curr_share = np.clip(curr_counts / max(len(curr), 1), PSI_EPS, None)
    return ref_share, curr_share


def population_stability_index(ref: pd.Series, curr: pd.Series) -> float:
    """PSI = Σ (curr − ref) · ln(curr / ref) по интервалам или значениям."""
    ref_share, curr_share = _psi_shares(ref.dropna(), curr.dropna())
    return float(np.sum((curr_share - ref_share) * np.log(curr_share / ref_share)))


def psi_level(psi: float) -> str:
    """Словесная оценка PSI по стандартной шкале."""
    if psi > PSI_SIGNIFICANT:
        return "значимый сдвиг"
    if psi >= PSI_MODERATE:
        return "умеренный сдвиг"
    return "стабильно"


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
        Сводка: по признаку — тест, статистика, p-value, PSI, признак
        дрейфа, средние; итог — число признаков с дрейфом и сообщение.
        Дрейф по признаку — если его показал статистический тест или
        PSI > 0.25.
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

        psi = population_stability_index(ref_values, curr_values)
        results[-1]["psi"] = round(psi, 4)
        results[-1]["psi_level"] = psi_level(psi)
        if psi > PSI_SIGNIFICANT:
            results[-1]["drift_detected"] = True

    drift_count = sum(1 for r in results if r["drift_detected"])
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_features": len(feature_columns),
        "drift_features": drift_count,
        "p_threshold": p_threshold,
        "psi_threshold": PSI_SIGNIFICANT,
        "max_psi": max((r["psi"] for r in results), default=0.0),
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
