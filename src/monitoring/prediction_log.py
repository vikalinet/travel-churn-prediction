"""
Журнал входящих данных API — «текущие» данные для анализа дрейфа.

Каждый клиент из /predict и /predict_batch дописывается в CSV в папке
выполнения в том же виде, что data/processed/processed_data.csv (без Target).
Когда записей наберётся достаточно, анализ дрейфа сравнивает их со всей
обучающей выборкой; до этого — эталонную и тестовую части датасета.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src import settings

logger = logging.getLogger(__name__)

# Меньше — сравнение долей и квантилей слишком шумное.
MIN_LIVE_ROWS = 50


def log_path() -> Path:
    return settings.runtime_dir() / "monitoring" / "current_data.csv"


def log_inputs(encoded: pd.DataFrame) -> None:
    """Дописать закодированных клиентов в журнал.

    Ошибка записи не должна ломать предсказание — только пишется в лог.
    """
    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded.to_csv(path, mode="a", header=not path.exists(), index=False)
    except OSError as e:
        logger.error(f"Не удалось записать входные данные в журнал: {e}")


def load_logged() -> Optional[pd.DataFrame]:
    """Накопленные входные данные или None, если журнала ещё нет."""
    path = log_path()
    if not path.exists():
        return None
    return pd.read_csv(path)
