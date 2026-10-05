"""
Имитация входящих запросов — проверка мониторинга дрейфа на запущенном API.

Отправляет клиентов в POST /api/v1/predict_batch (они попадают в журнал
runtime/monitoring/current_data.csv), затем запускает анализ дрейфа и
печатает итог. Клиенты берутся из сырого датасета; с --shift — «другая
аудитория»: старше и без статуса часто летающего, — и дрейф должен найтись.

    python scripts/simulate_traffic.py                       # без дрейфа
    python scripts/simulate_traffic.py --shift               # с дрейфом
    python scripts/simulate_traffic.py --url http://localhost:8000 -n 100

Журнал копится: чтобы начать заново, удалите runtime/monitoring/current_data.csv
(в Docker — том churn_runtime).
"""

import argparse
import glob
import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import settings  # noqa: E402

API_FIELDS = {
    "Age": "age",
    "FrequentFlyer": "frequent_flyer",
    "AnnualIncomeClass": "annual_income_class",
    "ServicesOpted": "services_opted",
    "AccountSyncedToSocialMedia": "account_synced_to_social_media",
    "BookedHotelOrNot": "booked_hotel_or_not",
}


def make_customers(n: int, shift: bool, seed: Optional[int] = None) -> list:
    raw = pd.read_csv(glob.glob(str(settings.RAW_DATA_DIR / "*.csv"))[0])
    sample = raw.sample(n=n, replace=n > len(raw), random_state=seed)
    if shift:
        sample = sample.assign(Age=sample["Age"] + 8, FrequentFlyer="No")
    return [
        {API_FIELDS[k]: v for k, v in row.items() if k in API_FIELDS}
        for row in sample.to_dict("records")
    ]


def call(url: str, data=None, method: str = "POST") -> dict:
    body = None if data is None else json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("-n", type=int, default=60, help="Сколько клиентов отправить")
    parser.add_argument("--shift", action="store_true", help="Сдвинутая аудитория")
    parser.add_argument(
        "--seed", type=int, default=None, help="Для повторяемой выборки клиентов"
    )
    args = parser.parse_args()

    customers = make_customers(args.n, args.shift, args.seed)
    predictions = call(f"{args.url}/api/v1/predict_batch", customers)["predictions"]
    churn = sum(p["prediction"] for p in predictions)
    print(f"Отправлено клиентов: {len(predictions)}, прогноз оттока: {churn}")

    call(f"{args.url}/api/v1/drift/analyze")
    summary = call(f"{args.url}/api/v1/drift/status", method="GET")
    print(f"\nСравниваются: {summary.get('source_label')}")
    print(f"Итог: {summary.get('message')}")
    print(f"Макс. PSI: {summary.get('max_psi')}")
    for row in summary.get("results", []):
        mark = "ДРЕЙФ" if row["drift_detected"] else "ok"
        print(
            f"  {row['feature']:<28} PSI {row['psi']:<8} {row['psi_level']:<16} {mark}"
        )
    print(f"\nДашборд: {args.url}/drift")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
