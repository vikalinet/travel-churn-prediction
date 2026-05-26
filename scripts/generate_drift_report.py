"""
Генерация HTML отчёта по мониторингу дрейфа данных.
"""

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


def generate_drift_html_report(data_path: str = "data/processed/processed_data.csv"):
    """Генерация HTML отчёта по дрейфу данных."""

    # Загрузка данных
    df = pd.read_csv(data_path)

    # Разделение на train и test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    feature_columns = [col for col in df.columns if col != "Target"]

    # Расчёт метрик дрейфа
    drift_results = []

    for column in feature_columns:
        ref_values = train_df[column].dropna()
        curr_values = test_df[column].dropna()

        if pd.api.types.is_numeric_dtype(ref_values):
            stat, p_value = stats.ks_2samp(ref_values, curr_values)
            drift_detected = p_value < 0.05

            drift_results.append(
                {
                    "feature": column,
                    "type": "numeric",
                    "ks_statistic": float(stat),
                    "p_value": float(p_value),
                    "drift_detected": drift_detected,
                    "ref_mean": float(ref_values.mean()),
                    "curr_mean": float(curr_values.mean()),
                    "ref_std": float(ref_values.std()),
                    "curr_std": float(curr_values.std()),
                }
            )
        else:
            ref_counts = ref_values.value_counts(normalize=True)
            curr_counts = curr_values.value_counts(normalize=True)

            all_categories = ref_counts.index.union(curr_counts.index)
            ref_normalized = ref_counts.reindex(all_categories, fill_value=0)
            curr_normalized = curr_counts.reindex(all_categories, fill_value=0)

            js_divergence = 0.5 * np.sum(np.abs(ref_normalized - curr_normalized))
            drift_detected = js_divergence > 0.2

            drift_results.append(
                {
                    "feature": column,
                    "type": "categorical",
                    "js_divergence": float(js_divergence),
                    "drift_detected": drift_detected,
                }
            )

    # Генерация HTML
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    drift_count = sum(1 for r in drift_results if r.get("drift_detected", False))

    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчёт по мониторингу дрейфа данных</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .timestamp {{
            margin-top: 10px;
            opacity: 0.9;
            font-size: 14px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }}
        .summary-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .status-ok {{
            color: #27ae60;
        }}
        .status-warning {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .drift-badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .drift-ok {{
            background-color: #d4edda;
            color: #155724;
        }}
        .drift-warning {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Отчёт по мониторингу дрейфа данных</h1>
        <div class="timestamp">Сгенерировано: {timestamp}</div>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3>Всего признаков</h3>
            <div class="value">{len(feature_columns)}</div>
        </div>
        <div class="summary-card">
            <h3>Признаков с дрейфом</h3>
            <div class="value {'status-warning' if drift_count > 0 else 'status-ok'}">{drift_count}</div>
        </div>
        <div class="summary-card">
            <h3>Размер Reference (Train)</h3>
            <div class="value">{len(train_df)}</div>
        </div>
        <div class="summary-card">
            <h3>Размер Current (Test)</h3>
            <div class="value">{len(test_df)}</div>
        </div>
    </div>

    <h2 style="color: #333; margin-bottom: 15px;">📊 Детальный анализ по признакам</h2>
    <table>
        <thead>
            <tr>
                <th>Признак</th>
                <th>Тип</th>
                <th>Статус</th>
                <th>Статистика</th>
                <th>p-value</th>
                <th>Reference (Mean)</th>
                <th>Current (Mean)</th>
            </tr>
        </thead>
        <tbody>
"""

    for result in drift_results:
        status_class = (
            "drift-warning" if result.get("drift_detected", False) else "drift-ok"
        )
        status_text = "⚠️ ДРЕЙФ" if result.get("drift_detected", False) else "✅ OK"

        if result["type"] == "numeric":
            stats_text = f"KS: {result['ks_statistic']:.4f}"
            ref_mean = f"{result['ref_mean']:.2f}"
            curr_mean = f"{result['curr_mean']:.2f}"
        else:
            stats_text = f"JS: {result.get('js_divergence', 0):.4f}"
            ref_mean = "-"
            curr_mean = "-"

        html_content += f"""
            <tr>
                <td><strong>{result['feature']}</strong></td>
                <td>{result['type']}</td>
                <td><span class="drift-badge {status_class}">{status_text}</span></td>
                <td>{stats_text}</td>
                <td>{result['p_value']:.4f}</td>
                <td>{ref_mean}</td>
                <td>{curr_mean}</td>
            </tr>
"""

    html_content += """
        </tbody>
    </table>

    <div class="footer">
        <p>Отчёт сгенерирован автоматически для проекта "Прогнозирование оттока клиентов"</p>
    </div>
</body>
</html>
"""

    # Сохранение
    output_path = "evidently_reports/drift_report.html"
    Path("evidently_reports").mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML отчёт сохранён: {output_path}")

    # Также сохраним JSON сводку
    json_path = "evidently_reports/drift_summary.json"

    # Преобразование numpy bool в Python bool
    json_results = []
    for r in drift_results:
        json_result = {}
        for k, v in r.items():
            if isinstance(v, (np.bool_, np.integer)):
                json_result[k] = int(v)
            elif isinstance(v, np.floating):
                json_result[k] = float(v)
            elif isinstance(v, np.ndarray):
                json_result[k] = v.tolist()
            else:
                json_result[k] = v
        json_results.append(json_result)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "total_features": len(feature_columns),
                "drift_features": drift_count,
                "results": json_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"JSON сводка сохранена: {json_path}")

    return drift_results


if __name__ == "__main__":
    # Установка UTF-8 кодировки для Windows
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=== Запуск генерации отчёта мониторинга ===")
    results = generate_drift_html_report()
    print("\n=== Отчёт сгенерирован ===")
