"""
Генерация отчёта о времени обучения и производительности модели.
Загружает актуальные результаты из training_results.csv.
"""

import io
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def generate_training_report():
    """Генерация отчёта на основе актуальных результатов обучения."""

    # Загрузка результатов из CSV (сгенерированы при обучении модели)
    results_path = "reports/training_results.csv"
    if not Path(results_path).exists():
        print(f"Файл {results_path} не найден! Запустите обучение модели сначала.")
        return

    results_df = pd.read_csv(results_path)

    print(f"Данные загружены: {len(results_df)} моделей")
    print(results_df.to_string(index=False))

    # Загрузка данных для информации
    data_path = "data/processed/processed_data.csv"
    if Path(data_path).exists():
        df = pd.read_csv(data_path)
        data_info = {
            "total_rows": len(df),
            "features": len(df.columns) - 1,
        }
    else:
        data_info = {"total_rows": "N/A", "features": "N/A"}

    # Генерация отчёта
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Нахождение лучшей модели (по F1-score, если есть, или по Accuracy)
    if "f1_score" in results_df.columns:
        best_row = results_df.loc[results_df["f1_score"].idxmax()]
        best_model_name = best_row["model_name"]
        best_f1 = best_row["f1_score"]
    else:
        best_row = results_df.loc[results_df["accuracy"].idxmax()]
        best_model_name = best_row["model_name"]
        best_f1 = best_row["accuracy"]
    # HTML отчёт
    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчёт по обучению моделей</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
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
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #27ae60;
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
        .best {{
            background-color: #d4edda;
            font-weight: bold;
        }}
        .info-box {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .info-box h3 {{
            margin-top: 0;
            color: #27ae60;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⏱️ Отчёт по обучению моделей</h1>
        <div class="timestamp">Сгенерировано: {timestamp}</div>
    </div>

    <div class="info-box">
        <h3>📊 Информация о данных</h3>
        <p><strong>Общее количество строк:</strong> {data_info['total_rows']}</p>
        <p><strong>Количество признаков:</strong> {data_info['features']}</p>
        <p><strong>Количество обученных моделей:</strong> {len(results_df)}</p>
    </div>

    <h2 style="color: #333; margin-bottom: 15px;">📈 Результаты обучения</h2>
    <table>
        <thead>
            <tr>
                <th>Модель</th>
                <th>Accuracy</th>
                <th>F1-Score</th>
                <th>ROC AUC</th>
                <th>Precision</th>
                <th>Recall</th>
            </tr>
        </thead>
        <tbody>
"""

    # Сортировка по F1-score (убывание)
    results_sorted = results_df.sort_values("f1_score", ascending=False)

    for _, r in results_sorted.iterrows():
        row_class = "best" if r["model_name"] == best_model_name else ""
        html_content += f"""
            <tr class="{row_class}">
                <td><strong>{r["model_name"]}</strong></td>
                <td>{r["accuracy"]:.4f}</td>
                <td>{r["f1_score"]:.4f}</td>
                <td>{r["roc_auc"]:.4f}</td>
                <td>{r.get("precision", 0):.4f}</td>
                <td>{r.get("recall", 0):.4f}</td>
            </tr>
"""

    html_content += f"""
        </tbody>
    </table>

    <div class="info-box" style="margin-top: 30px;">
        <h3>🏆 Лучшие результаты</h3>
        <p><strong>Лучшая модель по F1-Score:</strong> {best_model_name}</p>
        <p><strong>F1-Score:</strong> {best_f1:.4f}</p>
        <p><strong>Accuracy:</strong> {results_sorted.iloc[0]["accuracy"]:.4f}</p>
        <p><strong>ROC AUC:</strong> {results_sorted.iloc[0]["roc_auc"]:.4f}</p>
    </div>

    <div style="text-align: center; margin-top: 30px; color: #666; font-size: 14px;">
        <p>Отчёт сгенерирован автоматически для проекта "Прогнозирование оттока клиентов"</p>
    </div>
</body>
</html>
"""

    # Сохранение HTML
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    html_path = output_dir / "training_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ HTML отчёт сохранён: {html_path}")

    # Вывод итогов
    print("\n=== Итоги ===")
    print(results_sorted.to_string(index=False))


if __name__ == "__main__":
    # Установка UTF-8 кодировки для Windows
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=== Запуск генерации отчёта по обучению ===")
    generate_training_report()
    print("\n=== Отчёт сгенерирован ===")
