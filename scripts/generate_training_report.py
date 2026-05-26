"""
Генерация отчёта о времени обучения и производительности модели.
"""

import io
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def measure_training_time():
    """Измерение времени обучения моделей."""

    results = []

    # Загрузка данных
    data_path = "data/processed/processed_data.csv"
    if not Path(data_path).exists():
        print(f"Файл {data_path} не найден!")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=["Target"])
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Данные загружены: {len(df)} строк")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 1. GradientBoosting
    print("\nОбучение GradientBoosting...")
    start = time.time()

    model_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_gb.fit(X_train, y_train)
    time_gb = time.time() - start

    y_pred_gb = model_gb.predict(X_test)
    y_proba_gb = model_gb.predict_proba(X_test)[:, 1]

    results.append(
        {
            "model": "GradientBoosting",
            "training_time_sec": time_gb,
            "accuracy": accuracy_score(y_test, y_pred_gb),
            "f1_score": f1_score(y_test, y_pred_gb),
            "roc_auc": roc_auc_score(y_test, y_proba_gb),
        }
    )
    print(f"  Время: {time_gb:.2f} сек")

    # 2. RandomForest
    print("\nОбучение RandomForest...")
    start = time.time()

    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    time_rf = time.time() - start

    y_pred_rf = model_rf.predict(X_test)
    y_proba_rf = model_rf.predict_proba(X_test)[:, 1]

    results.append(
        {
            "model": "RandomForest",
            "training_time_sec": time_rf,
            "accuracy": accuracy_score(y_test, y_pred_rf),
            "f1_score": f1_score(y_test, y_pred_rf),
            "roc_auc": roc_auc_score(y_test, y_proba_rf),
        }
    )
    print(f"  Время: {time_rf:.2f} сек")

    # 3. XGBoost
    print("\nОбучение XGBoost...")
    start = time.time()

    model_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model_xgb.fit(X_train, y_train)
    time_xgb = time.time() - start

    y_pred_xgb = model_xgb.predict(X_test)
    y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

    results.append(
        {
            "model": "XGBoost",
            "training_time_sec": time_xgb,
            "accuracy": accuracy_score(y_test, y_pred_xgb),
            "f1_score": f1_score(y_test, y_pred_xgb),
            "roc_auc": roc_auc_score(y_test, y_proba_xgb),
        }
    )
    print(f"  Время: {time_xgb:.2f} сек")

    # 4. LogisticRegression
    print("\nОбучение LogisticRegression...")
    start = time.time()

    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train, y_train)
    time_lr = time.time() - start

    y_pred_lr = model_lr.predict(X_test)
    y_proba_lr = model_lr.predict_proba(X_test)[:, 1]

    results.append(
        {
            "model": "LogisticRegression",
            "training_time_sec": time_lr,
            "accuracy": accuracy_score(y_test, y_pred_lr),
            "f1_score": f1_score(y_test, y_pred_lr),
            "roc_auc": roc_auc_score(y_test, y_proba_lr),
        }
    )
    print(f"  Время: {time_lr:.2f} сек")

    # Генерация отчёта
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        <p><strong>Общее количество строк:</strong> {len(df)}</p>
        <p><strong>Количество признаков:</strong> {len(X.columns)}</p>
        <p><strong>Размер обучающей выборки:</strong> {len(X_train)}</p>
        <p><strong>Размер тестовой выборки:</strong> {len(X_test)}</p>
    </div>

    <h2 style="color: #333; margin-bottom: 15px;">📈 Результаты обучения</h2>
    <table>
        <thead>
            <tr>
                <th>Модель</th>
                <th>Время обучения (сек)</th>
                <th>Accuracy</th>
                <th>F1-Score</th>
                <th>ROC AUC</th>
            </tr>
        </thead>
        <tbody>
"""

    # Нахождение лучших моделей
    best_f1 = max(results, key=lambda x: x["f1_score"])

    for r in results:
        row_class = "best" if r["model"] == best_f1["model"] else ""
        html_content += f"""
            <tr class="{row_class}">
                <td><strong>{r["model"]}</strong></td>
                <td>{r["training_time_sec"]:.2f}</td>
                <td>{r["accuracy"]:.4f}</td>
                <td>{r["f1_score"]:.4f}</td>
                <td>{r["roc_auc"]:.4f}</td>
            </tr>
"""

    html_content += f"""
        </tbody>
    </table>

    <div class="info-box" style="margin-top: 30px;">
        <h3>🏆 Лучшие результаты</h3>
        <p><strong>Лучшая модель по F1-Score:</strong> {best_f1["model"]}</p>
        <p><strong>F1-Score:</strong> {best_f1["f1_score"]:.4f}</p>
        <p><strong>Время обучения:</strong> {best_f1["training_time_sec"]:.2f} сек</p>
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

    # Сохранение CSV
    results_df = pd.DataFrame(results)
    csv_path = output_dir / "training_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"✅ CSV результаты сохранены: {csv_path}")

    # Вывод итогов
    print("\n=== Итоги ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    # Установка UTF-8 кодировки для Windows
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=== Запуск измерения времени обучения ===")
    measure_training_time()
    print("\n=== Измерение завершено ===")
