import json
from datetime import datetime

# Загрузка данных
with open("evidently_reports/drift_summary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Чтение шаблона
with open("evidently_reports/drift_report_template.html", "r", encoding="utf-8") as f:
    html = f.read()

# Запись финального отчёта
with open("evidently_reports/drift_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"Отчёт создан: evidently_reports/drift_report.html")
print(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
