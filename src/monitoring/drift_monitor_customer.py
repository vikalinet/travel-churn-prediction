"""
Мониторинг дрейфа данных с помощью Evidently AI.
Адаптировано для датасета Customer Travel.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerTravelDriftMonitor:
    """Мониторинг дрейфа для датасета Customer Travel."""

    def __init__(self, reference_data: pd.DataFrame, feature_columns: List[str]):
        """
        Инициализация монитора.

        Args:
            reference_data: Базовый датасет (обучающая выборка)
            feature_columns: Список колонок для мониторинга
        """
        self.reference_data = reference_data[feature_columns].copy()
        self.feature_columns = feature_columns
        self.current_data = None
        self.report_count = 0

    def update_current_data(self, new_data: pd.DataFrame):
        """Обновление текущих данных."""
        if self.current_data is None:
            self.current_data = new_data[self.feature_columns].copy()
        else:
            self.current_data = pd.concat(
                [self.current_data, new_data[self.feature_columns].copy()],
                ignore_index=True,
            )

        logger.info(f"Обновлены текущие данные: {len(self.current_data)} записей")

    def generate_drift_report(self, output_path: Optional[str] = None) -> str:
        """
        Генерация отчёта о дрейфе данных.

        Args:
            output_path: Путь для сохранения HTML отчёта

        Returns:
            Путь к сохранённому отчёту
        """
        try:
            from evidently.report import Report
            from evidently.metrics import DataDriftTable

            logger.info("Генерация отчёта о дрейфе данных...")

            # Проверка на достаточный объём данных
            if len(self.reference_data) < 10 or len(self.current_data) < 10:
                logger.warning("Мало данных для генерации отчёта")
                return None

            # Создание отчёта
            report = Report(
                metrics=[
                    DataDriftTable(column_names=self.feature_columns),
                ]
            )

            report.run(
                reference_data=self.reference_data, current_data=self.current_data
            )

            # Сохранение отчёта
            self.report_count += 1
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"evidently_reports/drift_report_{timestamp}.html"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            report.save_html(output_path)

            logger.info(f"Отчёт о дрейфе сохранён: {output_path}")

            return output_path

        except ImportError:
            logger.error(
                "Evidently AI не установлен. Установите: pip install evidently"
            )
            return None
        except Exception as e:
            logger.error(f"Ошибка при генерации отчёта: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def calculate_drift_summary(self) -> dict:
        """
        Расчёт сводки по дрейфу.

        Returns:
            Словарь с метриками дрейфа
        """
        drift_summary = {
            "timestamp": datetime.now().isoformat(),
            "reference_size": len(self.reference_data),
            "current_size": len(self.current_data),
            "features": {},
        }

        for column in self.feature_columns:
            if (
                column not in self.reference_data.columns
                or column not in self.current_data.columns
            ):
                continue

            ref_values = self.reference_data[column].dropna()
            curr_values = self.current_data[column].dropna()

            # KS-тест для числовых признаков
            if pd.api.types.is_numeric_dtype(ref_values):
                from scipy import stats

                stat, p_value = stats.ks_2samp(ref_values, curr_values)
                drift_summary["features"][column] = {
                    "type": "numeric",
                    "ks_statistic": float(stat),
                    "p_value": float(p_value),
                    "drift_detected": p_value < 0.05,
                    "ref_mean": float(ref_values.mean()),
                    "curr_mean": float(curr_values.mean()),
                    "ref_std": float(ref_values.std()),
                    "curr_std": float(curr_values.std()),
                }

            # Для категориальных
            else:
                ref_counts = ref_values.value_counts(normalize=True)
                curr_counts = curr_values.value_counts(normalize=True)

                # Расстояние между распределениями
                all_categories = ref_counts.index.union(curr_counts.index)
                ref_normalized = ref_counts.reindex(all_categories, fill_value=0)
                curr_normalized = curr_counts.reindex(all_categories, fill_value=0)

                js_divergence = 0.5 * (np.sum(np.abs(ref_normalized - curr_normalized)))

                drift_summary["features"][column] = {
                    "type": "categorical",
                    "js_divergence": float(js_divergence),
                    "drift_detected": js_divergence > 0.2,
                    "ref_distribution": ref_counts.to_dict(),
                    "curr_distribution": curr_counts.to_dict(),
                }

        # Общее количество признаков с дрейфом
        drift_count = sum(
            1
            for f in drift_summary["features"].values()
            if f.get("drift_detected", False)
        )
        drift_summary["total_drift_features"] = drift_count
        drift_summary["drift_ratio"] = drift_count / len(self.feature_columns)

        return drift_summary

    def check_critical_drift(self, threshold: float = 0.05) -> bool:
        """
        Проверка наличия критического дрейфа.

        Args:
            threshold: Порог p-value для определения дрейфа

        Returns:
            True если обнаружен критический дрейф
        """
        summary = self.calculate_drift_summary()

        drift_features = [
            col
            for col, data in summary["features"].items()
            if data.get("drift_detected", False)
        ]

        if drift_features:
            logger.warning(f"Обнаружен дрейф в колонках: {drift_features}")
            return True

        logger.info("Критический дрейф не обнаружен")
        return False


def create_monitor_from_training_data(data_path: str, test_size: float = 0.2):
    """
    Создание монитора из обработанных данных обучения.

    Args:
        data_path: Путь к обработанному датасету
        test_size: Доля тестовой выборки

    Returns:
        CustomerTravelDriftMonitor объект
    """
    df = pd.read_csv(data_path)

    # Разделение на train (reference) и test (current)
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Признаки (без Target)
    feature_columns = [col for col in df.columns if col != "Target"]

    monitor = CustomerTravelDriftMonitor(
        reference_data=train_df, feature_columns=feature_columns
    )

    monitor.update_current_data(test_df)

    return monitor, feature_columns


def main():
    """Запуск мониторинга дрейфа."""
    import sys

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "data/processed/processed_data.csv"

    logger.info("=== Запуск мониторинга дрейфа данных ===")

    # Создание монитора
    monitor, feature_columns = create_monitor_from_training_data(data_file)

    logger.info(f"Признаки для мониторинга: {feature_columns}")
    logger.info(f"Reference размер: {len(monitor.reference_data)}")
    logger.info(f"Current размер: {len(monitor.current_data)}")

    # Генерация отчёта
    report_path = monitor.generate_drift_report()

    if report_path:
        logger.info(f"HTML отчёт: {report_path}")

    # Сводка
    summary = monitor.calculate_drift_summary()

    logger.info("\n=== Сводка по дрейфу ===")
    logger.info(f"Всего признаков: {len(feature_columns)}")
    logger.info(f"Признаков с дрейфом: {summary['total_drift_features']}")
    logger.info(f"Доля признаков с дрейфом: {summary['drift_ratio']:.2%}")

    logger.info("\nДетали по признакам:")
    for feature, data in summary["features"].items():
        drift_status = "⚠️ ДРЕЙФ" if data.get("drift_detected", False) else "✅ OK"
        logger.info(f"  {feature}: {drift_status}")

        if data["type"] == "numeric":
            logger.info(
                f"    KS-statistic: {data['ks_statistic']:.4f}, p-value: {data['p_value']:.4f}"
            )
            logger.info(
                f"    Ref: mean={data['ref_mean']:.2f}, std={data['ref_std']:.2f}"
            )
            logger.info(
                f"    Curr: mean={data['curr_mean']:.2f}, std={data['curr_std']:.2f}"
            )
        else:
            logger.info(f"    JS-divergence: {data['js_divergence']:.4f}")

    # Проверка критического дрейфа
    if monitor.check_critical_drift():
        logger.warning("⚠️ Обнаружен критический дрейф! Требуется переобучение модели.")
    else:
        logger.info("✅ Дрейф в пределах нормы")

    # Сохранение сводки
    import json

    summary_path = "evidently_reports/drift_summary.json"
    Path("evidently_reports").mkdir(exist_ok=True)

    # Преобразование для JSON
    json_summary = {
        "timestamp": summary["timestamp"],
        "reference_size": summary["reference_size"],
        "current_size": summary["current_size"],
        "total_drift_features": summary["total_drift_features"],
        "drift_ratio": summary["drift_ratio"],
        "features": {},
    }

    for feature, data in summary["features"].items():
        json_summary["features"][feature] = {
            "type": data["type"],
            "drift_detected": bool(data["drift_detected"]),
        }
        if data["type"] == "numeric":
            json_summary["features"][feature].update(
                {
                    "ks_statistic": float(data["ks_statistic"]),
                    "p_value": float(data["p_value"]),
                    "ref_mean": float(data["ref_mean"]),
                    "curr_mean": float(data["curr_mean"]),
                }
            )
        else:
            json_summary["features"][feature].update(
                {"js_divergence": float(data["js_divergence"])}
            )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\nСводка сохранена в {summary_path}")
    logger.info("=== Мониторинг дрейфа завершён ===")


if __name__ == "__main__":
    main()
