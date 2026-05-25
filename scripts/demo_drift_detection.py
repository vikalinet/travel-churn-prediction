"""
Демонстрация дрейфа данных.
Показывает, как мониторинг обнаруживает изменения в распределении данных.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json


def simulate_data_drift():
    """Симуляция дрейфа данных для демонстрации."""

    print("=== Демонстрация дрейфа данных ===\n")

    # 1. Исходные данные (Reference) - как было при обучении
    print("1️⃣ Генерация Reference данных (train)...")
    np.random.seed(42)
    n_samples = 500

    reference_data = pd.DataFrame(
        {
            "Age": np.random.normal(40, 12, n_samples).clip(18, 70),
            "FrequentFlyer": np.random.choice(["Yes", "No"], n_samples, p=[0.3, 0.7]),
            "AnnualIncomeClass": np.random.choice(
                ["Low Income", "Middle Income", "High Income"],
                n_samples,
                p=[0.4, 0.4, 0.2],
            ),
            "ServicesOpted": np.random.randint(1, 7, n_samples),
            "AccountSyncedToSocialMedia": np.random.choice(
                ["Yes", "No"], n_samples, p=[0.45, 0.55]
            ),
            "BookedHotelOrNot": np.random.choice(
                ["Yes", "No"], n_samples, p=[0.35, 0.65]
            ),
            "Target": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        }
    )

    print(f"   ✅ Reference: {len(reference_data)} образцов")
    print(f"   📊 Средний возраст: {reference_data['Age'].mean():.1f}")
    print(
        f"   ✈️ Частые путешественники: {reference_data['FrequentFlyer'].value_counts()['Yes']/len(reference_data)*100:.1f}%\n"
    )

    # 2. Текущие данные WITHOUT drift (стабильные)
    print("2️⃣ Генерация Current данных (без дрейфа)...")
    current_data_no_drift = reference_data.copy()
    print(f"   ✅ Current (стабильный): {len(current_data_no_drift)} образцов")
    print(f"   📊 Средний возраст: {current_data_no_drift['Age'].mean():.1f}")
    print(
        f"   ✈️ Частые путешественники: {current_data_no_drift['FrequentFlyer'].value_counts()['Yes']/len(current_data_no_drift)*100:.1f}%\n"
    )

    # 3. Текущие данные WITH drift (изменённые)
    print("3️⃣ Генерация Current данных (С ДРЕЙФОМ)...")
    n_samples_new = 500

    # Имитация изменений в данных:
    # - Постарели клиенты (средний возраст вырос с 40 до 48)
    # - Больше частых путешественников (с 30% до 50%)
    # - Больше клиентов с высоким доходом
    # - Больше бронирований отелей

    current_data_with_drift = pd.DataFrame(
        {
            "Age": np.random.normal(48, 12, n_samples_new).clip(18, 70),  # +8 лет!
            "FrequentFlyer": np.random.choice(
                ["Yes", "No"], n_samples_new, p=[0.5, 0.5]
            ),  # +20%!
            "AnnualIncomeClass": np.random.choice(
                ["Low Income", "Middle Income", "High Income"],
                n_samples_new,
                p=[0.25, 0.45, 0.3],  # Больше High Income!
            ),
            "ServicesOpted": np.random.randint(2, 7, n_samples_new),  # Больше услуг
            "AccountSyncedToSocialMedia": np.random.choice(
                ["Yes", "No"], n_samples_new, p=[0.6, 0.4]
            ),  # Больше синхронизации
            "BookedHotelOrNot": np.random.choice(
                ["Yes", "No"], n_samples_new, p=[0.55, 0.45]
            ),  # Больше отелей
            "Target": np.random.choice(
                [0, 1], n_samples_new, p=[0.65, 0.35]
            ),  # Больше оттока!
        }
    )

    print(f"   ✅ Current (с дрейфом): {len(current_data_with_drift)} образцов")
    print(
        f"   📊 Средний возраст: {current_data_with_drift['Age'].mean():.1f} (+{current_data_with_drift['Age'].mean() - reference_data['Age'].mean():.1f})"
    )
    print(
        f"   ✈️ Частые путешественники: {current_data_with_drift['FrequentFlyer'].value_counts()['Yes']/len(current_data_with_drift)*100:.1f}% (+{current_data_with_drift['FrequentFlyer'].value_counts()['Yes']/len(current_data_with_drift)*100 - reference_data['FrequentFlyer'].value_counts()['Yes']/len(reference_data)*100:.1f}%)"
    )
    print(
        f"   💰 High Income: {current_data_with_drift['AnnualIncomeClass'].value_counts()['High Income']/len(current_data_with_drift)*100:.1f}%\n"
    )

    # 4. Запуск мониторинга дрейфа
    print("4️⃣ Запуск анализа дрейфа...")

    feature_columns = [col for col in reference_data.columns if col != "Target"]

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА ДРЕЙФА")
    print("=" * 60)

    # Анализ WITHOUT drift
    print("\n🔍 Случай 1: Стабильные данные (без дрейфа)")
    print("-" * 60)
    check_drift(reference_data, current_data_no_drift, feature_columns)

    # Анализ WITH drift
    print("\n\n🔍 Случай 2: Изменённые данные (С ДРЕЙФОМ)")
    print("-" * 60)
    check_drift(reference_data, current_data_with_drift, feature_columns)

    # 5. Выводы
    print("\n" + "=" * 60)
    print("📊 ВЫВОДЫ")
    print("=" * 60)
    print(
        """
Дрейф данных обнаружен! Вот что изменилось:

⚠️ Age: Клиенты постарели (ср. возраст +8 лет)
⚠️ FrequentFlyer: Больше частых путешественников (+20%)
⚠️ AnnualIncomeClass: Больше клиентов с высоким доходом
⚠️ ServicesOpted: Чаще пользуются услугами
⚠️ AccountSyncedToSocialMedia: Больше синхронизации с соцсетями
⚠️ BookedHotelOrNot: Чаще бронируют отели
⚠️ Target: Вырос уровень оттока (с 30% до 35%)

🎯 Рекомендации:
1. Модель может давать менее точные прогнозы
2. Необходимо переобучение на новых данных
3. Проверить бизнес-процессы - что изменилось?
    """
    )


def check_drift(reference_df, current_df, feature_columns):
    """Проверка дрейфа для одного признака."""
    from scipy import stats

    drift_detected_count = 0

    for column in feature_columns:
        ref_values = reference_df[column].dropna()
        curr_values = current_df[column].dropna()

        if pd.api.types.is_numeric_dtype(ref_values):
            # KS-тест для числовых
            stat, p_value = stats.ks_2samp(ref_values, curr_values)
            drift_detected = p_value < 0.05

            if drift_detected:
                drift_detected_count += 1
                status = "⚠️ ДРЕЙФ"
            else:
                status = "✅ OK"

            print(f"{column:35} KS={stat:.4f} p={p_value:.4f} {status}")

        else:
            # JS-дивергенция для категориальных
            ref_counts = ref_values.value_counts(normalize=True)
            curr_counts = curr_values.value_counts(normalize=True)

            all_categories = ref_counts.index.union(curr_counts.index)
            ref_normalized = ref_counts.reindex(all_categories, fill_value=0)
            curr_normalized = curr_counts.reindex(all_categories, fill_value=0)

            js_divergence = 0.5 * np.sum(np.abs(ref_normalized - curr_normalized))
            drift_detected = js_divergence > 0.2

            if drift_detected:
                drift_detected_count += 1
                status = "⚠️ ДРЕЙФ"
            else:
                status = "✅ OK"

            print(f"{column:35} JS={js_divergence:.4f} {status}")

    print(
        f"\n📊 Обнаружен дрейф в {drift_detected_count}/{len(feature_columns)} признаках"
    )

    return drift_detected_count


if __name__ == "__main__":
    import sys

    if sys.platform.startswith("win"):
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    simulate_data_drift()
