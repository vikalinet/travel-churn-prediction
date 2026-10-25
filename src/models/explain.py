"""Локальная объяснимость прогноза через SHAP — без переобучения модели."""

from typing import Any, Dict, List

import pandas as pd
import shap

from src.features.engineering import FeatureEngineer

RU_FEATURE_NAMES = {
    "Age": "Возраст",
    "FrequentFlyer": "Частый перелётчик",
    "AnnualIncomeClass": "Уровень дохода",
    "ServicesOpted": "Число услуг",
    "AccountSyncedToSocialMedia": "Соцсети синхронизированы",
    "BookedHotelOrNot": "Бронировал отель",
    "services_per_age": "Услуги относительно возраста",
    "flyer_and_hotel": "Частый перелётчик и отель",
    "income_x_services": "Доход и число услуг",
}


def _readable_label(feature: str, poly_rename: Dict[str, str]) -> str:
    real_name = poly_rename.get(feature, feature)
    return " × ".join(RU_FEATURE_NAMES.get(part, part) for part in real_name.split(" "))


class PredictionExplainer:
    """SHAP TreeExplainer поверх уже обученной модели."""

    def __init__(self, model: Any, feature_engineer: FeatureEngineer):
        self._explainer = shap.TreeExplainer(model)
        self._poly_rename = dict(
            zip(
                feature_engineer.poly_feature_names or [],
                feature_engineer.poly_feature_labels(),
            )
        )

    def top_factors(self, X: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """Главные признаки, повлиявшие на прогноз для одной строки X."""
        shap_values = self._explainer.shap_values(X)
        contribution = pd.Series(shap_values[0], index=X.columns)
        ranked = contribution.reindex(
            contribution.abs().sort_values(ascending=False).index
        )
        return [
            {
                "feature": _readable_label(feature, self._poly_rename),
                "contribution": round(float(value), 4),
                "direction": "к оттоку" if value > 0 else "удерживает",
            }
            for feature, value in ranked.head(top_n).items()
        ]
