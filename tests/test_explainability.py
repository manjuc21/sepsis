from __future__ import annotations

from src.data import features
from src.explainability.shap_explainer import XGBoostExplainer, _readable_name
from src.models.baseline import XGBoostModel


def test_readable_name_covers_engineered_suffixes():
    assert "trend" in _readable_name("HR_roll_slope")
    assert "not measured" in _readable_name("Lactate_missing")
    assert "last" in _readable_name("MAP_roll_mean")


def test_xgboost_explainer_output_shape(synthetic_patients):
    X, y = features.build_tabular_feature_matrix(synthetic_patients)
    model = XGBoostModel().fit(X, y)
    explainer = XGBoostExplainer(model, list(X.columns))

    result = explainer.explain(X.iloc[[0]], top_n=5)

    assert isinstance(result, list)
    assert len(result) == 5
    assert all(isinstance(item, str) for item in result)
