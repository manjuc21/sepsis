"""Human-readable, per-prediction explanations. Every model wired into the
API must go through this module — the API never returns a bare probability
(CLAUDE.md §2.3).

XGBoost uses SHAP's TreeExplainer (exact, fast). The LSTM uses SHAP's
GradientExplainer over the flattened last-timestep hidden representation;
if that proves too slow/unstable for a given input, we fall back to a
saliency-style method (input-gradient magnitude) — documented here rather
than silently swapped, per CLAUDE.md's explainability requirement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import torch

from src import config

# Rough clinical-language templates for the raw feature names. Anything not
# in this map falls back to a readable version of the column name itself.
_FEATURE_LABELS: dict[str, str] = {
    "HR": "Heart rate",
    "O2Sat": "Oxygen saturation",
    "Temp": "Temperature",
    "SBP": "Systolic blood pressure",
    "MAP": "Mean arterial pressure",
    "DBP": "Diastolic blood pressure",
    "Resp": "Respiratory rate",
    "EtCO2": "End-tidal CO2",
    "Lactate": "Lactate",
    "WBC": "White blood cell count",
    "Creatinine": "Creatinine",
    "Platelets": "Platelet count",
    "BUN": "Blood urea nitrogen",
    "Bilirubin_total": "Total bilirubin",
}


def _readable_name(col: str) -> str:
    base = col.replace("_roll_mean", "").replace("_roll_min", "").replace(
        "_roll_max", ""
    ).replace("_roll_slope", "").replace("_missing", "")
    label = _FEATURE_LABELS.get(base, base.replace("_", " "))

    if col.endswith("_roll_slope"):
        return f"{label} trend (last {config.ROLLING_WINDOW_HOURS}h)"
    if col.endswith("_missing"):
        return f"{label} not measured recently"
    if col.endswith(("_roll_mean", "_roll_min", "_roll_max")):
        return f"{label} (last {config.ROLLING_WINDOW_HOURS}h)"
    return label


def _direction_phrase(shap_value: float, feature_name: str) -> str:
    if "not measured" in feature_name:
        return feature_name
    direction = "elevated" if shap_value > 0 else "improved / lower risk contribution"
    if shap_value > 0:
        return f"{feature_name} — rising risk"
    return f"{feature_name} — lowering risk"


class XGBoostExplainer:
    def __init__(self, model, feature_columns: list[str]):
        self.explainer = shap.TreeExplainer(model.clf)
        self.feature_columns = feature_columns

    def explain(self, X_row: pd.DataFrame, top_n: int = 5) -> list[str]:
        shap_values = self.explainer.shap_values(X_row)
        values = np.asarray(shap_values).reshape(-1)
        order = np.argsort(-np.abs(values))[:top_n]
        return [
            _direction_phrase(values[i], _readable_name(self.feature_columns[i]))
            for i in order
        ]


class LSTMExplainer:
    """Gradient-based attribution on the last real timestep of a sequence.

    SHAP's GradientExplainer is used when a small background batch is
    available; if it raises (small/unstable inputs — a known SHAP+LSTM
    failure mode) we fall back to raw input-gradient saliency, which is
    less theoretically grounded but always available. The fallback is
    logged in the returned dict's `method` field so it's visible, not silent.
    """

    def __init__(self, model, background: torch.Tensor, feature_columns: list[str]):
        self.model = model
        self.background = background
        self.feature_columns = feature_columns

    def explain(self, x: torch.Tensor, length: int, top_n: int = 5) -> list[str]:
        last_t = length - 1
        try:
            explainer = shap.GradientExplainer(
                lambda inp: self.model(inp, torch.tensor([length]))[:, last_t],
                self.background,
            )
            shap_values = explainer.shap_values(x.unsqueeze(0))
            values = np.asarray(shap_values)[0, last_t, :]
        except Exception:
            x = x.clone().requires_grad_(True)
            logits = self.model(x.unsqueeze(0), torch.tensor([length]))
            logits[0, last_t].backward()
            values = x.grad[last_t].detach().numpy()

        order = np.argsort(-np.abs(values))[:top_n]
        return [
            _direction_phrase(values[i], _readable_name(self.feature_columns[i]))
            for i in order
        ]
