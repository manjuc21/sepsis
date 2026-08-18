"""Loads the best available trained model artifact once at process startup
and wraps predict + explain into a single call used by the API layer."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd
import torch

from src import config
from src.api.schemas import PredictRequest, PredictResponse
from src.data import features
from src.explainability.shap_explainer import LSTMExplainer, XGBoostExplainer
from src.models.sequence import SepsisLSTM


def _pick_best_model_name() -> str:
    """Prefer whichever model has the best logged AUPRC in results/comparison.csv;
    fall back to whatever artifact exists on disk if no comparison table yet."""
    comparison_path = config.RESULTS_DIR / "comparison.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        if not df.empty and "auprc" in df.columns:
            best_row = df.loc[df["auprc"].idxmax()]
            name = best_row["model_name"]
            artifact = {
                "xgboost": config.XGBOOST_ARTIFACT,
                "logreg": config.LOGREG_ARTIFACT,
                "lstm": config.LSTM_ARTIFACT,
            }[name]
            if artifact.exists():
                return name

    for name, artifact in (
        ("xgboost", config.XGBOOST_ARTIFACT),
        ("logreg", config.LOGREG_ARTIFACT),
        ("lstm", config.LSTM_ARTIFACT),
    ):
        if artifact.exists():
            return name

    raise FileNotFoundError(
        "No trained model artifact found in models/. Run scripts/run_pipeline.py "
        "(or src/models/train.py) before starting the API."
    )


def _request_to_dataframe(request: PredictRequest) -> pd.DataFrame:
    rows = [row.model_dump() for row in request.window]
    df = pd.DataFrame(rows)
    df.insert(0, "patient_id", request.patient_id)
    df["Age"] = request.age
    df["Gender"] = request.gender
    df["Unit1"] = 0
    df["Unit2"] = 1
    df["HospAdmTime"] = request.hosp_adm_time
    df[config.LABEL_COLUMN] = 0  # unknown at inference time, required by shared feature code
    return df.sort_values("ICULOS").reset_index(drop=True)


class PredictionService:
    """Instantiate once at API startup — loading a model artifact and (for
    XGBoost) building the SHAP TreeExplainer is not cheap enough to redo
    per-request."""

    def __init__(self):
        self.model_name = _pick_best_model_name()

        if self.model_name in ("xgboost", "logreg"):
            with open(
                config.XGBOOST_ARTIFACT if self.model_name == "xgboost" else config.LOGREG_ARTIFACT,
                "rb",
            ) as f:
                artifact = pickle.load(f)
            self.model = artifact["model"]
            self.feature_columns = artifact["feature_columns"]
            self.explainer = (
                XGBoostExplainer(self.model, self.feature_columns)
                if self.model_name == "xgboost"
                else None
            )
        elif self.model_name == "lstm":
            checkpoint = torch.load(config.LSTM_ARTIFACT, map_location="cpu")
            self.model = SepsisLSTM(n_features=checkpoint["n_features"])
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self.max_len = checkpoint["max_len"]
            self.explainer = None  # built lazily per-request; needs a background sample

    def predict_and_explain(self, request: PredictRequest) -> PredictResponse:
        start = time.perf_counter()
        df = _request_to_dataframe(request)

        if self.model_name in ("xgboost", "logreg"):
            X, _ = features.build_tabular_feature_matrix(df)
            X = X[self.feature_columns]
            risk_score = float(self.model.predict_proba(X)[-1])
            if self.explainer is not None:
                top_features = self.explainer.explain(X.iloc[[-1]])
            else:
                top_features = ["Explanation unavailable for this model type"]
        else:
            X, y, mask, _ = features.build_sequence_tensors(df, self.max_len)
            length = int(mask[0].sum().item())
            lengths = mask.sum(dim=1)
            with torch.no_grad():
                logits = self.model(X, lengths)
                risk_score = float(torch.sigmoid(logits[0, length - 1]).item())

            explainer = LSTMExplainer(
                self.model, background=X[:1], feature_columns=list(range(X.shape[-1]))
            )
            top_features = explainer.explain(X[0], length)

        latency_ms = (time.perf_counter() - start) * 1000

        return PredictResponse(
            patient_id=request.patient_id,
            risk_score=risk_score,
            alert=risk_score >= config.ALERT_THRESHOLD,
            threshold=config.ALERT_THRESHOLD,
            top_features=top_features,
            model_name=self.model_name,
            latency_ms=latency_ms,
        )
