#!/usr/bin/env python3
"""Builds data/processed/demo_patients.json: a small, fixed set of patients
(a mix of septic and non-septic) with a precomputed hourly risk trend +
explanation, served by the API's /patients and /patients/{id}/timeline for
the dashboard demo. Precomputed rather than scored live per hour so the
dashboard doesn't need N /predict calls per patient during a live viva demo.

Usage:
    python scripts/prepare_demo_patients.py [--n-patients 12]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from src import config
from src.data import features
from src.explainability.shap_explainer import LSTMExplainer, XGBoostExplainer
from src.models.sequence import SepsisLSTM


def _load_best_model():
    comparison_path = config.RESULTS_DIR / "comparison.csv"
    name = "xgboost"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        if not df.empty:
            name = df.loc[df["auprc"].idxmax(), "model_name"]

    if name in ("xgboost", "logreg"):
        artifact_path = config.XGBOOST_ARTIFACT if name == "xgboost" else config.LOGREG_ARTIFACT
        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)
        return name, artifact["model"], artifact["feature_columns"]

    checkpoint = torch.load(config.LSTM_ARTIFACT, map_location="cpu")
    model = SepsisLSTM(n_features=checkpoint["n_features"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return name, model, checkpoint["max_len"]


def _score_patient_tabular(model, feature_columns, patient_df: pd.DataFrame, explainer) -> list[dict]:
    X, _ = features.build_tabular_feature_matrix(patient_df)
    X = X[feature_columns]
    probs = model.predict_proba(X)

    timeline = []
    for i, (_, row) in enumerate(patient_df.iterrows()):
        top_features = explainer.explain(X.iloc[[i]]) if explainer else []
        timeline.append({
            "hour": int(row["ICULOS"]),
            "risk_score": float(probs[i]),
            "label": int(row[config.LABEL_COLUMN]),
            "top_features": top_features,
        })
    return timeline


def _score_patient_lstm(model, max_len: int, patient_df: pd.DataFrame) -> list[dict]:
    X, y, mask, _ = features.build_sequence_tensors(patient_df, max_len)
    length = int(mask[0].sum().item())
    lengths = mask.sum(dim=1)

    with torch.no_grad():
        logits = model(X, lengths)
        probs = torch.sigmoid(logits[0]).numpy()

    explainer = LSTMExplainer(model, background=X[:1], feature_columns=list(range(X.shape[-1])))

    timeline = []
    for t in range(length):
        top_features = explainer.explain(X[0], t + 1) if t > 0 else []
        timeline.append({
            "hour": t + 1,
            "risk_score": float(probs[t]),
            "label": int(y[0, t].item()),
            "top_features": top_features,
        })
    return timeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patients", type=int, default=12)
    args = parser.parse_args()

    test_path = config.PROCESSED_DATA_DIR / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} not found — run scripts/run_pipeline.py first")

    test_df = pd.read_parquet(test_path)
    patient_labels = test_df.groupby("patient_id")[config.LABEL_COLUMN].max()

    septic_ids = patient_labels[patient_labels == 1].index.tolist()
    non_septic_ids = patient_labels[patient_labels == 0].index.tolist()

    n_septic = min(len(septic_ids), max(1, args.n_patients // 3))
    n_non_septic = min(len(non_septic_ids), args.n_patients - n_septic)
    chosen_ids = septic_ids[:n_septic] + non_septic_ids[:n_non_septic]

    model_name, model, extra = _load_best_model()
    print(f"Using model: {model_name}")

    if model_name in ("xgboost", "logreg"):
        explainer = XGBoostExplainer(model, extra) if model_name == "xgboost" else None

    patients_out = []
    for pid in chosen_ids:
        patient_df = test_df[test_df["patient_id"] == pid].sort_values("ICULOS")
        if model_name in ("xgboost", "logreg"):
            timeline = _score_patient_tabular(model, extra, patient_df, explainer)
        else:
            timeline = _score_patient_lstm(model, extra, patient_df)

        patients_out.append({"patient_id": pid, "timeline": timeline})

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DATA_DIR / "demo_patients.json"
    out_path.write_text(json.dumps({"model_name": model_name, "patients": patients_out}, indent=2))
    print(f"Wrote {len(patients_out)} demo patients to {out_path}")


if __name__ == "__main__":
    main()
