"""Runs a saved model against a test split and produces the full metric
report (CLAUDE.md §6). Same function for every model type — nothing model-
specific lives outside `_predict_on_split`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import torch

from src import config
from src.data import features
from src.evaluation import metrics
from src.models.sequence import SepsisLSTM


def _predict_baseline(artifact_path: Path, split: pd.DataFrame) -> pd.DataFrame:
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    X, y = features.build_tabular_feature_matrix(split)
    X = X[feature_columns]
    y_prob = model.predict_proba(X)

    return pd.DataFrame({
        "patient_id": split["patient_id"].to_numpy(),
        "ICULOS": split["ICULOS"].to_numpy(),
        config.LABEL_COLUMN: y.to_numpy(),
        "y_prob": y_prob,
    })


def _predict_lstm(artifact_path: Path, split: pd.DataFrame) -> pd.DataFrame:
    checkpoint = torch.load(artifact_path, map_location="cpu")
    model = SepsisLSTM(n_features=checkpoint["n_features"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    X, y, mask, patient_ids = features.build_sequence_tensors(split, checkpoint["max_len"])
    lengths = mask.sum(dim=1)

    with torch.no_grad():
        logits = model(X, lengths)
        probs = torch.sigmoid(logits).numpy()

    rows = []
    for i, pid in enumerate(patient_ids):
        length = int(lengths[i].item())
        for t in range(length):
            rows.append({
                "patient_id": pid,
                "ICULOS": t + 1,
                config.LABEL_COLUMN: int(y[i, t].item()),
                "y_prob": float(probs[i, t]),
            })
    return pd.DataFrame(rows)


def evaluate_model(
    model_name: str,
    artifact_path: Path,
    test_split: pd.DataFrame,
    threshold: float = config.ALERT_THRESHOLD,
) -> dict:
    """Returns AUROC, AUPRC, sensitivity/specificity, early-prediction gain,
    and utility score for the given model artifact against `test_split`."""
    if model_name in ("logreg", "xgboost"):
        predictions = _predict_baseline(artifact_path, test_split)
    elif model_name == "lstm":
        predictions = _predict_lstm(artifact_path, test_split)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    report = metrics.full_report(predictions, threshold=threshold)
    report["model_name"] = model_name
    return report


def append_to_comparison_table(report: dict, path: Path = config.RESULTS_DIR / "comparison.csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([report])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["model_name"] != report["model_name"]]
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(path, index=False)
