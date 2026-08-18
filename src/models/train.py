"""Shared training entrypoint. Every model type (baseline or sequence) plugs
into this same function and the same evaluate.py path — no per-model one-off
training scripts (CLAUDE.md §6).
"""

from __future__ import annotations

import gc
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.data import features
from src.models.baseline import LogRegModel, XGBoostModel
from src.models.sequence import SepsisLSTM, masked_bce_loss


@dataclass
class ModelConfig:
    name: str  # "logreg" | "xgboost" | "lstm"
    artifact_path: Path
    extra: dict = field(default_factory=dict)


def train_baseline(
    model_config: ModelConfig, train: pd.DataFrame, val: pd.DataFrame
) -> dict:
    X_train, y_train = features.build_tabular_feature_matrix(train)
    X_val, y_val = features.build_tabular_feature_matrix(val)

    model = LogRegModel() if model_config.name == "logreg" else XGBoostModel()

    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - start

    val_prob = model.predict_proba(X_val)

    model_config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_config.artifact_path, "wb") as f:
        pickle.dump({"model": model, "feature_columns": list(X_train.columns)}, f)

    result = {
        "model_name": model_config.name,
        "artifact_path": str(model_config.artifact_path),
        "train_seconds": train_seconds,
        "val_predictions": pd.DataFrame({
            "patient_id": val["patient_id"].to_numpy(),
            "ICULOS": val["ICULOS"].to_numpy(),
            config.LABEL_COLUMN: y_val.to_numpy(),
            "y_prob": val_prob,
        }),
    }
    del X_train, X_val, y_train, y_val
    gc.collect()
    return result


def train_lstm(
    model_config: ModelConfig, train: pd.DataFrame, val: pd.DataFrame
) -> dict:
    max_len = model_config.extra.get("max_len", config.MAX_SEQUENCE_LENGTH)

    X_train, y_train, mask_train, _ = features.build_sequence_tensors(train, max_len)
    X_val, y_val, mask_val, val_patient_ids = features.build_sequence_tensors(val, max_len)

    n_features = X_train.shape[-1]
    lengths_train = mask_train.sum(dim=1)
    lengths_val = mask_val.sum(dim=1)

    torch.manual_seed(config.RANDOM_SEED)
    model = SepsisLSTM(n_features=n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LSTM_LEARNING_RATE)

    n_pos = y_train[mask_train].sum()
    n_neg = mask_train.sum() - n_pos
    pos_weight = (n_neg / n_pos.clamp(min=1)).clamp(max=100.0)

    train_ds = TensorDataset(X_train, y_train, mask_train, lengths_train)
    train_loader = DataLoader(train_ds, batch_size=config.LSTM_BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience_left = config.LSTM_EARLY_STOP_PATIENCE
    start = time.time()

    for epoch in range(config.LSTM_MAX_EPOCHS):
        model.train()
        for xb, yb, mb, lb in train_loader:
            optimizer.zero_grad()
            logits = model(xb, lb)
            loss = masked_bce_loss(logits, yb, mb, pos_weight)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val, lengths_val)
            val_loss = masked_bce_loss(val_logits, y_val, mask_val, pos_weight).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = config.LSTM_EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = time.time() - start

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val, lengths_val)
        val_prob = torch.sigmoid(val_logits).numpy()

    rows = []
    for i, pid in enumerate(val_patient_ids):
        length = int(lengths_val[i].item())
        for t in range(length):
            rows.append({
                "patient_id": pid,
                "ICULOS": t + 1,
                config.LABEL_COLUMN: int(y_val[i, t].item()),
                "y_prob": float(val_prob[i, t]),
            })
    val_predictions = pd.DataFrame(rows)

    model_config.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "n_features": n_features, "max_len": max_len},
        model_config.artifact_path,
    )

    del X_train, X_val, y_train, y_val, mask_train, mask_val, train_ds, train_loader
    gc.collect()

    return {
        "model_name": model_config.name,
        "artifact_path": str(model_config.artifact_path),
        "train_seconds": train_seconds,
        "val_predictions": val_predictions,
    }


def train(model_config: ModelConfig, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    if model_config.name in ("logreg", "xgboost"):
        return train_baseline(model_config, train_df, val_df)
    if model_config.name == "lstm":
        return train_lstm(model_config, train_df, val_df)
    raise ValueError(f"Unknown model name: {model_config.name}")
