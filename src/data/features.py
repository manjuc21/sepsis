"""Feature engineering: rolling-window features (tabular, for tree models) and
padded/masked sequence tensors (for the LSTM).

Leakage rule: every feature at hour t must only use data from hours <= t.
Rolling windows use `min_periods=1` and pandas' backward-looking `rolling()`,
which by construction never sees t+1 or later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src import config


def add_rolling_window_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    window: int = config.ROLLING_WINDOW_HOURS,
) -> pd.DataFrame:
    """Add mean/min/max/slope over the trailing `window` hours, per patient,
    for each column. Causal by construction (backward-looking rolling)."""
    if columns is None:
        columns = config.VITAL_COLUMNS + config.LAB_COLUMNS

    out = df.sort_values(["patient_id", "ICULOS"])
    grouped = out.groupby("patient_id", sort=False)

    # Build every new column in a dict first, then concat once. Assigning
    # 100+ columns one at a time onto a million-row frame (the original
    # approach) triggers pandas' block-fragmentation, which transiently
    # multiplies peak memory well beyond the final frame size — a real
    # problem at this dataset's scale, not just a performance nitpick.
    new_cols: dict[str, pd.Series] = {}
    for col in columns:
        roll = grouped[col].rolling(window=window, min_periods=1)
        new_cols[f"{col}_roll_mean"] = roll.mean().reset_index(level=0, drop=True).astype(np.float32)
        new_cols[f"{col}_roll_min"] = roll.min().reset_index(level=0, drop=True).astype(np.float32)
        new_cols[f"{col}_roll_max"] = roll.max().reset_index(level=0, drop=True).astype(np.float32)

        # Slope = (current - value `window` hours ago) / window. NaN for the
        # first `window` hours of a stay is expected and filled with 0
        # (no measurable trend yet), never bfilled from the future.
        shifted = grouped[col].shift(window)
        slope = (out[col] - shifted) / window
        new_cols[f"{col}_roll_slope"] = slope.fillna(0.0).astype(np.float32)

    new_df = pd.DataFrame(new_cols, index=out.index)
    return pd.concat([out, new_df], axis=1)


def build_tabular_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Flatten to (X, y) for baseline models: raw features + rolling-window
    features + missingness indicators + demographics, one row per patient-hour."""
    df = add_rolling_window_features(df)

    feature_cols = (
        config.VITAL_COLUMNS
        + config.LAB_COLUMNS
        + [f"{c}_missing" for c in config.VITAL_COLUMNS + config.LAB_COLUMNS]
        + [f"{c}_roll_mean" for c in config.VITAL_COLUMNS + config.LAB_COLUMNS]
        + [f"{c}_roll_min" for c in config.VITAL_COLUMNS + config.LAB_COLUMNS]
        + [f"{c}_roll_max" for c in config.VITAL_COLUMNS + config.LAB_COLUMNS]
        + [f"{c}_roll_slope" for c in config.VITAL_COLUMNS + config.LAB_COLUMNS]
        + config.DEMOGRAPHIC_COLUMNS
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    # float32 throughout: halves memory vs. pandas' float64 default with no
    # meaningful precision loss for this model input, which matters at this
    # dataset's row count (see run_pipeline.py's memory-budget comments).
    X = df[feature_cols].astype(np.float32)
    y = df[config.LABEL_COLUMN].copy()
    return X, y


def build_sequence_tensors(
    df: pd.DataFrame,
    max_len: int = config.MAX_SEQUENCE_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Build padded (patients, max_len, n_features) tensors for the LSTM.

    Returns (X, y, mask, patient_ids) where:
      X: float tensor, zero-padded
      y: float tensor of per-timestep labels, zero-padded
      mask: bool tensor, True where a timestep is real (not padding)

    For patients whose stay exceeds `max_len`, only the first `max_len`
    hours are kept (the rest silently dropped, not truncated-and-flagged).
    This only affects the tail of the stay-length distribution when
    `max_len` is set below the true max (see scripts/run_pipeline.py, which
    caps at the 99th percentile for memory reasons) — documented as a known
    limitation: a small number of very-long-stay septic patients whose
    onset falls after `max_len` hours will appear label-negative in this
    tensor. Not an issue when max_len >= true max stay length.
    """
    feature_cols = (
        config.VITAL_COLUMNS
        + config.LAB_COLUMNS
        + [f"{c}_missing" for c in config.VITAL_COLUMNS + config.LAB_COLUMNS]
        + config.DEMOGRAPHIC_COLUMNS
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    patient_ids = df["patient_id"].unique().tolist()
    n_patients = len(patient_ids)
    n_features = len(feature_cols)

    X = np.zeros((n_patients, max_len, n_features), dtype=np.float32)
    y = np.zeros((n_patients, max_len), dtype=np.float32)
    mask = np.zeros((n_patients, max_len), dtype=bool)

    grouped = df.sort_values(["patient_id", "ICULOS"]).groupby("patient_id", sort=False)
    for i, (_, group) in enumerate(grouped):
        length = min(len(group), max_len)
        X[i, :length, :] = group[feature_cols].to_numpy(dtype=np.float32)[:length]
        y[i, :length] = group[config.LABEL_COLUMN].to_numpy(dtype=np.float32)[:length]
        mask[i, :length] = True

    if np.isnan(X[mask]).any():
        raise ValueError(
            "NaNs found in real (non-padded) sequence positions — call "
            "preprocess.preprocess() before build_sequence_tensors(). The "
            "LSTM must never silently train on unhandled missingness "
            "(CLAUDE.md anti-pattern list)."
        )

    return (
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(mask),
        patient_ids,
    )
