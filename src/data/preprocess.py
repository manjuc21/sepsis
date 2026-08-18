"""Missing-value handling and patient-level train/val/test splitting.

Two rules that must never be violated (see CLAUDE.md §2):
  1. No cross-patient imputation — only forward-fill within a patient, plus an
     explicit missingness indicator per feature.
  2. Splits are by patient_id, never by row — a patient's hours never cross
     split boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def add_missingness_indicators(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add a `<col>_missing` boolean column per feature, computed before any
    imputation. Missingness itself is informative in ICU data (labs are drawn
    on clinical suspicion, not at random)."""
    # Built as one concat rather than 34 sequential column assignments — see
    # features.py's identical fix for why that matters at this row count.
    missing_cols = {f"{col}_missing": df[col].isna().astype(int) for col in columns}
    return pd.concat([df, pd.DataFrame(missing_cols, index=df.index)], axis=1)


def forward_fill_within_patient(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Forward-fill each column within each patient_id only. Any value still
    missing after ffill (i.e. missing from the start of a patient's stay) is
    left as NaN here — imputed downstream using train-set statistics only."""
    out = df.sort_values(["patient_id", "ICULOS"]).copy()
    out[columns] = out.groupby("patient_id", sort=False)[columns].ffill()
    return out


def impute_remaining_with_train_stats(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Fill any still-missing values (start-of-stay gaps) using per-column
    medians computed on the TRAIN split only, then apply that same fitted
    statistic to val/test. Returns the fitted stats too, for artifact saving."""
    medians = train[columns].median(numeric_only=True).to_dict()

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[columns] = out[columns].fillna(medians)
        return out

    return _apply(train), _apply(val), _apply(test), medians


def patient_level_split(
    df: pd.DataFrame,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
    test_frac: float = config.TEST_FRAC,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by patient_id (stratified on whether the patient ever develops
    sepsis) so no patient's hours appear in more than one split."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9

    patient_labels = df.groupby("patient_id")[config.LABEL_COLUMN].max()
    rng = np.random.default_rng(seed)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for label_value in (0, 1):
        ids = patient_labels[patient_labels == label_value].index.to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    train = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    val = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)
    test = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)
    return train, val, test


def preprocess(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Full preprocessing pipeline: missingness indicators -> patient-level
    split -> within-patient ffill -> train-fit median imputation for
    remaining gaps. Returns (train, val, test, fitted_medians)."""
    feature_cols = config.VITAL_COLUMNS + config.LAB_COLUMNS

    df = add_missingness_indicators(df, feature_cols)
    df = forward_fill_within_patient(df, feature_cols)

    train, val, test = patient_level_split(df)

    train, val, test, medians = impute_remaining_with_train_stats(
        train, val, test, feature_cols
    )
    return train, val, test, medians
