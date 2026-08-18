"""Shared fixtures: small synthetic patient-hour dataframes matching the
PhysioNet schema, used across data/model/metric tests so we're not dependent
on the real (gitignored) dataset being present."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config


def _make_patient(patient_id: str, n_hours: int, positive: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(loc=80, scale=10, size=n_hours) for col in config.VITAL_COLUMNS}
    for col in config.LAB_COLUMNS:
        vals = rng.normal(loc=1.0, scale=0.5, size=n_hours)
        # simulate heavy missingness typical of labs
        mask = rng.random(n_hours) < 0.7
        vals[mask] = np.nan
        data[col] = vals

    data["Age"] = rng.integers(20, 90)
    data["Gender"] = rng.integers(0, 2)
    data["Unit1"] = 0
    data["Unit2"] = 1
    data["HospAdmTime"] = -rng.uniform(0, 5)
    data["ICULOS"] = np.arange(1, n_hours + 1)

    label = np.zeros(n_hours, dtype=int)
    if positive:
        onset = max(1, n_hours - 3)
        label[onset:] = 1
    data[config.LABEL_COLUMN] = label

    df = pd.DataFrame(data)
    df.insert(0, "patient_id", patient_id)
    return df


@pytest.fixture
def synthetic_patients() -> pd.DataFrame:
    """~1.8% positive prevalence at the patient level, matching real dataset."""
    frames = []
    n_patients = 60
    n_positive = 1  # roughly matches ~1.8% prevalence at small scale
    for i in range(n_patients):
        positive = i < n_positive
        n_hours = int(np.random.default_rng(i).integers(10, 40))
        frames.append(_make_patient(f"p{i:06d}", n_hours, positive, seed=i))
    return pd.concat(frames, ignore_index=True)
