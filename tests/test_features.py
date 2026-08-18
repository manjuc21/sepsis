from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data import features


def _single_patient_frame(values: list[float]) -> pd.DataFrame:
    n = len(values)
    data = {col: values if col == "HR" else [0.0] * n for col in config.VITAL_COLUMNS}
    for col in config.LAB_COLUMNS:
        data[col] = [0.0] * n
    data["Age"] = [50] * n
    data["Gender"] = [0] * n
    data["Unit1"] = [0] * n
    data["Unit2"] = [1] * n
    data["HospAdmTime"] = [0.0] * n
    data["ICULOS"] = list(range(1, n + 1))
    data[config.LABEL_COLUMN] = [0] * n
    df = pd.DataFrame(data)
    df.insert(0, "patient_id", "p000001")
    return df


def test_rolling_window_no_future_leakage():
    # HR strictly increasing 1..10; roll_max at hour t must equal value at t
    # (never a later, larger value) since the window only looks backward.
    df = _single_patient_frame([float(i) for i in range(1, 11)])
    out = features.add_rolling_window_features(df, columns=["HR"], window=3)

    for t in range(len(out)):
        assert out.loc[t, "HR_roll_max"] == out.loc[t, "HR"], (
            "roll_max at time t must never exceed HR at time t for a "
            "monotonically increasing series — a violation implies future leakage"
        )


def test_rolling_window_correct_shape():
    df = _single_patient_frame([1.0, 2.0, 3.0, 4.0, 5.0])
    out = features.add_rolling_window_features(df, columns=["HR"], window=3)
    assert len(out) == 5
    for stat in ("mean", "min", "max", "slope"):
        assert f"HR_roll_{stat}" in out.columns
    assert out["HR_roll_mean"].isna().sum() == 0


def test_build_tabular_feature_matrix_shapes(synthetic_patients):
    X, y = features.build_tabular_feature_matrix(synthetic_patients)
    assert len(X) == len(synthetic_patients)
    assert len(y) == len(synthetic_patients)
    assert set(y.unique()).issubset({0, 1})


def test_build_sequence_tensors_padding_and_mask(synthetic_patients):
    from src.data import preprocess

    train, _, _, _ = preprocess.preprocess(synthetic_patients)
    max_len = 50
    X, y, mask, patient_ids = features.build_sequence_tensors(train, max_len=max_len)
    n_patients = train["patient_id"].nunique()

    assert X.shape[0] == n_patients
    assert X.shape[1] == max_len
    assert mask.shape == (n_patients, max_len)

    # mask sum per patient must equal that patient's true sequence length
    lengths = train.groupby("patient_id").size()
    for i, pid in enumerate(patient_ids):
        assert mask[i].sum().item() == lengths[pid]
        # padded region must be exactly zero
        true_len = lengths[pid]
        if true_len < max_len:
            assert torch_all_zero(X[i, true_len:])


def torch_all_zero(tensor) -> bool:
    return bool((tensor == 0).all().item())
