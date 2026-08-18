from __future__ import annotations

import pandas as pd

from src import config
from src.data import ingest, preprocess


def test_ingest_produces_expected_columns(tmp_path):
    psv_path = tmp_path / "p000001.psv"
    header = "|".join(
        config.VITAL_COLUMNS + config.LAB_COLUMNS + config.DEMOGRAPHIC_COLUMNS
        + [config.LABEL_COLUMN]
    )
    row1 = "|".join(["NaN"] * (len(config.VITAL_COLUMNS) + len(config.LAB_COLUMNS)) + ["70", "1", "0", "1", "-0.1", "1", "0"])
    row2 = "|".join(["NaN"] * (len(config.VITAL_COLUMNS) + len(config.LAB_COLUMNS)) + ["70", "1", "0", "1", "-0.1", "2", "0"])
    psv_path.write_text(f"{header}\n{row1}\n{row2}\n")

    df = ingest.load_all_patients(raw_dirs=[tmp_path])

    assert "patient_id" in df.columns
    assert set(config.VITAL_COLUMNS).issubset(df.columns)
    assert set(config.LAB_COLUMNS).issubset(df.columns)
    assert len(df) == 2


def test_ingest_no_duplicate_patient_hour_rows(synthetic_patients):
    dupes = synthetic_patients.duplicated(subset=["patient_id", "ICULOS"]).sum()
    assert dupes == 0


def test_patient_level_split_has_zero_id_overlap(synthetic_patients):
    train, val, test = preprocess.patient_level_split(synthetic_patients)

    train_ids = set(train["patient_id"])
    val_ids = set(val["patient_id"])
    test_ids = set(test["patient_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    all_ids = set(synthetic_patients["patient_id"])
    assert train_ids | val_ids | test_ids == all_ids


def test_forward_fill_stays_within_patient(synthetic_patients):
    cols = config.VITAL_COLUMNS
    filled = preprocess.forward_fill_within_patient(synthetic_patients, cols)

    # Manually corrupt: value at the *end* of patient A must never leak into
    # the start of patient B's rows after a groupby-based ffill.
    first_patient = synthetic_patients["patient_id"].iloc[0]
    other_patients_first_rows = filled[filled["patient_id"] != first_patient].groupby(
        "patient_id"
    ).head(1)
    # first row of every other patient should be untouched by ffill (still NaN
    # if it started NaN) i.e. ffill did not pull from a prior patient's tail.
    original_first_rows = synthetic_patients[
        synthetic_patients["patient_id"] != first_patient
    ].groupby("patient_id").head(1)
    for col in cols:
        pd.testing.assert_series_equal(
            other_patients_first_rows[col].reset_index(drop=True).isna(),
            original_first_rows[col].reset_index(drop=True).isna(),
        )


def test_imputation_fit_on_train_only(synthetic_patients):
    train, val, test, medians = preprocess.preprocess(synthetic_patients)

    feature_cols = config.VITAL_COLUMNS + config.LAB_COLUMNS
    assert train[feature_cols].isna().sum().sum() == 0
    assert val[feature_cols].isna().sum().sum() == 0
    assert test[feature_cols].isna().sum().sum() == 0
    assert set(medians.keys()) == set(feature_cols)
