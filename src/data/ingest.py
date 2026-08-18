"""Load raw per-patient PSV files (PhysioNet/CinC 2019 Sepsis Challenge) into a
single long-format dataframe: one row per patient-hour, with a `patient_id` column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config


def _patient_id_from_path(path: Path) -> str:
    # "p000001.psv" -> "p000001"; setA and setB share no ids so the stem is unique.
    return path.stem


def load_patient_file(path: Path) -> pd.DataFrame:
    """Load a single patient's PSV file and attach patient_id."""
    df = pd.read_csv(path, sep="|")
    df.insert(0, "patient_id", _patient_id_from_path(path))
    return df


def load_all_patients(raw_dirs: list[Path] | None = None) -> pd.DataFrame:
    """Load every patient file under the given directories into one dataframe.

    Defaults to both training_setA and training_setB. Row order is patient_id
    then hour-of-stay (ICULOS), ascending.
    """
    if raw_dirs is None:
        raw_dirs = [config.RAW_SET_A_DIR, config.RAW_SET_B_DIR]

    files: list[Path] = []
    for d in raw_dirs:
        if not d.exists():
            continue
        files.extend(sorted(d.glob("*.psv")))

    if not files:
        raise FileNotFoundError(
            f"No .psv files found under {raw_dirs}. Run scripts/download_data.sh first."
        )

    frames = [load_patient_file(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    return combined


if __name__ == "__main__":
    df = load_all_patients()
    n_patients = df["patient_id"].nunique()
    n_positive = df.groupby("patient_id")[config.LABEL_COLUMN].max().sum()
    print(f"Loaded {len(df):,} patient-hours across {n_patients:,} patients")
    print(f"Patients with at least one positive label: {n_positive:,} "
          f"({100 * n_positive / n_patients:.2f}%)")
