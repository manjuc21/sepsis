#!/usr/bin/env python3
"""Single reproducible entrypoint: raw data -> preprocessed -> trained (all
models) -> evaluated -> comparison table. No manual notebook steps.

Usage:
    python scripts/run_pipeline.py [--models logreg,xgboost,lstm] [--max-len 336]
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.data import ingest, preprocess
from src.evaluation import evaluate
from src.models.train import ModelConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default="logreg,xgboost,lstm",
        help="Comma-separated model names to train: logreg, xgboost, lstm",
    )
    parser.add_argument("--threshold", type=float, default=config.ALERT_THRESHOLD)
    args = parser.parse_args()
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    print("=== Step 1/4: Ingest ===")
    t0 = time.time()
    raw = ingest.load_all_patients()
    print(f"Loaded {len(raw):,} patient-hours, {raw['patient_id'].nunique():,} patients "
          f"({time.time() - t0:.1f}s)")

    print("=== Step 2/4: Preprocess (patient-level split, leakage-safe imputation) ===")
    t0 = time.time()
    train_df, val_df, test_df, medians = preprocess.preprocess(raw)
    print(f"train={train_df['patient_id'].nunique():,} val={val_df['patient_id'].nunique():,} "
          f"test={test_df['patient_id'].nunique():,} patients ({time.time() - t0:.1f}s)")

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(config.PROCESSED_DATA_DIR / "train.parquet")
    val_df.to_parquet(config.PROCESSED_DATA_DIR / "val.parquet")
    test_df.to_parquet(config.PROCESSED_DATA_DIR / "test.parquet")

    del raw
    gc.collect()

    artifact_paths = {
        "logreg": config.LOGREG_ARTIFACT,
        "xgboost": config.XGBOOST_ARTIFACT,
        "lstm": config.LSTM_ARTIFACT,
    }

    # Cap sequence padding at the 99th percentile of observed stay length,
    # not the true max. Real distribution here: mean ~39h, 99th pct ~134h,
    # true max 336h (a handful of long-stay outliers). Padding every patient
    # to the true max would waste ~8x memory/compute for no signal gain on
    # 99% of patients, and risks OOM on memory-constrained hosts. The <1% of
    # patients longer than the cap keep their most recent `lstm_max_len`
    # hours (most clinically relevant window), not silently dropped.
    stay_lengths = pd.concat([
        train_df.groupby("patient_id")["ICULOS"].max(),
        val_df.groupby("patient_id")["ICULOS"].max(),
        test_df.groupby("patient_id")["ICULOS"].max(),
    ])
    lstm_max_len = min(config.MAX_SEQUENCE_LENGTH, int(stay_lengths.quantile(0.99)))
    print(
        f"LSTM sequence padding length: {lstm_max_len} "
        f"(99th pct stay length; true max observed: {int(stay_lengths.max())}h)"
    )

    print(f"=== Step 3/4: Train ({', '.join(model_names)}) ===")
    for name in model_names:
        t0 = time.time()
        extra = {"max_len": lstm_max_len} if name == "lstm" else {}
        result = train(
            ModelConfig(name=name, artifact_path=artifact_paths[name], extra=extra),
            train_df, val_df,
        )
        print(f"  {name}: trained in {result['train_seconds']:.1f}s -> {result['artifact_path']}")
        del result
        gc.collect()

    print("=== Step 4/4: Evaluate on held-out test split ===")
    for name in model_names:
        report = evaluate.evaluate_model(name, artifact_paths[name], test_df, threshold=args.threshold)
        evaluate.append_to_comparison_table(report)
        print(
            f"  {name}: AUROC={report['auroc']:.3f} AUPRC={report['auprc']:.3f} "
            f"sensitivity={report['sensitivity']:.3f} specificity={report['specificity']:.3f} "
            f"early_gain={report['mean_early_gain_hours']:.1f}h "
            f"utility={report['utility_score']:.3f}"
        )

    print(f"\nComparison table: {config.RESULTS_DIR / 'comparison.csv'}")
    print("Done. Run `pytest` to confirm the suite passes, then "
          "`python scripts/prepare_demo_patients.py` for the dashboard demo data.")


if __name__ == "__main__":
    main()
