# Sepsis Early Prediction System

Predicts sepsis risk from ICU time-series data (vitals + labs) hours before
clinical onset, explains *why* it's flagging a patient, and serves that
through a demo dashboard. Built as a BE final-year major project — the goal
is a working, defensible, well-tested system, not a production clinical
product. See `docs/requirements_spec.md` for the full design rationale and
`CLAUDE.md` for the operational conventions this codebase follows.

**Dataset:** [PhysioNet/Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)
— ~40,000 de-identified per-patient ICU records, ~1.8% sepsis prevalence,
public with no data-use agreement required.

## Architecture

```
raw PSV files ─▶ ingest ─▶ preprocess ─▶ features ─▶ train ─▶ evaluate
                (patient-level split, leakage-safe imputation)
                                                          │
                                                          ▼
                                              models/*.pkl, *.pt
                                              results/comparison.csv
                                                          │
                                                          ▼
                                         FastAPI /predict (+ SHAP explanation)
                                                          │
                                                          ▼
                                              React dashboard (Vite)
```

| Layer | Choice |
|---|---|
| Data processing | Python, pandas, numpy |
| Baseline models | scikit-learn (Logistic Regression), XGBoost |
| Sequence model | PyTorch LSTM (padded/masked per-patient sequences) |
| Explainability | SHAP (TreeExplainer for XGBoost, gradient-based for the LSTM) |
| Backend API | FastAPI |
| Database | PostgreSQL (prediction/audit log) |
| Frontend | React (Vite) |
| Testing | pytest |

## Non-negotiables

1. **No data leakage** — any scaler/imputer/imbalance-handling is fit on the
   train split only, then applied to val/test.
2. **Accuracy alone is never reported** — every evaluation includes AUROC,
   AUPRC, sensitivity/specificity at a stated threshold, and an
   early-prediction-gain metric (sepsis prevalence is ~1.8%).
3. **Every prediction is explainable** — the API never returns a bare
   probability, only probability + top contributing features.
4. **Reproducible from one command** — `scripts/train.sh` takes raw data to
   trained, evaluated models with no manual notebook steps.

## Quickstart

```bash
scripts/download_data.sh   # fetch the ~40K PhysioNet PSV files into data/raw/
scripts/setup.sh           # create .venv, install deps (GPU torch auto-detected)
scripts/train.sh           # raw -> preprocessed -> trained -> evaluated
scripts/test.sh            # run the test suite
scripts/demo.sh            # docker compose up: Postgres + API + dashboard
```

Dashboard: http://localhost:5173 · API: http://localhost:8000 (docs at `/docs`)

### Management scripts

| Script | Purpose |
|---|---|
| `scripts/download_data.sh` | Downloads the PhysioNet 2019 dataset into `data/raw/` |
| `scripts/setup.sh` | Creates `.venv`; installs CUDA-enabled torch if an NVIDIA GPU is detected, CPU-only otherwise |
| `scripts/train.sh [--models logreg,xgboost,lstm]` | Runs `scripts/run_pipeline.py` end to end |
| `scripts/test.sh [pytest args]` | Runs the test suite |
| `scripts/serve_api.sh [--reload]` | Runs the API directly on the host (no Docker) against whatever's in `models/` |
| `scripts/demo.sh` | Generates demo patient data, brings up the full stack via `docker-compose` |
| `scripts/stop.sh` | `docker compose down` |

Training runs on the host/venv (not in a container — too slow to iterate on
inside Docker); `scripts/demo.sh` is the one-command way to serve whatever
was last trained. If Postgres isn't reachable, the API skips prediction
logging automatically rather than crashing — `scripts/serve_api.sh` alone
(no `docker compose`) is enough to exercise `/predict`.

## Directory structure

```
src/
├── config.py             # paths, random seed, thresholds — single source of truth
├── data/                 # ingest, preprocess (leakage-safe), feature engineering
├── models/                # LogReg/XGBoost baselines, LSTM, shared train.py entrypoint
├── evaluation/            # AUROC/AUPRC/sensitivity/early-gain/utility metrics
├── explainability/        # SHAP wrappers, human-readable feature explanations
└── api/                   # FastAPI app: schemas, predict, main
frontend/                  # React dashboard (Vite)
scripts/                   # download_data.sh, run_pipeline.py, and the management scripts above
tests/                     # one test file per src/ module
docs/requirements_spec.md  # full requirements & design rationale
tasks.md                   # phase-by-phase execution plan and progress
```

## Evaluation contract

Any trained model must pass through `src/evaluation/evaluate.py`, which reports:
- AUROC, AUPRC
- Sensitivity/specificity at the configured alert threshold
- Early-prediction gain (mean hours between alert and true onset, across true positives)
- A simplified utility score (`docs/requirements_spec.md` §10)

Results accumulate in `results/comparison.csv` so every model type is
compared on the same metric set.

## Status

Code and tests for all phases (data → preprocessing → baselines → LSTM →
evaluation → explainability → API → dashboard) are in place — see
`tasks.md` for the current per-item checklist. Run `scripts/train.sh` to
produce trained artifacts and populate `results/comparison.csv` if they
don't exist yet.

## Out of scope

This is not a real clinical deployment: no auth/HIPAA-grade security, no
real patient-identifiable data handling, no claims of clinical validation.
See `docs/requirements_spec.md` §4.
