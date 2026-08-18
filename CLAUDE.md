# CLAUDE.md — Sepsis Early Prediction System

This file is the standing context for Claude Code on this project. Read it before starting any task. `tasks.md` is the execution plan — work through it phase by phase, don't jump ahead.

Full requirements/design rationale lives in `docs/requirements_spec.md`. This file is the condensed, operational version of that doc.

---

## 1. Mission

Build a system that predicts sepsis risk from ICU time-series data (vitals + labs) **hours before clinical onset**, explains *why* it's flagging a patient, and serves that through a working demo dashboard. This is a BE final-year major project — the bar is a working, defensible, well-tested system, not a production-grade clinical product.

## 2. Non-Negotiables (read these before writing any model code)

1. **No data leakage.** Any imbalance-handling (SMOTE, class weights, etc.) is fit on the training split only, never on validation/test. Any scaler/normalizer is fit on train only. This is the single most common mistake in this domain — check it explicitly in every preprocessing PR.
2. **Accuracy is not an acceptable primary metric.** Sepsis prevalence is ~1.8% in this dataset. Every model evaluation must report AUROC, AUPRC, sensitivity at a stated threshold, and (once built) the early-prediction-gain metric. See `src/evaluation/metrics.py`.
3. **Every prediction must be explainable.** The API never returns a bare probability — it returns probability + top contributing features. If a model can't produce that (via SHAP or attention weights), it's not ready to wire into the API.
4. **The pipeline must be reproducible from one command.** `scripts/run_pipeline.py` should take raw data to trained model to evaluation report with no manual notebook steps in between. Notebooks are for exploration only — nothing in `notebooks/` should be a dependency of the actual pipeline.
5. **This is not a real clinical deployment.** Don't add auth/HIPAA-grade security theater, real patient-identifiable data handling, or claims of clinical validation anywhere in code, comments, or docs. The dataset is already de-identified; keep it that way conceptually too.

## 3. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Data processing | Python, pandas, numpy | Standard, matches dataset format (per-patient CSVs) |
| Baseline models | scikit-learn, XGBoost | Interpretable, fast to iterate, strong tabular baseline |
| Sequence models | PyTorch (LSTM, optionally a TCN) | Needed for genuine temporal modeling — this is what differentiates the project |
| Explainability | SHAP | TreeExplainer for XGBoost, gradient/kernel explainer for the LSTM |
| Backend API | FastAPI | Async, typed, fast to build a clean prediction endpoint |
| Database | PostgreSQL | Prediction/audit log storage |
| Frontend | React (Vite, plain CSS or Tailwind — no heavyweight UI kit needed) | Dashboard for demo; keep it simple, this isn't the graded core |
| Testing | pytest (backend/ML), Vitest or plain manual testing (frontend, lower priority) | |
| Env management | `requirements.txt` + `venv`, or `uv` if available | |

Don't introduce Celery, Redis, or Kubernetes — that complexity doesn't buy anything for a single-demo academic project and will eat time better spent on the model comparison and evaluation, which is what gets graded.

**Update (2026-08-18):** the serving stack (Postgres + API + frontend) runs via `docker-compose` — see `docker-compose.yml` and `docker/`. Model training (`scripts/run_pipeline.py`) still runs directly on the host/venv, not in a container — training is iterative and too slow to develop inside compose. `docker-compose up --build` after a local training run is the one-command way to get the demo running.

## 4. Directory Structure

```
sepsis-prediction/
├── data/
│   ├── raw/                  # PhysioNet CSVs — gitignored, see data/README.md for download steps
│   ├── processed/            # feature-engineered parquet/csv — gitignored
│   └── README.md
├── notebooks/                # EDA only, never imported by src/
├── src/
│   ├── config.py             # paths, thresholds, random seed — single source of truth
│   ├── data/
│   │   ├── ingest.py         # load raw per-patient files into one frame
│   │   ├── preprocess.py     # missing-value handling, normalization
│   │   └── features.py       # rolling-window features, missingness indicators
│   ├── models/
│   │   ├── baseline.py       # LogisticRegression, XGBoost
│   │   ├── sequence.py       # LSTM (and optionally TCN)
│   │   └── train.py          # shared training entrypoint, takes a model config
│   ├── evaluation/
│   │   ├── metrics.py        # AUROC, AUPRC, early-prediction gain, utility score
│   │   └── evaluate.py       # runs a saved model against the test split, writes report
│   ├── explainability/
│   │   └── shap_explainer.py
│   └── api/
│       ├── main.py           # FastAPI app
│       ├── schemas.py        # pydantic request/response models
│       └── predict.py        # loads model artifact, wraps predict + explain
├── frontend/                 # React dashboard
├── models/                   # saved model artifacts — gitignored, name with model+date
├── tests/
│   ├── test_preprocess.py
│   ├── test_features.py
│   ├── test_metrics.py
│   ├── test_models.py
│   └── test_api.py
├── scripts/
│   ├── run_pipeline.py       # single reproducible entrypoint: raw → trained → evaluated
│   └── download_data.sh
├── docs/
│   └── requirements_spec.md
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.frontend
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── tasks.md
```

## 5. Coding Conventions

- Python 3.11+, type hints on all function signatures in `src/`.
- No logic in notebooks that isn't also in `src/` — notebooks call `src/` functions, they don't reimplement them.
- Every module in `src/` gets a corresponding test file in `tests/`. A task in `tasks.md` isn't done until its test task is also done.
- Config values (thresholds, paths, random seed, model hyperparameters) live in `src/config.py`, not hardcoded inline — this matters for reproducibility, which is a graded criterion.
- Commit messages: short imperative (`add rolling-window feature engineering`), no need for conventional-commit prefixes for a solo/small academic project.
- Random seed fixed and reused everywhere (`config.RANDOM_SEED`) — results need to be reproducible for the report.

## 6. Evaluation Contract

Any model training task is not "done" until `src/evaluation/evaluate.py` can run against it and produce:
- AUROC, AUPRC
- Sensitivity/specificity at the configured threshold
- Early-prediction gain (mean hours between alert and actual onset, across true positives)
- A simplified utility score (see `docs/requirements_spec.md` §10 for the definition to implement)

If a new model type is added, it must plug into this same evaluation path — don't write a one-off evaluation script per model.

## 7. What "Done" Looks Like Per Component

- **Preprocessing pipeline:** deterministic, handles missing labs without crashing, has tests covering at least one malformed/missing-heavy patient record.
- **A model:** trained, saved as a versioned artifact, passes through `evaluate.py`, has its metrics logged somewhere comparable to the other models (a simple `results/comparison.csv` or similar is fine).
- **API:** returns probability + top-N contributing features for a given patient time-series window in under 500ms; has at least one test hitting the endpoint with a real feature vector.
- **Dashboard:** can show a list of simulated patients, a risk trend chart for one, and an alert state when risk crosses threshold. Polish is secondary to this working end-to-end.

## 8. Anti-Patterns to Avoid

- Fitting any preprocessing/imbalance-handling step on the full dataset before splitting.
- Reporting only accuracy or only AUROC without AUPRC (imbalance context is always required).
- Letting the LSTM implementation silently swallow NaNs instead of handling missingness explicitly upstream.
- Building the frontend before the API contract (schemas.py) is stable — wire the contract first.
- Scope creep into real deployment concerns (auth systems, HIPAA compliance docs, live hospital integration) — explicitly out of scope, see `docs/requirements_spec.md` §4.
