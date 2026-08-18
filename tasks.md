# tasks.md — Sepsis Early Prediction System

Work through phases in order. Each phase has a Definition of Done — don't move to the next phase until it's met. Testing tasks are embedded inside each phase, not deferred to the end. Check off tasks as completed.

Read `CLAUDE.md` before starting Phase 0.

---

## Phase 0 — Project Scaffolding

- [x] Create directory structure exactly as specified in `CLAUDE.md` §4
- [x] Set up `venv`, write `requirements.txt` (pandas, numpy, scikit-learn, xgboost, torch, shap, fastapi, uvicorn, psycopg2 or sqlalchemy, pytest, python-dotenv)
- [x] Write `src/config.py` with: `RANDOM_SEED`, data paths, alert threshold (placeholder, tune later), model artifact paths
- [x] Write `.env.example` (DB connection string placeholder, API port)
- [x] Write `data/README.md` with PhysioNet 2019 Challenge download instructions
- [x] Initialize git repo, add `.gitignore` (data/raw, data/processed, models/*.pkl, .env, __pycache__, node_modules)

**Definition of Done:** `pip install -r requirements.txt` works cleanly in a fresh venv; directory structure matches spec; git repo initialized with first commit.

---

## Phase 1 — Data Acquisition & EDA

- [x] Write `scripts/download_data.sh` to fetch the PhysioNet/CinC 2019 Sepsis dataset
- [x] Write `src/data/ingest.py`: load all per-patient PSV/CSV files into a single dataframe with a `patient_id` column
- [x] `tests/test_preprocess.py` (start it here): test that `ingest.py` produces the expected columns and no duplicate patient-hour rows
- [ ] EDA notebook (`notebooks/01_eda.ipynb`): class balance, missingness heatmap per feature, distribution of vitals/labs, distribution of hours-to-onset for positive cases — notebook scaffolded but not yet executed (no saved outputs)
- [ ] Document 3–5 key EDA findings in `docs/eda_findings.md` (these go straight into your report's EDA chapter) — file does not exist yet

**Definition of Done:** raw dataset loads into a single clean dataframe; EDA findings documented; class imbalance and missingness rates quantified and written down (you'll need these exact numbers in the report).

---

## Phase 2 — Preprocessing & Feature Engineering

- [x] `src/data/preprocess.py`: missing-value strategy (forward-fill within patient + missingness indicator columns — do not silently mean-impute across patients)
- [x] `src/data/preprocess.py`: train/val/test split **at the patient level** (never split by row — a patient's hours must not leak across splits)
- [x] `src/data/features.py`: rolling-window features for tree models (mean/min/max/slope over last N hours per vital/lab)
- [x] `src/data/features.py`: sequence-formatted tensors for the LSTM (padded/masked per-patient sequences)
- [x] Verify and enforce: any fitted transformer (scaler, imputer stats) is fit on train split only, then applied to val/test
- [x] `tests/test_preprocess.py`: test patient-level split has zero patient_id overlap between splits
- [x] `tests/test_features.py`: test rolling-window features produce correct shape and no leakage (a feature at hour t must not use data from hour t+1 or later)

**Definition of Done:** two ready-to-train feature sets exist (flattened tabular for baselines, padded sequences for LSTM); leakage tests pass; split is patient-level and reproducible via fixed seed.

---

## Phase 3 — Baseline Models

- [x] `src/models/baseline.py`: Logistic Regression (class-weighted)
- [x] `src/models/baseline.py`: XGBoost (class-weighted or scale_pos_weight tuned)
- [x] `src/models/train.py`: shared training entrypoint that takes a model config and produces a saved artifact + logged metrics
- [x] `tests/test_models.py`: smoke test — model trains on a small synthetic/sampled subset without error and produces predictions in [0,1]
- [ ] Run both baselines through the (not-yet-built) evaluation path — code is wired end-to-end but has not actually been executed yet: `models/` and `results/` are empty, no `results/comparison.csv`

**Definition of Done:** two trained, saved baseline models with AUROC/AUPRC logged; results written to `results/comparison.csv` (or similar).

---

## Phase 4 — Sequence Model

- [x] `src/models/sequence.py`: LSTM architecture (input: padded per-patient hourly sequence; output: per-timestep risk score)
- [x] Handle variable-length sequences properly (padding + masking, not truncation that throws away signal)
- [x] Train LSTM using the same `train.py` entrypoint pattern as baselines (add a case for sequence models — don't fork a separate training script)
- [x] `tests/test_models.py`: smoke test for LSTM forward pass with a padded batch, checks output shape
- [ ] (Optional, if time permits) Add a TCN as a third comparison point — not started, optional
- [ ] Run through evaluation path, log results alongside baselines in `results/comparison.csv` — not yet run (no trained artifacts or results file)

**Definition of Done:** LSTM trained and evaluated with the same metrics as baselines, all three (or two) models in one comparison table. This comparison table is a core artifact for your report — don't lose the intermediate results.

---

## Phase 5 — Evaluation Framework (formalize, if still stubbed)

- [x] `src/evaluation/metrics.py`: AUROC, AUPRC (use sklearn)
- [x] `src/evaluation/metrics.py`: sensitivity/specificity at configurable threshold
- [x] `src/evaluation/metrics.py`: early-prediction gain — for each true positive, hours between first alert crossing threshold and actual onset time, averaged
- [x] `src/evaluation/metrics.py`: simplified utility score per `docs/requirements_spec.md` §10
- [x] `src/evaluation/evaluate.py`: single function that takes a model artifact + test split, returns all four metric groups
- [x] `tests/test_metrics.py`: unit tests for each metric function with hand-computable small examples (e.g., a 4-row confusion matrix where you know the expected sensitivity by hand)
- [ ] Re-run all trained models (Phase 3, 4) through the finalized evaluation path, confirm numbers match earlier stubbed results — blocked on models actually being trained first

**Definition of Done:** all metric functions independently unit-tested; final comparison table regenerated with the complete metric set for every model.

---

## Phase 6 — Explainability

- [x] `src/explainability/shap_explainer.py`: TreeExplainer wired to the XGBoost model
- [x] `src/explainability/shap_explainer.py`: explainer for the LSTM (gradient-based SHAP, or fall back to attention-weight inspection if SHAP proves too slow/unstable on the sequence model — document whichever choice you make and why)
- [x] Function that takes a single patient window + trained model, returns top-N contributing features in human-readable form (e.g., `"Respiratory rate rising", "MAP falling"`, not raw SHAP arrays)
- [x] `tests/test_models.py` (or a new `test_explainability.py`): test that explanation output has the expected shape/type for a sample input
- [ ] Save 2–3 example explanation outputs (screenshots or printed examples) for the report — this is a chapter examiners specifically look for in healthcare ML projects — not yet done, needs a trained model first

**Definition of Done:** every model that will be wired into the API can produce a human-readable explanation for a given input.

---

## Phase 7 — API

- [x] `src/api/schemas.py`: pydantic models for request (patient time-series window) and response (risk score, alert boolean, top contributing features)
- [x] `src/api/predict.py`: loads best model artifact once at startup, wraps predict + explain into one function
- [x] `src/api/main.py`: FastAPI app with `/predict` endpoint, `/patients` endpoint (serves the simulated patient list for the demo), `/health` endpoint
- [x] Wire prediction logging: every call to `/predict` writes a row to the PostgreSQL prediction log table (timestamp, patient_id, risk score, top features)
- [x] `tests/test_api.py`: test `/predict` with a real feature vector returns 200 and expected schema; test malformed input returns a clean 4xx, not a crash
- [ ] Confirm latency: `/predict` responds in under 500ms for a single request (time it, note the number for your non-functional requirements section) — not yet measured/recorded

**Definition of Done:** API runs locally, `/predict` works end-to-end from a real patient window to a scored, explained, logged response.

---

## Phase 8 — Dashboard (Frontend)

- [x] Scaffold React app (Vite) in `frontend/`
- [x] Patient list view — pulls from `/patients`
- [x] Patient detail view — risk trend line chart over time (any lightweight charting lib), pulling from `/predict` per timestep or a batch endpoint
- [x] Alert banner — visibly changes state when risk crosses threshold
- [x] Feature explanation panel — shows the top contributing features returned by the API, in plain language
- [ ] Manual test pass: walk through a full simulated patient going from low risk to a triggered alert, confirm UI updates correctly at every step — not yet done (needs a trained model + demo patient data running end-to-end)

**Definition of Done:** a non-technical person could open the dashboard, click a patient, and understand why the system is (or isn't) concerned about them within 2 minutes — this is the actual bar from the requirements doc, use it literally when reviewing your own UI.

---

## Phase 9 — Integration, Testing Pass, Report Support

- [ ] Full pipeline run via `scripts/run_pipeline.py`: raw data → preprocessed → trained (all models) → evaluated → comparison table, single command, no manual steps — script is written but has not actually been run end-to-end (`models/` and `results/` are still empty)
- [x] Run the entire test suite (`pytest`), confirm everything passes — 26/26 passing as of 2026-08-18
- [ ] Generate final charts for the report: ROC curves, PR curves, comparison bar chart across models, one example patient risk-trend chart with explanation
- [ ] Write up final numbers: AUROC/AUPRC per model, early-prediction gain, utility score, API latency — these all go directly into your results chapter
- [ ] Record a short demo video/walkthrough as a backup in case live demo has issues on viva day

**Definition of Done:** everything in `docs/requirements_spec.md` §13 (Deliverables Checklist) is complete and reproducible from a clean checkout.

---

## Notes for Claude Code

- If a task reveals that an earlier assumption was wrong (e.g., the LSTM needs a different padding strategy than planned), fix it at the source (Phase 2) rather than patching around it downstream — leakage and shape bugs compound badly in ML pipelines.
- Don't skip the "Definition of Done" checks to move faster — an examiner asking "how did you handle data leakage" and getting a vague answer is a worse outcome than a slower but defensible pipeline.
- Flag any deviation from `CLAUDE.md` conventions in the relevant task's commit message so it's easy to explain in the report later.
