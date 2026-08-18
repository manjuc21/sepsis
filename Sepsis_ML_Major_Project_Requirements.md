# Project Requirements Specification
## Early Sepsis Prediction and Clinical Alert System Using Machine Learning

**Document type:** BE Final Year Major Project — Requirements & System Design
**Prepared for:** VTU-affiliated BE Major Project (2-phase evaluation model)

> **Assumptions stated up front** (change any of these and the rest of the doc adjusts easily):
> - Team size: 1–4 students (works for solo or group)
> - Dataset: PhysioNet/CinC 2019 Early Sepsis Prediction Challenge dataset (public, ICU time-series, purpose-built for this exact problem)
> - Timeline: standard VTU 2-semester major project (Phase 1 review ~ literature + design + partial implementation; Phase 2 review ~ full implementation + testing + report)
> - Deployment target: local/cloud demo (not a certified clinical product — this matters for scope, see §9)

---

## 1. Problem Statement

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. Mortality rises by roughly 4–8% for every hour treatment is delayed after onset, but early symptoms overlap with many other conditions, making early clinical recognition hard. ICUs generate continuous vitals and lab data that, if modeled well, can flag deterioration hours before it's clinically obvious.

**Goal:** Build a system that ingests ICU patient time-series data (vitals + labs + demographics), predicts sepsis risk ahead of onset, and surfaces that risk to clinical staff through a dashboard with an explainable, actionable alert — not just a black-box probability score.

---

## 2. Why This Qualifies as a "Major" Project (not mini)

A common failure mode: submitting a Jupyter notebook with `.fit()` and a confusion matrix. To be defensible as a major project, this needs to go beyond model training into a full system. The scope below includes:

- A **real system architecture** (ingestion → pipeline → model serving → UI), not just an offline notebook
- **Time-series modeling**, not flat tabular classification — this is technically harder and more defensible in a viva
- **Explainability** (SHAP/attention weights) — increasingly expected for any healthcare ML project
- **A working demo application** (dashboard), giving examiners something interactive to see
- **A rigorous, clinically-relevant evaluation metric** (not just accuracy — sepsis is a rare-event problem, so this needs justification)

If your guide wants a lighter scope, §9 tells you exactly what to cut and what that costs you.

---

## 3. Objectives

1. Curate and preprocess ICU time-series data for sepsis onset prediction.
2. Engineer clinically meaningful features (vital trends, lab abnormalities, missingness patterns — missingness itself is informative in ICU data).
3. Train and compare multiple model families (gradient-boosted trees vs. sequence models) for early prediction, not just detection at onset.
4. Produce risk explanations per prediction (which features drove the alert).
5. Serve predictions through an API and a clinician-facing dashboard with a working alert mechanism.
6. Evaluate using metrics appropriate to early, rare-event prediction (not plain accuracy).

---

## 4. Scope

**In scope:**
- Retrospective prediction using structured EHR time-series (vitals, labs, demographics)
- Prediction horizon: risk score updated per time step (hourly), targeting alert several hours before clinical onset
- Binary classification: sepsis / no sepsis within prediction window
- Explainability layer
- Web dashboard for demo purposes

**Out of scope (explicitly state this in your report to preempt viva questions):**
- Real hospital deployment / integration with live EHR systems (regulatory, liability, and data-access barriers — not feasible for a BE project)
- Clinical validation / IRB-approved trials
- Free-text clinical notes / NLP (unless you want to add it as a stretch goal — see §11)
- Multi-hospital generalization testing (single dataset only)

---

## 5. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | System shall ingest patient time-series records (vitals, labs) in structured format |
| FR2 | System shall preprocess data: handle missing values, normalize, generate rolling-window features |
| FR3 | System shall train and persist an ML model that outputs sepsis risk probability per time step |
| FR4 | System shall expose a prediction API accepting a patient's time-series window and returning risk score + explanation |
| FR5 | System shall generate a feature-level explanation (e.g., SHAP values) for each prediction |
| FR6 | System shall display patient risk trends over time on a dashboard |
| FR7 | System shall trigger a visible/audible alert state when risk crosses a configurable threshold |
| FR8 | System shall allow browsing multiple simulated patients (since this is a demo, not live intake) |
| FR9 | System shall log all predictions with timestamps for later audit/evaluation |

---

## 6. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| Performance | Inference latency under 500ms per prediction request |
| Scalability | Should handle at least 50 simulated concurrent patient streams for demo purposes |
| Reliability | Pipeline should handle missing/malformed input rows without crashing (ICU data is notoriously messy — this will happen constantly) |
| Explainability | Every prediction must return a human-readable top-contributing-features list, not just a number |
| Usability | Dashboard must be understandable by a non-technical evaluator within 2 minutes (this matters a lot for your demo/viva) |
| Reproducibility | Full pipeline (data → features → model → metrics) must be re-runnable from a single script/config, for your report's reproducibility section |
| Security (demo-level) | No real patient-identifiable data — dataset is already de-identified, but reinforce this in your report given the domain sensitivity |

---

## 7. Dataset

**Recommended: PhysioNet/Computing in Cardiology Challenge 2019 — Early Prediction of Sepsis**
- Purpose-built for exactly this problem (unlike MIMIC-III, which needs heavy adaptation)
- ~40,000 ICU patients, hourly time-series, includes the actual sepsis-onset label (SOFA-score + suspected infection based, per Sepsis-3 clinical criteria)
- Features: 8 vital signs (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2), 26 lab values, 6 demographics
- Freely downloadable, no data-use agreement delay (unlike MIMIC, which requires a completed CITI training certificate — worth avoiding for a time-boxed project)

**Key challenge you should name explicitly in your report:** severe class imbalance (~1.8% of ICU stays develop sepsis) and heavy missingness in lab values (some labs drawn only every several hours). This is a legitimate technical contribution point — how you handle it (e.g., forward-fill with missingness indicators, class-weighted loss, SMOTE variants) is exactly the kind of thing examiners probe on.

---

## 8. System Architecture

```
┌─────────────────┐
│  Data Source     │  PhysioNet CSV files (per-patient time series)
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Preprocessing     │  Missing-value handling, normalization,
│ Pipeline          │  rolling-window feature engineering
│ (Python/Pandas)   │
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Model Training     │  XGBoost / LSTM / Temporal CNN (compare 2-3)
│ (offline, notebook  │  → saved as versioned model artifact
│  + scripts)         │
└────────┬─────────┘
         │
┌────────▼─────────┐        ┌──────────────────┐
│ Prediction API     │◄──────│ Explainability     │
│ (FastAPI)           │       │ module (SHAP)      │
└────────┬─────────┘        └──────────────────┘
         │  REST/JSON
┌────────▼─────────┐
│ Dashboard (React)   │  Patient list → risk trend chart →
│                      │  alert banner → feature explanation panel
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Prediction Log DB   │  PostgreSQL — stores every prediction
│                      │  for later evaluation/audit
└──────────────────┘
```

Given your existing FastAPI/Celery/PostgreSQL experience from the answer-booklet system, this stack should feel familiar — same shape: ingestion → processing pipeline → model inference → API → frontend, just healthcare data instead of OCR'd exam scripts.

---

## 9. Model Design — Deep Dive

**Baseline models (must-have, easy to justify in report):**
- Logistic Regression — interpretable baseline
- Random Forest / XGBoost on flattened rolling-window features (e.g., last-6-hours mean/min/max/slope per vital)

**Stronger models (this is what pushes it from "mini" to "major"):**
- LSTM or GRU over the raw hourly sequence — models temporal dependency directly instead of hand-engineered windows
- Optionally: a simple Temporal Convolutional Network (TCN) as a third comparison point

**Recommended framing for your report:** don't just pick one model — run the comparison (tree-based vs. sequence-based) and discuss *why* one wins. That comparison, done rigorously, is often the strongest chapter in these reports.

**Explainability:**
- SHAP (TreeExplainer for XGBoost, KernelExplainer or attention weights for the LSTM) — output the top 3–5 contributing features per alert (e.g., "Rising respiratory rate + falling MAP over last 4 hours")

**Class imbalance handling (pick and justify one):**
- Class-weighted loss function
- SMOTE / ADASYN on the training set only (never on validation/test — a classic mistake examiners will catch)
- Threshold tuning on the precision-recall curve instead of default 0.5

---

## 10. Evaluation Metrics

Accuracy alone is meaningless here (a model predicting "no sepsis" always would be ~98% accurate and useless). Use:

- **AUROC and AUPRC** (AUPRC especially, given class imbalance)
- **Sensitivity/Recall at a clinically reasonable threshold** — missing a sepsis case is much worse than a false alarm
- **Early prediction gain**: how many hours before actual onset the model raises an alert, averaged across true positives — this is the metric that actually matters clinically and is a strong result to headline in your report
- **Utility score** — the PhysioNet 2019 Challenge defined a custom clinical utility function that rewards early true positives and penalizes late detection/false alarms; adopting it (or a simplified version) shows you understood the domain, not just the ML

---

## 11. Optional Stretch Goals (only if time permits after core system works)

- Add clinical notes via a lightweight NLP model (even TF-IDF + logistic regression) fused with the time-series model
- Federated-learning simulation angle (multiple simulated "hospitals") — good if you want an extra novelty angle for a paper/publication
- Deploy on a small cloud instance with a public demo link for your viva

Don't start these until FR1–FR9 and the core model comparison are done and working end-to-end. A polished core system beats a half-finished ambitious one in every VTU viva.

---

## 12. Suggested Timeline (2-semester VTU structure)

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| Literature survey + problem finalization | 1–3 | Survey document, finalized problem statement |
| Dataset acquisition + EDA | 4–6 | Cleaned dataset, EDA report, missingness/imbalance analysis |
| Feature engineering + baseline model | 7–10 | Baseline model (LogReg/XGBoost) with metrics |
| **Phase 1 Review** | ~Week 10 | Problem def, architecture, baseline results |
| Sequence model (LSTM/TCN) + comparison | 11–15 | Trained sequence model, comparison table |
| Explainability integration | 16–17 | SHAP integration into pipeline |
| API + Dashboard build | 18–21 | Working FastAPI + React demo |
| Testing + evaluation writeup | 22–24 | Final metrics, utility score, report draft |
| **Phase 2 Review / Final Viva** | ~Week 25 | Full report, working demo, presentation |

---

## 13. Deliverables Checklist

- [ ] Final report (problem, literature survey, design, implementation, results, conclusion)
- [ ] Working codebase (preprocessing, training, API, dashboard) — version controlled
- [ ] Trained model artifacts + reproducible training script
- [ ] Live/recorded demo of the dashboard with a simulated patient going from low → high risk
- [ ] Comparison table of models with justified metric choice
- [ ] Explainability screenshots/examples in the report
- [ ] (Optional but strong) A short paper draft if your guide wants a publication angle

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Class imbalance tanks naive metrics | Use AUPRC/utility score from the start, not accuracy |
| LSTM overfits on relatively small positive class | Use dropout, early stopping, and compare against XGBoost baseline honestly |
| Dashboard scope creep eats implementation time | Build a minimal working dashboard first (list + risk chart), polish only if time remains |
| Examiners ask "would this work in a real hospital?" | Have the out-of-scope section (§4) ready as your answer — regulatory/integration reality, not a technical gap |
