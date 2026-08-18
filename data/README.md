# Dataset

**PhysioNet/Computing in Cardiology Challenge 2019 — Early Prediction of Sepsis**
https://physionet.org/content/challenge-2019/1.0.0/

Public, de-identified, no data-use agreement or CITI training required.

## Download

```bash
./scripts/download_data.sh
```

This fetches ~40,000 per-patient `.psv` files into:

```
data/raw/training_setA/p000001.psv ...
data/raw/training_setB/p100001.psv ...
```

Both `data/raw/` and `data/processed/` are gitignored — never commit raw or processed
patient data.

## Format

Each `.psv` (pipe-separated) file is one ICU patient, one row per hour, columns:

- **Vitals (8):** HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2
- **Labs (26):** BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos,
  Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium,
  Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC,
  Fibrinogen, Platelets
- **Demographics (6):** Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS
- **Label:** SepsisLabel (1 = sepsis onset within the next 6 hours, per Sepsis-3 criteria)

Heavy missingness in labs is expected and clinically meaningful (labs are drawn on
demand, not continuously) — see `src/data/preprocess.py` for how it's handled
(forward-fill within patient + explicit missingness indicators, never cross-patient
imputation).

Class imbalance: ~1.8% of patient-hours are labeled positive. See
`docs/eda_findings.md` (generated in Phase 1) for exact numbers on this copy of the
dataset.
