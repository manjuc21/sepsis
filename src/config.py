"""Single source of truth for paths, seeds, and thresholds. Nothing below should
be hardcoded again elsewhere in src/."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Reproducibility -------------------------------------------------------
RANDOM_SEED: int = 42

# --- Paths -------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
RAW_SET_A_DIR: Path = RAW_DATA_DIR / "training_setA"
RAW_SET_B_DIR: Path = RAW_DATA_DIR / "training_setB"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# --- Dataset schema (PhysioNet/CinC 2019 Sepsis Challenge) -----------------
VITAL_COLUMNS: list[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
]
LAB_COLUMNS: list[str] = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
DEMOGRAPHIC_COLUMNS: list[str] = [
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]
# Demographic columns that PhysioNet leaves genuinely missing for some
# patients (e.g. hospital system B never populates Unit1/Unit2). These need
# the same train-fit imputation as vitals/labs, or downstream consumers that
# reject NaNs (e.g. features.build_sequence_tensors) fail outright.
DEMOGRAPHIC_IMPUTE_COLUMNS: list[str] = ["Unit1", "Unit2", "HospAdmTime"]
LABEL_COLUMN: str = "SepsisLabel"
FEATURE_COLUMNS: list[str] = VITAL_COLUMNS + LAB_COLUMNS + DEMOGRAPHIC_COLUMNS

# --- Splits -------------------------------------------------------------
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15

# --- Feature engineering -------------------------------------------------------
ROLLING_WINDOW_HOURS: int = 6
MAX_SEQUENCE_LENGTH: int = 336  # 14 days at hourly resolution, generous cap

# --- Modeling / evaluation -------------------------------------------------------
ALERT_THRESHOLD: float = 0.5  # placeholder; tuned in Phase 5 via PR curve
LSTM_HIDDEN_SIZE: int = 64
LSTM_NUM_LAYERS: int = 1
LSTM_DROPOUT: float = 0.3
LSTM_BATCH_SIZE: int = 64
LSTM_LEARNING_RATE: float = 1e-3
LSTM_MAX_EPOCHS: int = 30
LSTM_EARLY_STOP_PATIENCE: int = 4

# Utility score window, per PhysioNet 2019 Challenge definition (docs/requirements_spec.md §10)
UTILITY_EARLY_WINDOW_HOURS: int = 12
UTILITY_LATE_WINDOW_HOURS: int = 3

# --- Model artifact naming -------------------------------------------------------
LOGREG_ARTIFACT: Path = MODELS_DIR / "logreg.pkl"
XGBOOST_ARTIFACT: Path = MODELS_DIR / "xgboost.pkl"
LSTM_ARTIFACT: Path = MODELS_DIR / "lstm.pt"

# --- Database (Postgres, via docker-compose service "db") -------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://sepsis:sepsis@localhost:5432/sepsis",
)

# --- API -------------------------------------------------------------
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_LATENCY_BUDGET_MS: int = 500
