"""Pydantic request/response contracts. Defined before the frontend is built
(CLAUDE.md §8 — wire the contract first)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src import config


class _BaseSchema(BaseModel):
    # PredictResponse/HealthResponse use a `model_*` field name, which
    # collides with pydantic's own reserved "model_" namespace by default.
    model_config = ConfigDict(protected_namespaces=())


class VitalsLabsRow(BaseModel):
    """One hour of a patient's time series. All fields optional except
    ICULOS — ICU data is heavily missing by nature (CLAUDE.md §7)."""

    ICULOS: int
    HR: float | None = None
    O2Sat: float | None = None
    Temp: float | None = None
    SBP: float | None = None
    MAP: float | None = None
    DBP: float | None = None
    Resp: float | None = None
    EtCO2: float | None = None
    BaseExcess: float | None = None
    HCO3: float | None = None
    FiO2: float | None = None
    pH: float | None = None
    PaCO2: float | None = None
    SaO2: float | None = None
    AST: float | None = None
    BUN: float | None = None
    Alkalinephos: float | None = None
    Calcium: float | None = None
    Chloride: float | None = None
    Creatinine: float | None = None
    Bilirubin_direct: float | None = None
    Glucose: float | None = None
    Lactate: float | None = None
    Magnesium: float | None = None
    Phosphate: float | None = None
    Potassium: float | None = None
    Bilirubin_total: float | None = None
    TroponinI: float | None = None
    Hct: float | None = None
    Hgb: float | None = None
    PTT: float | None = None
    WBC: float | None = None
    Fibrinogen: float | None = None
    Platelets: float | None = None


class PredictRequest(BaseModel):
    patient_id: str
    age: float
    gender: int = Field(ge=0, le=1)
    hosp_adm_time: float = 0.0
    window: list[VitalsLabsRow] = Field(
        min_length=1,
        description="Hourly rows for this patient, oldest first. Only the "
        "trailing window up to the current hour is needed.",
    )


class FeatureContribution(BaseModel):
    description: str


class PredictResponse(_BaseSchema):
    patient_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    alert: bool
    threshold: float = config.ALERT_THRESHOLD
    top_features: list[str]
    model_name: str
    latency_ms: float


class PatientSummary(BaseModel):
    patient_id: str
    current_risk_score: float
    alert: bool
    last_updated_hour: int


class HealthResponse(_BaseSchema):
    status: str = "ok"
    model_loaded: bool
