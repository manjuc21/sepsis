"""FastAPI app: /predict, /patients, /patients/{id}/timeline, /health.

Every /predict call is logged to Postgres for audit (FR9). DB/model
unavailability degrades gracefully (returns a clear error / flag) rather than
crashing the process, since this API needs to be startable in dev/test
environments where Postgres or a trained model may not be present yet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.api import db
from src.api.predict import PredictionService
from src.api.schemas import HealthResponse, PatientSummary, PredictRequest, PredictResponse

logger = logging.getLogger("sepsis.api")

app = FastAPI(title="Sepsis Early Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo project, not a real deployment (CLAUDE.md §2.5)
    allow_methods=["*"],
    allow_headers=["*"],
)

_service_holder: dict[str, PredictionService | None] = {"service": None}
_db_available_holder: dict[str, bool] = {"available": False}

DEMO_PATIENTS_PATH = config.PROCESSED_DATA_DIR / "demo_patients.json"


@app.on_event("startup")
def startup() -> None:
    try:
        _service_holder["service"] = PredictionService()
        logger.info("Loaded model: %s", _service_holder["service"].model_name)
    except FileNotFoundError as e:
        logger.warning("No trained model available at startup: %s", e)

    try:
        db.init_db()
        _db_available_holder["available"] = True
    except Exception as e:
        logger.warning("Postgres unavailable, prediction logging disabled: %s", e)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(model_loaded=_service_holder["service"] is not None)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    service = _service_holder["service"]
    if service is None:
        raise HTTPException(status_code=503, detail="No trained model loaded yet")

    try:
        response = service.predict_and_explain(request)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not score request: {e}")

    if _db_available_holder["available"]:
        try:
            session = db.SessionLocal()
            db.log_prediction(session, response)
            session.close()
        except Exception as e:
            logger.warning("Failed to log prediction: %s", e)

    return response


def _load_demo_patients() -> list[dict]:
    if not DEMO_PATIENTS_PATH.exists():
        return []
    return json.loads(DEMO_PATIENTS_PATH.read_text())["patients"]


@app.get("/patients", response_model=list[PatientSummary])
def list_patients() -> list[PatientSummary]:
    patients = _load_demo_patients()
    return [
        PatientSummary(
            patient_id=p["patient_id"],
            current_risk_score=p["timeline"][-1]["risk_score"],
            alert=p["timeline"][-1]["risk_score"] >= config.ALERT_THRESHOLD,
            last_updated_hour=p["timeline"][-1]["hour"],
        )
        for p in patients
    ]


@app.get("/patients/{patient_id}/timeline")
def patient_timeline(patient_id: str) -> dict:
    for p in _load_demo_patients():
        if p["patient_id"] == patient_id:
            return p
    raise HTTPException(status_code=404, detail="Unknown patient_id")
