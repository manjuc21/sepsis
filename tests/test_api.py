from __future__ import annotations

import pickle

import pytest
from fastapi.testclient import TestClient

from src import config
from src.data import features
from src.models.baseline import XGBoostModel


@pytest.fixture
def api_client(tmp_path, monkeypatch, synthetic_patients):
    X, y = features.build_tabular_feature_matrix(synthetic_patients)
    model = XGBoostModel().fit(X, y)

    artifact_path = tmp_path / "xgboost.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump({"model": model, "feature_columns": list(X.columns)}, f)

    monkeypatch.setattr(config, "XGBOOST_ARTIFACT", artifact_path)
    monkeypatch.setattr(config, "LOGREG_ARTIFACT", tmp_path / "no_logreg.pkl")
    monkeypatch.setattr(config, "LSTM_ARTIFACT", tmp_path / "no_lstm.pt")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)
    # point at an unroutable DB so db.init_db() fails fast and predictable
    # rather than hanging on a real connection attempt in CI/test envs.
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://x:x@127.0.0.1:1/x")

    from src.api import main as main_module

    with TestClient(main_module.app) as client:
        yield client


def _sample_request(synthetic_patients):
    p0 = synthetic_patients[synthetic_patients["patient_id"] == synthetic_patients["patient_id"].iloc[0]]
    window = []
    for _, row in p0.iterrows():
        record = {"ICULOS": int(row["ICULOS"])}
        for col in config.VITAL_COLUMNS + config.LAB_COLUMNS:
            val = row[col]
            record[col] = None if val != val else float(val)  # NaN check
        window.append(record)
    return {
        "patient_id": "test-patient",
        "age": float(p0["Age"].iloc[0]),
        "gender": int(p0["Gender"].iloc[0]),
        "hosp_adm_time": float(p0["HospAdmTime"].iloc[0]),
        "window": window,
    }


def test_health_reports_model_loaded(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_200_and_expected_schema(api_client, synthetic_patients):
    response = api_client.post("/predict", json=_sample_request(synthetic_patients))
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert isinstance(body["alert"], bool)
    assert isinstance(body["top_features"], list)
    assert len(body["top_features"]) > 0
    assert body["model_name"] == "xgboost"
    assert body["latency_ms"] < config.API_LATENCY_BUDGET_MS


def test_predict_malformed_input_returns_clean_4xx(api_client):
    response = api_client.post("/predict", json={"patient_id": "bad"})  # missing required fields
    assert 400 <= response.status_code < 500


def test_patients_endpoint_returns_list(api_client):
    response = api_client.get("/patients")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
