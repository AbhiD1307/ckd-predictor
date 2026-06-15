"""
Tests for the FastAPI server.
Uses httpx TestClient (sync) — no running server needed.
Skips if model artifacts are not present.
"""

from __future__ import annotations


import pytest

from ckd.config import ARTIFACT_PIPELINE, ARTIFACTS_DIR

ARTIFACT_EXISTS = (ARTIFACTS_DIR / ARTIFACT_PIPELINE).exists()
skip_no_artifact = pytest.mark.skipif(
    not ARTIFACT_EXISTS,
    reason="Model artifact not found — run scripts/train.py first"
)

# Lazy import so import errors in server.py don't break collection
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from ckd.api.server import app
    return TestClient(app)


@skip_no_artifact
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@skip_no_artifact
def test_models_endpoint(client):
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert "available_models" in data
    assert "default_model" in data
    assert len(data["available_models"]) > 0


@skip_no_artifact
def test_predict_high_risk(client):
    payload = {
        "age": 65, "bp": 90, "sg": 1.005, "al": 4, "su": 2,
        "bgr": 220, "bu": 80, "sc": 4.5, "sod": 125, "pot": 6.0, "hemo": 7.0,
        "htn": "yes", "dm": "yes", "appet": "poor", "pe": "yes", "ane": "yes",
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in ("ckd", "notckd")
    assert 0.0 <= data["probability_ckd"] <= 1.0
    assert data["risk_level"] in ("High", "Medium", "Low")


@skip_no_artifact
def test_predict_low_risk(client):
    payload = {
        "age": 30, "bp": 70, "sg": 1.020, "al": 0, "su": 0,
        "bgr": 90, "bu": 15, "sc": 0.8, "sod": 140, "pot": 4.5, "hemo": 15.0,
        "htn": "no", "dm": "no", "appet": "good", "pe": "no", "ane": "no",
        "rbc": "normal", "pc": "normal", "pcc": "notpresent",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["probability_not_ckd"] > 0.0


@skip_no_artifact
def test_predict_batch(client):
    patients = [
        {"age": 60, "bp": 80, "htn": "yes", "dm": "yes", "sc": 3.0},
        {"age": 25, "bp": 70, "htn": "no",  "dm": "no",  "sc": 0.9},
    ]
    r = client.post("/predict/batch", json=patients)
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 2


@skip_no_artifact
def test_predict_empty_payload(client):
    """Empty dict should succeed — all fields will use pipeline defaults."""
    r = client.post("/predict", json={})
    assert r.status_code == 200


@skip_no_artifact
def test_batch_size_limit(client):
    patients = [{"age": 40}] * 501
    r = client.post("/predict/batch", json=patients)
    assert r.status_code == 400
