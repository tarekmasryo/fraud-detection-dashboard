from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraud_dashboard.api.main import app


def _load_demo_record() -> dict:
    """Build a minimal valid payload using the shipped schema.

    This keeps CI/data policy clean: the repo does not ship any third-party dataset.
    """

    repo_root = Path(__file__).resolve().parents[1]
    meta_json = repo_root / "artifacts" / "metadata.json"

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    features = meta["schema"]["features"]

    # Minimal valid numeric payload.
    return {f: 0.0 for f in features}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "message" in data


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metadata(client: TestClient) -> None:
    r = client.get("/metadata")
    assert r.status_code == 200
    data = r.json()
    assert "available_models" in data
    assert "schema" in data
    assert "features" in data["schema"]
    assert len(data["schema"]["features"]) > 10
    assert "thresholds_by_model" in data


def test_predict_success_rf(client: TestClient) -> None:
    rec = _load_demo_record()
    payload = {"record": rec, "model": "rf"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert 0.0 <= out["proba_fraud"] <= 1.0
    assert out["label"] in (0, 1)


def test_predict_success_xgb(client: TestClient) -> None:
    rec = _load_demo_record()
    payload = {"record": rec, "model": "xgb"}
    r = client.post("/predict", json=payload)

    # xgb artifact exists in the repo; if the runtime lacks xgboost, API may error.
    # In GitHub Actions (ubuntu), it should pass.
    assert r.status_code == 200
    out = r.json()
    assert 0.0 <= out["proba_fraud"] <= 1.0
    assert out["label"] in (0, 1)


def test_predict_query_param_overrides_body_model(client: TestClient) -> None:
    rec = _load_demo_record()
    payload = {"record": rec, "model": "rf"}
    r = client.post("/predict?model=xgb", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["model"] == "xgb"


def test_predict_query_param_overrides_body_threshold(client: TestClient) -> None:
    rec = _load_demo_record()
    payload = {"record": rec, "model": "rf", "threshold": 0.1}
    r = client.post("/predict?threshold=0.9", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert abs(out["threshold"] - 0.9) < 1e-12


def test_predict_batch_success(client: TestClient) -> None:
    rec = _load_demo_record()
    payload = {"records": [rec, rec], "model": "rf"}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert "results" in out
    assert len(out["results"]) == 2
    for item in out["results"]:
        assert 0.0 <= item["proba_fraud"] <= 1.0
        assert item["label"] in (0, 1)


def test_predict_batch_query_param_overrides_body_model(client: TestClient) -> None:
    rec = _load_demo_record()
    payload = {"records": [rec, rec], "model": "rf"}
    r = client.post("/predict/batch?model=xgb", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["model"] == "xgb"


def test_predict_missing_column_400(client: TestClient) -> None:
    rec = _load_demo_record()
    # remove a required field
    k = next(iter(rec.keys()))
    rec.pop(k)
    r = client.post("/predict", json={"record": rec, "model": "rf"})
    assert r.status_code == 400
    assert "Missing columns" in r.text


def test_predict_unknown_model_400(client: TestClient) -> None:
    rec = _load_demo_record()
    r = client.post("/predict", json={"record": rec, "model": "nope"})
    assert r.status_code == 400
    assert "Unknown model" in r.text


def test_predict_non_numeric_400(client: TestClient) -> None:
    rec = _load_demo_record()
    # Break one numeric field
    k = next(iter(rec.keys()))
    rec[k] = "not-a-number"
    r = client.post("/predict", json={"record": rec, "model": "rf"})
    assert r.status_code == 400
    assert "Non-numeric" in r.text or "missing values" in r.text


def test_predict_invalid_threshold_500(client: TestClient) -> None:
    rec = _load_demo_record()
    r = client.post("/predict", json={"record": rec, "model": "rf", "threshold": 2.0})
    assert r.status_code == 500
    assert "Invalid threshold" in r.text
