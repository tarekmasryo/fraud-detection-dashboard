from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud_dashboard.api import main as api_main
from fraud_dashboard.api.main import app

FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

POLICY = {
    "policy_version": "test-policy-v1",
    "default_model": "rf",
    "default_policy": "min_cost",
    "policies": {
        "min_cost": {"thresholds": {"rf": 0.2, "xgb": 0.3}},
        "balanced": {"thresholds": {"rf": 0.5, "xgb": 0.6}},
    },
}


class _FakeModel:
    def __init__(self, base_score: float) -> None:
        self.base_score = base_score

    def predict_proba(self, X):
        scores = np.full(len(X), self.base_score, dtype=float)
        return np.column_stack([1.0 - scores, scores])


def _bundle() -> dict:
    return {
        "available_models": ["rf", "xgb"],
        "metadata": {"env": {}, "models": {}},
        "schema": {"features": FEATURES},
        "thresholds": {
            "default_model": "rf",
            "models": {"rf": {"threshold": 0.2}, "xgb": {"threshold": 0.3}},
        },
        "policy": POLICY,
        "models": {"rf": _FakeModel(0.1), "xgb": _FakeModel(0.4)},
    }


def _load_demo_record() -> dict[str, float]:
    return {feature: 0.0 for feature in FEATURES}


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    from fraud_dashboard.core import config
    from fraud_dashboard.platform import store

    monkeypatch.setattr(api_main, "_bundle", _bundle)
    monkeypatch.setattr(api_main, "_policy", lambda: POLICY)
    config._SETTINGS = config.Settings(database_url=f"sqlite:///{tmp_path / 'ops.db'}")  # type: ignore[misc]
    store._STORE = None  # type: ignore[misc]

    with TestClient(app) as test_client:
        yield test_client

    store._STORE = None  # type: ignore[misc]
    config._SETTINGS = None


def test_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metadata(client: TestClient) -> None:
    r = client.get("/metadata")
    assert r.status_code == 200
    data = r.json()
    assert data["available_models"] == ["rf", "xgb"]
    assert "features" in data["schema"]
    assert len(data["schema"]["features"]) > 10
    assert "thresholds_by_model" in data


def test_predict_success_rf(client: TestClient) -> None:
    payload = {"record": _load_demo_record(), "model": "rf"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["model"] == "rf"
    assert 0.0 <= out["proba_fraud"] <= 1.0
    assert out["label"] in (0, 1)
    assert isinstance(out["latency_ms"], int)


def test_predict_success_xgb(client: TestClient) -> None:
    payload = {"record": _load_demo_record(), "model": "xgb"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["model"] == "xgb"
    assert 0.0 <= out["proba_fraud"] <= 1.0
    assert out["label"] in (0, 1)


def test_predict_query_param_overrides_body_model(client: TestClient) -> None:
    payload = {"record": _load_demo_record(), "model": "rf"}
    r = client.post("/predict?model=xgb", json=payload)
    assert r.status_code == 200
    assert r.json()["model"] == "xgb"


def test_predict_query_param_overrides_body_threshold(client: TestClient) -> None:
    payload = {"record": _load_demo_record(), "model": "rf", "threshold": 0.1}
    r = client.post("/predict?threshold=0.9", json=payload)
    assert r.status_code == 200
    assert abs(r.json()["threshold"] - 0.9) < 1e-12


def test_predict_batch_success(client: TestClient) -> None:
    rec = _load_demo_record()
    payload = {"records": [rec, rec], "model": "rf"}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert len(out["results"]) == 2
    for item in out["results"]:
        assert 0.0 <= item["proba_fraud"] <= 1.0
        assert item["label"] in (0, 1)


def test_predict_batch_query_param_overrides_body_model(client: TestClient) -> None:
    rec = _load_demo_record()
    payload = {"records": [rec, rec], "model": "rf"}
    r = client.post("/predict/batch?model=xgb", json=payload)
    assert r.status_code == 200
    assert r.json()["model"] == "xgb"


def test_predict_missing_column_400(client: TestClient) -> None:
    rec = _load_demo_record()
    rec.pop("Time")
    r = client.post("/predict", json={"record": rec, "model": "rf"})
    assert r.status_code == 400
    assert "Missing columns" in r.text


def test_predict_unknown_model_400(client: TestClient) -> None:
    r = client.post("/predict", json={"record": _load_demo_record(), "model": "nope"})
    assert r.status_code == 400
    assert "Unknown model" in r.text


def test_predict_unknown_policy_400(client: TestClient) -> None:
    r = client.post(
        "/v1/predictions",
        json={"record": _load_demo_record(), "model": "rf", "policy": "nope"},
    )
    assert r.status_code == 400
    assert "Unknown policy" in r.text


def test_predict_non_numeric_400(client: TestClient) -> None:
    rec = _load_demo_record()
    rec["Time"] = "not-a-number"  # type: ignore[assignment]
    r = client.post("/predict", json={"record": rec, "model": "rf"})
    assert r.status_code == 400
    assert "Non-numeric" in r.text or "missing values" in r.text


def test_predict_invalid_threshold_400_or_422(client: TestClient) -> None:
    r = client.post(
        "/predict", json={"record": _load_demo_record(), "model": "rf", "threshold": 2.0}
    )
    assert r.status_code in {400, 422}
    assert "Invalid threshold" in r.text or "less than or equal to 1" in r.text


def test_metadata_exposes_batch_limit(client: TestClient) -> None:
    r = client.get("/metadata")
    assert r.status_code == 200
    assert r.json()["limits"]["max_batch_records"] >= 1
