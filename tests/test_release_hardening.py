from __future__ import annotations

from fastapi.testclient import TestClient

from fraud_dashboard.api import main as api_main
from fraud_dashboard.api.main import app
from fraud_dashboard.core import config
from fraud_dashboard.core.security import hash_api_key
from fraud_dashboard.platform import store
from fraud_dashboard.platform.store import OpsStore
from tests.test_api import POLICY, _bundle, _load_demo_record


def _reset_state() -> None:
    store._STORE = None  # type: ignore[misc]
    config._SETTINGS = None


def test_legacy_scoring_routes_are_protected_when_auth_is_required(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api_main, "_bundle", _bundle)
    monkeypatch.setattr(api_main, "_policy", lambda: POLICY)
    config._SETTINGS = config.Settings(  # type: ignore[misc]
        database_url=f"sqlite:///{tmp_path / 'ops.db'}",
        require_auth=True,
        jwt_secret_key="x" * 32,
        demo_api_key="strong-local-api-key",
        admin_password="strong-local-password",
    )
    store._STORE = None  # type: ignore[misc]

    payload = {"record": _load_demo_record(), "model": "rf"}
    batch_payload = {"records": [_load_demo_record()], "model": "rf"}
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 401
        assert client.post("/predict/batch", json=batch_payload).status_code == 401
        headers = {"X-API-Key": "strong-local-api-key"}
        assert client.post("/predict", json=payload, headers=headers).status_code == 200
        assert client.post("/predict/batch", json=batch_payload, headers=headers).status_code == 200

    _reset_state()


def test_hashed_api_key_is_accepted_when_configured(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api_main, "_bundle", _bundle)
    monkeypatch.setattr(api_main, "_policy", lambda: POLICY)
    config._SETTINGS = config.Settings(  # type: ignore[misc]
        database_url=f"sqlite:///{tmp_path / 'ops.db'}",
        require_auth=True,
        jwt_secret_key="x" * 32,
        api_key_hash_secret="hash-secret-" + "y" * 24,
        demo_api_key="ignored-plaintext-value",
        demo_api_key_hash="",
        admin_password="strong-local-password",
    )
    digest = hash_api_key("hashed-api-key-value")
    config._SETTINGS = config.Settings(  # type: ignore[misc]
        database_url=f"sqlite:///{tmp_path / 'ops.db'}",
        require_auth=True,
        jwt_secret_key="x" * 32,
        api_key_hash_secret="hash-secret-" + "y" * 24,
        demo_api_key="ignored-plaintext-value",
        demo_api_key_hash=digest,
        admin_password="strong-local-password",
    )
    store._STORE = None  # type: ignore[misc]

    payload = {"record": _load_demo_record(), "model": "rf"}
    with TestClient(app) as client:
        assert client.post("/v1/predictions", json=payload).status_code == 401
        ok = client.post(
            "/v1/predictions", json=payload, headers={"X-API-Key": "hashed-api-key-value"}
        )
        assert ok.status_code == 200

    _reset_state()


def test_unsupported_database_url_fails_loudly() -> None:
    try:
        OpsStore(database_url="postgresql://user:pass@localhost/db")
    except ValueError as exc:
        assert "Only SQLite DATABASE_URL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected unsupported DATABASE_URL to fail loudly")


def test_settings_read_environment_at_creation_time(monkeypatch) -> None:
    config._SETTINGS = None
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    monkeypatch.setenv("MAX_BATCH_RECORDS", "7")
    settings = config.Settings()
    assert settings.require_auth is True
    assert settings.max_batch_records == 7
    config._SETTINGS = None


def test_ui_api_config_sends_api_key_header() -> None:
    from fraud_dashboard.ui.api_client import ApiConfig, _auth_headers

    cfg = ApiConfig(base_url="http://api:8000", api_key="local-key", bearer_token="Bearer jwt")
    assert _auth_headers(cfg) == {
        "X-API-Key": "local-key",
        "Authorization": "Bearer jwt",
    }


def test_metadata_requires_auth_when_protected_mode_is_enabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api_main, "_bundle", _bundle)
    monkeypatch.setattr(api_main, "_policy", lambda: POLICY)
    config._SETTINGS = config.Settings(  # type: ignore[misc]
        database_url=f"sqlite:///{tmp_path / 'ops.db'}",
        require_auth=True,
        jwt_secret_key="x" * 32,
        demo_api_key="strong-local-api-key",
        admin_password="strong-local-password",
    )
    store._STORE = None  # type: ignore[misc]

    with TestClient(app) as client:
        assert client.get("/metadata").status_code == 401
        ok = client.get("/metadata", headers={"X-API-Key": "strong-local-api-key"})
        assert ok.status_code == 200
        assert "artifact_checksums" in ok.json()

    _reset_state()


def test_metrics_summary_exposes_persisted_operational_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api_main, "_bundle", _bundle)
    monkeypatch.setattr(api_main, "_policy", lambda: POLICY)
    config._SETTINGS = config.Settings(database_url=f"sqlite:///{tmp_path / 'ops.db'}")  # type: ignore[misc]
    store._STORE = None  # type: ignore[misc]

    with TestClient(app) as client:
        payload = {"record": _load_demo_record(), "model": "rf", "policy": "min_cost"}
        assert client.post("/v1/predictions", json=payload).status_code == 200
        summary = client.get("/v1/metrics/summary")
        assert summary.status_code == 200
        data = summary.json()
        assert data["prediction_requests"] >= 1
        assert data["predictions"] >= 1
        assert data["latest_policy_version"] == "test-policy-v1"
        assert data["active_model_versions"] >= 0
        assert "jobs" in data

    _reset_state()


def test_reference_tables_are_seeded_from_artifacts(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api_main, "_bundle", _bundle)
    monkeypatch.setattr(api_main, "_policy", lambda: POLICY)
    config._SETTINGS = config.Settings(database_url=f"sqlite:///{tmp_path / 'ops.db'}")  # type: ignore[misc]
    store._STORE = None  # type: ignore[misc]

    with TestClient(app) as client:
        assert client.get("/v1/metrics/summary").status_code == 200
        current_store = store.get_store()
        assert current_store.list_threshold_policies()
        assert isinstance(current_store.list_model_versions(), list)

    _reset_state()


def test_prod_env_requires_auth_enabled() -> None:
    settings = config.Settings(app_env="prod", require_auth=False)
    try:
        config.validate_settings(settings)
    except RuntimeError as exc:
        assert "REQUIRE_AUTH" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected APP_ENV=prod without auth to fail fast")


def test_prod_env_rejects_wildcard_cors() -> None:
    settings = config.Settings(
        app_env="prod",
        require_auth=True,
        jwt_secret_key="x" * 32,
        api_key_hash_secret="y" * 32,
        demo_api_key="strong-local-api-key",
        admin_password="strong-local-password",
        cors_allow_origins=("*",),
    )
    try:
        config.validate_settings(settings)
    except RuntimeError as exc:
        assert "CORS_ALLOW_ORIGINS" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected wildcard CORS to fail in prod")


def test_audit_logs_support_offset(tmp_path) -> None:
    ops = OpsStore(database_url=f"sqlite:///{tmp_path / 'ops.db'}")
    for idx in range(3):
        ops.audit(action=f"action_{idx}")

    first = ops.latest_audit_logs(limit=1, offset=0)
    second = ops.latest_audit_logs(limit=1, offset=1)

    assert len(first) == 1
    assert len(second) == 1
    assert first[0]["id"] != second[0]["id"]
