from __future__ import annotations

import pytest
from fastapi import HTTPException

from fraud_dashboard.core import config
from fraud_dashboard.core.security import create_access_token, require_role, verify_access_token
from fraud_dashboard.core.thresholds import get_model_threshold, normalize_model_key, pick_model_key
from fraud_dashboard.platform import queue
from fraud_dashboard.platform.store import OpsStore
from fraud_dashboard.workers import worker


class _FakeSocket:
    def __init__(self, responses: bytes) -> None:
        self._responses = bytearray(responses)
        self.sent: list[bytes] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def settimeout(self, _timeout: float) -> None:
        return None

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def makefile(self, _mode: str):
        return self

    def readline(self) -> bytes:
        if not self._responses:
            return b""
        end = self._responses.find(b"\r\n")
        if end == -1:
            out = bytes(self._responses)
            self._responses.clear()
            return out
        out = bytes(self._responses[: end + 2])
        del self._responses[: end + 2]
        return out

    def read(self, n: int) -> bytes:
        out = bytes(self._responses[:n])
        del self._responses[:n]
        return out


def test_threshold_helpers_cover_nested_and_fallback_paths() -> None:
    assert normalize_model_key("Calibrated_RF") == "rf"
    assert normalize_model_key("XGBoost") == "xgb"
    assert get_model_threshold({"models": {"rf": {"threshold": 0.42}}}, "random_forest") == 0.42
    assert get_model_threshold({"xgb": 0.33}, "xgboost") == 0.33
    assert pick_model_key({"xgb": object()}, {"default_model": "rf"}) == "xgb"
    assert pick_model_key({"custom": object()}, {}) == "custom"


def test_security_token_and_role_paths() -> None:
    config._SETTINGS = config.Settings(jwt_secret_key="z" * 32)  # type: ignore[misc]
    token = create_access_token("alice", role="analyst", expires_in_s=60)
    payload = verify_access_token(token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "analyst"

    with pytest.raises(HTTPException):
        verify_access_token(token + "broken")

    dependency = require_role("admin")
    with pytest.raises(HTTPException):
        dependency({"role": "viewer"})
    assert dependency({"role": "admin"})["role"] == "admin"
    config._SETTINGS = None


def test_queue_endpoint_and_command_protocol(monkeypatch) -> None:
    ep = queue._endpoint("redis://user:pass@redis.example:6380/2")
    assert ep.host == "redis.example"
    assert ep.port == 6380
    assert ep.db == 2
    assert ep.username == "user"
    assert ep.password == "pass"
    assert queue._encode_command("PING") == b"*1\r\n$4\r\nPING\r\n"

    fake = _FakeSocket(b"+OK\r\n+OK\r\n:1\r\n")

    def fake_connection(address, timeout):
        assert address == ("redis.example", 6380)
        assert timeout == 3.0
        return fake

    config._SETTINGS = config.Settings(redis_url="redis://user:pass@redis.example:6380/2")  # type: ignore[misc]
    monkeypatch.setattr(queue.socket, "create_connection", fake_connection)
    assert queue._command("LPUSH", queue.QUEUE_KEY, "job_1") == 1
    sent = b"".join(fake.sent)
    assert b"AUTH" in sent
    assert b"SELECT" in sent
    assert b"LPUSH" in sent
    config._SETTINGS = None


def test_queue_helpers_with_mocked_command(monkeypatch) -> None:
    monkeypatch.setattr(queue, "_command", lambda *args, **kwargs: 1)
    assert queue.enqueue_job("job_1") is True
    monkeypatch.setattr(queue, "_command", lambda *args, **kwargs: [queue.QUEUE_KEY, "job_2"])
    assert queue.dequeue_job(timeout_s=1) == "job_2"
    monkeypatch.setattr(queue, "_command", lambda *args, **kwargs: None)
    assert queue.dequeue_job(timeout_s=1) is None


def test_worker_run_once_claims_redis_job(monkeypatch) -> None:
    processed: list[str] = []

    class Store:
        def claim_job(self, job_id: str):
            return {"id": job_id, "request": {"records": [{"Amount": 1.0}], "model": "rf"}}

        def claim_next_job(self):
            raise AssertionError("claim_next_job should not be called")

    monkeypatch.setattr(worker, "get_store", lambda: Store())
    monkeypatch.setattr(worker, "dequeue_job", lambda timeout_s: "job_123")
    monkeypatch.setattr(worker, "_process_job", lambda job: processed.append(job["id"]))
    config._SETTINGS = config.Settings(worker_poll_seconds=1)  # type: ignore[misc]
    assert worker.run_once() is True
    assert processed == ["job_123"]
    config._SETTINGS = None


def test_worker_run_once_falls_back_to_sqlite_poll(monkeypatch) -> None:
    processed: list[str] = []

    class Store:
        def claim_job(self, job_id: str):
            return None

        def claim_next_job(self):
            return {"id": "job_sqlite", "request": {"records": [{"Amount": 1.0}], "model": "rf"}}

    monkeypatch.setattr(worker, "get_store", lambda: Store())
    monkeypatch.setattr(worker, "dequeue_job", lambda timeout_s: None)
    monkeypatch.setattr(worker, "_process_job", lambda job: processed.append(job["id"]))
    config._SETTINGS = config.Settings(worker_poll_seconds=1)  # type: ignore[misc]
    assert worker.run_once() is True
    assert processed == ["job_sqlite"]
    config._SETTINGS = None


def test_store_update_and_read_paths(tmp_path) -> None:
    db_path = tmp_path / "ops.db"
    store = OpsStore(database_url=f"sqlite:///{db_path}")
    job_id = store.create_job(
        model_key="rf", threshold=0.5, total_records=1, request={"model": "rf"}
    )
    assert store.get_job(job_id)["status"] == "queued"  # type: ignore[index]
    store.update_job(job_id, status="completed", processed_records=1, result={"ok": True})
    completed = store.get_job(job_id)
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"]["ok"] is True
    store.audit(action="test_action", resource_type="unit", resource_id="resource")
    assert any(item["action"] == "test_action" for item in store.latest_audit_logs())
