from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    timeout_s: float = 10.0


class ApiError(RuntimeError):
    """Raised when the FastAPI backend cannot be reached or returns an error."""


def get_api_config() -> ApiConfig:
    """Resolve the FastAPI base URL from env vars.

    - Preferred: FRAUD_API_URL
    - Back-compat: API_BASE_URL
    """

    base_url = (
        os.getenv("FRAUD_API_URL") or os.getenv("API_BASE_URL") or "http://127.0.0.1:8000"
    ).rstrip("/")
    return ApiConfig(base_url=base_url)


def _client(cfg: ApiConfig) -> httpx.Client:
    return httpx.Client(base_url=cfg.base_url, timeout=cfg.timeout_s)


def _request_json(
    method: str, path: str, *, cfg: ApiConfig, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        with _client(cfg) as c:
            r = c.request(method, path, json=payload)
        r.raise_for_status()
        data = r.json()
        # Attach a lightweight client-side timing too (useful for UI debugging).
        data.setdefault("client_latency_ms", (time.perf_counter() - t0) * 1000.0)
        return data
    except httpx.RequestError as e:
        raise ApiError(f"Cannot reach API at {cfg.base_url}: {e}") from e
    except httpx.HTTPStatusError as e:
        body = e.response.text
        raise ApiError(f"API error {e.response.status_code} on {path}: {body}") from e
    except ValueError as e:
        raise ApiError(f"API returned non-JSON response on {path}") from e


def ping(cfg: ApiConfig) -> dict[str, Any]:
    """GET /health"""

    return _request_json("GET", "/health", cfg=cfg)


def ping_ok(cfg: ApiConfig) -> bool:
    """Return True if GET /health succeeds."""
    try:
        data = ping(cfg)
        return bool(data.get("status") == "ok")
    except ApiError:
        return False


def fetch_metadata(cfg: ApiConfig) -> dict[str, Any]:
    """GET /metadata"""

    return _request_json("GET", "/metadata", cfg=cfg)


def predict_one(*, cfg: ApiConfig, model: str, record: dict[str, float]) -> dict[str, Any]:
    """POST /predict"""

    payload = {"model": model, "record": record}
    return _request_json("POST", "/predict", cfg=cfg, payload=payload)


def predict_batch(*, cfg: ApiConfig, model: str, records: list[dict[str, float]]) -> dict[str, Any]:
    """POST /predict/batch"""

    payload = {"model": model, "records": records}
    return _request_json("POST", "/predict/batch", cfg=cfg, payload=payload)
