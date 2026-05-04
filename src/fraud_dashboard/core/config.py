from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _str_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or tuple(item.strip() for item in default.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    app_env: str = field(default_factory=lambda: _str_env("APP_ENV", "dev"))
    log_level: str = field(default_factory=lambda: _str_env("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: _str_env("LOG_FORMAT", "text"))
    model_artifact_dir: str = field(
        default_factory=lambda: _str_env("MODEL_ARTIFACT_DIR", "artifacts")
    )
    database_url: str = field(
        default_factory=lambda: _str_env("DATABASE_URL", "sqlite:///./data/fraud_ops.db")
    )
    redis_url: str = field(default_factory=lambda: _str_env("REDIS_URL", "redis://redis:6379/0"))
    require_auth: bool = field(default_factory=lambda: _bool_env("REQUIRE_AUTH", False))
    jwt_secret_key: str = field(
        default_factory=lambda: _str_env("JWT_SECRET_KEY", "dev-only-change-me")
    )
    api_key_hash_secret: str = field(
        default_factory=lambda: _str_env("API_KEY_HASH_SECRET", "dev-only-api-secret")
    )
    demo_api_key: str = field(default_factory=lambda: _str_env("DEMO_API_KEY", "dev-api-key"))
    demo_api_key_hash: str = field(default_factory=lambda: _str_env("DEMO_API_KEY_HASH", ""))
    admin_username: str = field(default_factory=lambda: _str_env("ADMIN_USERNAME", "admin"))
    admin_password: str = field(
        default_factory=lambda: _str_env("ADMIN_PASSWORD", "change-me-for-protected-mode")
    )
    max_batch_records: int = field(default_factory=lambda: _int_env("MAX_BATCH_RECORDS", 1000))
    run_jobs_in_api: bool = field(default_factory=lambda: _bool_env("RUN_JOBS_IN_API", True))
    worker_poll_seconds: int = field(default_factory=lambda: _int_env("WORKER_POLL_SECONDS", 2))
    allow_local_fallback: bool = field(
        default_factory=lambda: _bool_env("ALLOW_LOCAL_FALLBACK", True)
    )
    strict_artifact_runtime: bool = field(
        default_factory=lambda: _bool_env("STRICT_ARTIFACT_RUNTIME", True)
    )
    allow_artifact_compatibility_fallback: bool = field(
        default_factory=lambda: _bool_env("ALLOW_ARTIFACT_COMPATIBILITY_FALLBACK", False)
    )
    prometheus_enabled: bool = field(default_factory=lambda: _bool_env("PROMETHEUS_ENABLED", True))
    cors_allow_origins: tuple[str, ...] = field(
        default_factory=lambda: _csv_env(
            "CORS_ALLOW_ORIGINS", "http://127.0.0.1:8501,http://localhost:8501"
        )
    )

    @property
    def data_dir(self) -> Path:
        return Path("data")


_SETTINGS: Settings | None = None
_SETTINGS_LOCK = Lock()


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        with _SETTINGS_LOCK:
            if _SETTINGS is None:
                _SETTINGS = Settings()
    return _SETTINGS


def validate_settings(settings: Settings | None = None) -> None:
    resolved = settings or get_settings()
    normalized_env = resolved.app_env.strip().lower()
    prod_like = normalized_env in {"prod", "production"}
    if prod_like and not resolved.require_auth:
        raise RuntimeError("REQUIRE_AUTH must be true when APP_ENV=prod or APP_ENV=production")
    if prod_like and "*" in resolved.cors_allow_origins:
        raise RuntimeError("CORS_ALLOW_ORIGINS must not contain '*' when APP_ENV=prod")
    if not resolved.require_auth:
        return
    insecure_values = {
        "dev-only-change-me",
        "dev-only-api-secret",
        "dev-api-key",
        "admin",
        "change-me-for-protected-mode",
        "change-me-in-real-deployments",
        "change-me-too",
        "replace-with-a-strong-hmac-secret",
        "replace-with-a-strong-local-key",
    }
    problems: list[str] = []
    if resolved.jwt_secret_key in insecure_values or len(resolved.jwt_secret_key) < 24:
        problems.append(
            "JWT_SECRET_KEY must be replaced with a strong secret when REQUIRE_AUTH=true"
        )
    if resolved.api_key_hash_secret in insecure_values or len(resolved.api_key_hash_secret) < 24:
        problems.append(
            "API_KEY_HASH_SECRET must be replaced with a strong HMAC secret when REQUIRE_AUTH=true"
        )
    if not resolved.demo_api_key_hash and (
        resolved.demo_api_key in insecure_values or len(resolved.demo_api_key) < 16
    ):
        problems.append(
            "DEMO_API_KEY or DEMO_API_KEY_HASH must be replaced with a strong value when REQUIRE_AUTH=true"
        )
    if resolved.admin_password in insecure_values or len(resolved.admin_password) < 12:
        problems.append(
            "ADMIN_PASSWORD must be replaced with a strong password when REQUIRE_AUTH=true"
        )
    if problems:
        raise RuntimeError("; ".join(problems))
