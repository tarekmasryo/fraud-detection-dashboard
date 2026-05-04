from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from fraud_dashboard.core.config import get_settings


def _sqlite_path_from_url(url: str) -> Path:
    if url.startswith("sqlite:///"):
        return Path(url.replace("sqlite:///", "", 1))
    if url == ":memory:":
        return Path(":memory:")
    raise ValueError(
        "Only SQLite DATABASE_URL values are supported in this release. "
        "Use sqlite:///./data/fraud_ops.db or :memory:."
    )


class OpsStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or get_settings().database_url
        self.path = _sqlite_path_from_url(self.database_url)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        if str(self.path) != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS prediction_requests (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    model_key TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    record_count INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    policy_name TEXT,
                    policy_version TEXT
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    proba_fraud REAL NOT NULL,
                    label INTEGER NOT NULL,
                    decision TEXT,
                    reason_codes_json TEXT,
                    input_hash TEXT,
                    FOREIGN KEY(request_id) REFERENCES prediction_requests(id)
                );
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    actor_id TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    model_key TEXT,
                    threshold REAL,
                    total_records INTEGER NOT NULL DEFAULT 0,
                    processed_records INTEGER NOT NULL DEFAULT 0,
                    failed_records INTEGER NOT NULL DEFAULT 0,
                    request_json TEXT,
                    result_json TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS model_versions (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    model_key TEXT NOT NULL,
                    artifact_file TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS threshold_policies (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    policy_name TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    model_key TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                """
            )
            job_cols = {row[1] for row in conn.execute("PRAGMA table_info(batch_jobs)").fetchall()}
            if "request_json" not in job_cols:
                conn.execute("ALTER TABLE batch_jobs ADD COLUMN request_json TEXT")

            request_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(prediction_requests)").fetchall()
            }
            if "policy_name" not in request_cols:
                conn.execute("ALTER TABLE prediction_requests ADD COLUMN policy_name TEXT")
            if "policy_version" not in request_cols:
                conn.execute("ALTER TABLE prediction_requests ADD COLUMN policy_version TEXT")

            prediction_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
            }
            if "decision" not in prediction_cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN decision TEXT")
            if "reason_codes_json" not in prediction_cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN reason_codes_json TEXT")
            if "input_hash" not in prediction_cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN input_hash TEXT")

    def record_prediction_request(
        self,
        *,
        model_key: str,
        threshold: float,
        probabilities: list[float],
        labels: list[int],
        latency_ms: int,
        source: str,
        status: str = "completed",
        policy_name: str | None = None,
        policy_version: str | None = None,
        decisions: list[str] | None = None,
        reason_codes: list[list[str]] | None = None,
        input_hashes: list[str] | None = None,
    ) -> str:
        request_id = f"pred_{uuid.uuid4().hex[:16]}"
        now = time.time()
        decisions = decisions or ["review" if int(label) else "approve" for label in labels]
        reason_codes = reason_codes or [[] for _ in probabilities]
        input_hashes = input_hashes or [None for _ in probabilities]
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_requests
                (id, created_at, model_key, threshold, record_count, latency_ms, source, status, policy_name, policy_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    now,
                    model_key,
                    float(threshold),
                    len(probabilities),
                    latency_ms,
                    source,
                    status,
                    policy_name,
                    policy_version,
                ),
            )
            for idx, (prob, label) in enumerate(zip(probabilities, labels, strict=False)):
                conn.execute(
                    """
                    INSERT INTO predictions
                    (id, request_id, row_index, proba_fraud, label, decision, reason_codes_json, input_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"p_{uuid.uuid4().hex[:16]}",
                        request_id,
                        idx,
                        float(prob),
                        int(label),
                        decisions[idx] if idx < len(decisions) else None,
                        json.dumps(reason_codes[idx] if idx < len(reason_codes) else []),
                        input_hashes[idx] if idx < len(input_hashes) else None,
                    ),
                )
        self.audit(
            action="prediction_requested",
            resource_type="prediction_request",
            resource_id=request_id,
            metadata={
                "model_key": model_key,
                "record_count": len(probabilities),
                "policy_name": policy_name,
                "policy_version": policy_version,
            },
        )
        return request_id

    def audit(
        self,
        *,
        action: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        actor_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        audit_id = f"audit_{uuid.uuid4().hex[:16]}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs
                (id, created_at, actor_id, action, resource_type, resource_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    time.time(),
                    actor_id,
                    action,
                    resource_type,
                    resource_id,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
        return audit_id

    def create_job(
        self,
        *,
        model_key: str,
        threshold: float,
        total_records: int,
        request: dict[str, Any] | None = None,
    ) -> str:
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO batch_jobs
                (id, created_at, updated_at, status, model_key, threshold, total_records, request_json)
                VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    now,
                    now,
                    model_key,
                    float(threshold),
                    int(total_records),
                    json.dumps(request or {}, ensure_ascii=False),
                ),
            )
        self.audit(
            action="batch_job_created",
            resource_type="batch_job",
            resource_id=job_id,
            metadata={"model_key": model_key, "total_records": total_records},
        )
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        processed_records: int = 0,
        failed_records: int = 0,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE batch_jobs
                SET updated_at = ?, status = ?, processed_records = ?, failed_records = ?, result_json = ?, error = ?
                WHERE id = ?
                """,
                (
                    time.time(),
                    status,
                    processed_records,
                    failed_records,
                    json.dumps(result or {}, ensure_ascii=False),
                    error,
                    job_id,
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM batch_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        if out.get("result_json"):
            out["result"] = json.loads(out["result_json"])
        if out.get("request_json"):
            out["request"] = json.loads(out["request_json"])
        out.pop("result_json", None)
        out.pop("request_json", None)
        return out

    def claim_job(self, job_id: str) -> dict[str, Any] | None:
        """Atomically claim a specific queued job by id."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM batch_jobs WHERE id = ? AND status = 'queued'", (job_id,)
            ).fetchone()
            if row is None:
                return None
            now = time.time()
            conn.execute(
                "UPDATE batch_jobs SET status = 'running', updated_at = ? WHERE id = ?",
                (now, job_id),
            )
        out = dict(row)
        out["status"] = "running"
        if out.get("request_json"):
            out["request"] = json.loads(out["request_json"])
        out.pop("request_json", None)
        if out.get("result_json"):
            out["result"] = json.loads(out["result_json"])
        out.pop("result_json", None)
        return out

    def claim_next_job(self) -> dict[str, Any] | None:
        """Atomically claim the oldest queued job for a worker process."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM batch_jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            now = time.time()
            conn.execute(
                "UPDATE batch_jobs SET status = 'running', updated_at = ? WHERE id = ?",
                (now, row["id"]),
            )
        out = dict(row)
        out["status"] = "running"
        if out.get("request_json"):
            out["request"] = json.loads(out["request_json"])
        out.pop("request_json", None)
        if out.get("result_json"):
            out["result"] = json.loads(out["result_json"])
        out.pop("result_json", None)
        return out

    def upsert_model_versions(self, rows: list[dict[str, Any]]) -> None:
        """Persist the active model artifact inventory for this local release."""
        now = time.time()
        with self.connect() as conn:
            for row in rows:
                model_key = str(row["model_key"])
                artifact_file = str(row["artifact_file"])
                record_id = f"model_{model_key}_{artifact_file}".replace(".", "_").replace("-", "_")
                conn.execute(
                    """
                    INSERT INTO model_versions
                    (id, created_at, model_key, artifact_file, active, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        model_key = excluded.model_key,
                        artifact_file = excluded.artifact_file,
                        active = excluded.active,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        record_id,
                        now,
                        model_key,
                        artifact_file,
                        1 if row.get("active", True) else 0,
                        json.dumps(row.get("metadata") or {}, ensure_ascii=False),
                    ),
                )

    def upsert_threshold_policies(self, rows: list[dict[str, Any]]) -> None:
        """Persist named policy thresholds for operational inspection."""
        now = time.time()
        with self.connect() as conn:
            for row in rows:
                policy_name = str(row["policy_name"])
                policy_version = str(row["policy_version"])
                model_key = str(row["model_key"])
                record_id = f"policy_{policy_version}_{policy_name}_{model_key}".replace(
                    ".", "_"
                ).replace("-", "_")
                conn.execute(
                    """
                    INSERT INTO threshold_policies
                    (id, created_at, policy_name, policy_version, threshold, model_key, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        policy_name = excluded.policy_name,
                        policy_version = excluded.policy_version,
                        threshold = excluded.threshold,
                        model_key = excluded.model_key,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        record_id,
                        now,
                        policy_name,
                        policy_version,
                        float(row["threshold"]),
                        model_key,
                        json.dumps(row.get("metadata") or {}, ensure_ascii=False),
                    ),
                )

    def list_model_versions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_versions ORDER BY active DESC, model_key ASC"
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw_meta = item.pop("metadata_json", "{}")
            try:
                item["metadata"] = json.loads(raw_meta or "{}")
            except json.JSONDecodeError:
                item["metadata"] = {}
            item["active"] = bool(item.get("active"))
            out.append(item)
        return out

    def list_threshold_policies(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM threshold_policies
                ORDER BY policy_version DESC, policy_name ASC, model_key ASC
                """
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw_meta = item.pop("metadata_json", "{}")
            try:
                item["metadata"] = json.loads(raw_meta or "{}")
            except json.JSONDecodeError:
                item["metadata"] = {}
            out.append(item)
        return out

    def metrics_summary(self) -> dict[str, Any]:
        """Aggregate persisted operational state for API/UI consumers."""
        with self.connect() as conn:
            request_count = int(
                conn.execute("SELECT COUNT(*) AS n FROM prediction_requests").fetchone()["n"]
            )
            prediction_count = int(
                conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
            )
            high_risk_count = int(
                conn.execute("SELECT COUNT(*) AS n FROM predictions WHERE label = 1").fetchone()[
                    "n"
                ]
            )
            avg_latency = conn.execute(
                "SELECT AVG(latency_ms) AS value FROM prediction_requests"
            ).fetchone()["value"]
            policy_rows = conn.execute(
                """
                SELECT policy_name, COUNT(*) AS n
                FROM prediction_requests
                WHERE policy_name IS NOT NULL
                GROUP BY policy_name
                ORDER BY n DESC, policy_name ASC
                """
            ).fetchall()
            model_rows = conn.execute(
                """
                SELECT model_key, COUNT(*) AS n
                FROM prediction_requests
                GROUP BY model_key
                ORDER BY n DESC, model_key ASC
                """
            ).fetchall()
            job_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS n, COALESCE(SUM(processed_records), 0) AS processed,
                       COALESCE(SUM(failed_records), 0) AS failed
                FROM batch_jobs
                GROUP BY status
                """
            ).fetchall()
            latest_policy = conn.execute(
                """
                SELECT policy_version
                FROM threshold_policies
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
            active_models = int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM model_versions WHERE active = 1"
                ).fetchone()["n"]
            )

        jobs = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "processed_records": 0,
            "failed_records": 0,
        }
        for row in job_rows:
            status = str(row["status"])
            jobs[status] = int(row["n"])
            jobs["processed_records"] += int(row["processed"] or 0)
            jobs["failed_records"] += int(row["failed"] or 0)

        return {
            "prediction_requests": request_count,
            "predictions": prediction_count,
            "high_risk_predictions": high_risk_count,
            "high_risk_rate": round(high_risk_count / prediction_count, 6)
            if prediction_count
            else 0.0,
            "average_latency_ms": round(float(avg_latency), 3) if avg_latency is not None else 0.0,
            "jobs": jobs,
            "policy_usage": {str(row["policy_name"]): int(row["n"]) for row in policy_rows},
            "model_usage": {str(row["model_key"]): int(row["n"]) for row in model_rows},
            "latest_policy_version": latest_policy["policy_version"] if latest_policy else None,
            "active_model_versions": active_models,
        }

    def latest_audit_logs(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (int(limit), int(offset)),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw_meta = item.pop("metadata_json", "{}")
            try:
                item["metadata"] = json.loads(raw_meta or "{}")
            except json.JSONDecodeError:
                item["metadata"] = {}
            out.append(item)
        return out


_STORE: OpsStore | None = None
_STORE_LOCK = Lock()


def get_store() -> OpsStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = OpsStore()
    return _STORE
