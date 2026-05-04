from __future__ import annotations

from typing import Any

from fraud_dashboard.core.thresholds import normalize_model_key
from fraud_dashboard.platform.store import OpsStore


def seed_reference_data(
    store: OpsStore,
    *,
    bundle: dict[str, Any],
    policy: dict[str, Any],
    checksums: dict[str, str] | None = None,
) -> None:
    """Persist active model and policy references for operational inspection.

    The release intentionally keeps artifacts file-based for local reproducibility,
    but the ops store should still expose the active model/policy inventory so the
    schema is backed by real runtime records instead of decorative tables.
    """

    metadata = bundle.get("metadata") or {}
    models = metadata.get("models") or {}
    checksum_map = checksums or {}

    model_rows: list[dict[str, Any]] = []
    for raw_key, model_meta in models.items():
        artifact_file = str(model_meta.get("file") or "")
        model_key = normalize_model_key(raw_key) or str(raw_key)
        model_rows.append(
            {
                "model_key": model_key,
                "artifact_file": artifact_file,
                "active": True,
                "metadata": {
                    "source_key": raw_key,
                    "pipeline_file": model_meta.get("pipeline_file"),
                    "artifact_checksum": checksum_map.get(artifact_file),
                    "artifact_version": metadata.get("artifact_version"),
                    "trained_at": metadata.get("trained_at"),
                },
            }
        )

    policy_rows: list[dict[str, Any]] = []
    policy_version = str(policy.get("policy_version", "policy"))
    for policy_name, policy_doc in (policy.get("policies") or {}).items():
        thresholds = policy_doc.get("thresholds") or {}
        for raw_model_key, threshold in thresholds.items():
            model_key = normalize_model_key(raw_model_key) or str(raw_model_key)
            policy_rows.append(
                {
                    "policy_name": str(policy_name),
                    "policy_version": policy_version,
                    "model_key": model_key,
                    "threshold": float(threshold),
                    "metadata": {
                        "description": policy_doc.get("description"),
                        "default_policy": policy.get("default_policy"),
                        "release_stage": policy.get("release_stage"),
                    },
                }
            )

    store.upsert_model_versions(model_rows)
    store.upsert_threshold_policies(policy_rows)
