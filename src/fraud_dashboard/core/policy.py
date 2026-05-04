from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from fraud_dashboard.core.artifacts import artifacts_dir
from fraud_dashboard.core.thresholds import normalize_model_key


def load_policy(root: Path | None = None) -> dict[str, Any]:
    path = artifacts_dir(root) / "policy.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return normalize_policy(raw)
    # Backward-compatible fallback from thresholds.json.
    from fraud_dashboard.core.artifacts import load_thresholds

    thresholds = load_thresholds(root)
    rf = float(thresholds.get("RF_Thr_MinCost", 0.0534831589433206))
    xgb = float(thresholds.get("XGB_Thr_MinCost", 0.22812701761722565))
    return normalize_policy(
        {
            "policy_version": thresholds.get("artifact_version", "legacy-thresholds"),
            "default_model": thresholds.get("default_model", "rf"),
            "default_policy": "min_cost",
            "costs": {
                "false_positive": thresholds.get("COST_FP", 1.0),
                "false_negative": thresholds.get("COST_FN", 10.0),
            },
            "policies": {"min_cost": {"thresholds": {"rf": rf, "xgb": xgb}}},
        }
    )


def normalize_policy(policy: dict[str, Any]) -> dict[str, Any]:
    out = dict(policy)
    out["default_model"] = normalize_model_key(out.get("default_model")) or "rf"
    policies = out.get("policies") or {}
    normalized: dict[str, Any] = {}
    for policy_name, cfg in policies.items():
        if not isinstance(cfg, dict):
            continue
        cfg = dict(cfg)
        thresholds = cfg.get("thresholds") or {}
        cfg["thresholds"] = {
            normalize_model_key(model) or str(model): float(value)
            for model, value in thresholds.items()
            if isinstance(value, (int, float))
        }
        normalized[str(policy_name)] = cfg
    out["policies"] = normalized
    return out


def list_policy_names(policy: dict[str, Any]) -> list[str]:
    return sorted((policy.get("policies") or {}).keys())


def get_default_policy_name(policy: dict[str, Any]) -> str:
    name = str(policy.get("default_policy") or "min_cost")
    if name in (policy.get("policies") or {}):
        return name
    names = list_policy_names(policy)
    if not names:
        raise KeyError("policy.json does not contain any policies")
    return names[0]


def get_policy_threshold(
    policy: dict[str, Any], *, model_key: str, policy_name: str | None = None
) -> float:
    mk = normalize_model_key(model_key) or model_key
    pname = policy_name or get_default_policy_name(policy)
    cfg = (policy.get("policies") or {}).get(pname)
    if not isinstance(cfg, dict):
        raise KeyError(f"Unknown policy '{pname}'")
    thresholds = cfg.get("thresholds") or {}
    if mk in thresholds:
        return float(thresholds[mk])
    raise KeyError(f"Policy '{pname}' does not define a threshold for model '{mk}'")


def policy_to_thresholds(policy: dict[str, Any]) -> dict[str, Any]:
    """Expose policy in the legacy thresholds shape used by UI widgets."""
    costs = policy.get("costs") or {}
    out: dict[str, Any] = {
        "artifact_version": policy.get("policy_version", "policy"),
        "default_model": policy.get("default_model", "rf"),
        "default_policy": get_default_policy_name(policy),
        "COST_FP": float(costs.get("false_positive", 1.0)),
        "COST_FN": float(costs.get("false_negative", 10.0)),
        "policies": policy.get("policies", {}),
    }
    # Backward-compatible keys for the existing UI.
    for key, alias, model in [
        ("strict", "Strict", "rf"),
        ("balanced", "RF_Thr_P90", "rf"),
        ("min_cost", "RF_Thr_MinCost", "rf"),
        ("lenient", "Lenient", "rf"),
    ]:
        with suppress(Exception):
            out[alias] = get_policy_threshold(policy, model_key=model, policy_name=key)
    with suppress(Exception):
        out["XGB_Thr_P90"] = get_policy_threshold(policy, model_key="xgb", policy_name="balanced")
        out["XGB_Thr_MinCost"] = get_policy_threshold(
            policy, model_key="xgb", policy_name="min_cost"
        )
    out["models"] = {
        model: {"threshold": get_policy_threshold(policy, model_key=model)}
        for model in ("rf", "xgb")
        if model in (policy.get("models") or {"rf": {}, "xgb": {}})
    }
    return out
