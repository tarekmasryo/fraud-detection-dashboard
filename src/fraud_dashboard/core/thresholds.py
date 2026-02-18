from __future__ import annotations

from typing import Any


def normalize_model_key(model_name: str | None) -> str | None:
    """Normalize model identifiers across UI / API / artifact naming.

    Canonical keys used by this repo:
      - rf
      - xgb
    """

    if model_name is None:
        return None

    m = str(model_name).strip().lower()
    aliases = {
        "rf": "rf",
        "random_forest": "rf",
        "randomforest": "rf",
        "calibrated_rf": "rf",
        "xgb": "xgb",
        "xgboost": "xgb",
        "calibrated_xgb": "xgb",
    }
    return aliases.get(m, m)


def get_primary_threshold(thresholds: Any, *, model_name: str | None = None) -> float:
    """Return a usable threshold value from multiple tolerated formats.

    This function is intentionally permissive because threshold policies tend to
    drift across versions.

    Supported inputs (examples from tests):
      - 0.42
      - {"primary": 0.33}
      - {"rf": 0.21}
      - [0.1, 0.2, 0.3]

    Additional tolerated formats:
      - {"primary_threshold": 0.5}
      - {"models": {"rf": {"threshold": 0.5}, "xgb": {"threshold": 0.2}}}
    """

    model_key = normalize_model_key(model_name)

    # scalar
    if isinstance(thresholds, (int, float)):
        return float(thresholds)

    # list/tuple: pick first numeric
    if isinstance(thresholds, (list, tuple)):
        for v in thresholds:
            if isinstance(v, (int, float)):
                return float(v)
        raise KeyError("No numeric threshold found in sequence.")

    if isinstance(thresholds, dict):
        # common single-key formats
        for k in ("primary", "primary_threshold", "threshold"):
            if k in thresholds and isinstance(thresholds[k], (int, float)):
                return float(thresholds[k])

        # model-specific direct mapping: {"rf": 0.21}
        if (
            model_key
            and model_key in thresholds
            and isinstance(thresholds[model_key], (int, float))
        ):
            return float(thresholds[model_key])

        # nested models config: {"models": {"rf": {"threshold": 0.5}}}
        models_cfg = thresholds.get("models")
        if isinstance(models_cfg, dict):
            if model_key:
                cfg = models_cfg.get(model_key)
                if isinstance(cfg, dict) and isinstance(cfg.get("threshold"), (int, float)):
                    return float(cfg["threshold"])

            # fallback: first model threshold
            for _, cfg in models_cfg.items():
                if isinstance(cfg, dict) and isinstance(cfg.get("threshold"), (int, float)):
                    return float(cfg["threshold"])

        # last resort: any numeric value in dict
        for _, v in thresholds.items():
            if isinstance(v, (int, float)):
                return float(v)

    raise KeyError("threshold policy does not contain a usable threshold")


def get_model_threshold(thresholds: dict[str, Any], model_key: str) -> float:
    """Get threshold for a specific model key (canonical 'rf'/'xgb')."""

    mk = normalize_model_key(model_key) or model_key

    # nested models format
    models_cfg = thresholds.get("models")
    if isinstance(models_cfg, dict):
        cfg = models_cfg.get(mk)
        if isinstance(cfg, dict) and isinstance(cfg.get("threshold"), (int, float)):
            return float(cfg["threshold"])

    # mapping format {"rf": 0.5}
    if mk in thresholds and isinstance(thresholds[mk], (int, float)):
        return float(thresholds[mk])

    # fallback to primary
    return get_primary_threshold(thresholds, model_name=mk)


def get_default_model_key(thresholds: dict[str, Any]) -> str | None:
    """Return default model key from a thresholds config."""

    dm = thresholds.get("default_model")
    if isinstance(dm, str):
        return normalize_model_key(dm)
    return None


def pick_model_key(available_models: dict[str, Any], thresholds: dict[str, Any]) -> str:
    """Pick a model key for inference when request doesn't specify one."""

    dm = get_default_model_key(thresholds)
    if dm and dm in available_models:
        return dm

    # common preference order
    for k in ("rf", "xgb"):
        if k in available_models:
            return k

    # deterministic fallback
    return sorted(available_models.keys())[0]
