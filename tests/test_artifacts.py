from __future__ import annotations

from fraud_dashboard.core.artifacts import (
    check_env_versions,
    load_bundle,
    load_models,
    project_root,
)


def test_project_root_exists() -> None:
    p = project_root()
    assert p.exists()
    assert (p / "artifacts").exists()


def test_load_bundle_has_expected_keys() -> None:
    b = load_bundle()
    assert "available_models" in b
    assert "metadata" in b
    assert "schema" in b
    assert "thresholds" in b
    assert "models" in b
    assert set(b["models"].keys()).issuperset({"rf", "xgb"})


def test_load_models_direct() -> None:
    models = load_models()
    assert "rf" in models
    assert "xgb" in models


def test_check_env_versions_mismatch_warnings() -> None:
    # Force mismatches to cover warning paths
    meta = {"env": {"python": "0.0.0", "scikit_learn": "0.0.0", "xgboost": "0.0.0"}}
    warnings = check_env_versions(meta)
    assert len(warnings) >= 1
