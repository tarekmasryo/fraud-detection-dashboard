from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _pick_threshold_for_review_rate(proba: np.ndarray, review_rate: float) -> float:
    review_rate = float(np.clip(review_rate, 0.0, 1.0))
    if proba.size == 0:
        return 0.5
    q = 1.0 - review_rate
    return float(np.quantile(proba, q))


def _eval_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "review_rate": float((tp + fp) / max(len(y_true), 1)),
    }


def train(
    data_path: Path,
    out_dir: Path,
    label_col: str,
    seed: int,
    test_size: float,
    target_review_rate: float,
    cost_fp: float,
    cost_fn: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {data_path}")

    df = df.dropna()
    y = df[label_col].astype(int).to_numpy()
    X = _ensure_numeric(df.drop(columns=[label_col]))

    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    rf_cal = CalibratedClassifierCV(rf, cv="prefit", method="sigmoid")
    rf_cal.fit(X_cal, y_cal)

    # XGBoost
    if xgb is None:
        raise RuntimeError(
            "xgboost is not installed. Install it (see requirements.txt) to train the XGBoost model."
        )

    neg = float((y_train == 0).sum())
    pos = float((y_train == 1).sum())
    scale_pos_weight = (neg / max(pos, 1.0)) if pos > 0 else 1.0

    xgb_clf = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    xgb_clf.fit(X_train, y_train)
    xgb_cal = CalibratedClassifierCV(xgb_clf, cv="prefit", method="sigmoid")
    xgb_cal.fit(X_cal, y_cal)

    # Thresholds (simple review-rate targeting)
    rf_proba = rf_cal.predict_proba(X_cal)[:, 1]
    xgb_proba = xgb_cal.predict_proba(X_cal)[:, 1]

    rf_thr = _pick_threshold_for_review_rate(rf_proba, target_review_rate)
    xgb_thr = _pick_threshold_for_review_rate(xgb_proba, target_review_rate)

    thresholds = {
        "models": {
            "random_forest": {
                "threshold": float(rf_thr),
                "description": "Chosen to match target review rate on calibration split.",
                "evaluation": _eval_at_threshold(y_cal, rf_proba, rf_thr),
            },
            "xgboost": {
                "threshold": float(xgb_thr),
                "description": "Chosen to match target review rate on calibration split.",
                "evaluation": _eval_at_threshold(y_cal, xgb_proba, xgb_thr),
            },
        },
        "cost_settings": {
            "cost_fp": float(cost_fp),
            "cost_fn": float(cost_fn),
            "target_review_rate": float(target_review_rate),
        },
    }

    metadata = {
        "artifact_version": "v1",
        "trained_at": None,
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": __import__("sklearn").__version__,
            "xgboost": getattr(__import__("xgboost"), "__version__", None),
        },
        "schema": {"label": label_col, "features": list(X.columns)},
        "thresholds_file": "thresholds.json",
        "models": {
            "random_forest": {"file": "rf_calibrated.joblib", "pipeline_file": "rf_pipe.joblib"},
            "xgboost": {"file": "xgb_calibrated.joblib", "pipeline_file": "xgb_pipe.joblib"},
        },
        "metrics": {
            "rf_auc": float(roc_auc_score(y_cal, rf_proba)),
            "xgb_auc": float(roc_auc_score(y_cal, xgb_proba)),
        },
    }

    # Save artifacts (match dashboard expectations)
    joblib.dump(rf, out_dir / "rf_pipe.joblib")
    joblib.dump(rf_cal, out_dir / "rf_calibrated.joblib")
    joblib.dump(xgb_clf, out_dir / "xgb_pipe.joblib")
    joblib.dump(xgb_cal, out_dir / "xgb_calibrated.joblib")
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train calibrated RF + XGBoost fraud models and export artifacts."
    )
    p.add_argument(
        "--data", type=Path, required=True, help="Path to CSV (must include label column)."
    )
    p.add_argument(
        "--out", type=Path, default=Path("artifacts"), help="Output artifacts directory."
    )
    p.add_argument("--label", type=str, default="Class", help="Label column name.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--test-size", type=float, default=0.2, help="Calibration split size.")
    p.add_argument(
        "--target-review-rate", type=float, default=0.01, help="Target flagged transaction rate."
    )
    p.add_argument("--cost-fp", type=float, default=5.0, help="False positive cost.")
    p.add_argument("--cost-fn", type=float, default=500.0, help="False negative cost.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data_path=args.data,
        out_dir=args.out,
        label_col=args.label,
        seed=args.seed,
        test_size=args.test_size,
        target_review_rate=args.target_review_rate,
        cost_fp=args.cost_fp,
        cost_fn=args.cost_fn,
    )


if __name__ == "__main__":
    main()
