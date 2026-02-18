from __future__ import annotations

import numpy as np
import pandas as pd


def validate_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Validate that required columns exist.

    Returns the missing columns (empty list if OK) and raises ValueError if any are missing.
    """

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return missing


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Best-effort conversion to numeric.

    Non-numeric values become NaN. Use `validate_no_nan()` after coercion if you
    want strict input validation.
    """

    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def validate_no_nan(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any NaN exists in the required numeric columns."""

    bad = [c for c in cols if df[c].isna().any()]
    if bad:
        raise ValueError(
            "Non-numeric or missing values detected after coercion in columns: "
            + ", ".join(bad)
        )


def validate_finite(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any +/- inf exists in required numeric columns."""

    bad = []
    for c in cols:
        s = df[c]
        if (~np.isfinite(s.to_numpy())).any():
            bad.append(c)
    if bad:
        raise ValueError("Non-finite values detected in columns: " + ", ".join(bad))
