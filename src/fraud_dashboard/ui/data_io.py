from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_dashboard.data.synthetic import SyntheticCreditcardSpec, generate_synthetic_creditcard

DEFAULT_DATA_PATHS = [
    "data/demo_creditcard.csv",
    "creditcard.csv",
    "data/creditcard.csv",
    "/mnt/data/creditcard.csv",
]


def read_csv_any(file) -> pd.DataFrame:
    name = getattr(file, "name", "") or ""
    data = file.read() if hasattr(file, "read") else file
    buf = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else None
    if isinstance(file, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(file))
    if name.endswith(".gz") and buf is not None:
        buf.seek(0)
        return pd.read_csv(buf, compression="gzip")
    if buf is not None:
        buf.seek(0)
        return pd.read_csv(buf)
    return pd.read_csv(file)


@st.cache_data(show_spinner=False)
def try_load_default_dataset() -> tuple[pd.DataFrame | None, str]:
    for p in DEFAULT_DATA_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df, f"Loaded default dataset: {Path(p).name}"
            except Exception:
                continue
    spec = SyntheticCreditcardSpec()
    df = generate_synthetic_creditcard(spec)
    return df, "Loaded demo dataset (synthetic). Upload a real CSV for meaningful results."


def get_active_dataframe(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        try:
            df = read_csv_any(uploaded_file)
            return df, f"Using uploaded dataset: {getattr(uploaded_file, 'name', 'uploaded')}"
        except Exception as e:
            st.error(f"Failed to read uploaded file: {type(e).__name__}")
            return pd.DataFrame(), "Upload failed."
    df0, msg = try_load_default_dataset()
    if df0 is None:
        return pd.DataFrame(), msg
    return df0, msg
