import os
import io
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Tuple, Optional

# Plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Metrics
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance

# ----------------------
# App Config
# ----------------------
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🧭",
    layout="wide",
)

# ----------------------
# Single Global Theme (Aurora Light)
# ----------------------
PRIMARY = "#3B82F6"  

# Color sets (colorblind-friendly)
AURORA_COLORWAY = ["#3366CC","#DC3912","#FF9900","#109618","#990099","#3B3EAC",
                   "#0099C6","#DD4477","#66AA00","#B82E2E"]
AURORA_CS = "Tealrose"

base_layout = dict(
    font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=13),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(namelength=-1, font_size=12),
    legend=dict(title=None, orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
    yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
    paper_bgcolor="white",
    plot_bgcolor="rgba(248,250,252,1)",
    colorway=AURORA_COLORWAY
)

pio.templates["tarek_theme"] = go.layout.Template(layout=base_layout)
px.defaults.template = "tarek_theme"
px.defaults.color_continuous_scale = AURORA_CS

def style_fig(fig: go.Figure, title: str = None):
    fig.update_layout(template="tarek_theme")
    if title:
        fig.update_layout(title=dict(text=title, x=0.02, xanchor="left", font_size=18))
    fig.update_layout(modebar_add=["togglespikelines"], hovermode="x unified")
    return fig

# ----------------------
# CSS (Glass UI) 
# ----------------------
st.markdown(f"""
<style>
:root {{
  --glass-bg: rgba(255,255,255,0.55);
  --glass-br: 16px;
  --shadow: 0 10px 30px rgba(2, 6, 23, 0.10);
}}
html, body, [data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 800px at 10% 10%, rgba(130,170,255,0.12), transparent),
              radial-gradient(1000px 700px at 90% 10%, rgba(255,130,180,0.10), transparent);
}}
[data-testid="stHeader"] {{ background: transparent; }}
section.main > div {{ padding-top: 0.5rem; }}
.block-container {{ padding-top: 1rem; }}

/* Header (text-only) */
.brand-title h2 {{ margin:0; }}
.brand-sub {{ opacity:.75; font-size: 0.95rem; }}

/* Tabs */
.stTabs [role="tab"] {{
  font-size: 0.95rem; font-weight: 700; letter-spacing: .2px;
  padding: 0.6rem 1rem; border-bottom: 2px solid transparent;
  border-radius: 10px 10px 0 0 !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.35), rgba(255,255,255,0.15));
  border: 1px solid rgba(120,120,180,0.20);
  margin-right: .25rem;
}}
.stTabs [role="tab"][aria-selected="true"] {{
  background: rgba(59,130,246,0.12);
  border-bottom-color: {PRIMARY};
}}

/* Cards */
.card {{
  background: var(--glass-bg); border-radius: var(--glass-br);
  border: 1px solid rgba(120, 120, 180, 0.18); box-shadow: var(--shadow);
  padding: 14px 16px; margin-bottom: 12px;
}}

/* Buttons */
.stButton>button {{
  border-radius: 12px; font-weight: 700; letter-spacing: .2px;
  box-shadow: 0 6px 16px rgba(33, 150, 243, 0.15);
  border: 1px solid rgba(59,130,246,0.40);
}}

/* DataFrame header */
[data-testid="stStyledTable"] thead tr th {{
  background: rgba(59,130,246,0.10) !important;
}}
section[data-testid="stSidebar"] button {{ margin-top: 0.25rem; }}
</style>
""", unsafe_allow_html=True)

# ----------------------
# Header / Branding 
# ----------------------
st.markdown(f"""
<div class="brand-title">
  <h2>Fraud Detection Dashboard</h2>
  <div class="brand-sub">Calibrated RF/XGB · Threshold Optimization · Cost Control</div>
</div>
""", unsafe_allow_html=True)

# ----------------------
# Paths & Registry
# ----------------------
APP_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(APP_DIR, "artifacts")
THRESH_FILE = os.path.join(ARTIFACTS_DIR, "thresholds.json")

DEFAULT_DATA_PATHS = [
    os.path.join(APP_DIR, "creditcard.csv"),
    "/mnt/data/creditcard.csv",
]

MODEL_FILES = {
    "RandomForest (Calibrated)": "rf_calibrated.joblib",
    "XGBoost (Calibrated)": "xgb_calibrated.joblib",
}
FALLBACK_FILES = {
    "RandomForest (Calibrated)": "rf_pipe.joblib",
    "XGBoost (Calibrated)": "xgb_pipe.joblib",
}

BASE_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ----------------------
# IO Utils
# ----------------------
def read_csv_any(file) -> "pd.DataFrame":
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

def try_load_default_dataset() -> Tuple[Optional[pd.DataFrame], str]:
    for p in DEFAULT_DATA_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df, f"Loaded default dataset: {os.path.basename(p)}"
            except Exception:
                continue
    return None, "Default dataset not found. Please upload a CSV."

# ----------------------
# Feature Engineering
# ----------------------
def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Amount" in out.columns and "_log_amount" not in out.columns:
        out["_log_amount"] = np.log1p(out["Amount"].astype(float))
    if "Time" in out.columns and "Hour_from_start_mod24" not in out.columns:
        hours = (out["Time"].astype(float) / 3600.0) % 24.0
        out["Hour_from_start_mod24"] = hours.astype(int)
    if "Hour_from_start_mod24" in out.columns:
        h = out["Hour_from_start_mod24"].astype(int)
        out["is_business_hours_proxy"] = ((h >= 9) & (h <= 17)).astype(int)
        out["is_night_proxy"] = ((h <= 6) | (h >= 22)).astype(int)
    return out

def get_expected_features(model) -> list:
    for attr in ["feature_names_in_", "features_in_"]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, (list, np.ndarray)):
                return list(val)
    for a in ["base_estimator_", "estimator", "classifier"]:
        if hasattr(model, a):
            inner = getattr(model, a)
            if hasattr(inner, "feature_names_in_"):
                return list(inner.feature_names_in_)
    return [f"V{i}" for i in range(1,29)] + ["Amount", "_log_amount",
            "Hour_from_start_mod24", "is_business_hours_proxy", "is_night_proxy"]

# ----------------------
# Model & Thresholds
# ----------------------
@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    fname = MODEL_FILES.get(model_key, "")
    # main → fallback
    tried = []
    for candidate in [fname, FALLBACK_FILES.get(model_key, "")]:
        if not candidate:
            continue
        fpath = os.path.join(ARTIFACTS_DIR, candidate)
        tried.append(os.path.basename(fpath))
        if os.path.exists(fpath):
            try:
                model = joblib.load(fpath)
                return model, os.path.basename(fpath)
            except Exception as e:
                st.warning(f"Failed to load {os.path.basename(fpath)}: {type(e).__name__}")
                continue
    st.warning(f"Model artifacts not found. Tried: {tried}. Running without a pre-trained model.")
    return None, "N/A"

@st.cache_resource(show_spinner=False)
def load_thresholds() -> Dict[str, float]:
    if not os.path.exists(THRESH_FILE):
        return {
            "RF_Thr_P90": 0.65, "XGB_Thr_P90": 0.75,
            "RF_Thr_MinCost": 0.07, "XGB_Thr_MinCost": 0.17,
            "COST_FP": 5.0, "COST_FN": 200.0
        }
    with open(THRESH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_features(df: pd.DataFrame, model) -> Tuple[pd.DataFrame, list]:
    df2 = build_engineered_features(df)
    expected = get_expected_features(model)
    cols = [c for c in expected if c in df2.columns]
    missing = [c for c in expected if c not in df2.columns]
    if missing:
        st.warning(f"Missing expected columns (engineered): {missing}")
    return df2[cols].copy(), cols

def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    raise ValueError("Model does not support probability outputs.")

def expected_cost(y_true: np.ndarray, y_prob: np.ndarray, thr: float, cost_fp: float, cost_fn: float) -> float:
    y_pred = (y_prob >= thr).astype(int)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(fp) * float(cost_fp) + float(fn) * float(cost_fn)

def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp

# ----------------------
# Sidebar Controls (only what's useful)
# ----------------------
st.sidebar.header("Model")
model_key = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()), index=0)
model, model_loaded_from = load_model(model_key)

st.sidebar.header("Threshold & Costs")
thresholds = load_thresholds()

cost_fp = st.sidebar.number_input("Cost of False Positive (COST_FP)", min_value=0.0, max_value=10000.0,
                                  value=float(thresholds.get("COST_FP", 5.0)), step=1.0)
cost_fn = st.sidebar.number_input("Cost of False Negative (COST_FN)", min_value=0.0, max_value=100000.0,
                                  value=float(thresholds.get("COST_FN", 200.0)), step=5.0)

if "RandomForest" in model_key:
    thr_p90_default = float(thresholds.get("RF_Thr_P90", 0.65))
    thr_mincost_default = float(thresholds.get("RF_Thr_MinCost", 0.07))
else:
    thr_p90_default = float(thresholds.get("XGB_Thr_P90", 0.75))
    thr_mincost_default = float(thresholds.get("XGB_Thr_MinCost", 0.17))

thr = st.sidebar.slider("Decision threshold", 0.0, 1.0, float(thr_p90_default), 0.001)

st.sidebar.markdown("**Presets**")
preset = st.sidebar.radio("Select preset", ["Strict", "Balanced", "Lenient"], horizontal=True, label_visibility="collapsed")
if preset == "Strict":
    thr = 0.90
elif preset == "Balanced":
    thr = float(thr_p90_default)
else:
    thr = 0.20

st.sidebar.markdown("---")
st.sidebar.markdown("**Threshold Finder**")
target_precision = st.sidebar.slider("Target Precision", 0.50, 0.99, 0.90, 0.01)
target_recall = st.sidebar.slider("Target Recall", 0.50, 0.99, 0.80, 0.01)

# FP warning threshold
FP_WARN = st.sidebar.number_input("FP warning threshold", min_value=0.0, max_value=100000.0, value=2000.0, step=100.0)

# ----------------------
# Data Loading (Default + Upload)
# ----------------------
def get_active_dataframe(uploaded_file):
    if uploaded_file is not None:
        df = read_csv_any(uploaded_file)
        src = f"Using uploaded file: {getattr(uploaded_file, 'name', 'uploaded.csv')}"
        return df, src
    df_def, msg = try_load_default_dataset()
    if df_def is not None:
        return df_def, msg
    return pd.DataFrame(), "No data available. Upload a CSV in 'DATA OVERVIEW'."

def get_df_for_tab5():
    state = st.session_state.get("last_run")
    if state is not None:
        return state["df"], state["y_true"], state["y_prob"], "Using data from last prediction run."
    df_def, msg = try_load_default_dataset()
    if df_def is not None:
        y_true = df_def["Class"].values if "Class" in df_def.columns else None
        return df_def, y_true, None, msg
    return pd.DataFrame(), None, None, "No data available. Upload a CSV in 'DATA OVERVIEW' or run 'PREDICTION ENGINE'."

# ----------------------
# Title
# ----------------------
st.title("Fraud Detection")
st.caption("Default dataset loads automatically (creditcard.csv). You can still upload your own CSV to override.")

# ----------------------
# Tabs
# ----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA OVERVIEW",
    "PREDICTION ENGINE",
    "MODEL METRICS",
    "MODEL INSIGHTS",
    "DATA QUALITY & SEGMENTS",
])

# ----------------------
# DATA OVERVIEW
# ----------------------
with tab1:
    st.subheader("Dataset")
    up1 = st.file_uploader("Upload CSV (.csv or .csv.gz)", type=["csv","gz"], key="csv_overview")
    df, src_msg = get_active_dataframe(up1)
    st.info(src_msg)
    if df.empty:
        st.warning("No data loaded yet.")
    else:
        st.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(50))

        if "Class" in df.columns:
            pos = int(df["Class"].sum())
            total = len(df)
            st.write(f"Class distribution: Fraud = {pos:,} / {total:,} ({pos/total:.4%})")

        exp_cols = [f"V{i}" for i in range(1,29)] + ["Amount", "Time (optional)"]
        schema_df = pd.DataFrame({"expected_columns": exp_cols})
        st.download_button("Download expected schema (columns).csv",
                           schema_df.to_csv(index=False).encode("utf-8"),
                           file_name="expected_schema.csv",
                           mime="text/csv")

# ----------------------
# PREDICTION ENGINE
# ----------------------
with tab2:
    st.subheader("Run Batch Prediction")
    up2 = st.file_uploader("Upload CSV (.csv or .csv.gz)", type=["csv","gz"], key="csv_predict")
    df_pred, src_msg2 = get_active_dataframe(up2)
    st.info(src_msg2)

    if df_pred.empty:
        st.warning("No data to predict on yet.")
    else:
        X, used_cols = ensure_features(df_pred, model)
        start = time.time()
        probs = predict_proba(model, X)
        elapsed = time.time() - start

        preds = (probs >= thr).astype(int)
        out = df_pred.copy()
        out["fraud_proba"] = probs
        out["fraud_pred"] = preds
        y_true = out["Class"].values if "Class" in out.columns else None

        if y_true is not None:
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            cost_now = expected_cost(y_true, probs, thr, float(cost_fp), float(cost_fn))
            st.success(f"Inference: {elapsed:.3f}s — Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | Expected Cost: {cost_now:,.0f}")
        else:
            st.info(f"Inference: {elapsed:.3f}s — Labels not found; metrics skipped.")

        st.session_state["last_run"] = {
            "df": df_pred,
            "X": X,
            "features": used_cols,
            "y_true": y_true,
            "y_prob": probs,
            "thr": float(thr),
            "cost_fp": float(cost_fp),
            "cost_fn": float(cost_fn),
            "model_key": model_key,
            "elapsed": elapsed,
        }

        preview_all = st.checkbox("Return all rows (preview full table)", value=False)
        preview = out if preview_all else out.head(50)
        st.dataframe(preview)
        st.download_button("Download predictions.csv", out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")

# ----------------------
# MODEL METRICS
# ----------------------
with tab3:
    st.subheader("Threshold & Cost Analysis")
    state = st.session_state.get("last_run", None)
    if not state or state["y_true"] is None:
        st.info("Upload a labeled dataset (with 'Class') and run prediction to view model metrics.")
    else:
        y_true = state["y_true"]
        y_prob = state["y_prob"]

        y_pred = (y_prob >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        cost_now = expected_cost(y_true, y_prob, thr, float(cost_fp), float(cost_fn))

        # Metrics block in glass card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Precision", f"{prec:.3f}")
        k2.metric("Recall", f"{rec:.3f}")
        k3.metric("F1", f"{f1:.3f}")
        k4.metric("Expected Cost", f"{cost_now:,.0f}")
        k5.metric("True Positives", f"{tp:,}")
        st.markdown('</div>', unsafe_allow_html=True)

        # KPI Sparklines (trend across thresholds)
        st.markdown("#### KPI Trends (across thresholds)")
        s1, s2, s3 = st.columns(3)
        grid_spark = np.linspace(0.0, 1.0, 51)

        def kpi_sparkline(values, title, fmt="{:.3f}"):
            fig = go.Figure(go.Scatter(y=values, mode="lines", fill="tozeroy"))
            fig.update_layout(height=80, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            fig = style_fig(fig, None)
            st.metric(title, fmt.format(values[-1]))
            st.plotly_chart(fig, use_container_width=True)

        prec_list, rec_list, cost_list = [], [], []
        for t in grid_spark:
            y_hat = (y_prob >= t).astype(int)
            prec_list.append(precision_score(y_true, y_hat, zero_division=0))
            rec_list.append(recall_score(y_true, y_hat, zero_division=0))
            cost_list.append(expected_cost(y_true, y_prob, t, float(cost_fp), float(cost_fn)))

        with s1: kpi_sparkline(prec_list, "Precision (sweep)")
        with s2: kpi_sparkline(rec_list, "Recall (sweep)")
        with s3: kpi_sparkline(cost_list, "Expected Cost (sweep)", fmt="{:,.0f}")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        # Confusion Matrix
        warn_color = "red" if fp >= FP_WARN else PRIMARY
        cm = np.array([[tn, fp],[fn, tp]])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Pred: Non-Fraud","Pred: Fraud"],
            y=["Actual: Non-Fraud","Actual: Fraud"],
            text=cm, texttemplate="%{text}",
            colorscale="RdBu", reversescale=True, showscale=False
        ))
        fig_cm.update_xaxes(side="top")
        fig_cm = style_fig(fig_cm, "Confusion Matrix")
        fig_cm.add_annotation(
            text=f"FP={fp:,} (warn≥{int(FP_WARN)})",
            x=0.95, y=-0.18, xref="paper", yref="paper",
            showarrow=False, font=dict(color=warn_color, size=12)
        )
        c1.plotly_chart(fig_cm, use_container_width=True)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))
        fig_roc.update_xaxes(title="False Positive Rate")
        fig_roc.update_yaxes(title="True Positive Rate")
        fig_roc = style_fig(fig_roc, "ROC Curve")
        c2.plotly_chart(fig_roc, use_container_width=True)

        # PR + current operating point
        precision_arr, recall_arr, thr_arr = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_arr, precision_arr)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall_arr, y=precision_arr, mode="lines", name=f"PR (AUC={pr_auc:.3f})"))
        if thr_arr.size > 0:
            idx_op = np.argmin(np.abs(thr_arr - thr))
            op_p = precision_arr[idx_op]
            op_r = recall_arr[idx_op]
        else:
            op_p, op_r = prec, rec
        fig_pr.add_trace(go.Scatter(x=[op_r], y=[op_p], mode="markers",
                                    marker=dict(size=10, line=dict(width=1), color=PRIMARY),
                                    name=f"Operating @ thr={thr:.3f}"))
        fig_pr.update_xaxes(title="Recall"); fig_pr.update_yaxes(title="Precision")
        fig_pr = style_fig(fig_pr, "Precision–Recall Curve")
        c3.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("---")
        grid = np.linspace(0.0, 1.0, 201)
        costs = [expected_cost(y_true, y_prob, t, float(cost_fp), float(cost_fn)) for t in grid]
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(x=grid, y=costs, mode="lines", name="Expected Cost"))
        cur_cost = expected_cost(y_true, y_prob, thr, float(cost_fp), float(cost_fn))
        fig_cost.add_vline(x=thr, line_dash="dash", line_color=warn_color,
                           annotation_text=f"thr={thr:.3f}\\nFP={fp:,}\\ncost={cur_cost:,.0f}",
                           annotation_position="top right")
        fig_cost.add_vline(x=thr_p90_default, line_color="green", annotation_text=f"P@90%≈{thr_p90_default:.3f}", line_dash="dot")
        fig_cost.add_vline(x=thr_mincost_default, line_color="orange", annotation_text=f"MinCost≈{thr_mincost_default:.3f}", line_dash="dot")
        fig_cost.update_layout(xaxis_title="Threshold", yaxis_title="Expected Cost")
        fig_cost = style_fig(fig_cost, "Expected Cost vs Threshold")
        st.plotly_chart(fig_cost, use_container_width=True)

        # Threshold helpers
        cta1, cta2, cta3 = st.columns(3)
        if cta1.button("Find Min-Cost Threshold"):
            idx = int(np.argmin(costs))
            best_thr = float(grid[idx])
            st.info(f"Min-Cost threshold ≈ {best_thr:.3f}  (Expected Cost={costs[idx]:,.0f})")
        if cta2.button("Find Threshold @ Target Precision"):
            feasible = []
            for t in grid:
                y_hat = (y_prob >= t).astype(int)
                feasible.append((t, precision_score(y_true, y_hat, zero_division=0)))
            good = [t for t, p in feasible if p >= target_precision]
            st.info(f"Threshold @ Precision ≥ {target_precision:.2f} → {min(good):.3f}" if good else "No threshold achieves the requested precision.")
        if cta3.button("Find Threshold @ Target Recall"):
            feasible = []
            for t in grid:
                y_hat = (y_prob >= t).astype(int)
                feasible.append((t, recall_score(y_true, y_hat, zero_division=0)))
            good = [t for t, r in feasible if r >= target_recall]
            st.info(f"Threshold @ Recall ≥ {target_recall:.2f} → {max(good):.3f}" if good else "No threshold achieves the requested recall.")

        # Optional: Animate threshold sweep on PR (moving operating point)
        if st.button("Animate Threshold Sweep (PR)"):
            frames = []
            thr_sweep = np.linspace(0, 1, 30)
            for t in thr_sweep:
                if thr_arr.size > 0:
                    idx_t = np.argmin(np.abs(thr_arr - t))
                    r_t = float(recall_arr[idx_t])
                    p_t = float(precision_arr[idx_t])
                else:
                    yhat_t = (y_prob >= t).astype(int)
                    p_t = float(precision_score(y_true, yhat_t, zero_division=0))
                    r_t = float(recall_score(y_true, yhat_t, zero_division=0))
                frames.append(go.Frame(data=[
                    go.Scatter(x=recall_arr, y=precision_arr, mode="lines", name="PR"),
                    go.Scatter(x=[r_t], y=[p_t], mode="markers", marker=dict(size=10), name=f"thr={t:.2f}")
                ], name=f"{t:.2f}"))
            fig_anim = go.Figure(
                data=[go.Scatter(x=recall_arr, y=precision_arr, mode="lines", name="PR"),
                      go.Scatter(x=[op_r], y=[op_p], mode="markers", name=f"thr={thr:.2f}")],
                frames=frames
            )
            fig_anim.update_layout(
                xaxis_title="Recall", yaxis_title="Precision",
                updatemenus=[dict(type="buttons",
                                  buttons=[dict(label="Play", method="animate", args=[None])])]
            )
            fig_anim = style_fig(fig_anim, "PR Curve · Threshold Sweep")
            st.plotly_chart(fig_anim, use_container_width=True)

# ----------------------
# MODEL INSIGHTS
# ----------------------
with tab4:
    st.subheader("Permutation Importance (Top Features)")
    state = st.session_state.get("last_run", None)
    if not state or state["y_true"] is None:
        st.info("Upload labeled data and run prediction to compute feature importances.")
    else:
        X = state["X"]
        y_true = state["y_true"]
        # Keep the UI snappy by subsampling for permutation importance
        max_n = 6000
        if len(X) > max_n:
            idx = np.random.RandomState(42).choice(len(X), size=max_n, replace=False)
            X_sub = X.iloc[idx]
            y_sub = y_true[idx]
        else:
            X_sub, y_sub = X, y_true

        with st.spinner("Computing permutation importances..."):
            res = permutation_importance(model, X_sub, y_sub, n_repeats=5, random_state=42, scoring="roc_auc")
            imp = pd.DataFrame({
                "feature": X_sub.columns,
                "importance_mean": res.importances_mean,
                "importance_std": res.importances_std,
            }).sort_values("importance_mean", ascending=False).head(20)

        fig_imp = px.bar(
            imp, x="importance_mean", y="feature",
            error_x="importance_std", orientation="h", title=None
        )
        fig_imp.update_traces(marker_line_width=0.5, opacity=0.95)
        fig_imp = style_fig(fig_imp, "Top 20 Feature Importances (ROC AUC drop)")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption("Interpret with care (correlated features may share importance).")

# ----------------------
# DATA QUALITY & SEGMENTS
# ----------------------
with tab5:
    st.subheader("Profiling & Segmented Performance")
    df, y_true, y_prob, src_msg = get_df_for_tab5()
    st.info(src_msg)

    if df.empty:
        st.warning("No data available for profiling yet.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Rows", f"{len(df):,}")
        with c2: st.metric("Columns", f"{df.shape[1]:,}")
        with c3: st.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

        st.markdown("### Key Distributions")
        col1, col2 = st.columns(2)
        # Sample large datasets to keep histograms responsive
        plot_df = df
        if len(df) > 120_000:
            plot_df = df.sample(120_000, random_state=42)

        if "Amount" in df.columns:
            h1 = px.histogram(plot_df, x="Amount", nbins=60, marginal="box", title=None)
            h1.update_traces(opacity=0.9)
            h1 = style_fig(h1, "Amount Distribution")
            col1.plotly_chart(h1, use_container_width=True)

        if "Time" in df.columns:
            df_tmp = plot_df.copy()
            if "Hour_from_start_mod24" not in df_tmp.columns:
                df_tmp = build_engineered_features(df_tmp)
            h2 = px.histogram(df_tmp, x="Hour_from_start_mod24", nbins=24, title=None)
            h2 = style_fig(h2, "Hour from start (mod 24)")
            col2.plotly_chart(h2, use_container_width=True)

        if y_true is not None and y_prob is not None and "Amount" in df.columns:
            st.markdown("### Segmented Metrics by Amount (quintiles)")
            seg = pd.DataFrame({"Amount": df["Amount"].values, "y_true": y_true, "y_prob": y_prob})
            # Guard against constant Amount (qcut fails)
            if seg["Amount"].nunique() > 1:
                seg["bin"] = pd.qcut(seg["Amount"], q=5, duplicates="drop")
                rows = []
                for b, sub in seg.groupby("bin", observed=True):
                    yb = sub["y_true"].values
                    pb = sub["y_prob"].values
                    yhat = (pb >= thr).astype(int)
                    rows.append({
                        "Amount bin": str(b),
                        "Precision": precision_score(yb, yhat, zero_division=0),
                        "Recall": recall_score(yb, yhat, zero_division=0),
                        "F1": f1_score(yb, yhat, zero_division=0),
                        "Count": len(sub)
                    })
                st.dataframe(pd.DataFrame(rows))
            else:
                st.info("Skipping segmented metrics: 'Amount' has a single unique value.")

st.markdown("---")
st.caption("Fraud Detection · Calibrated RF/XGB · © Tarek Masryo")
