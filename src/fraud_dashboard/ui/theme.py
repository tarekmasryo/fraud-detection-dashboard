from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PRIMARY = "#3B82F6"

AURORA_COLORWAY = [
    "#3366CC",
    "#DC3912",
    "#FF9900",
    "#109618",
    "#990099",
    "#3B3EAC",
    "#0099C6",
    "#DD4477",
    "#66AA00",
    "#B82E2E",
]
AURORA_CS = "Tealrose"


def init_plotly_theme() -> None:
    base_layout = dict(
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=13),
        margin=dict(l=50, r=30, t=60, b=50),
        hoverlabel=dict(namelength=-1, font_size=12),
        legend=dict(title=None, orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,250,252,1)",
        colorway=AURORA_COLORWAY,
    )
    pio.templates["tarek_theme"] = go.layout.Template(layout=base_layout)
    px.defaults.template = "tarek_theme"
    px.defaults.color_continuous_scale = AURORA_CS


def style_fig(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(template="tarek_theme")
    if title:
        fig.update_layout(title=dict(text=title, x=0.02, xanchor="left", font_size=18))
    fig.update_layout(modebar_add=["togglespikelines"], hovermode="x unified")
    return fig


def inject_css() -> None:
    st.markdown(
        f"""
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

/* Header */
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
""",
        unsafe_allow_html=True,
    )
