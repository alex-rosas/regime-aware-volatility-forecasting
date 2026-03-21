"""
app/streamlit_app.py
--------------------
Overview page + entry point for the volatility_regimes Streamlit app.

Run from the project root:
    streamlit run app/streamlit_app.py

Additional pages are auto-discovered from app/pages/:
    1_Volatility_Regimes.py
    2_Forecast_Intervals.py
    3_Backtesting.py
"""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Regime-Aware Volatility Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT    = Path(__file__).resolve().parent.parent
METRICS = ROOT / "metrics.json"
FIG_DIR = ROOT / "assets" / "figures"


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------

def _load_metrics() -> dict:
    if not METRICS.exists():
        return {}
    with open(METRICS) as f:
        return json.load(f)


def _fmt(v, pct: bool = False, decimals: int = 4) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v:.2f}%"
    return f"{v:.{decimals}f}"


# -----------------------------------------------------------------------
# page
# -----------------------------------------------------------------------

st.title("📈 Regime-Aware Volatility Forecasting — MXN/USD")
st.caption(
    "GARCH · EGARCH · 3-state HMM · XGBoost Hybrid · "
    "Asymmetric Conformal Prediction Intervals"
)
st.divider()

m = _load_metrics()

if not m:
    st.warning(
        "metrics.json not found.  "
        "Run `dvc repro` to generate pipeline artifacts."
    )
else:
    # ---- row 1: point forecast ----------------------------------------
    st.subheader("Point Forecast — Test Set")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "RMSE — Hybrid XGBoost",
        _fmt(m.get("rmse_hybrid")),
        delta=f'{m.get("rmse_improvement_pct", 0):+.2f}% vs GARCH',
        delta_color="normal",
    )
    c2.metric("RMSE — GARCH",  _fmt(m.get("rmse_garch")))
    c3.metric("RMSE — EGARCH", _fmt(m.get("rmse_egarch")))
    c4.metric(
        "DM p-value vs GARCH",
        _fmt(m.get("dm_pval_vs_garch"), decimals=4),
        delta="statistically significant" if (m.get("dm_pval_vs_garch") or 1) < 0.05 else "not significant",
        delta_color="normal" if (m.get("dm_pval_vs_garch") or 1) < 0.05 else "off",
    )

    st.divider()

    # ---- row 2: QLIKE + conformal coverage ----------------------------
    col_q, col_cov = st.columns(2)

    with col_q:
        st.subheader("QLIKE Loss")
        qa, qb, qc = st.columns(3)
        qa.metric("Hybrid",  _fmt(m.get("qlike_hybrid"),  decimals=4))
        qb.metric("GARCH",   _fmt(m.get("qlike_garch"),   decimals=4))
        qc.metric("EGARCH",  _fmt(m.get("qlike_egarch"),  decimals=4))

    with col_cov:
        st.subheader("Conformal Coverage (asymmetric, φ=0.7)")
        ca, cb, cc = st.columns(3)
        ca.metric("80% interval", _fmt(m.get("coverage_80"), pct=True, decimals=2) if m.get("coverage_80") else "—")
        cb.metric("90% interval", _fmt(m.get("coverage_90"), pct=True, decimals=2) if m.get("coverage_90") else "—")
        cc.metric("95% interval", _fmt(m.get("coverage_95"), pct=True, decimals=2) if m.get("coverage_95") else "—")

    st.divider()

    # ---- row 3: walk-forward ------------------------------------------
    st.subheader("Walk-Forward Validation (5-fold expanding window)")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric(
        "WF RMSE — Hybrid",
        _fmt(m.get("wf_rmse_hybrid")),
        delta=f'{m.get("wf_rmse_improvement_pct", 0):+.2f}% vs GARCH',
        delta_color="normal",
    )
    w2.metric("WF RMSE — GARCH",  _fmt(m.get("wf_rmse_garch")))
    w3.metric("WF RMSE — EGARCH", _fmt(m.get("wf_rmse_egarch")))
    w4.metric("WF QLIKE — Hybrid", _fmt(m.get("wf_qlike_hybrid"), decimals=4))

st.divider()

# ---- pipeline diagram -------------------------------------------------
st.subheader("DVC Pipeline")
diag = FIG_DIR / "readme" / "pipeline_diagram.png"
if diag.exists():
    st.image(str(diag), use_container_width=True)
else:
    st.info("Pipeline diagram not built yet.  Run `dvc repro build_figures`.")
