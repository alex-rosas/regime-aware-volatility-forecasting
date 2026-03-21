"""
pages/2_Forecast_Intervals.py
------------------------------
Test-set forecast comparison + asymmetric conformal prediction intervals.

Alpha selector loads a pre-built PNG — zero re-render latency.
"""

from pathlib import Path
import streamlit as st

ROOT    = Path(__file__).resolve().parent.parent.parent
FIG_DIR = ROOT / "assets" / "figures" / "dark"

st.title("🎯 Forecast & Prediction Intervals")
st.divider()

# -----------------------------------------------------------------------
# forecast comparison (always shown)
# -----------------------------------------------------------------------
st.subheader("Point Forecast — Test Set")
st.caption(
    "Hybrid XGBoost vs GARCH and EGARCH benchmarks. "
    "Background bands show HMM volatility regime."
)

fig_forecast = FIG_DIR / "02_forecast.png"
if fig_forecast.exists():
    st.image(str(fig_forecast), use_container_width=True)
else:
    st.warning("Figure not found. Run `dvc repro build_figures`.")

st.divider()

# -----------------------------------------------------------------------
# conformal intervals — pre-built per alpha level
# -----------------------------------------------------------------------
st.subheader("Asymmetric Conformal Prediction Intervals  (φ = 0.7)")
st.caption(
    "The asymmetric predictor allocates 70% of the error budget to the "
    "upper tail (φ = 0.7), producing wider upper bounds during stress "
    "periods. Select the coverage level below."
)

LEVELS = {
    "80%  —  α_upper = 14%": 80,
    "90%  —  α_upper = 7%":  90,
    "95%  —  α_upper = 3.5%": 95,
}

choice = st.radio(
    "Coverage level",
    options=list(LEVELS.keys()),
    index=1,          # default: 90%
    horizontal=True,
)
coverage = LEVELS[choice]

fig_interval = FIG_DIR / f"03_intervals_{coverage}.png"
if fig_interval.exists():
    st.image(str(fig_interval), use_container_width=True)
else:
    st.warning(
        f"Figure `03_intervals_{coverage}.png` not found. "
        "Run `dvc repro build_figures`."
    )

st.divider()

# -----------------------------------------------------------------------
# explainer
# -----------------------------------------------------------------------
with st.expander("How asymmetric conformal intervals work"):
    st.markdown(
        """
        Standard (symmetric) split conformal prediction uses the
        absolute residuals from a calibration set to construct an
        interval $[\\hat{y} - q, \\hat{y} + q]$ where $q$ is the
        $(1-\\alpha)$ quantile of the non-conformity scores.

        **Asymmetric conformal** uses the *signed* residuals instead,
        splitting the error budget $\\alpha$ between the two tails:

        | Tail  | Budget | Formula |
        |-------|--------|---------|
        | Lower | $(1-\\phi)\\alpha$ | $q_\\text{lower} = $ quantile at $(1-\\phi)\\alpha$ of signed residuals |
        | Upper | $\\phi\\alpha$     | $q_\\text{upper} = $ quantile at $1-\\phi\\alpha$ of signed residuals |

        With $\\phi = 0.7$ the upper bound is **wider** — it absorbs
        70% of the error budget — which is the risk-management oriented
        choice: we care more about underestimating upward volatility
        spikes than about the lower bound.

        The Kupiec POF test evaluates the upper bound against the
        correct null $\\alpha_\\text{upper} = \\phi\\alpha$, not against
        the full $\\alpha$.
        """
    )
