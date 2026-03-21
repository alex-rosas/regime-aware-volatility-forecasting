"""
pages/02_vol_regimes.py
-----------------------
Full-period conditional volatility with HMM regime bands.
"""

from pathlib import Path
import streamlit as st

ROOT    = Path(__file__).resolve().parent.parent.parent
FIG_DIR = ROOT / "assets" / "figures" / "dark"

st.title("📊 Conditional Volatility & HMM Regimes")
st.caption("GARCH(1,1)-t and EGARCH(1,1)-t conditional volatility coloured by 3-state HMM regime.")
st.divider()

fig = FIG_DIR / "01_vol_regimes.png"
if fig.exists():
    st.image(str(fig), use_column_width=True)
else:
    st.warning("Figure not found. Run `dvc repro build_figures`.")

st.divider()

st.subheader("Regime interpretation")
col_low, col_med, col_high = st.columns(3)

with col_low:
    st.markdown(
        """
        <div style="border-left: 4px solid #66BB6A; padding-left: 12px;">
        <b style="color:#66BB6A">Low volatility</b><br/>
        Calm, trending market. GARCH/EGARCH produce tight volatility
        estimates. Conformal intervals are wide relative to realised
        moves — safe but capital-inefficient.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_med:
    st.markdown(
        """
        <div style="border-left: 4px solid #FFA726; padding-left: 12px;">
        <b style="color:#FFA726">Medium volatility</b><br/>
        Normal market conditions. Hybrid XGBoost achieves its best
        calibration here. Kupiec POF test passes at all levels.
        Coverage close to the asymmetric target.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_high:
    st.markdown(
        """
        <div style="border-left: 4px solid #EF5350; padding-left: 12px;">
        <b style="color:#EF5350">High volatility (crisis)</b><br/>
        Stress periods. Fixed-width conformal intervals undercover the
        upper tail — empirical breach rate exceeds the 7% target.
        The strongest argument for locally adaptive intervals.
        </div>
        """,
        unsafe_allow_html=True,
    )
