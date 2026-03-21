"""
pages/3_Backtesting.py
-----------------------
Backtesting: walk-forward validation, Kupiec POF test,
and regime-conditioned coverage gap.
"""

from pathlib import Path
import streamlit as st

ROOT    = Path(__file__).resolve().parent.parent.parent
FIG_DIR = ROOT / "assets" / "figures" / "dark"


def _show(filename: str, caption: str = "") -> None:
    fig = FIG_DIR / filename
    if fig.exists():
        st.image(str(fig), use_column_width=True)
        if caption:
            st.caption(caption)
    else:
        st.warning(f"`{filename}` not found. Run `dvc repro build_figures`.")


st.title("🧪 Backtesting")
st.divider()

# -----------------------------------------------------------------------
# walk-forward
# -----------------------------------------------------------------------
st.subheader("Walk-Forward Validation — Expanding Window")
st.caption(
    "XGBoost refitted on each expanding fold. HMM and GARCH features "
    "are pre-computed and held fixed across folds. Dotted vertical "
    "lines mark fold boundaries."
)
_show("04_walkforward.png")

st.divider()

# -----------------------------------------------------------------------
# Kupiec + regime coverage side by side
# -----------------------------------------------------------------------
col_kup, col_reg = st.columns(2)

with col_kup:
    st.subheader("Kupiec POF Test")
    st.caption(
        "Empirical upper violation rate vs asymmetric target "
        "$\\alpha_\\text{upper} = \\phi \\cdot \\alpha$ (horizontal bar). "
        "Green = below target (conservative). "
        "Red = above target (undercoverage)."
    )
    _show("05_kupiec.png")

with col_reg:
    st.subheader("Regime-Conditioned Coverage")
    st.caption(
        "Gap between empirical upper violation rate and the 7% target "
        "(90% interval, φ=0.7) broken out by HMM regime. "
        "Positive = undercoverage. Negative = conservative."
    )
    _show("06_regime_coverage.png")

st.divider()

# -----------------------------------------------------------------------
# key findings
# -----------------------------------------------------------------------
with st.expander("Key findings"):
    st.markdown(
        """
        **Walk-forward validation** confirms the hybrid model's
        out-of-sample edge: walk-forward RMSE improvement over GARCH
        is consistent with the single test-split result, showing the
        gain is not specific to one particular test window.

        **Kupiec POF test** — using the correct asymmetric null
        $\\alpha_\\text{upper} = 0.7\\alpha$ — does not reject H₀ at
        any of the three coverage levels, confirming the upper bounds
        are well-calibrated in aggregate.

        **Regime-conditioned coverage** reveals the residual weakness:
        the high-volatility regime shows a positive gap — the fixed-width
        interval breaches the 7% upper target during crisis periods.
        This is the empirical motivation for locally adaptive conformal
        prediction as a future extension.
        """
    )
