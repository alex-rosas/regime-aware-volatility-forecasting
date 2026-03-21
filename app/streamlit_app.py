"""
app/streamlit_app.py
--------------------
Entry point for the volatility_regimes Streamlit app.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title = "Volatility Regimes",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

overview    = st.Page("pages/01_overview.py",    title="Overview",             icon="🏠")
vol_regimes = st.Page("pages/02_vol_regimes.py", title="Volatility & Regimes", icon="📊")
forecast    = st.Page("pages/03_forecast.py",    title="Forecast & Intervals", icon="🎯")
backtesting = st.Page("pages/04_backtesting.py", title="Backtesting",          icon="🧪")

pg = st.navigation([overview, vol_regimes, forecast, backtesting])
pg.run()
