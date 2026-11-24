"""AirAware AI — Clean Air Intelligence. Thin multipage shell: loads shared state,
injects the design system, renders sidebar nav, dispatches to ui/ page modules.
Business logic lives in src/.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DAILY_CSV, RESULTS, load_config
from ui import forecast, live_quality, overview, performance
from ui.theme import inject_css

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="AirAware AI", page_icon="🌬️", layout="wide", initial_sidebar_state="expanded")
inject_css()


@st.cache_data(ttl=600)
def load_daily() -> pd.DataFrame:
    return pd.read_csv(DAILY_CSV, parse_dates=["date"], index_col="date")


@st.cache_resource
def load_static_config() -> dict:
    cfg = load_config()
    selected = json.loads((RESULTS / "selected_model.json").read_text())
    return {"cfg": cfg, "selected": selected}


state = load_static_config()
cfg, selected = state["cfg"], state["selected"]
daily = load_daily()

PAGES = {
    "Overview": "overview",
    "Live Air Quality": "live_quality",
    "PM2.5 Forecast": "forecast",
    "Model Performance": "performance",
}
ICONS = {
    "Overview": "🏠", "Live Air Quality": "📡",
    "PM2.5 Forecast": "📈", "Model Performance": "🧪",
}

with st.sidebar:
    logo_svg = (ASSETS / "logo.svg").read_text()
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.2rem;">'
        f'{logo_svg}<span style="font-size:1.25rem;font-weight:800;color:#EAF3F1;">AirAware AI</span></div>'
        f'<div style="color:#B9DED5;font-size:0.82rem;margin-bottom:1.1rem;">Clean Air Intelligence</div>',
        unsafe_allow_html=True,
    )
    choice = st.radio(
        "Navigate", list(PAGES.keys()), label_visibility="collapsed",
        format_func=lambda p: f"{ICONS[p]}  {p}",
    )
    st.markdown("---")
    st.markdown(
        '<span class="aa-secondary">Research prototype for educational use. Not an official '
        "government warning service or medical advice.</span>",
        unsafe_allow_html=True,
    )

page = PAGES[choice]
if page == "overview":
    overview.render(cfg, daily)
elif page == "live_quality":
    live_quality.render(cfg)
elif page == "forecast":
    forecast.render(cfg, daily)
elif page == "performance":
    performance.render(selected)
