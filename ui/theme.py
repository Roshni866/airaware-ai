"""Shared design tokens and CSS injection for every page. One place to keep the palette
and component styling consistent, instead of repeating CSS per page."""
from __future__ import annotations

import streamlit as st

COLORS = {
    "forest": "#102C2B",
    "teal": "#247C72",
    "mint": "#B9DED5",
    "mist": "#EAF3F1",
    "white": "#FFFFFF",
    "amber": "#F0B65A",
    "coral": "#D96C5F",
    "text": "#172D2B",
    "text_secondary": "#6C8580",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "font": {"family": "Inter, -apple-system, 'Segoe UI', sans-serif", "color": COLORS["text"]},
        "paper_bgcolor": COLORS["white"],
        "plot_bgcolor": COLORS["white"],
        "colorway": [COLORS["teal"], COLORS["amber"], COLORS["coral"], COLORS["forest"], COLORS["mint"]],
        "xaxis": {"gridcolor": "#E3EDEA", "zerolinecolor": "#E3EDEA"},
        "yaxis": {"gridcolor": "#E3EDEA", "zerolinecolor": "#E3EDEA"},
        "margin": {"t": 48, "b": 36, "l": 48, "r": 24},
    }
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
        }}
        .stApp {{
            background-color: {COLORS["mist"]};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {COLORS["forest"]};
        }}
        section[data-testid="stSidebar"] * {{
            color: {COLORS["mint"]} !important;
        }}
        section[data-testid="stSidebar"] .stRadio label span {{
            font-size: 0.97rem;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(185,222,213,0.25);
        }}

        h1, h2, h3, h4 {{
            color: {COLORS["text"]};
            font-weight: 700;
            letter-spacing: -0.01em;
        }}
        p, li, span, div {{
            color: {COLORS["text"]};
        }}
        .aa-secondary {{
            color: {COLORS["text_secondary"]} !important;
            font-size: 0.92rem;
        }}

        .aa-card {{
            background: {COLORS["white"]};
            border-radius: 16px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 2px 10px rgba(16,44,43,0.07);
            border: 1px solid rgba(16,44,43,0.05);
            height: 100%;
        }}
        .aa-card .aa-label {{
            color: {COLORS["text_secondary"]};
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.3rem;
        }}
        .aa-card .aa-value {{
            color: {COLORS["forest"]};
            font-size: 1.9rem;
            font-weight: 800;
            line-height: 1.15;
        }}
        .aa-card .aa-unit {{
            font-size: 0.95rem;
            font-weight: 500;
            color: {COLORS["text_secondary"]};
            margin-left: 0.25rem;
        }}
        .aa-card .aa-sub {{
            color: {COLORS["text_secondary"]};
            font-size: 0.85rem;
            margin-top: 0.3rem;
        }}
        .aa-badge {{
            display: inline-block;
            padding: 0.15rem 0.65rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .aa-badge-fresh {{ background: #E3F3EC; color: #1F7A4D; }}
        .aa-badge-delayed {{ background: #FEF3DF; color: #A3700E; }}
        .aa-badge-stale {{ background: #FBE7E4; color: #B3392D; }}
        .aa-badge-unavailable {{ background: #ECECEC; color: #6C6C6C; }}

        .aa-alert {{
            border-radius: 14px;
            padding: 1rem 1.25rem;
            margin: 0.6rem 0 1rem 0;
            border-left: 5px solid;
        }}
        .aa-alert-elevated {{ background: #FBEAE7; border-color: {COLORS["coral"]}; }}
        .aa-alert-potential {{ background: #FEF5E6; border-color: {COLORS["amber"]}; }}
        .aa-alert-below {{ background: #EAF5F1; border-color: {COLORS["teal"]}; }}
        .aa-alert-title {{ font-weight: 700; font-size: 1rem; margin-bottom: 0.2rem; }}

        .aa-hero {{
            background: linear-gradient(120deg, {COLORS["forest"]} 0%, {COLORS["teal"]} 100%);
            border-radius: 20px;
            padding: 2.2rem 2.4rem;
            color: {COLORS["white"]};
            margin-bottom: 1.4rem;
        }}
        .aa-hero h1, .aa-hero h1 * {{
            color: {COLORS["white"]} !important;
            font-size: 2.1rem;
            margin-bottom: 0.3rem;
        }}
        .aa-hero p, .aa-hero p * {{
            color: {COLORS["mint"]} !important;
            font-size: 1.02rem;
            max-width: 46rem;
        }}

        div[data-testid="stMetricValue"] {{
            color: {COLORS["forest"]};
        }}

        /* st.info / st.warning / st.error boxes: force a readable light background +
        dark text regardless of the viewer's OS/browser dark-mode setting, independent
        of .streamlit/config.toml (belt-and-suspenders — works even if that file isn't
        picked up, e.g. on some deployment setups). */
        div[data-testid="stAlert"] {{
            background-color: {COLORS["white"]} !important;
            border: 1px solid rgba(16,44,43,0.08);
        }}
        div[data-testid="stAlert"] p, div[data-testid="stAlert"] span,
        div[data-testid="stAlert"] div, div[data-testid="stAlert"] li {{
            color: {COLORS["text"]} !important;
        }}

        button[kind="primary"] {{
            background-color: {COLORS["teal"]} !important;
            border-color: {COLORS["teal"]} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, unit: str = "", sub: str = "", col=None) -> None:
    target = col if col is not None else st
    target.markdown(
        f"""
        <div class="aa-card">
            <div class="aa-label">{label}</div>
            <div><span class="aa-value">{value}</span><span class="aa-unit">{unit}</span></div>
            <div class="aa-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def freshness_badge(level: str) -> str:
    labels = {"fresh": "Fresh", "delayed": "Delayed", "stale": "Stale", "unavailable": "Unavailable", "error": "Error"}
    css = {"fresh": "fresh", "delayed": "delayed", "stale": "stale", "unavailable": "unavailable", "error": "unavailable"}
    cls = css.get(level, "unavailable")
    label = labels.get(level, level.title())
    return f'<span class="aa-badge aa-badge-{cls}">{label}</span>'


def alert_panel(level: str, title: str, message: str) -> None:
    css_cls = {"elevated": "elevated", "potential": "potential", "below_threshold": "below"}.get(level, "below")
    st.markdown(
        f"""
        <div class="aa-alert aa-alert-{css_cls}">
            <div class="aa-alert-title">{title}</div>
            <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
