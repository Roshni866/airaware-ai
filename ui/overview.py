"""Overview — public-facing homepage."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import live_data
from src.aqi import calculate_aqi
from src.predict import classify_alert, generate_forecast
from ui.theme import COLORS, PLOTLY_TEMPLATE, alert_panel, freshness_badge, metric_card

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def render(cfg: dict, daily: pd.DataFrame) -> None:
    hero_svg = (ASSETS / "hero.svg").read_text()
    st.markdown(
        f"""
        <div class="aa-hero">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:1.5rem; flex-wrap:wrap;">
                <div style="flex: 1 1 320px;">
                    <h1>Breathe better. Stay informed.</h1>
                    <p>Monitor air quality, understand pollution patterns and explore next-day PM2.5
                    predictions powered by machine learning — for {cfg['station']['city']}, {cfg['station']['country']}.</p>
                </div>
                <div style="flex: 0 0 auto; max-width: 320px; opacity: 0.95;">{hero_svg}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    live = live_data.fetch_live_readings(cfg["station"]["openaq_location_id"])
    forecast = generate_forecast()

    last_obs_date = daily["pm25_mean"].dropna().index.max()
    last_obs_value = daily.loc[last_obs_date, "pm25_mean"]
    st.markdown(
        f"📍 {cfg['station']['name']} &nbsp;·&nbsp; Archived data last refreshed through "
        f"<b>{last_obs_date.date()}</b> &nbsp;·&nbsp; Live feed: "
        f"{freshness_badge(live.status if live.status == 'ok' else 'unavailable')}",
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3, c4 = st.columns(4)

    # 1. Current overall AQI — ONLY from live data; never invented.
    with c1:
        if live.ok and live.readings:
            conc = {p: r.value for p, r in live.readings.items()}
            result = calculate_aqi(conc)
            if result.overall_aqi is not None:
                label = "Current Overall AQI" if result.is_overall else f"Current AQI ({result.dominant_pollutant}-only)"
                metric_card(label, f"{result.overall_aqi:.0f}", "", f"Category: {result.category}")
            else:
                metric_card("Current Overall AQI", "Data unavailable", "", "No valid live pollutant readings.")
        else:
            metric_card("Current Overall AQI", "Data unavailable", "",
                        live.reason or "Live feed not configured.")

    # 2. Current PM2.5 — live if available, else the latest archived observation (clearly labeled).
    with c2:
        if live.ok and "PM2.5" in live.readings:
            r = live.readings["PM2.5"]
            metric_card("Current PM2.5 (live)", f"{r.value:.0f}", "µg/m³",
                        f"OpenAQ · {r.datetime_utc.strftime('%Y-%m-%d %H:%M UTC')}")
        else:
            metric_card("Latest PM2.5 (archived)", f"{last_obs_value:.0f}", "µg/m³",
                        f"Not live — last available daily mean, {last_obs_date.date()}")

    # 3. Next-day prediction — always from the ML model, clearly distinct from measurements.
    with c3:
        if forecast.point_forecast is not None:
            sub = f"For {forecast.target_date}" + (" · ⚠ stale input data" if forecast.status == "stale_data" else "")
            metric_card("Next-day PM2.5 (ML forecast)", f"{forecast.point_forecast:.0f}", "µg/m³", sub)
        else:
            metric_card("Next-day PM2.5 (ML forecast)", "Unavailable", "", forecast.reason)

    # 4. 90% prediction interval
    with c4:
        if forecast.interval_low is not None:
            metric_card("90% Prediction Interval", f"{forecast.interval_low:.0f}–{forecast.interval_high:.0f}", "µg/m³",
                        "Conformal interval (log-space)")
        else:
            metric_card("90% Prediction Interval", "Unavailable", "", "")

    if forecast.point_forecast is not None:
        alert = classify_alert(forecast, cfg["events"]["primary_threshold"])
        titles = {
            "elevated": "⚠ Elevated-pollution forecast",
            "potential": "⚠ Potential elevated pollution",
            "below_threshold": "✓ Below threshold",
        }
        if alert["level"] in titles:
            alert_panel(alert["level"], titles[alert["level"]], alert["message"])

    st.markdown("### Pollution trend")
    window = st.radio(
        "Show last", ["90 days", "180 days", "1 year", "3 years", "All history"],
        index=0, horizontal=True, label_visibility="collapsed", key="overview_window",
    )
    _trend_chart(daily, forecast, cfg, window)


_WINDOW_DAYS = {"90 days": 90, "180 days": 180, "1 year": 365, "3 years": 365 * 3, "All history": None}


def _trend_chart(daily: pd.DataFrame, forecast, cfg: dict, window: str = "90 days") -> None:
    full = daily["pm25_mean"].dropna()
    n = _WINDOW_DAYS[window]
    hist = full if n is None else full.tail(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, mode="lines", name="Observed PM2.5",
        line=dict(color=COLORS["forest"], width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f} µg/m³<extra>Observed</extra>",
    ))
    thr = cfg["events"]["primary_threshold"]
    fig.add_hline(y=thr, line_dash="dash", line_color=COLORS["coral"],
                  annotation_text=f"Event threshold ({thr} µg/m³)", annotation_position="top left")

    if forecast.point_forecast is not None:
        target = pd.Timestamp(forecast.target_date)
        fig.add_trace(go.Scatter(
            x=[hist.index[-1], target], y=[hist.values[-1], forecast.point_forecast],
            mode="lines+markers", name="Next-day forecast",
            line=dict(color=COLORS["amber"], width=2, dash="dot"),
            marker=dict(size=[0, 9]),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f} µg/m³<extra>Forecast</extra>",
        ))
        if forecast.interval_low is not None:
            fig.add_trace(go.Scatter(
                x=[target, target], y=[forecast.interval_low, forecast.interval_high],
                mode="lines", name="90% interval", line=dict(color=COLORS["amber"], width=8),
                opacity=0.3, hovertemplate="Interval: %{y:.0f} µg/m³<extra></extra>",
            ))

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"], height=380, hovermode="x unified",
        yaxis_title="PM2.5 (µg/m³)", legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)