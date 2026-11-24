"""PM2.5 Forecast — next-day prediction, interval, and the uncertainty-aware alert."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import classify_alert, generate_forecast
from ui.theme import COLORS, PLOTLY_TEMPLATE, alert_panel, metric_card


def render(cfg: dict, daily: pd.DataFrame) -> None:
    st.markdown("## Next-Day PM2.5 Forecast")
    st.markdown(
        '<p class="aa-secondary">This page predicts the <b>24-hour mean PM2.5 concentration '
        "(µg/m³)</b> for tomorrow. It is not an overall CPCB AQI — see the README for how PM2.5 "
        "relates to the full AQI methodology.</p>",
        unsafe_allow_html=True,
    )

    forecast = generate_forecast()

    if forecast.status == "no_data":
        st.error(f"**No forecast available.** {forecast.reason}")
        return
    if forecast.status == "schema_mismatch":
        st.error(f"**Forecast pipeline error.** {forecast.reason}")
        return

    if forecast.status == "stale_data":
        st.warning(
            f"⚠ **Input data is stale** — the most recent valid observation is from "
            f"**{forecast.origin_date}** ({forecast.data_age_days} days old). The forecast below is "
            f"the last one the model could validly produce from that data; it is **not** a current "
            f"prediction for today. Run the data refresh to update it."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Next-day prediction", f"{forecast.point_forecast:.0f}", "µg/m³",
                    f"Target date: {forecast.target_date}")
    with c2:
        metric_card("90% prediction interval", f"{forecast.interval_low:.0f}–{forecast.interval_high:.0f}", "µg/m³",
                    "Log-space split conformal, calibrated on 2024")
    with c3:
        metric_card("Latest observation (day t)", f"{forecast.observed_pm25_today:.0f}", "µg/m³",
                    f"Observed on {forecast.origin_date}")

    st.markdown(
        f'<p class="aa-secondary">Model: <b>{forecast.model_name.upper()}</b>, feature set '
        f"<b>{forecast.feature_set}</b> (pollution lags + calendar + weather) &nbsp;·&nbsp; "
        f"Forecast generated: {forecast.generated_at_utc}</p>",
        unsafe_allow_html=True,
    )

    alert = classify_alert(forecast, cfg["events"]["primary_threshold"])
    titles = {
        "elevated": "⚠ Elevated-pollution forecast",
        "potential": "⚠ Potential elevated pollution (uncertainty-aware alert)",
        "below_threshold": "✓ Forecast is below the configured threshold",
    }
    if alert["level"] in titles:
        alert_panel(alert["level"], titles[alert["level"]], alert["message"])
    with st.expander("How is this alert decided?"):
        st.markdown(
            f"""
The configured pollution-event threshold is **{cfg['events']['primary_threshold']} µg/m³** (CPCB's
PM2.5 "Very Poor or worse" 24-hour breakpoint) — this is a **PM2.5 concentration threshold**, not a
CPCB overall AQI category, and it is unrelated to this page's PM2.5-vs-AQI distinction above.

1. **Elevated-pollution forecast** — the point forecast itself exceeds the threshold.
2. **Potential elevated pollution** — the point forecast is below the threshold, but the upper
   bound of the 90% prediction interval crosses it. The point estimate alone would miss this risk.
3. **Below threshold** — both the point forecast and the interval's upper bound stay under the
   threshold, though forecast uncertainty is never zero.

This re-uses the model's existing conformal prediction interval; it is not a new ML algorithm, and
the interval's **90% coverage is an empirical estimate** from the calibration period — it can drift
if pollution patterns change (see Model Performance → Conformal coverage).
            """
        )

    st.markdown("### Observed vs. forecast")
    window = st.radio(
        "Show last", ["120 days", "1 year", "3 years", "All history"],
        index=0, horizontal=True, label_visibility="collapsed", key="forecast_window",
    )
    _forecast_chart(daily, forecast, cfg, window)


_WINDOW_DAYS = {"120 days": 120, "1 year": 365, "3 years": 365 * 3, "All history": None}


def _forecast_chart(daily: pd.DataFrame, forecast, cfg: dict, window: str = "120 days") -> None:
    full = daily["pm25_mean"].dropna()
    n = _WINDOW_DAYS[window]
    hist = full if n is None else full.tail(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Historical PM2.5",
                              line=dict(color=COLORS["forest"], width=2)))
    target = pd.Timestamp(forecast.target_date)
    fig.add_trace(go.Scatter(x=[target], y=[forecast.point_forecast], mode="markers", name="Point prediction",
                              marker=dict(color=COLORS["amber"], size=12, symbol="diamond")))
    fig.add_trace(go.Scatter(
        x=[target, target], y=[forecast.interval_low, forecast.interval_high], mode="lines+markers",
        name="90% interval", line=dict(color=COLORS["amber"], width=6), opacity=0.5,
        marker=dict(size=8, color=COLORS["amber"]),
    ))
    fig.add_hline(y=cfg["events"]["primary_threshold"], line_dash="dash", line_color=COLORS["coral"],
                  annotation_text="Event threshold")
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=400, yaxis_title="PM2.5 (µg/m³)",
                       legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)