"""Live Air Quality — direct view of the OpenAQ live feed, honestly labeled."""
from __future__ import annotations

import os

import streamlit as st

from src import live_data
from src.aqi import CATEGORY_COLORS, calculate_aqi
from ui.theme import freshness_badge, metric_card


def render(cfg: dict) -> None:
    st.markdown("## Live Air Quality")
    st.markdown(
        '<p class="aa-secondary">Direct, current-as-available readings from OpenAQ for '
        f'{cfg["station"]["name"]}. This is different from the ML forecast on the PM2.5 '
        "Forecast page — these are measurements, not predictions.</p>",
        unsafe_allow_html=True,
    )

    if st.button("🔄 Refresh live data", type="primary"):
        _cached_fetch.clear()
        st.rerun()

    result = _cached_fetch(cfg["station"]["openaq_location_id"])

    if result.status == "unavailable":
        st.warning(
            "**Live data is unavailable.** " + result.reason + "\n\n"
            "To enable it: register a free API key at https://openaq.org, then set it as the "
            "`OPENAQ_API_KEY` environment variable (local) or in `.streamlit/secrets.toml` / the "
            "Streamlit Cloud app's **Secrets** settings (deployed) — never commit it to the repo."
        )
        _show_env_hint()
        return

    if result.status == "error":
        st.error(f"**Could not reach the live data provider.** {result.reason}\n\nShowing no data rather than a "
                 "guess — try refreshing again shortly.")
        return

    if not result.readings:
        st.info("The live feed responded but reported no current pollutant readings for this station.")
        return

    freshest = max(r.datetime_utc for r in result.readings.values())
    age_level = live_data.classify_freshness(freshest)
    st.markdown(
        f"**Source:** {result.source} &nbsp;·&nbsp; **Station:** location {result.location_id} &nbsp;·&nbsp; "
        f"**Most recent reading:** {freshest.strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp; "
        f"**Freshness:** {freshness_badge(age_level)} &nbsp;·&nbsp; "
        f"**Retrieved:** {result.retrieved_at_utc.strftime('%Y-%m-%d %H:%M UTC')}",
        unsafe_allow_html=True,
    )
    st.caption(
        "\"Freshness\" reflects how old the latest available OpenAQ observation is, not a live "
        "streaming sensor — the station reports periodically, not continuously."
    )

    aqi_result = calculate_aqi({p: r.value for p, r in result.readings.items()})
    st.markdown("### Pollutant readings")
    cols = st.columns(4)
    for i, (param, reading) in enumerate(sorted(result.readings.items())):
        with cols[i % 4]:
            lvl = live_data.classify_freshness(reading.datetime_utc)
            metric_card(param, f"{reading.value:.1f}", reading.unit,
                        f"{reading.datetime_utc.strftime('%Y-%m-%d %H:%M UTC')} " + freshness_badge(lvl))

    st.markdown("### AQI assessment")
    if aqi_result.is_overall:
        color = CATEGORY_COLORS.get(aqi_result.category, "#999")
        st.markdown(
            f"<div class='aa-card'><div class='aa-label'>Overall AQI</div>"
            f"<span class='aa-value'>{aqi_result.overall_aqi:.0f}</span> "
            f"<span class='aa-badge' style='background:{color}22;color:{color};'>{aqi_result.category}</span>"
            f"<div class='aa-sub'>Dominant pollutant: {aqi_result.dominant_pollutant}</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(f"Not enough pollutants for an official overall AQI. {aqi_result.reason}")
        if aqi_result.overall_aqi is not None:
            st.markdown(
                f"Best available **{aqi_result.dominant_pollutant}-specific** sub-index: "
                f"**{aqi_result.overall_aqi:.0f}** ({aqi_result.category}) — not the overall CPCB AQI."
            )


@st.cache_data(ttl=600)
def _cached_fetch(location_id: int):
    return live_data.fetch_live_readings(location_id)


def _show_env_hint() -> None:
    configured = bool(os.environ.get("OPENAQ_API_KEY"))
    st.caption(f"Environment check: `OPENAQ_API_KEY` env var {'is' if configured else 'is not'} set in this session.")
