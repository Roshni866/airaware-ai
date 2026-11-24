"""Live air-quality data integration — OpenAQ v3 API (see README for why this provider
was chosen over CPCB/WAQI).

Requires a real API key, read from the `OPENAQ_API_KEY` environment variable or
`st.secrets["OPENAQ_API_KEY"]`. With no key configured, returns an explicit
"unavailable" status — never fabricates or substitutes a cached value for a live one.

Two OpenAQ calls are combined because `latest` reports only `sensorsId` + a raw value,
not the pollutant name:
1. `GET /v3/locations/{id}/sensors` -> sensorsId -> {parameter name, units}
2. `GET /v3/locations/{id}/latest`  -> sensorsId -> {value, datetime}
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

OPENAQ_BASE = "https://api.openaq.org/v3"
REQUEST_TIMEOUT = 15

# OpenAQ parameter name -> the name src/aqi.py's BREAKPOINTS dict expects.
PARAMETER_NAME_MAP = {
    "pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "so2": "SO2",
    "co": "CO", "o3": "O3", "nh3": "NH3", "pb": "Pb",
}
# OpenAQ commonly reports CO in ppm; CPCB breakpoints are in mg/m³. 1 ppm CO ~= 1.145 mg/m3 at 25C/1atm.
CO_PPM_TO_MGM3 = 1.145

FRESHNESS_FRESH_HOURS = 3
FRESHNESS_DELAYED_HOURS = 24


@dataclass
class PollutantReading:
    parameter: str          # normalized name, e.g. "PM2.5"
    value: float
    unit: str
    datetime_utc: datetime
    sensor_id: int


@dataclass
class LiveReadingResult:
    status: str                              # "ok" | "unavailable" | "error"
    reason: str = ""
    location_name: str | None = None
    location_id: int | None = None
    source: str = "OpenAQ v3 (api.openaq.org)"
    retrieved_at_utc: datetime | None = None
    readings: dict[str, PollutantReading] = field(default_factory=dict)  # by normalized name

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _api_key() -> str | None:
    key = os.environ.get("OPENAQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("OPENAQ_API_KEY")  # type: ignore[union-attr]
    except Exception:
        return None


def classify_freshness(obs_time_utc: datetime, now_utc: datetime | None = None) -> str:
    """'fresh' | 'delayed' | 'stale' based on age of the observation."""
    now_utc = now_utc or datetime.now(timezone.utc)
    if obs_time_utc.tzinfo is None:
        obs_time_utc = obs_time_utc.replace(tzinfo=timezone.utc)
    age_hours = (now_utc - obs_time_utc).total_seconds() / 3600
    if age_hours < 0:
        return "fresh"  # clock skew guard; don't report negative age as stale
    if age_hours <= FRESHNESS_FRESH_HOURS:
        return "fresh"
    if age_hours <= FRESHNESS_DELAYED_HOURS:
        return "delayed"
    return "stale"


def _get(path: str, api_key: str, params: dict | None = None) -> dict:
    r = requests.get(
        f"{OPENAQ_BASE}{path}", headers={"X-API-Key": api_key}, params=params or {}, timeout=REQUEST_TIMEOUT,
    )
    if r.status_code == 401:
        raise PermissionError("OpenAQ rejected the API key (401 Unauthorized).")
    if r.status_code == 429:
        raise RuntimeError("OpenAQ rate limit exceeded (429). Try again shortly.")
    r.raise_for_status()
    return r.json()


def fetch_live_readings(location_id: int, retries: int = 2) -> LiveReadingResult:
    """Fetch the latest available reading for every pollutant OpenAQ reports for this
    station. Returns status='unavailable' (no key) or 'error' (API/network problem)
    rather than ever inventing a value."""
    api_key = _api_key()
    if not api_key:
        return LiveReadingResult(
            status="unavailable",
            reason="No OPENAQ_API_KEY configured (set it as an environment variable or in "
                   "Streamlit secrets). Register a free key at https://openaq.org to enable live data.",
            location_id=location_id,
        )

    last_err = None
    for attempt in range(retries + 1):
        try:
            sensors_js = _get(f"/locations/{location_id}/sensors", api_key)
            sensor_meta = {
                s["id"]: (s.get("parameter", {}).get("name", "").lower(), s.get("parameter", {}).get("units", ""))
                for s in sensors_js.get("results", [])
            }
            latest_js = _get(f"/locations/{location_id}/latest", api_key)
            results = latest_js.get("results", [])
            if not results:
                return LiveReadingResult(status="unavailable", reason="OpenAQ returned no sensor readings for this station.",
                                          location_id=location_id, retrieved_at_utc=datetime.now(timezone.utc))

            readings: dict[str, PollutantReading] = {}
            for row in results:
                sid = row.get("sensorsId")
                param_raw, unit = sensor_meta.get(sid, (None, None))
                norm = PARAMETER_NAME_MAP.get(param_raw) if param_raw else None
                if norm is None:
                    continue  # pollutant not in CPCB's 8 (or unmapped sensor) — skip, don't guess
                value = row.get("value")
                dt_str = (row.get("datetime") or {}).get("utc")
                if value is None or dt_str is None:
                    continue
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if norm == "CO" and unit and "ppm" in unit.lower():
                    value = value * CO_PPM_TO_MGM3
                    unit = "mg/m³ (converted from ppm)"
                # Keep the most recent reading if a pollutant has multiple sensors.
                if norm not in readings or dt > readings[norm].datetime_utc:
                    readings[norm] = PollutantReading(norm, float(value), unit or "", dt, sid)

            return LiveReadingResult(
                status="ok", location_id=location_id, readings=readings,
                retrieved_at_utc=datetime.now(timezone.utc),
            )
        except PermissionError as e:
            return LiveReadingResult(status="error", reason=str(e), location_id=location_id)
        except (requests.RequestException, RuntimeError, ValueError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
    return LiveReadingResult(status="error", reason=f"OpenAQ request failed after {retries + 1} attempts: {last_err}",
                              location_id=location_id)
