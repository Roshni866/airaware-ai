"""Tests for src/live_data.py — API parsing, freshness, graceful failure. No network calls."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import requests

from src import live_data


def test_no_api_key_returns_unavailable_not_fabricated(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    monkeypatch.setattr(live_data, "_api_key", lambda: None)
    result = live_data.fetch_live_readings(8118)
    assert result.status == "unavailable"
    assert result.readings == {}
    assert "API" in result.reason or "key" in result.reason.lower()


@pytest.mark.parametrize(
    "age_hours,expected",
    [(0, "fresh"), (2.9, "fresh"), (3.1, "delayed"), (23.9, "delayed"), (24.1, "stale"), (240, "stale"), (-1, "fresh")],
)
def test_freshness_classification_boundaries(age_hours, expected):
    now = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
    obs = now - timedelta(hours=age_hours)
    assert live_data.classify_freshness(obs, now_utc=now) == expected


def test_freshness_handles_naive_datetime():
    now = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
    obs_naive = datetime(2026, 11, 24, 11, 0)  # no tzinfo
    assert live_data.classify_freshness(obs_naive, now_utc=now) == "fresh"


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def _fake_get_factory(sensors_payload, latest_payload):
    def _fake_get(url, headers=None, params=None, timeout=None):
        if url.endswith("/sensors"):
            return _FakeResponse(200, sensors_payload)
        if url.endswith("/latest"):
            return _FakeResponse(200, latest_payload)
        raise AssertionError(f"unexpected URL {url}")
    return _fake_get


def test_parses_valid_response_and_maps_sensor_to_pollutant(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    sensors = {"results": [
        {"id": 1, "parameter": {"name": "pm25", "units": "µg/m³"}},
        {"id": 2, "parameter": {"name": "no2", "units": "µg/m³"}},
    ]}
    latest = {"results": [
        {"sensorsId": 1, "value": 88.5, "datetime": {"utc": "2026-11-24T10:00:00Z"}, "locationsId": 8118},
        {"sensorsId": 2, "value": 45.0, "datetime": {"utc": "2026-11-24T10:00:00Z"}, "locationsId": 8118},
    ]}
    with patch("requests.get", side_effect=_fake_get_factory(sensors, latest)):
        result = live_data.fetch_live_readings(8118)
    assert result.status == "ok"
    assert set(result.readings) == {"PM2.5", "NO2"}
    assert result.readings["PM2.5"].value == 88.5
    assert result.readings["PM2.5"].sensor_id == 1


def test_unmapped_pollutant_is_skipped_not_guessed(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    sensors = {"results": [{"id": 9, "parameter": {"name": "bc", "units": "µg/m³"}}]}  # black carbon: not CPCB's 8
    latest = {"results": [{"sensorsId": 9, "value": 1.2, "datetime": {"utc": "2026-11-24T10:00:00Z"}}]}
    with patch("requests.get", side_effect=_fake_get_factory(sensors, latest)):
        result = live_data.fetch_live_readings(8118)
    assert result.status == "ok"
    assert result.readings == {}


def test_co_unit_conversion_ppm_to_mgm3(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    sensors = {"results": [{"id": 3, "parameter": {"name": "co", "units": "ppm"}}]}
    latest = {"results": [{"sensorsId": 3, "value": 1.0, "datetime": {"utc": "2026-11-24T10:00:00Z"}}]}
    with patch("requests.get", side_effect=_fake_get_factory(sensors, latest)):
        result = live_data.fetch_live_readings(8118)
    assert result.readings["CO"].value == pytest.approx(live_data.CO_PPM_TO_MGM3)


def test_401_returns_error_status_not_crash(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "bad-key")
    def _unauthorized(url, headers=None, params=None, timeout=None):
        return _FakeResponse(401, {"message": "Unauthorized"})
    with patch("requests.get", side_effect=_unauthorized):
        result = live_data.fetch_live_readings(8118)
    assert result.status == "error"


def test_empty_results_returns_unavailable(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    with patch("requests.get", side_effect=_fake_get_factory({"results": []}, {"results": []})):
        result = live_data.fetch_live_readings(8118)
    assert result.status == "unavailable"


def test_network_error_retries_then_errors(monkeypatch):
    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise requests.ConnectionError("boom")

    with patch("requests.get", side_effect=_boom), patch("time.sleep", return_value=None):
        result = live_data.fetch_live_readings(8118, retries=1)
    assert result.status == "error"
    assert calls["n"] == 2  # initial + 1 retry
