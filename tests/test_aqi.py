"""Tests for the CPCB AQI calculator (src/aqi.py)."""
import pytest

from src import aqi


# --- breakpoint boundary correctness (spot-checked against the CPCB table) ---
@pytest.mark.parametrize(
    "pollutant,conc,expected_index,expected_category",
    [
        ("PM2.5", 0, 0, "Good"),
        ("PM2.5", 30, 50, "Good"),
        ("PM2.5", 31, 51, "Satisfactory"),
        ("PM2.5", 60, 100, "Satisfactory"),
        ("PM2.5", 90, 200, "Moderate"),
        ("PM2.5", 120, 300, "Poor"),
        ("PM2.5", 250, 400, "Very Poor"),
        ("PM10", 50, 50, "Good"),
        ("PM10", 430, 400, "Very Poor"),
        ("CO", 1.0, 50, "Good"),
        ("CO", 34, 400, "Very Poor"),
        ("Pb", 0.5, 50, "Good"),
    ],
)
def test_known_breakpoints(pollutant, conc, expected_index, expected_category):
    r = aqi.calculate_sub_index(pollutant, conc)
    assert r.sub_index == pytest.approx(expected_index, abs=0.5)
    assert r.category == expected_category
    assert r.out_of_range is False


def test_midband_interpolation_pm25():
    # Moderate band for PM2.5: conc 61-90 -> index 101-200. Midpoint check.
    r = aqi.calculate_sub_index("PM2.5", 75.5)
    lo_c, hi_c = 61, 90
    lo_i, hi_i = 101, 200
    expected = (hi_i - lo_i) / (hi_c - lo_c) * (75.5 - lo_c) + lo_i
    assert r.sub_index == pytest.approx(expected, abs=0.1)


def test_severe_band_is_extrapolated_and_flagged():
    r = aqi.calculate_sub_index("PM2.5", 400)  # well above the 251 Severe lower bound
    assert r.category == "Severe"
    assert r.sub_index > 401
    assert r.out_of_range is True


def test_negative_concentration_rejected():
    with pytest.raises(ValueError):
        aqi.calculate_sub_index("PM2.5", -5)


def test_unknown_pollutant_rejected():
    with pytest.raises(ValueError):
        aqi.calculate_sub_index("Radon", 10)


# --- overall AQI aggregation rules ---
def test_overall_aqi_requires_three_pollutants():
    result = aqi.calculate_aqi({"PM2.5": 80})
    assert result.is_overall is False
    assert "3" in result.reason or "three" in result.reason.lower()
    assert result.overall_aqi is not None  # still returns a PM2.5-specific value


def test_overall_aqi_requires_pm25_or_pm10():
    result = aqi.calculate_aqi({"NO2": 100, "SO2": 100, "CO": 2})
    assert result.is_overall is False
    assert "PM2.5" in result.reason or "PM10" in result.reason


def test_overall_aqi_valid_with_three_including_pm25():
    result = aqi.calculate_aqi({"PM2.5": 200, "NO2": 50, "CO": 1})
    assert result.is_overall is True
    assert result.dominant_pollutant == "PM2.5"
    assert result.overall_aqi == max(r.sub_index for r in result.sub_indices)


def test_dominant_pollutant_is_max_sub_index():
    result = aqi.calculate_aqi({"PM2.5": 50, "PM10": 430, "NO2": 50})
    # PM10 at 430 -> AQI 400 (Very Poor), far higher than PM2.5 at 50 -> AQI ~83
    assert result.is_overall is True
    assert result.dominant_pollutant == "PM10"


def test_empty_input_handled_gracefully():
    result = aqi.calculate_aqi({})
    assert result.is_overall is False
    assert result.overall_aqi is None
    assert result.sub_indices == []


def test_none_values_are_skipped_not_errors():
    result = aqi.calculate_aqi({"PM2.5": 80, "PM10": None, "NO2": None})
    assert len(result.sub_indices) == 1
    assert result.is_overall is False  # only 1 valid pollutant
