"""CPCB National Air Quality Index (NAQI) calculator.

Official methodology (CPCB, "National Air Quality Index", 2014) — breakpoints verified
against https://www.aqicn.org/images/aqi-scales/india-national-air-quality-standard.pdf
(CPCB's own published scale). Pure calculation logic, no Streamlit/UI code here.

Sub-index formula (linear interpolation within a breakpoint band):

    I_p = (IHI - ILO) / (BHI - BLO) * (C_p - BLO) + ILO

Overall AQI = max of the individual (valid) sub-indices; the pollutant that produced it
is the "dominant pollutant". CPCB rule: an overall AQI is reported only when at least
three pollutants have valid readings, of which at least one must be PM2.5 or PM10 —
otherwise there isn't enough coverage to represent the city's air quality, not just one
pollutant's. This module enforces that rule rather than silently computing from fewer.

**Severe band is officially open-ended.** CPCB's published breakpoint table (e.g. the
2014 CPCB report, Table 3.11) gives the Severe category (AQI 401-500) only a *lower*
concentration bound per pollutant ("250+ µg/m³" for PM2.5, etc.) — it does not publish an
upper concentration for AQI=500, because Indian cities routinely exceed it. Rather than
inventing an upper bound (which would misrepresent an official standard), this module
extrapolates linearly past the Severe lower bound using the slope of the last *fully
defined* band (Poor -> Very Poor), and flags the result as an estimate
(`SubIndexResult.out_of_range = True`) rather than presenting it as an official CPCB number.
"""
from __future__ import annotations

from dataclasses import dataclass

AQI_BANDS = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CATEGORY_COLORS = {
    "Good": "#4CAF50",
    "Satisfactory": "#8BC34A",
    "Moderate": "#F0B65A",
    "Poor": "#EF6C00",
    "Very Poor": "#D96C5F",
    "Severe": "#7B241C",
}

# {pollutant: unit, averaging period, and the five FULLY DEFINED bands (Good..Very Poor)
# plus the Severe band's lower bound only (its upper bound is officially open-ended).
# Source: CPCB "National Air Quality Index" (2014), breakpoint table.
BREAKPOINTS: dict[str, dict] = {
    "PM2.5": {"unit": "µg/m³", "period": "24-hour", "bands": [(0, 30), (31, 60), (61, 90), (91, 120), (121, 250)], "severe_lower": 251},
    "PM10":  {"unit": "µg/m³", "period": "24-hour", "bands": [(0, 50), (51, 100), (101, 250), (251, 350), (351, 430)], "severe_lower": 431},
    "NO2":   {"unit": "µg/m³", "period": "24-hour", "bands": [(0, 40), (41, 80), (81, 180), (181, 280), (281, 400)], "severe_lower": 401},
    "SO2":   {"unit": "µg/m³", "period": "24-hour", "bands": [(0, 40), (41, 80), (81, 380), (381, 800), (801, 1600)], "severe_lower": 1601},
    "O3":    {"unit": "µg/m³", "period": "8-hour",  "bands": [(0, 50), (51, 100), (101, 168), (169, 208), (209, 748)], "severe_lower": 749},
    "CO":    {"unit": "mg/m³", "period": "8-hour",  "bands": [(0, 1.0), (1.1, 2.0), (2.1, 10), (10.1, 17), (17.1, 34)], "severe_lower": 34.1},
    "NH3":   {"unit": "µg/m³", "period": "24-hour", "bands": [(0, 200), (201, 400), (401, 800), (801, 1200), (1201, 1800)], "severe_lower": 1801},
    "Pb":    {"unit": "µg/m³", "period": "24-hour", "bands": [(0, 0.5), (0.6, 1.0), (1.1, 2.0), (2.1, 3.0), (3.1, 3.5)], "severe_lower": 3.6},
}

POLLUTANTS = list(BREAKPOINTS.keys())
# CPCB: overall AQI requires >=3 pollutants, at least one of which is PM2.5 or PM10.
MIN_POLLUTANTS_FOR_OVERALL_AQI = 3
REQUIRED_ANY_OF = {"PM2.5", "PM10"}


@dataclass
class SubIndexResult:
    pollutant: str
    concentration: float
    unit: str
    sub_index: float
    category: str
    out_of_range: bool = False  # concentration above the top published breakpoint (extrapolated)


@dataclass
class AQIResult:
    sub_indices: list[SubIndexResult]
    overall_aqi: float | None
    category: str | None
    dominant_pollutant: str | None
    is_overall: bool  # True = valid CPCB overall AQI; False = insufficient pollutants (partial only)
    reason: str = ""


def valid_pollutants() -> list[str]:
    return POLLUTANTS


def calculate_sub_index(pollutant: str, concentration: float) -> SubIndexResult:
    """Linear interpolation of one pollutant's concentration into its CPCB sub-index.

    For concentrations at or above the pollutant's Severe lower bound, the sub-index is
    an *extrapolation* (see module docstring) and ``out_of_range`` is set True; the
    category is still correctly reported as "Severe" either way.
    """
    if pollutant not in BREAKPOINTS:
        raise ValueError(f"Unknown pollutant '{pollutant}'. Expected one of {POLLUTANTS}.")
    if concentration is None or concentration < 0:
        raise ValueError(f"Concentration for {pollutant} must be a non-negative number, got {concentration!r}.")

    spec = BREAKPOINTS[pollutant]
    bands = spec["bands"]  # 5 fully-defined bands: Good..Very Poor
    severe_lower = spec["severe_lower"]

    if concentration >= severe_lower:
        # Officially "Severe" with no published upper concentration bound. Extrapolate
        # using the slope of the last fully-defined band (Poor -> Very Poor) rather than
        # inventing an upper bound; this is clearly flagged, not presented as official.
        c_lo, c_hi = bands[-1]
        i_lo, i_hi = AQI_BANDS[-2]  # Very Poor's index range (301-400)
        slope = (i_hi - i_lo) / (c_hi - c_lo)
        sub_index = 401 + slope * (concentration - severe_lower)
        out_of_range = True
    else:
        band_idx = 0
        for i, (lo, hi) in enumerate(bands):
            if lo <= concentration <= hi:
                band_idx = i
                break
        c_lo, c_hi = bands[band_idx]
        i_lo, i_hi = AQI_BANDS[band_idx]
        sub_index = i_lo if c_hi == c_lo else (i_hi - i_lo) / (c_hi - c_lo) * (concentration - c_lo) + i_lo
        out_of_range = False

    category = _category_for_index(sub_index)
    return SubIndexResult(
        pollutant=pollutant, concentration=float(concentration), unit=spec["unit"],
        sub_index=round(sub_index, 1), category=category, out_of_range=out_of_range,
    )


def _category_for_index(index: float) -> str:
    for (lo, hi), cat in zip(AQI_BANDS, CATEGORIES):
        if lo <= index <= hi:
            return cat
    return CATEGORIES[-1] if index > AQI_BANDS[-1][1] else CATEGORIES[0]


def calculate_aqi(concentrations: dict[str, float]) -> AQIResult:
    """concentrations: {pollutant_name: value}. Pollutants with None/missing are skipped.

    Returns an AQIResult. ``is_overall`` is False (reason explains why) when CPCB's
    minimum-coverage rule (>=3 pollutants, >=1 of PM2.5/PM10) is not met; in that case
    ``overall_aqi`` is still returned as the best available PM2.5/PM10 sub-index if present,
    so the caller can label it explicitly as a *pollutant-specific* assessment, never as
    "the overall AQI".
    """
    valid = {p: c for p, c in concentrations.items() if c is not None and p in BREAKPOINTS}
    sub_results = [calculate_sub_index(p, c) for p, c in valid.items()]

    if not sub_results:
        return AQIResult([], None, None, None, False, reason="No valid pollutant concentrations supplied.")

    has_required_anchor = any(p in REQUIRED_ANY_OF for p in valid)
    enough_count = len(valid) >= MIN_POLLUTANTS_FOR_OVERALL_AQI

    best = max(sub_results, key=lambda r: r.sub_index)

    if enough_count and has_required_anchor:
        return AQIResult(sub_results, best.sub_index, best.category, best.pollutant, True)

    reasons = []
    if not enough_count:
        reasons.append(f"only {len(valid)} pollutant(s) supplied (CPCB requires >=3 for an overall AQI)")
    if not has_required_anchor:
        reasons.append("neither PM2.5 nor PM10 was supplied (CPCB requires at least one)")
    reason = "Cannot compute an official overall AQI: " + "; ".join(reasons) + "."

    # Report the best available sub-index, but flagged as partial, not an overall AQI.
    anchor = next((r for r in sub_results if r.pollutant in REQUIRED_ANY_OF), None)
    partial = anchor if anchor is not None else best
    return AQIResult(sub_results, partial.sub_index, partial.category, partial.pollutant, False, reason=reason)
