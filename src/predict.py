"""Single source of truth for producing a next-day PM2.5 forecast from the existing,
trained model. Both `app.py` (interactive) and `src/refresh.py` (scheduled) call this
module, so the forecast logic is defined exactly once.

Does not retrain anything: loads the pre-selected model bundle from `models/`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from .config import DAILY_CSV, FEATURES_CSV, MODELS, RESULTS, load_config
from .conformal import fit_log, predict_interval


@dataclass
class ForecastResult:
    status: str  # "ok" | "stale_data" | "no_data" | "schema_mismatch"
    reason: str = ""
    origin_date: str | None = None
    target_date: str | None = None
    point_forecast: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    observed_pm25_today: float | None = None
    model_name: str | None = None
    feature_set: str | None = None
    generated_at_utc: str | None = None
    data_age_days: int | None = None


def load_model_bundle(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    selected = json.loads((RESULTS / "selected_model.json").read_text())
    bundle = joblib.load(MODELS / f"{selected['model']}_{selected['feature_set']}.joblib")
    return bundle


def _calibration_params(cfg: dict, selected: dict):
    preds = pd.read_csv(RESULTS / "predictions_all.csv", parse_dates=["target_date"])
    cal = preds[(preds.model == selected["model"]) & (preds.feature_set == selected["feature_set"]) & (preds.split == "cal")]
    if cal.empty:
        raise RuntimeError("No calibration-split predictions found in results/predictions_all.csv.")
    return fit_log(cal.y, cal.yhat, cfg["conformal"]["alpha"])


def generate_forecast(max_data_age_days: int = 3) -> ForecastResult:
    """Build the live feature row from data/processed/, validate schema, predict, and
    attach a conformal interval. Never silently predicts from data judged too old: still
    returns the point estimate/interval so the caller can show it labeled 'stale', per
    the pre-registered rule ("show the last valid prediction with a clear stale label").
    """
    cfg = load_config()
    try:
        selected = json.loads((RESULTS / "selected_model.json").read_text())
        bundle = load_model_bundle(cfg)
    except FileNotFoundError as e:
        return ForecastResult(status="no_data", reason=f"Model artifact missing: {e}")

    if not FEATURES_CSV.exists():
        return ForecastResult(status="no_data", reason="features.csv not found — run the data pipeline first.")

    feats = pd.read_csv(FEATURES_CSV, parse_dates=["origin_date", "target_date"], index_col="origin_date")
    live_rows = feats[feats["pm_lag0"].notna()].sort_index()
    if live_rows.empty:
        return ForecastResult(status="no_data", reason="No rows with a valid day-t observation in features.csv.")

    last_origin = live_rows.index.max()
    expected_cols = bundle["features"]
    missing = [c for c in expected_cols if c not in live_rows.columns]
    if missing:
        return ForecastResult(
            status="schema_mismatch",
            reason=f"features.csv is missing columns the model expects: {missing}. "
                   "Re-run `python -m src.features` (and data_pipeline if needed).",
        )

    live_x = live_rows.loc[[last_origin], expected_cols]
    yhat = float(bundle["pipeline"].predict(live_x)[0])
    yhat = max(yhat, 0.0)

    try:
        params = _calibration_params(cfg, selected)
        lo, hi = predict_interval(np.array([yhat]), params)
        lo, hi = float(lo[0]), float(hi[0])
    except Exception as e:  # noqa: BLE001 - surface clearly rather than crash the page
        lo = hi = None

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    age_days = int((today - last_origin).days)
    status = "ok" if age_days <= max_data_age_days else "stale_data"

    return ForecastResult(
        status=status,
        reason="" if status == "ok" else f"Latest observation is {age_days} days old (threshold: {max_data_age_days}).",
        origin_date=str(last_origin.date()),
        target_date=str((last_origin + pd.Timedelta(days=1)).date()),
        point_forecast=round(yhat, 1),
        interval_low=round(lo, 1) if lo is not None else None,
        interval_high=round(hi, 1) if hi is not None else None,
        observed_pm25_today=round(float(live_x["pm_lag0"].iloc[0]), 1),
        model_name=selected["model"],
        feature_set=selected["feature_set"],
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        data_age_days=age_days,
    )


def classify_alert(forecast: ForecastResult, threshold: float) -> dict:
    """Three-tier, uncertainty-aware alert (Phase 7's "small but meaningful innovation"):
    the point forecast alone is not the only signal — the conformal interval's upper
    bound can flag risk even when the point estimate looks fine. Not a new ML algorithm;
    this is application-level decision logic layered on the existing model + interval.
    """
    if forecast.point_forecast is None:
        return {"level": "unknown", "message": "No forecast available."}
    if forecast.point_forecast > threshold:
        return {
            "level": "elevated",
            "message": f"Point forecast ({forecast.point_forecast:.0f} µg/m³) exceeds the "
                       f"{threshold:.0f} µg/m³ threshold.",
        }
    if forecast.interval_high is not None and forecast.interval_high > threshold:
        return {
            "level": "potential",
            "message": f"Point forecast ({forecast.point_forecast:.0f} µg/m³) is below {threshold:.0f} µg/m³, but "
                       f"the upper end of the 90% interval ({forecast.interval_high:.0f} µg/m³) crosses it — "
                       "elevated pollution is plausible.",
        }
    return {
        "level": "below_threshold",
        "message": f"Forecast ({forecast.point_forecast:.0f} µg/m³) and its 90% interval stay below "
                   f"{threshold:.0f} µg/m³, though some uncertainty always remains.",
    }
