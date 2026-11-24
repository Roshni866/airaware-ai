"""Tests for src/predict.py — model loading, schema compatibility, prediction,
conformal interval, alert logic, and stale/missing-data behaviour."""
import pandas as pd
import pytest

from src.config import FEATURES_CSV
from src.predict import ForecastResult, classify_alert, generate_forecast, load_model_bundle


def test_model_bundle_loads_and_matches_feature_sets_json():
    bundle = load_model_bundle()
    assert bundle["model"] == "rf"
    assert bundle["feature_set"] == "C"
    assert len(bundle["features"]) > 0
    assert hasattr(bundle["pipeline"], "predict")


def test_generate_forecast_against_real_processed_data():
    # Exercises the real data/processed/features.csv shipped with the repo.
    result = generate_forecast()
    assert result.status in ("ok", "stale_data")
    assert result.point_forecast is not None and result.point_forecast >= 0
    assert result.interval_low <= result.point_forecast <= result.interval_high
    assert result.model_name == "rf"
    assert result.feature_set == "C"


def test_missing_features_csv_handled_gracefully(tmp_path, monkeypatch):
    import src.predict as predict_mod
    monkeypatch.setattr(predict_mod, "FEATURES_CSV", tmp_path / "does_not_exist.csv")
    result = predict_mod.generate_forecast()
    assert result.status == "no_data"
    assert "pipeline" in result.reason.lower() or "not found" in result.reason.lower()


def test_schema_mismatch_detected(tmp_path, monkeypatch):
    import src.predict as predict_mod
    df = pd.read_csv(FEATURES_CSV, parse_dates=["origin_date", "target_date"])
    df = df.drop(columns=["wx_temp_mean"])  # remove a column the model expects
    fake_csv = tmp_path / "features.csv"
    df.to_csv(fake_csv, index=False)
    monkeypatch.setattr(predict_mod, "FEATURES_CSV", fake_csv)
    result = predict_mod.generate_forecast()
    assert result.status == "schema_mismatch"
    assert "wx_temp_mean" in result.reason


def test_stale_data_flagged_but_still_returns_last_valid_prediction():
    result = generate_forecast(max_data_age_days=0)  # force staleness for any real data
    assert result.status == "stale_data"
    assert result.point_forecast is not None  # still returned, per spec


def test_no_rows_with_observed_target_handled(tmp_path, monkeypatch):
    import src.predict as predict_mod
    df = pd.read_csv(FEATURES_CSV, parse_dates=["origin_date", "target_date"])
    df["pm_lag0"] = None
    fake_csv = tmp_path / "features.csv"
    df.to_csv(fake_csv, index=False)
    monkeypatch.setattr(predict_mod, "FEATURES_CSV", fake_csv)
    result = predict_mod.generate_forecast()
    assert result.status == "no_data"


# --- uncertainty-aware alert logic (Phase 7 innovation) ---
def test_alert_elevated_when_point_forecast_exceeds_threshold():
    f = ForecastResult(status="ok", point_forecast=150, interval_low=120, interval_high=180)
    alert = classify_alert(f, threshold=120)
    assert alert["level"] == "elevated"


def test_alert_potential_when_only_interval_crosses_threshold():
    f = ForecastResult(status="ok", point_forecast=90, interval_low=60, interval_high=130)
    alert = classify_alert(f, threshold=120)
    assert alert["level"] == "potential"


def test_alert_below_threshold_when_interval_stays_under():
    f = ForecastResult(status="ok", point_forecast=50, interval_low=30, interval_high=80)
    alert = classify_alert(f, threshold=120)
    assert alert["level"] == "below_threshold"


def test_alert_unknown_when_no_forecast():
    f = ForecastResult(status="no_data", point_forecast=None)
    alert = classify_alert(f, threshold=120)
    assert alert["level"] == "unknown"
