"""Day 2 — leakage-safe feature construction and chronological split.

Each row is a forecast ORIGIN day t. The target is the daily mean PM2.5 of day t+1.
All PM2.5 and weather features are computed from data on or before day t (verified by
tests/test_features.py). Calendar features describe the target date, which is known in
advance. The ``C_oracle`` set additionally uses observed weather on day t+1 and is a
diagnostic upper bound only.

Usage:  python -m src.features
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import DAILY_CSV, FEATURES_CSV, RESULTS, load_config

WEATHER_COLS = ["temp_mean", "rh_mean", "wind_speed_mean", "wind_u_mean", "wind_v_mean", "precip_sum"]
SPLITS = ["train", "val", "cal", "test"]


def build_features(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return one row per origin date t (index), with target, features and split label."""
    daily = daily.asfreq("D")  # guarantee a gap-free calendar so shifts are in days
    pm = daily["pm25_mean"]
    f = pd.DataFrame(index=daily.index)
    f.index.name = "origin_date"
    f["target_date"] = f.index + pd.Timedelta(days=1)
    f["target"] = pm.shift(-1)

    # --- A: pollution history (all end at day t) ---
    for k in cfg["features"]["pm_lags"]:
        f[f"pm_lag{k}"] = pm.shift(k)
    for w in cfg["features"]["rolling_windows"]:
        r = pm.rolling(w, min_periods=max(2, int(np.ceil(w / 2))))
        f[f"pm_roll_mean{w}"] = r.mean()
        f[f"pm_roll_std{w}"] = r.std()
    f["pm_max_hourly_lag0"] = daily["pm25_max_hourly"]

    # --- B: calendar of the target date (known in advance) ---
    td = f["target_date"]
    f["dow_sin"] = np.sin(2 * np.pi * td.dt.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * td.dt.dayofweek / 7)
    f["month_sin"] = np.sin(2 * np.pi * (td.dt.month - 1) / 12)
    f["month_cos"] = np.cos(2 * np.pi * (td.dt.month - 1) / 12)
    f["doy_sin"] = np.sin(2 * np.pi * td.dt.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * td.dt.dayofyear / 365.25)

    # --- C: weather observed on origin day t ---
    for c in WEATHER_COLS:
        f[f"wx_{c}"] = daily[c]
    # --- C_oracle: weather observed on target day t+1 (diagnostic only) ---
    for c in WEATHER_COLS:
        f[f"wxnext_{c}"] = daily[c].shift(-1)

    # --- split by TARGET date ---
    s = cfg["splits"]
    edges = [pd.Timestamp(s["train_end"]), pd.Timestamp(s["val_end"]), pd.Timestamp(s["cal_end"])]
    f["split"] = np.select(
        [td <= edges[0], td <= edges[1], td <= edges[2]], ["train", "val", "cal"], default="test"
    )
    f = f[f["target_date"] >= pd.Timestamp(cfg["data"]["start_date"]) + pd.Timedelta(days=1)]
    return f


def feature_sets(columns) -> dict[str, list[str]]:
    a = [c for c in columns if c.startswith("pm_")]
    cal = ["dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]
    wx = [c for c in columns if c.startswith("wx_")]
    wxn = [c for c in columns if c.startswith("wxnext_")]
    return {"A": a, "B": a + cal, "C": a + cal + wx, "C_oracle": a + cal + wx + wxn}


def modelling_rows(f: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for evaluation: observed target AND observed y_t (so every model,
    including persistence, is scored on the identical set of days)."""
    return f[f["target"].notna() & f["pm_lag0"].notna()]


def split_summary(f: pd.DataFrame) -> pd.DataFrame:
    m = modelling_rows(f)
    rows = []
    for s in SPLITS:
        all_s, use = f[f["split"] == s], m[m["split"] == s]
        rows.append(
            {
                "split": s,
                "first_target_date": str(all_s["target_date"].min().date()),
                "last_target_date": str(all_s["target_date"].max().date()),
                "calendar_days": len(all_s),
                "target_observed": int(all_s["target"].notna().sum()),
                "usable_rows": len(use),
                "dropped_missing_y_t": int(all_s["target"].notna().sum() - len(use)),
                "mean_target": round(float(use["target"].mean()), 1),
            }
        )
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    cfg = load_config()
    daily = pd.read_csv(DAILY_CSV, parse_dates=["date"], index_col="date")
    f = build_features(daily, cfg)
    f.to_csv(FEATURES_CSV)
    summ = split_summary(f)
    summ.to_csv(RESULTS / "split_summary.csv", index=False)
    (RESULTS / "feature_sets.json").write_text(json.dumps(feature_sets(f.columns), indent=2))
    print(summ.to_string(index=False))
    return f


if __name__ == "__main__":
    run()
