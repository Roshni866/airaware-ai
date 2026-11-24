"""Leakage and correctness tests. Run:  python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.conformal import conformal_quantile
from src.features import WEATHER_COLS, build_features, feature_sets

CFG = {
    "data": {"start_date": "2020-01-01"},
    "features": {"pm_lags": [0, 1, 2, 3, 7], "rolling_windows": [3, 7]},
    "splits": {"train_end": "2020-03-31", "val_end": "2020-04-30", "cal_end": "2020-05-31"},
}


@pytest.fixture
def daily():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", "2020-07-31", freq="D", name="date")
    d = pd.DataFrame({"pm25_mean": rng.gamma(4, 25, len(idx))}, index=idx)
    d["pm25_max_hourly"] = d["pm25_mean"] * 1.5
    for c in WEATHER_COLS:
        d[c] = rng.normal(size=len(idx))
    d.iloc[[10, 11, 50]] = np.nan  # gaps
    return d


def test_no_future_information_in_deployable_features(daily):
    t0 = pd.Timestamp("2020-04-15")
    base = build_features(daily, CFG)
    future = daily.copy()
    future.loc[future.index > t0] = 9999.0  # corrupt everything after the origin
    pert = build_features(future, CFG)
    cols = feature_sets(base.columns)["C"]
    a = base.loc[:t0, cols]
    b = pert.loc[:t0, cols]
    pd.testing.assert_frame_equal(a, b)


def test_target_is_next_day(daily):
    f = build_features(daily, CFG)
    t = pd.Timestamp("2020-03-05")
    assert f.loc[t, "target"] == daily.loc[t + pd.Timedelta(days=1), "pm25_mean"]
    assert f.loc[t, "pm_lag0"] == daily.loc[t, "pm25_mean"]
    assert f.loc[t, "pm_lag7"] == daily.loc[t - pd.Timedelta(days=7), "pm25_mean"]


def test_gaps_are_not_filled(daily):
    f = build_features(daily, CFG)
    gap_day = daily.index[10]
    assert np.isnan(f.loc[gap_day, "pm_lag0"])
    assert np.isnan(f.loc[gap_day - pd.Timedelta(days=1), "target"])


def test_split_is_chronological_and_disjoint(daily):
    f = build_features(daily, CFG)
    order = {"train": 0, "val": 1, "cal": 2, "test": 3}
    codes = f["split"].map(order).to_numpy()
    assert (np.diff(codes) >= 0).all()
    assert f.loc[f["split"] == "train", "target_date"].max() <= pd.Timestamp("2020-03-31")


def test_conformal_quantile_finite_sample():
    scores = np.arange(1, 101, dtype=float)  # n = 100
    # ceil((n+1)(1-alpha)) = ceil(90.9) = 91 -> 91st smallest score
    assert conformal_quantile(scores, 0.10) == 91.0
    assert conformal_quantile(np.arange(1, 6, dtype=float), 0.10) == np.inf
