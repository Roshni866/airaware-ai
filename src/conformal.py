"""Split conformal prediction intervals (absolute and log-space variants)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score.
    Returns +inf when n is too small for the requested level."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(scores[k - 1])


def fit_absolute(y_cal, yhat_cal, alpha: float) -> dict:
    q = conformal_quantile(np.abs(np.asarray(y_cal) - np.asarray(yhat_cal)), alpha)
    return {"method": "absolute", "alpha": alpha, "q": q, "n_cal": int(len(y_cal))}


def fit_log(y_cal, yhat_cal, alpha: float) -> dict:
    """Residuals on log1p scale -> multiplicative intervals (width grows with level)."""
    s = np.abs(np.log1p(np.asarray(y_cal)) - np.log1p(np.clip(np.asarray(yhat_cal), 0, None)))
    q = conformal_quantile(s, alpha)
    return {"method": "log", "alpha": alpha, "q": q, "n_cal": int(len(y_cal))}


def predict_interval(yhat, params: dict) -> tuple[np.ndarray, np.ndarray]:
    yhat = np.clip(np.asarray(yhat, dtype=float), 0, None)
    q = params["q"]
    if params["method"] == "absolute":
        lo, hi = yhat - q, yhat + q
    else:
        lo, hi = np.expm1(np.log1p(yhat) - q), np.expm1(np.log1p(yhat) + q)
    return np.clip(lo, 0, None), hi  # PM2.5 cannot be negative


def coverage_table(y, lo, hi, groups: pd.Series | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "lo": np.asarray(lo), "hi": np.asarray(hi)})
    df["covered"] = (df["y"] >= df["lo"]) & (df["y"] <= df["hi"])
    df["width"] = df["hi"] - df["lo"]
    df["group"] = "all" if groups is None else np.asarray(groups)
    out = df.groupby("group").agg(n=("y", "size"), coverage=("covered", "mean"), mean_width=("width", "mean"))
    if groups is not None:
        tot = pd.DataFrame(
            {"n": [len(df)], "coverage": [df["covered"].mean()], "mean_width": [df["width"].mean()]},
            index=pd.Index(["all"], name="group"),
        )
        out = pd.concat([tot, out])
    return out
