"""Day 3–4 — chronological benchmark and feature ablation.

Protocol (see docs/PREREGISTRATION.md):
1. For every (model, feature set), hyper-parameters are chosen by validation MAE with the
   model fitted on the training period only.
2. The chosen configuration is re-fitted on train + validation.
3. The re-fitted model predicts the calibration and test periods.
4. The deployed model is the candidate with the lowest validation MAE (feature sets A–C).

Usage:  python -m src.train
"""
from __future__ import annotations

import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import FEATURES_CSV, MODELS, RESULTS, load_config
from .features import feature_sets, modelling_rows

FEATURE_SETS = ["A", "B", "C", "C_oracle"]
SIMPLICITY = {"-": 0, "A": 1, "B": 2, "C": 3, "C_oracle": 4}


def grid(model: str) -> list[dict]:
    if model == "linear":
        return [{}]
    if model == "rf":
        g = {"max_depth": [None, 8, 14], "min_samples_leaf": [1, 5, 15], "max_features": [0.33, 0.6, 1.0]}
    elif model == "xgb":
        g = {
            "max_depth": [2, 3, 5],
            "learning_rate": [0.03, 0.1],
            "n_estimators": [200, 600],
            "min_child_weight": [1, 10],
        }
    else:
        raise ValueError(model)
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*g.values())]


def make_model(model: str, params: dict, seed: int) -> Pipeline:
    """Imputer (fitted on training rows only, inside the pipeline) + estimator."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if model == "linear":
        steps += [("scale", StandardScaler()), ("est", LinearRegression())]
    elif model == "rf":
        steps += [("est", RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=seed, **params))]
    elif model == "xgb":
        steps += [
            (
                "est",
                XGBRegressor(
                    subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
                    n_jobs=-1, random_state=seed, **params,
                ),
            )
        ]
    return Pipeline(steps)


def run() -> None:
    cfg = load_config()
    seed = cfg["random_seed"]
    f = pd.read_csv(FEATURES_CSV, parse_dates=["origin_date", "target_date"], index_col="origin_date")
    fsets = feature_sets(f.columns)
    m = modelling_rows(f)
    tr, va = m[m["split"] == "train"], m[m["split"] == "val"]
    trva = m[m["split"].isin(["train", "val"])]
    later = m[m["split"].isin(["cal", "test"])]

    preds, tuning, summary = [], [], []

    def record(df, split_name, model, fs, yhat):
        preds.append(pd.DataFrame({
            "origin_date": df.index, "target_date": df["target_date"].values, "split": split_name,
            "model": model, "feature_set": fs, "y": df["target"].values, "yhat": np.asarray(yhat),
        }))

    # Persistence (no fitting, no feature set)
    for name, df in [("val", va), ("cal", later[later.split == "cal"]), ("test", later[later.split == "test"])]:
        record(df, name, "persistence", "-", df["pm_lag0"].values)
    summary.append({"model": "persistence", "feature_set": "-", "params": "{}",
                    "val_mae": mean_absolute_error(va["target"], va["pm_lag0"])})

    for fs in FEATURE_SETS:
        cols = fsets[fs]
        for model in ["linear", "rf", "xgb"]:
            t0 = time.time()
            best = (np.inf, None, None)
            for params in grid(model):
                est = make_model(model, params, seed).fit(tr[cols], tr["target"])
                yv = est.predict(va[cols])
                mae = mean_absolute_error(va["target"], yv)
                tuning.append({"model": model, "feature_set": fs, "params": json.dumps(params), "val_mae": mae})
                if mae < best[0]:
                    best = (mae, params, yv)
            val_mae, params, yv = best
            record(va, "val", model, fs, yv)  # validation predictions from the train-only fit
            final = make_model(model, params, seed).fit(trva[cols], trva["target"])
            for name in ["cal", "test"]:
                df = later[later.split == name]
                record(df, name, model, fs, final.predict(df[cols]))
            joblib.dump({"pipeline": final, "features": cols, "model": model, "feature_set": fs, "params": params},
                        MODELS / f"{model}_{fs}.joblib")
            summary.append({"model": model, "feature_set": fs, "params": json.dumps(params), "val_mae": val_mae})
            print(f"[train] {model:6s} {fs:9s} val MAE {val_mae:6.2f}  params={params}  ({time.time()-t0:.0f}s)")

    pd.concat(preds).to_csv(RESULTS / "predictions_all.csv", index=False)
    pd.DataFrame(tuning).to_csv(RESULTS / "tuning.csv", index=False)
    summ = pd.DataFrame(summary)

    # Pre-registered selection rule
    excl = set(cfg["selection"]["exclude_feature_sets"])
    cand = summ[~summ["feature_set"].isin(excl)].copy()
    cand["simplicity"] = cand["feature_set"].map(SIMPLICITY)
    cand = cand.sort_values(["val_mae", "simplicity"])
    sel = cand.iloc[0]
    summ["selected"] = (summ["model"] == sel["model"]) & (summ["feature_set"] == sel["feature_set"])
    summ.to_csv(RESULTS / "selection.csv", index=False)
    selected = {"model": sel["model"], "feature_set": sel["feature_set"], "val_mae": float(sel["val_mae"]),
                "params": json.loads(sel["params"]), "rule": "lowest validation MAE among feature sets A-C"}
    (RESULTS / "selected_model.json").write_text(json.dumps(selected, indent=2))
    print(f"[train] selected: {selected}")


if __name__ == "__main__":
    run()
