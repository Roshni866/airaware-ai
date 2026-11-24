"""Day 4–5 — final evaluation: benchmark metrics, feature ablation, conformal
coverage, high-pollution error analysis, figures, and a time-aware bootstrap.

Nothing here is used to pick a model or a threshold: both were fixed in
docs/PREREGISTRATION.md / config.yaml before this script ever touched the test set.

Usage:  python -m src.evaluate
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import FIGURES, RESULTS, load_config
from .conformal import coverage_table, fit_absolute, fit_log, predict_interval

plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "font.size": 10})
COLORS = {"persistence": "#9aa0a6", "linear": "#4285f4", "rf": "#34a853", "xgb": "#ea4335"}


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def bias(y, yhat):
    return float(np.mean(np.asarray(yhat) - np.asarray(y)))


# --------------------------------------------------------------------------- #
# 1. Benchmark + ablation tables (every model, every feature set — including losers)
# --------------------------------------------------------------------------- #
def benchmark_table(preds: pd.DataFrame, split: str) -> pd.DataFrame:
    df = preds[preds["split"] == split]
    rows = []
    for (model, fs), g in df.groupby(["model", "feature_set"]):
        rows.append({
            "model": model, "feature_set": fs, "n": len(g),
            "mae": mae(g.y, g.yhat), "rmse": rmse(g.y, g.yhat), "bias": bias(g.y, g.yhat),
        })
    out = pd.DataFrame(rows).sort_values("mae")
    return out


def ablation_table(bench_test: pd.DataFrame) -> pd.DataFrame:
    piv = bench_test[bench_test.feature_set != "-"].pivot(index="model", columns="feature_set", values="mae")
    for pair in [("B", "A"), ("C", "B"), ("C_oracle", "C")]:
        if pair[0] in piv and pair[1] in piv:
            piv[f"delta_{pair[0]}_minus_{pair[1]}"] = piv[pair[0]] - piv[pair[1]]
    return piv.reset_index()


# --------------------------------------------------------------------------- #
# 2. Conformal intervals for the selected model
# --------------------------------------------------------------------------- #
def conformal_eval(preds: pd.DataFrame, selected: dict, cfg: dict) -> dict:
    model, fs = selected["model"], selected["feature_set"]
    sub = preds[(preds.model == model) & (preds.feature_set == fs)]
    cal, test = sub[sub.split == "cal"], sub[sub.split == "test"]
    alpha = cfg["conformal"]["alpha"]

    out = {}
    for method, fitter in [("absolute", fit_absolute), ("log", fit_log)]:
        params = fitter(cal.y, cal.yhat, alpha)
        lo, hi = predict_interval(test.yhat.values, params)
        thr = cfg["events"]["primary_threshold"]
        regime = np.where(test.y.values > thr, f">{thr}", f"<={thr}")
        season = test.target_date.astype(str).str[5:7].map(
            lambda m: {"12": "winter", "01": "winter", "02": "winter"}.get(m, "other")
        )
        cov_regime = coverage_table(test.y.values, lo, hi, groups=regime)
        cov_season = coverage_table(test.y.values, lo, hi, groups=season.values)
        out[method] = {
            "params": params,
            "overall": {"coverage": float(((test.y >= lo) & (test.y <= hi)).mean()),
                        "mean_width": float(np.mean(hi - lo)), "n": int(len(test))},
            "by_regime": cov_regime.reset_index().to_dict(orient="records"),
            "by_season": cov_season.reset_index().to_dict(orient="records"),
        }
        np.save(FIGURES.parent / f"_interval_{method}_lo.npy", lo)
        np.save(FIGURES.parent / f"_interval_{method}_hi.npy", hi)
    return out


# --------------------------------------------------------------------------- #
# 3. High-pollution event analysis (threshold fixed BEFORE seeing test results)
# --------------------------------------------------------------------------- #
def event_metrics(y_true_event, y_pred_event) -> dict:
    yt, yp = np.asarray(y_true_event), np.asarray(y_pred_event)
    tp = int(np.sum(yt & yp)); fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp)); tn = int(np.sum(~yt & ~yp))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": f1, "n_events": int(yt.sum()), "false_alarms": fp, "missed_events": fn}


def high_pollution_analysis(preds: pd.DataFrame, selected: dict, cfg: dict, interval_hi: np.ndarray) -> dict:
    model, fs = selected["model"], selected["feature_set"]
    test = preds[(preds.model == model) & (preds.feature_set == fs) & (preds.split == "test")].reset_index(drop=True)
    thr = cfg["events"]["primary_threshold"]
    thr2 = cfg["events"]["secondary_threshold"]
    event = test.y.values > thr
    point_flag = test.yhat.values > thr
    interval_flag = interval_hi > thr

    out = {
        "threshold_ugm3": thr, "secondary_threshold_ugm3": thr2,
        "n_test_days": len(test), "n_event_days": int(event.sum()),
        "point_forecast_rule": event_metrics(event, point_flag),
        "precautionary_interval_rule": event_metrics(event, interval_flag),
        "mae_on_event_days": mae(test.y[event], test.yhat[event]) if event.any() else None,
        "mae_on_nonevent_days": mae(test.y[~event], test.yhat[~event]) if (~event).any() else None,
        "bias_on_event_days": bias(test.y[event], test.yhat[event]) if event.any() else None,
        "severe_event_days": int((test.y > thr2).sum()),
        "severe_mae": mae(test.y[test.y > thr2], test.yhat[test.y > thr2]) if (test.y > thr2).any() else None,
    }
    missed = test[event & ~point_flag][["target_date", "y", "yhat"]]
    out["missed_event_dates"] = missed.assign(
        target_date=missed.target_date.astype(str), y=missed.y.round(1), yhat=missed.yhat.round(1)
    ).to_dict(orient="records")
    return out


# --------------------------------------------------------------------------- #
# 4. Time-aware (moving-block) bootstrap for MAE differences on the test period
# --------------------------------------------------------------------------- #
def block_bootstrap_mae_diff(y, a, b, block=14, n_boot=2000, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    n = len(y)
    err_a, err_b = np.abs(y - a), np.abs(y - b)
    diffs = []
    n_blocks = int(np.ceil(n / block))
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        diffs.append(err_a[idx].mean() - err_b[idx].mean())
    diffs = np.array(diffs)
    point = err_a.mean() - err_b.mean()
    return {
        "point_estimate_mae_diff": float(point),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "n_boot": n_boot, "block_days": block,
        "note": "a-b MAE difference; positive means a is worse (higher error) than b",
    }


# --------------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------------- #
def fig_actual_vs_predicted(test_sel: pd.DataFrame, lo, hi, cfg: dict, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    d = test_sel.target_date.values
    ax.fill_between(d, lo, hi, color="#34a853", alpha=0.15, label="90% conformal interval")
    ax.plot(d, test_sel.y, color="#222", lw=1, label="Observed PM2.5")
    ax.plot(d, test_sel.yhat, color="#ea4335", lw=1, label="Forecast (RF, feature set C)")
    ax.axhline(cfg["events"]["primary_threshold"], color="#9aa0a6", ls="--", lw=1,
               label=f"Event threshold ({cfg['events']['primary_threshold']} µg/m³)")
    ax.set_ylabel("PM2.5 (µg/m³)"); ax.set_title("Test period: observed vs. forecast with 90% interval")
    ax.legend(loc="upper right", fontsize=8); fig.autofmt_xdate()
    fig.savefig(path); plt.close(fig)


def fig_residuals(test_sel: pd.DataFrame, path):
    resid = test_sel.yhat.values - test_sel.y.values
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(test_sel.y, resid, s=10, alpha=0.5, color="#4285f4")
    axes[0].axhline(0, color="#222", lw=1)
    axes[0].set_xlabel("Observed PM2.5 (µg/m³)"); axes[0].set_ylabel("Forecast − observed")
    axes[0].set_title("Residuals vs. observed level")
    axes[1].hist(resid, bins=30, color="#4285f4", alpha=0.8)
    axes[1].axvline(0, color="#222", lw=1)
    axes[1].set_xlabel("Forecast − observed"); axes[1].set_title("Residual distribution")
    fig.suptitle("Test-period residuals (RF, feature set C)")
    fig.savefig(path); plt.close(fig)


def fig_ablation(piv: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(piv))
    width = 0.25
    for i, fs in enumerate(["A", "B", "C"]):
        if fs in piv:
            ax.bar(x + i * width, piv[fs], width, label=f"Set {fs}")
    ax.set_xticks(x + width); ax.set_xticklabels(piv["model"])
    ax.set_ylabel("Test MAE (µg/m³)"); ax.set_title("Feature ablation: test MAE by model and feature set")
    ax.legend()
    fig.savefig(path); plt.close(fig)


def fig_high_pollution(test_sel: pd.DataFrame, cfg: dict, path):
    thr = cfg["events"]["primary_threshold"]
    fig, ax = plt.subplots(figsize=(6, 5))
    event = test_sel.y > thr
    ax.scatter(test_sel.y[~event], test_sel.yhat[~event], s=12, alpha=0.4, color="#4285f4", label="Non-event day")
    ax.scatter(test_sel.y[event], test_sel.yhat[event], s=16, alpha=0.8, color="#ea4335", label="Event day (>thr)")
    lims = [0, max(test_sel.y.max(), test_sel.yhat.max()) * 1.05]
    ax.plot(lims, lims, color="#222", lw=1, ls="--", label="Perfect forecast")
    ax.axhline(thr, color="#9aa0a6", lw=1, ls=":"); ax.axvline(thr, color="#9aa0a6", lw=1, ls=":")
    ax.set_xlabel("Observed PM2.5 (µg/m³)"); ax.set_ylabel("Forecast PM2.5 (µg/m³)")
    ax.set_title(f"High-pollution performance (threshold = {thr} µg/m³)"); ax.legend(fontsize=8)
    fig.savefig(path); plt.close(fig)


def fig_coverage_by_regime(conf: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, method in zip(axes, ["absolute", "log"]):
        rows = conf[method]["by_regime"]
        labels = [r["group"] for r in rows]
        cov = [r["coverage"] * 100 for r in rows]
        ax.bar(labels, cov, color=["#9aa0a6" if lab == "all" else "#4285f4" for lab in labels])
        ax.axhline(90, color="#ea4335", ls="--", lw=1, label="Nominal 90%")
        ax.set_ylim(0, 100); ax.set_title(f"{method} conformal — coverage by regime"); ax.legend(fontsize=8)
        ax.set_ylabel("Empirical coverage (%)")
    fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- #
def run() -> None:
    cfg = load_config()
    preds = pd.read_csv(RESULTS / "predictions_all.csv", parse_dates=["target_date"])
    selected = json.loads((RESULTS / "selected_model.json").read_text())

    bench_val = benchmark_table(preds, "val")
    bench_test = benchmark_table(preds, "test")
    bench_val.to_csv(RESULTS / "benchmark_val.csv", index=False)
    bench_test.to_csv(RESULTS / "benchmark_test.csv", index=False)
    ablation_table(bench_test).to_csv(RESULTS / "ablation_test.csv", index=False)

    conf = conformal_eval(preds, selected, cfg)
    (RESULTS / "conformal.json").write_text(json.dumps(conf, indent=2, default=float))

    lo = np.load(RESULTS / "_interval_log_lo.npy")
    hi = np.load(RESULTS / "_interval_log_hi.npy")
    sel_test = preds[(preds.model == selected["model"]) & (preds.feature_set == selected["feature_set"])
                      & (preds.split == "test")].reset_index(drop=True)
    hp = high_pollution_analysis(preds, selected, cfg, hi)
    (RESULTS / "high_pollution.json").write_text(json.dumps(hp, indent=2, default=float))

    # Bootstrap: selected model vs persistence, on the identical test rows
    pers = preds[(preds.model == "persistence") & (preds.split == "test")].sort_values("target_date")
    sel_sorted = sel_test.sort_values("target_date")
    common = np.intersect1d(pers.target_date.values, sel_sorted.target_date.values)
    pers_c = pers[pers.target_date.isin(common)].sort_values("target_date")
    sel_c = sel_sorted[sel_sorted.target_date.isin(common)].sort_values("target_date")
    boot = block_bootstrap_mae_diff(
        sel_c.y.values, pers_c.yhat.values, sel_c.yhat.values,
        block=cfg["evaluation"]["bootstrap_block_days"], n_boot=cfg["evaluation"]["bootstrap_resamples"],
    )
    boot["comparison"] = "persistence_minus_selected_model"
    (RESULTS / "bootstrap_mae_diff.json").write_text(json.dumps(boot, indent=2))

    fig_actual_vs_predicted(sel_test, lo, hi, cfg, FIGURES / "actual_vs_predicted.png")
    fig_residuals(sel_test, FIGURES / "residuals.png")
    fig_ablation(ablation_table(bench_test), FIGURES / "ablation.png")
    fig_high_pollution(sel_test, cfg, FIGURES / "high_pollution_scatter.png")
    fig_coverage_by_regime(conf, FIGURES / "coverage_by_regime.png")

    for p in RESULTS.glob("_interval_*.npy"):
        p.unlink()

    print("\n=== TEST benchmark (all models, all feature sets) ===")
    print(bench_test.to_string(index=False))
    print(f"\nSelected: {selected['model']} / {selected['feature_set']}")
    print(f"Conformal (absolute) overall coverage: {conf['absolute']['overall']}")
    print(f"High-pollution point-rule: {hp['point_forecast_rule']}")
    print(f"Bootstrap persistence-vs-selected MAE diff: {boot}")


if __name__ == "__main__":
    run()
