# AirAware AI — Research Report

**Uncertainty-aware next-day PM2.5 forecasting and high-pollution error analysis**
*New Delhi-8118 (U.S. Embassy Chanakyapuri, AirNow reference monitor) · generated 2026-10-09*

> *AirAware AI is a research prototype for uncertainty-aware next-day PM2.5 forecasting.
> Unlike a dashboard that reports only a point estimate, it evaluates forecast accuracy
> against a persistence baseline, measures the incremental value of weather features, reports
> prediction-interval coverage, and analyzes errors during high-pollution periods. Its
> contribution is transparent, reproducible evaluation — not an unverified claim of superior
> operational accuracy.*

All thresholds, splits and the model-selection rule were fixed **before** the test period was
evaluated; see `docs/PREREGISTRATION.md`. This report is generated from the saved outputs in
`results/` by re-running `src/data_pipeline.py → src/features.py → src/train.py →
src/evaluate.py` in order (also `make reproduce`).

## 1. Data

| | |
|---|---|
| PM2.5 source | OpenAQ archive (AWS S3), hourly, OpenAQ location 8118 |
| Weather source | Open-Meteo Historical Weather API (reanalysis), hourly |
| Period used | 2018-01-01 – 2026-09-25 (3,189 calendar days) |
| Daily coverage | 2,569 / 3,189 days (80.6%); requires ≥18 of 24 valid hours |
| Longest gap | 135 days, starting 2022-05-10 (2022 overall coverage: 57.3%) |

Two data-quality issues were found and corrected during validation (see
`results/data_quality.md` for full counts):
* **Off-the-hour duplicate stream.** From 2023-07-13 to 2023-11-03 the archive also contains
  `:30`-stamped rows that duplicate the hourly series at a different alignment; 2,535 such rows
  were dropped, keeping only on-the-hour records.
* **Sentinel/error values.** 4,549 hourly rows carried a `-999` missing-data sentinel or a
  repeated `1985.0` error code (plus a handful of other out-of-range values) and were dropped
  as invalid, not treated as real readings.
* **Boundary-layer height excluded.** Open-Meteo's `boundary_layer_height` is null for all of
  2024-01 through 2024-06 across every reanalysis model tried — a gap that falls inside the
  calibration period — so it was dropped from the weather feature set rather than imputed.

No missing day was ever filled or interpolated; gaps are visible as missing rows throughout.

## 2. Pre-registered protocol summary

| Period | Target dates | Rows usable | Purpose |
|---|---|---:|---|
| Train | 2018-01-02 – 2022-12-31 | 1,279 | fit models |
| Validation | 2023-01-01 – 2023-12-31 | 244 | model/hyper-parameter selection |
| Calibration | 2024-01-01 – 2024-12-31 | 275 | conformal residual quantiles only |
| Test | 2025-01-01 – 2026-09-25 | 594 | final evaluation (touched once) |

Event threshold (fixed from training-period CPCB categories, not test outcomes): **>120 µg/m³**
("Very Poor" or worse; 33.1% of training days). Secondary: **>250 µg/m³** ("Severe").

## 3. H1 — Do ML models beat a persistence baseline?

**Supported.** Every fitted model beats persistence on the untouched test set.

| Model | Feature set | Test MAE | Test RMSE | Bias |
|---|---|---:|---:|---:|
| **Random Forest** | **C (lags+calendar+weather)** | **17.08** | 26.60 | +4.59 |
| XGBoost | B | 17.24 | 27.38 | +2.93 |
| XGBoost | C | 17.29 | 26.59 | +5.20 |
| Random Forest | B | 17.30 | 27.62 | +2.52 |
| Random Forest | C_oracle (diagnostic) | 17.32 | 26.73 | +6.98 |
| Linear Regression | A | 17.80 | 28.52 | +2.00 |
| Random Forest | A | 17.83 | 28.38 | +2.32 |
| **Persistence** | **—** | **18.24** | **30.05** | **−0.10** |
| Linear Regression | C | 18.86 | 27.87 | +5.46 |
| Linear Regression | C_oracle | 20.11 | 28.03 | +9.71 |

Full table (all 13 rows): `results/benchmark_test.csv`.

**Selected model** (pre-registered rule: lowest validation MAE among feature sets A–C, ties to
the simpler set): **Random Forest, feature set C** (validation MAE 22.03; test MAE 17.08).

A moving-block bootstrap (14-day blocks, 2,000 resamples) on the MAE difference between
persistence and the selected model over the identical test days gives:

* Persistence − selected MAE = **+1.16 µg/m³**, 95% CI **[0.34, 1.96]** — the interval
  excludes zero, so the improvement over persistence is not explainable by test-period
  sampling noise alone under this resampling scheme (not a formal significance test).

## 4. H2 — Does adding weather help beyond pollution history and calendar?

**Weakly supported for the tree models; not supported for linear regression.**

| Model | A (lags) | B (+calendar) | C (+weather) | Δ B−A | Δ C−B |
|---|---:|---:|---:|---:|---:|
| Linear | 17.80 | 18.06 | 18.86 | +0.26 | +0.80 |
| Random Forest | 17.83 | 17.30 | **17.08** | **−0.53** | **−0.22** |
| XGBoost | 17.88 | 17.24 | 17.29 | **−0.64** | +0.05 |

For the tree-based models, calendar features deliver most of the gain over lags-only, and
weather adds a further small improvement for Random Forest only. Linear regression gets worse
with more features (likely unmodeled nonlinearity/collinearity in the raw weather variables).
The `C_oracle` diagnostic — using the target day's own observed weather instead of the origin
day's — improves Random Forest only marginally further (17.08 → 17.32 is actually *worse*
on test, though best on validation at 21.45 vs 22.03), suggesting that, at this station,
next-day forecast-time weather accuracy is not the main bottleneck: **recent pollution
persistence and seasonality explain most of what weather would add.**
See `results/ablation_test.csv` and `results/figures/ablation.png`.

## 5. H3 — Are the prediction intervals reliable?

Split conformal, nominal 90% (α = 0.10), calibrated on 2024, evaluated once on test.

| Method | Overall coverage | Mean width (µg/m³) | Coverage on event days (>120) | Coverage on non-event days |
|---|---:|---:|---:|---:|
| Absolute residuals | 95.8% | 107.8 | **84.8%** | 98.3% |
| Log-residuals (multiplicative) | 93.4% | **93.7** | **95.5%** | 92.9% |

Overall coverage is **conservative** for both methods (above the 90% nominal target), but the
two methods fail differently by regime. The absolute-residual interval, with one fixed width
for every day, **under-covers exactly on the high-pollution days that matter most (84.8%,
below nominal)** while over-covering ordinary days (98.3%, needlessly wide). The log-space
interval — width scaling with the forecast level — fixes this: it is both **narrower overall**
(93.7 vs. 107.8 µg/m³) and keeps event-day coverage at nominal (95.5%). **The log-space
conformal method is therefore the better-behaved choice for this application**, despite the
project plan treating absolute residuals as the default. Coverage by regime and by season is
broken out in `results/conformal.json` and plotted in
`results/figures/coverage_by_regime.png`; empirical coverage, not the nominal level, is the
number that should be trusted, since conformal guarantees assume exchangeability that
time-series data only approximately satisfies.

## 6. H4 — Does average performance hide high-pollution errors?

**Supported, substantially.**

| | Non-event days | Event days (>120 µg/m³) | Severe days (>250 µg/m³) |
|---|---:|---:|---:|
| n | 482 | 112 | 23 |
| MAE (µg/m³) | **12.6** | **36.5** | **50.5** |
| Bias | — | −1.7 (slight under-forecast) | — |

Average test MAE (17.08) is driven down by the many low-pollution days; **error on event days
is nearly 3× higher**, and on severe days nearly 4× higher, than on ordinary days.

Event detection (point forecast > 120 µg/m³, against 112 actual event days):

| Rule | Precision | Recall | F1 | False alarms | Missed events |
|---|---:|---:|---:|---:|---:|
| Point forecast > threshold | 78.9% | 93.8% | 85.7% | 28 | **7** |
| Upper 90% interval bound > threshold (precautionary, log-space interval) | 45.5% | **99.1%** | 62.4% | 133 | **1** |

The precautionary rule (flag whenever the interval's upper bound crosses the threshold) misses
almost no events at the cost of more false alarms — a usable trade-off for a health-warning
system that prefers over-alerting to missing a severe day. (With the absolute-residual
interval this rule had 0 missed events but 185 false alarms; the better-calibrated log-space
interval trades 1 additional missed event for 52 fewer false alarms.) The 7 missed events
under the point rule are listed in `results/high_pollution.json → missed_event_dates`; all are
within 10–60 µg/m³ of the threshold except one (2025-05-15, observed 211 vs. forecast 70 — a
sharp unforecast spike).

## 7. Figures

| File | Shows |
|---|---|
| `results/figures/actual_vs_predicted.png` | Test-period time series: observed, forecast, 90% interval |
| `results/figures/residuals.png` | Residual vs. level, and residual distribution |
| `results/figures/ablation.png` | Test MAE by model × feature set |
| `results/figures/high_pollution_scatter.png` | Observed vs. forecast, event days highlighted |
| `results/figures/coverage_by_regime.png` | Empirical coverage by pollution regime, both conformal methods |

## 8. Limitations

* Single station, single city; no claim of generalization elsewhere.
* Weather features used here are **observed/reanalysis** values, not the forecasts that would
  be available at prediction time in a live system; a deployed version would need weather
  forecasts, whose own error would reduce (not necessarily eliminate) the gains shown here.
  The `C_oracle` diagnostic bounds the best case and shows it is close to feature set C's
  result — weather-forecast error is not expected to be the dominant limitation.
* ~19% of days are missing, concentrated in a few long gaps (worst: 2022, 57% coverage);
  results could shift with a more complete record.
* Conformal coverage is empirical and measured once; no claim that 90% nominal will hold going
  forward under further distribution drift.
* No comparison to CPCB, SAFAR or IQAir forecasts was performed — matching dates, horizons and
  locations for such a comparison was out of scope for a one-week project — so no claim of
  superior accuracy is made.
* This is a research prototype; it issues no medical or clinical guidance.

## 9. Reproduction

```bash
pip install -r requirements.txt
python -m pytest -q                 # leakage + conformal unit tests
python -m src.data_pipeline         # Day 1 (downloads are cached under data/raw/)
python -m src.features              # Day 2
python -m src.train                 # Days 3-4
python -m src.evaluate              # Days 4-5
streamlit run app.py                # Day 6
```

All numbers in this report were produced by exactly this sequence on 2026-10-09 and are
reproduced bit-for-bit on re-run against the cached raw data (random_seed=42 throughout).
