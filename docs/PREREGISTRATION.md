# Pre-registered evaluation protocol

Written on 2026-10-09 after the Day-1 data audit and **before** any model was fitted or any
test-period outcome was examined. Only training-period statistics (2018–2022) were inspected
when fixing the thresholds below. All values are also encoded in `config.yaml`.

## Station and target
* OpenAQ location 8118 — New Delhi, U.S. Embassy (Chanakyapuri), AirNow reference monitor.
* Target: next-day (t+1) daily mean PM2.5 (µg/m³), local IST calendar day, ≥ 18 valid hours.
* Forecast origin: end of day t. Every feature uses information up to and including day t,
  except the calendar features of the target date (known in advance) and the diagnostic
  `C_oracle` feature set (see below).

## Chronological split (no shuffling)
| Period | Dates | Use |
|---|---|---|
| Train | 2018-01-01 – 2022-12-31 | fit models |
| Validation | 2023-01-01 – 2023-12-31 | hyper-parameter and model selection |
| Calibration | 2024-01-01 – 2024-12-31 | conformal residual quantiles only |
| Test | 2025-01-01 – last available day | final evaluation, used for nothing else |

Final point models are re-fitted on Train + Validation with the selected hyper-parameters,
then calibrated on Calibration and evaluated once on Test. Rows are assigned to a period by
the **target date** (t+1), so no target value appears in two periods.

## Models and feature sets
* Models: Persistence (ŷ_{t+1} = y_t), Linear Regression, Random Forest, XGBoost.
* A — PM2.5 lags (t, t−1, t−2, t−3, t−7) + 3- and 7-day trailing mean and std.
* B — A + calendar of the target date (day-of-week, month, day-of-year; cyclic encoding).
* C — B + observed weather on the origin day t (temperature, relative humidity, wind speed,
  wind u/v components, precipitation).
* C_oracle (diagnostic, **not eligible for selection**) — C + observed weather on the target
  day t+1. It approximates a perfect next-day weather forecast and bounds what weather
  forecasts could add. It is not a deployable configuration.

## Selection rule
The deployed model is the (model, feature set) pair with the lowest **validation MAE**
among A, B, C. Ties broken toward the simpler feature set.

## High-pollution events (fixed before test evaluation)
* Primary event: daily mean PM2.5 **> 120 µg/m³** — CPCB National AQI “Very Poor” or worse.
  33.1% of training days exceed this.
* Secondary event: **> 250 µg/m³** — CPCB “Severe”. 7.1% of training days.
* Alert rules evaluated: (i) point forecast > threshold; (ii) upper 90% interval bound > threshold
  (precautionary).
* Reported: precision, recall, F1, false alarms, missed events, and MAE/bias on event days.

## Uncertainty
* Split conformal prediction, nominal 90% (α = 0.10), absolute residuals on the Calibration year.
* Secondary variant: the same procedure on log-transformed values (multiplicative intervals),
  added because PM2.5 errors are expected to scale with concentration.
* Coverage and mean width reported overall, by pollution regime and by season. Nominal
  coverage is not a guarantee under temporal dependence and drift.

## Uncertainty about metric differences
Moving-block bootstrap over test days (block = 14 days, 2,000 resamples) for MAE differences.
Reported as 95% percentile intervals, not as significance tests.
