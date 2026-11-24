# Data-quality report

*Generated 2026-11-24 08:41 UTC by `src/data_pipeline.py`.*

**Station:** New Delhi-8118 (U.S. Embassy Chanakyapuri, AirNow reference monitor) — OpenAQ location 8118, (28.63576, 77.22445), time zone Asia/Kolkata
**Period:** 2018-01-01 to 2026-11-20

## Hourly validation

| Check | Count |
|---|---:|
| Raw archive rows | 71,503 |
| PM2.5 rows | 71,503 |
| Units found | µg/m³ |
| Sensor IDs | 23534 |
| Dropped: unexpected units | 0 |
| Dropped: off-the-hour timestamps (duplicated UTC stream, 2023-07-13 to 2023-11-03) | 2,535 |
| Dropped: invalid value (-999 sentinel, repeated 1985 error value, other <0 or >999) | 4,549 |
| Dropped: exact duplicate timestamp+value | 0 |
| Timestamps with conflicting duplicates (averaged) | 0 |
| Clean hourly observations | 64,419 |

## Daily coverage

Daily mean computed over the local (IST) day; requires ≥ 18 of 24 valid hours.

| Metric | Value |
|---|---:|
| Calendar days | 3,246 |
| Observed days | 2,604 (80.22%) |
| Missing days | 642 |
| …of which had some hours but < 18 | 328 |
| Number of gaps | 180 |
| Last observed day | 2026-11-20 |

Longest gaps: 2022-05-10 (135 d); 2019-02-11 (37 d); 2023-02-16 (26 d); 2024-04-10 (17 d); 2024-03-23 (17 d)

| Year | Days | Observed | Coverage % | Mean PM2.5 (µg/m³) |
|---|---:|---:|---:|---:|
| 2018 | 365 | 277 | 75.9 | 110.2 |
| 2019 | 365 | 303 | 83.0 | 106.5 |
| 2020 | 366 | 328 | 89.6 | 92.2 |
| 2021 | 365 | 295 | 80.8 | 108.3 |
| 2022 | 365 | 209 | 57.3 | 123.0 |
| 2023 | 365 | 268 | 73.4 | 104.3 |
| 2024 | 366 | 286 | 78.1 | 109.6 |
| 2025 | 365 | 348 | 95.3 | 88.7 |
| 2026 | 324 | 290 | 89.5 | 79.6 |

## Missing-data policy

* Missing target days are dropped from training and evaluation; they are never filled.
* Lag/rolling features that fall on a missing day are left missing and imputed with the training-period median inside the model pipeline (fit on training data only).
* No interpolation across gaps; no backward fill.

Weather days missing (any variable): temp_mean=0, rh_mean=0, wind_speed_mean=0, wind_u_mean=0, wind_v_mean=0, precip_sum=0
