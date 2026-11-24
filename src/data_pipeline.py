"""Day 1 — data acquisition, validation and daily alignment.

Sources
-------
* PM2.5: OpenAQ public data archive on AWS S3 (``openaq-data-archive``), one
  gzipped CSV per location per day. No API key is required for the archive.
  OpenAQ timestamps are the END of the hourly averaging period, in local time
  with an explicit UTC offset.
* Weather: Open-Meteo Historical Weather API (reanalysis-based), hourly, in
  local time (Asia/Kolkata).

Daily definition
----------------
A local calendar day D (IST) contains the 24 hourly PM2.5 averages whose
averaging PERIOD STARTS on D, i.e. hour-ending stamps D 01:00 .. D+1 00:00.
The daily mean is reported only if at least ``min_hours_per_day`` valid hours
exist; otherwise the day is missing. Missing days are NEVER filled from
future observations or interpolated.

Usage:  python -m src.data_pipeline [--refresh]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import DAILY_CSV, DATA_RAW, RESULTS, ensure_dirs, load_config

S3_BASE = "https://openaq-data-archive.s3.amazonaws.com/"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
SESSION = requests.Session()


# --------------------------------------------------------------------------- #
# OpenAQ archive
# --------------------------------------------------------------------------- #
def list_archive_keys(location_id: int, year: int) -> list[str]:
    """List all daily archive objects for one location and year."""
    keys, token = [], None
    while True:
        params = {
            "list-type": "2",
            "prefix": f"records/csv.gz/locationid={location_id}/year={year}/",
            "max-keys": "1000",
        }
        if token:
            params["continuation-token"] = token
        r = SESSION.get(S3_BASE, params=params, timeout=60)
        r.raise_for_status()
        keys += re.findall(r"<Key>(.*?)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", r.text)
        if not m:
            return keys
        token = m.group(1)


_local = threading.local()


def _thread_session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def _download(key: str, dest_dir: Path) -> Path | None:
    """Download one archive object (cached). Transient errors (incl. spurious 404s
    seen under concurrent access) are retried with backoff; returns None if the
    object stays unavailable, which is then reported as a missing day."""
    dest = dest_dir / Path(key).name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for attempt in range(6):
        try:
            r = _thread_session().get(S3_BASE + key, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return dest
        except requests.RequestException:
            time.sleep(min(2**attempt, 20) + random.random())
    print(f"[openaq] WARNING: could not download {key}")
    return None


def download_openaq(cfg: dict, refresh: bool = False) -> pd.DataFrame:
    """Download (with on-disk cache) and concatenate the station's hourly records."""
    loc = cfg["station"]["openaq_location_id"]
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"]) if cfg["data"]["end_date"] else pd.Timestamp.today()
    raw_dir = DATA_RAW / f"openaq_location_{loc}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    keys: list[str] = []
    # Include the day before start: its last hour-ending stamp (00:00) is not needed,
    # but the day *after* end may hold the 00:00 stamp closing the final day.
    for year in range(start.year, end.year + 1):
        listing_cache = raw_dir / f"_keys_{year}.json"
        if listing_cache.exists() and not refresh and year < dt.date.today().year:
            year_keys = json.loads(listing_cache.read_text())
        else:
            year_keys = list_archive_keys(loc, year)
            listing_cache.write_text(json.dumps(year_keys))
        keys += year_keys
    print(f"[openaq] {len(keys)} daily archive files listed for location {loc}")

    workers = cfg["data"].get("download_workers", 8)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        paths = list(ex.map(lambda k: _download(k, raw_dir), keys))
    failed = sum(p is None for p in paths)
    if failed:
        print(f"[openaq] {failed} files unavailable after retries (re-run to try again)")

    frames = []
    for p in (p for p in paths if p is not None):
        with gzip.open(p, "rb") as f:
            content = f.read()
        if content.strip():
            frames.append(pd.read_csv(io.BytesIO(content)))
    df = pd.concat(frames, ignore_index=True)
    print(f"[openaq] {len(df):,} raw rows loaded")
    return df


# --------------------------------------------------------------------------- #
# Validation + daily aggregation
# --------------------------------------------------------------------------- #
def clean_hourly(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Filter to PM2.5, validate units/values, resolve duplicates. Returns (hourly, report)."""
    rep: dict = {"raw_rows": int(len(raw))}
    tz = cfg["station"]["timezone"]

    df = raw[raw["parameter"] == cfg["station"]["parameter"]].copy()
    rep["rows_pm25"] = int(len(df))
    rep["units_found"] = sorted(df["units"].dropna().unique().tolist())
    rep["sensor_ids"] = sorted(int(s) for s in df["sensors_id"].unique())
    rep["location_names"] = sorted(df["location"].unique().tolist())
    bad_units = ~df["units"].isin(["µg/m³", "ug/m3", "µg/m3"])
    rep["rows_bad_units_dropped"] = int(bad_units.sum())
    df = df[~bad_units]

    # Parse hour-ending timestamps (with offset) -> IST, then shift to period start.
    ts_end = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(tz)
    df["period_end"] = ts_end
    df["period_start"] = ts_end - pd.Timedelta(hours=1)

    # Hourly averages end on the full hour. In mid-2023 the archive also carries
    # ':30'-stamped records that reproduce the same series offset by 5.5 h (a UTC
    # stream mislabeled as IST). Off-the-hour records are therefore discarded.
    off_hour = (ts_end.dt.minute != 0) | (ts_end.dt.second != 0)
    rep["rows_off_hour_timestamp_dropped"] = int(off_hour.sum())
    rep["off_hour_dates"] = (
        [str(ts_end[off_hour].min().date()), str(ts_end[off_hour].max().date())] if off_hour.any() else []
    )
    df = df[~off_hour]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    vmin, vmax = cfg["data"]["valid_min"], cfg["data"]["valid_max"]
    invalid = df["value"].isna() | (df["value"] < vmin) | (df["value"] > vmax)
    rep["rows_invalid_value_dropped"] = int(invalid.sum())
    rep["invalid_value_examples"] = df.loc[invalid, "value"].value_counts().head(5).to_dict()
    df = df[~invalid]

    exact_dup = df.duplicated(subset=["period_end", "value"])
    rep["rows_exact_duplicate_dropped"] = int(exact_dup.sum())
    df = df[~exact_dup]
    conflict = df.duplicated(subset=["period_end"], keep=False)
    rep["timestamps_with_conflicting_values_averaged"] = int(df.loc[conflict, "period_end"].nunique())
    hourly = (
        df.groupby("period_start", as_index=True)["value"].mean().sort_index().to_frame("pm25")
    )
    rep["clean_hourly_rows"] = int(len(hourly))
    return hourly, rep


def aggregate_daily(hourly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Local-day mean with a minimum-completeness rule. Index: naive local date."""
    min_hours = cfg["data"]["min_hours_per_day"]
    local_date = hourly.index.tz_localize(None).normalize()
    g = hourly.groupby(local_date)["pm25"]
    daily = pd.DataFrame({"pm25_mean": g.mean(), "n_hours": g.count(), "pm25_max_hourly": g.max()})
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"]) if cfg["data"]["end_date"] else daily.index.max()
    # The final local day is only complete once its last hour (ending 00:00 next day) exists.
    full_index = pd.date_range(start, end, freq="D", name="date")
    daily = daily.reindex(full_index)
    daily["n_hours"] = daily["n_hours"].fillna(0).astype(int)
    insufficient = daily["n_hours"] < min_hours
    daily.loc[insufficient, ["pm25_mean", "pm25_max_hourly"]] = np.nan
    return daily


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def fetch_weather(cfg: dict, start: pd.Timestamp, end: pd.Timestamp, refresh: bool = False) -> pd.DataFrame:
    """Hourly weather in local time, fetched year by year and cached."""
    st = cfg["station"]
    out_dir = DATA_RAW / "open_meteo"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    this_year = dt.date.today().year
    for year in range(start.year, end.year + 1):
        cache = out_dir / f"weather_hourly_{year}.csv"
        if cache.exists() and not refresh and year < this_year:
            cached = pd.read_csv(cache, parse_dates=["time"])
            if set(cfg["weather"]["hourly_vars"]) <= set(cached.columns):
                frames.append(cached)
                continue
        y0 = max(pd.Timestamp(f"{year}-01-01"), start)
        y1 = min(pd.Timestamp(f"{year}-12-31"), end)
        params = {
            "latitude": st["latitude"],
            "longitude": st["longitude"],
            "start_date": y0.strftime("%Y-%m-%d"),
            "end_date": y1.strftime("%Y-%m-%d"),
            "hourly": ",".join(cfg["weather"]["hourly_vars"]),
            "timezone": st["timezone"],
            "wind_speed_unit": "ms",
        }
        for attempt in range(4):
            try:
                r = SESSION.get(WEATHER_URL, params=params, timeout=180)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
        js = r.json()
        w = pd.DataFrame(js["hourly"])
        w["time"] = pd.to_datetime(w["time"])
        w.to_csv(cache, index=False)
        frames.append(w)
        print(f"[weather] {year}: {len(w)} hourly rows")
    return pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")


def aggregate_weather_daily(w: pd.DataFrame) -> pd.DataFrame:
    """Daily (local) weather summary. Wind direction is averaged as a vector."""
    w = w.set_index("time")
    rad = np.deg2rad(w["wind_direction_10m"])
    # Meteorological convention: direction the wind blows FROM.
    w["wind_u"] = -w["wind_speed_10m"] * np.sin(rad)
    w["wind_v"] = -w["wind_speed_10m"] * np.cos(rad)
    day = w.index.normalize()
    agg = pd.DataFrame(
        {
            "temp_mean": w["temperature_2m"].groupby(day).mean(),
            "rh_mean": w["relative_humidity_2m"].groupby(day).mean(),
            "wind_speed_mean": w["wind_speed_10m"].groupby(day).mean(),
            "wind_u_mean": w["wind_u"].groupby(day).mean(),
            "wind_v_mean": w["wind_v"].groupby(day).mean(),
            "precip_sum": w["precipitation"].groupby(day).sum(min_count=18),
            "weather_hours": w["temperature_2m"].groupby(day).count(),
        }
    )
    agg.index.name = "date"
    agg.loc[agg["weather_hours"] < 18, agg.columns.drop("weather_hours")] = np.nan
    return agg


# --------------------------------------------------------------------------- #
# Quality report
# --------------------------------------------------------------------------- #
def quality_report(daily: pd.DataFrame, hourly_rep: dict, cfg: dict) -> dict:
    s = daily["pm25_mean"]
    obs = s.notna()
    # gap lengths in consecutive missing days
    runs = (obs != obs.shift()).cumsum()
    gaps = (~obs).groupby(runs).sum()
    gaps = gaps[gaps > 0]
    gap_starts = {int(k): str(daily.index[runs == k][0].date()) for k in gaps.index}
    longest = gaps.sort_values(ascending=False).head(5)

    per_year = daily.assign(observed=obs).groupby(daily.index.year).agg(
        days=("observed", "size"), observed_days=("observed", "sum"), mean_pm25=("pm25_mean", "mean")
    )
    per_year["coverage_pct"] = (100 * per_year["observed_days"] / per_year["days"]).round(1)

    weather_cols = ["temp_mean", "rh_mean", "wind_speed_mean", "wind_u_mean", "wind_v_mean", "precip_sum"]
    rep = {
        "station": cfg["station"],
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "period": [str(daily.index.min().date()), str(daily.index.max().date())],
        "hourly": hourly_rep,
        "daily": {
            "calendar_days": int(len(daily)),
            "observed_days": int(obs.sum()),
            "missing_days": int((~obs).sum()),
            "coverage_pct": round(100 * obs.mean(), 2),
            "min_hours_rule": cfg["data"]["min_hours_per_day"],
            "days_dropped_for_incomplete_hours": int(((daily["n_hours"] > 0) & ~obs).sum()),
            "number_of_gaps": int(len(gaps)),
            "longest_gaps": [
                {"start": gap_starts[int(k)], "days": int(v)} for k, v in longest.items()
            ],
            "last_observed_day": str(s.dropna().index.max().date()),
            "pm25_summary": {k: round(float(v), 2) for k, v in s.describe().items()},
        },
        "per_year": per_year.reset_index(names="year").round(2).to_dict(orient="records"),
        "weather_missing_days": {c: int(daily[c].isna().sum()) for c in weather_cols if c in daily},
    }
    return rep


def write_quality_markdown(rep: dict, path: Path) -> None:
    h, d = rep["hourly"], rep["daily"]
    lines = [
        "# Data-quality report",
        "",
        f"*Generated {rep['retrieved_utc']} by `src/data_pipeline.py`.*",
        "",
        f"**Station:** {rep['station']['name']} — OpenAQ location {rep['station']['openaq_location_id']}, "
        f"({rep['station']['latitude']}, {rep['station']['longitude']}), time zone {rep['station']['timezone']}",
        f"**Period:** {rep['period'][0]} to {rep['period'][1]}",
        "",
        "## Hourly validation",
        "",
        "| Check | Count |",
        "|---|---:|",
        f"| Raw archive rows | {h['raw_rows']:,} |",
        f"| PM2.5 rows | {h['rows_pm25']:,} |",
        f"| Units found | {', '.join(h['units_found'])} |",
        f"| Sensor IDs | {', '.join(map(str, h['sensor_ids']))} |",
        f"| Dropped: unexpected units | {h['rows_bad_units_dropped']:,} |",
        f"| Dropped: off-the-hour timestamps (duplicated UTC stream, {' to '.join(h['off_hour_dates'])}) "
        f"| {h['rows_off_hour_timestamp_dropped']:,} |",
        f"| Dropped: invalid value (-999 sentinel, repeated 1985 error value, other <0 or >999) | {h['rows_invalid_value_dropped']:,} |",
        f"| Dropped: exact duplicate timestamp+value | {h['rows_exact_duplicate_dropped']:,} |",
        f"| Timestamps with conflicting duplicates (averaged) | {h['timestamps_with_conflicting_values_averaged']:,} |",
        f"| Clean hourly observations | {h['clean_hourly_rows']:,} |",
        "",
        "## Daily coverage",
        "",
        f"Daily mean computed over the local (IST) day; requires ≥ {d['min_hours_rule']} of 24 valid hours.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Calendar days | {d['calendar_days']:,} |",
        f"| Observed days | {d['observed_days']:,} ({d['coverage_pct']}%) |",
        f"| Missing days | {d['missing_days']:,} |",
        f"| …of which had some hours but < {d['min_hours_rule']} | {d['days_dropped_for_incomplete_hours']:,} |",
        f"| Number of gaps | {d['number_of_gaps']} |",
        f"| Last observed day | {d['last_observed_day']} |",
        "",
        "Longest gaps: " + "; ".join(f"{g['start']} ({g['days']} d)" for g in d["longest_gaps"]),
        "",
        "| Year | Days | Observed | Coverage % | Mean PM2.5 (µg/m³) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rep["per_year"]:
        lines.append(
            f"| {r['year']} | {r['days']} | {r['observed_days']} | {r['coverage_pct']} | {r['mean_pm25']:.1f} |"
        )
    lines += [
        "",
        "## Missing-data policy",
        "",
        "* Missing target days are dropped from training and evaluation; they are never filled.",
        "* Lag/rolling features that fall on a missing day are left missing and imputed with the "
        "training-period median inside the model pipeline (fit on training data only).",
        "* No interpolation across gaps; no backward fill.",
        "",
        "Weather days missing (any variable): "
        + ", ".join(f"{k}={v}" for k, v in rep["weather_missing_days"].items()),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_doc(cfg: dict, rep: dict) -> None:
    loc = cfg["station"]["openaq_location_id"]
    text = f"""# Raw data sources

Retrieved: {rep['retrieved_utc']}

## PM2.5 — OpenAQ archive
* Bucket listing: {S3_BASE}?list-type=2&prefix=records/csv.gz/locationid={loc}/
* Files: `openaq_location_{loc}/location-{loc}-YYYYMMDD.csv.gz` (one per day, hourly rows)
* Station: {cfg['station']['name']}
* Original provider: U.S. Department of State AirNow program (via OpenAQ).
* Terms: OpenAQ platform terms (https://openaq.org/terms) and the provider's terms.
  U.S. government AirNow data are generally in the public domain; confirm before redistribution.
* Timestamps: hour-ENDING, local time with +05:30 offset. Units: µg/m³.
* Documentation: https://docs.openaq.org/ · https://docs.openaq.org/aws/about

## Weather — Open-Meteo Historical Weather API
* Endpoint: {WEATHER_URL}
* Variables: {', '.join(cfg['weather']['hourly_vars'])}
* Underlying data: reanalysis (ERA5 family, with operational analysis for the most recent days).
* License: Open-Meteo data CC BY 4.0 (attribution: "Weather data by Open-Meteo.com");
  ERA5 under the Copernicus licence.
* IMPORTANT: these are OBSERVED/REANALYSIS values, not forecasts that would have been
  available at prediction time. See REPORT.md, "Limitations".
"""
    (DATA_RAW / "SOURCES.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
def run(refresh: bool = False) -> pd.DataFrame:
    ensure_dirs()
    cfg = load_config()
    raw = download_openaq(cfg, refresh=refresh)
    hourly, hrep = clean_hourly(raw, cfg)
    hourly.to_csv(DATA_RAW.parent / "processed" / "hourly_pm25.csv.gz")
    daily = aggregate_daily(hourly, cfg)

    w = fetch_weather(cfg, daily.index.min(), daily.index.max() + pd.Timedelta(days=1), refresh=refresh)
    wd = aggregate_weather_daily(w)
    daily = daily.join(wd, how="left")
    daily.index.name = "date"
    daily.to_csv(DAILY_CSV)

    rep = quality_report(daily, hrep, cfg)
    (RESULTS / "data_quality.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    write_quality_markdown(rep, RESULTS / "data_quality.md")
    write_source_doc(cfg, rep)
    d = rep["daily"]
    print(
        f"[daily] {d['observed_days']}/{d['calendar_days']} days observed ({d['coverage_pct']}%), "
        f"last observed {d['last_observed_day']}"
    )
    return daily


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-list and re-fetch the current year")
    run(refresh=ap.parse_args().refresh)
