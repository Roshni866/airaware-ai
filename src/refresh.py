"""Refresh orchestration: pull latest observations -> rebuild features -> forecast.
Never retrains the model (that's a separate, manual step: `src/train.py`).

Usage:  python -m src.refresh [--skip-download]

Non-zero exit on failure so a scheduler can alert on it. Appends to
results/refresh_log.jsonl and updates results/last_refresh.json on success.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone

from .config import RESULTS, ensure_dirs
from .predict import generate_forecast

REFRESH_LOG = RESULTS / "refresh_log.jsonl"
LAST_REFRESH = RESULTS / "last_refresh.json"


def _append_log(entry: dict) -> None:
    ensure_dirs()
    with open(REFRESH_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def run(skip_download: bool = False) -> int:
    started = datetime.now(timezone.utc)
    steps: list[dict] = []

    def step(name, fn):
        t0 = datetime.now(timezone.utc)
        try:
            fn()
            steps.append({"step": name, "ok": True, "seconds": (datetime.now(timezone.utc) - t0).total_seconds()})
        except Exception as e:  # noqa: BLE001
            steps.append({
                "step": name, "ok": False, "error": str(e),
                "traceback": traceback.format_exc(), "seconds": (datetime.now(timezone.utc) - t0).total_seconds(),
            })
            raise

    try:
        if not skip_download:
            from . import data_pipeline
            step("data_pipeline", data_pipeline.run)
        from . import features as features_mod
        step("features", features_mod.run)

        forecast_holder = {}

        def _predict():
            forecast_holder["result"] = generate_forecast()

        step("predict", _predict)
        forecast = forecast_holder["result"]
        ok = forecast.status in ("ok", "stale_data")  # a stale-but-valid forecast is not a pipeline failure
        entry = {
            "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
            "success": ok, "steps": steps, "forecast_status": forecast.status, "forecast_reason": forecast.reason,
            "origin_date": forecast.origin_date, "target_date": forecast.target_date,
            "point_forecast": forecast.point_forecast,
        }
        _append_log(entry)
        if ok:
            LAST_REFRESH.write_text(json.dumps(entry, indent=2, default=str))
        if not ok:
            print(f"[refresh] FAILED: forecast status = {forecast.status} ({forecast.reason})", file=sys.stderr)
            return 1
        print(f"[refresh] OK — origin {forecast.origin_date}, forecast {forecast.point_forecast} µg/m³ "
              f"for {forecast.target_date} (status={forecast.status})")
        return 0
    except Exception as e:  # noqa: BLE001
        entry = {
            "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
            "success": False, "steps": steps, "error": str(e),
        }
        _append_log(entry)
        print(f"[refresh] FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true", help="reuse cached data/processed/daily.csv as-is")
    sys.exit(run(skip_download=ap.parse_args().skip_download))
