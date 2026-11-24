"""Configuration loading and canonical project paths."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"

DAILY_CSV = DATA_PROCESSED / "daily.csv"
FEATURES_CSV = DATA_PROCESSED / "features.csv"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES, MODELS):
        p.mkdir(parents=True, exist_ok=True)
