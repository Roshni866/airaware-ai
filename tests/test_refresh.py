"""Tests for src/refresh.py — orchestration, logging, failure handling. Mocks the
expensive data_pipeline step so this runs fast and offline."""
import json
from unittest.mock import patch

from src.config import RESULTS


def test_refresh_skip_download_succeeds_and_logs(tmp_path, monkeypatch):
    from src import refresh as refresh_mod
    log_path = tmp_path / "refresh_log.jsonl"
    last_path = tmp_path / "last_refresh.json"
    monkeypatch.setattr(refresh_mod, "REFRESH_LOG", log_path)
    monkeypatch.setattr(refresh_mod, "LAST_REFRESH", last_path)

    code = refresh_mod.run(skip_download=True)
    assert code == 0
    assert log_path.exists()
    entry = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert entry["success"] is True
    assert last_path.exists()


def test_refresh_does_not_call_data_pipeline_when_skipped():
    from src import refresh as refresh_mod
    with patch("src.data_pipeline.run") as mock_dp:
        refresh_mod.run(skip_download=True)
        mock_dp.assert_not_called()


def test_refresh_never_calls_train():
    """Non-negotiable: refresh must never retrain the model."""
    import src.refresh as refresh_src
    source = open(refresh_src.__file__).read()
    assert "import train" not in source and "from . import train" not in source


def test_refresh_handles_feature_build_failure(tmp_path, monkeypatch):
    from src import refresh as refresh_mod
    monkeypatch.setattr(refresh_mod, "REFRESH_LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr(refresh_mod, "LAST_REFRESH", tmp_path / "last.json")

    def _boom():
        raise RuntimeError("simulated feature-build failure")

    with patch("src.features.run", side_effect=_boom):
        code = refresh_mod.run(skip_download=True)
    assert code == 1
    entry = json.loads((tmp_path / "log.jsonl").read_text().strip().splitlines()[-1])
    assert entry["success"] is False
