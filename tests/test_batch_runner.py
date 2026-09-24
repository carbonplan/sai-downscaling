"""Unit tests for the batch_runner entry point with DownscalingPipeline mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from saidownscale.batch_manifest import write_manifest
from saidownscale.batch_runner import _load_config_dict, run_stage
from saidownscale.downscaling_config import DownscalingConfig

_MINIMAL_CONFIG = {
    "gcm": "CESM2-WACCM6",
    "downscaling_method": "BCSD",
    "variable": "tas",
    "ensemble_member": "r1i1p1f1",
    "scenario": "ssp245",
    "predict_period_start": 2015,
    "predict_period_end": 2100,
}
_STAGES = ("prepare_observations", "fit_historical", "transform_scenario")
_CONFIG_ENV_VARS = ("CONFIG_JSON", "CONFIG_MANIFEST_URI", "AWS_BATCH_JOB_ARRAY_INDEX")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_run_stage_rejects_bad_config_json(monkeypatch, subtests):
    cases = {
        "empty string": "",
        "invalid json": "not-valid-json{",
        "invalid fields": json.dumps(
            {"gcm": "X", "variable": "sfcWind", "ensemble_member": "r1i1p1f1"}
        ),
    }
    for name, value in cases.items():
        with subtests.test(name):
            monkeypatch.setenv("CONFIG_JSON", value)
            with pytest.raises(ValueError, match="CONFIG_JSON" if not value else None):
                run_stage("prepare_observations")


def test_run_stage_routes_to_the_pipeline_method_and_returns_its_path(
    monkeypatch, capsys, subtests
):
    monkeypatch.setenv("CONFIG_JSON", json.dumps(_MINIMAL_CONFIG))
    for stage in _STAGES:
        with subtests.test(stage=stage):
            mock_pipeline = MagicMock()
            getattr(mock_pipeline, stage).return_value = f"s3://bucket/{stage}.icechunk"
            with patch(
                "saidownscale.batch_runner.DownscalingPipeline", return_value=mock_pipeline
            ) as MockPipeline:
                result = run_stage(stage)

            MockPipeline.assert_called_once()
            config = MockPipeline.call_args.args[0]
            assert isinstance(config, DownscalingConfig)
            assert (config.gcm, config.variable, config.ensemble_member) == (
                "CESM2-WACCM6",
                "tas",
                "r1i1p1f1",
            )
            for other in _STAGES:
                assert getattr(mock_pipeline, other).call_count == (other == stage)
            assert result == f"s3://bucket/{stage}.icechunk"
            assert result in capsys.readouterr().out


def test_run_stage_rejects_an_unknown_stage(monkeypatch):
    monkeypatch.setenv("CONFIG_JSON", json.dumps(_MINIMAL_CONFIG))
    with patch("saidownscale.batch_runner.DownscalingPipeline", return_value=MagicMock()):
        with pytest.raises(ValueError, match="Unknown stage"):
            run_stage("nonexistent_stage")


def test_load_config_dict_prefers_config_json_over_manifest(monkeypatch, tmp_path):
    uri = str(tmp_path / "manifest.json")
    write_manifest(uri, "fit_historical", [{"variable": "pr"}])
    monkeypatch.setenv("CONFIG_JSON", json.dumps({"variable": "tas"}))
    monkeypatch.setenv("CONFIG_MANIFEST_URI", uri)
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "0")
    assert _load_config_dict() == {"variable": "tas"}


def test_load_config_dict_reads_manifest_entry_by_array_index(monkeypatch, tmp_path):
    uri = str(tmp_path / "manifest.json")
    write_manifest(uri, "fit_historical", [{"variable": "tas"}, {"variable": "pr"}])
    monkeypatch.setenv("CONFIG_MANIFEST_URI", uri)
    monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "1")
    assert _load_config_dict() == {"variable": "pr"}


def test_load_config_dict_raises_without_a_complete_source(monkeypatch, tmp_path, subtests):
    with subtests.test("neither source"):
        with pytest.raises(ValueError, match="CONFIG_JSON"):
            _load_config_dict()
    with subtests.test("manifest without array index"):
        monkeypatch.setenv("CONFIG_MANIFEST_URI", str(tmp_path / "manifest.json"))
        with pytest.raises(ValueError, match="AWS_BATCH_JOB_ARRAY_INDEX"):
            _load_config_dict()
