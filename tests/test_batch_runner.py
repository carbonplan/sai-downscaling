"""
Unit tests for the batch_runner CLI entry point.

Tests focus on:
- CONFIG_JSON environment variable is read and parsed correctly
- Missing or malformed CONFIG_JSON raises immediately
- Stage argument routes to the correct BCSDPipeline method
- Unknown stage names raise ValueError
- The result path is printed and returned
- The stage argument is passed correctly to the pipeline

BCSDPipeline is always mocked so no real compute or S3 access is required.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from saidownscale.batch_runner import run_stage
from saidownscale.bcsd_config import BCSDConfig

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG = {
    "gcm": "CESM2-WACCM",
    "variable": "tas",
    "ensemble_member": "r1i1p1f1",
    "scenario": "ssp245",
    "predict_period_start": 2015,
    "predict_period_end": 2100,
}


@pytest.fixture
def valid_config_json() -> str:
    """JSON string for a minimal valid BCSDConfig."""
    return json.dumps(_MINIMAL_CONFIG)


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure CONFIG_JSON is not leaked between tests."""
    old = os.environ.pop("CONFIG_JSON", None)
    yield
    if old is not None:
        os.environ["CONFIG_JSON"] = old
    else:
        os.environ.pop("CONFIG_JSON", None)


# ---------------------------------------------------------------------------
# CONFIG_JSON environment variable handling
# ---------------------------------------------------------------------------


class TestConfigJsonReading:
    def test_raises_when_config_json_not_set(self):
        with pytest.raises(ValueError, match="CONFIG_JSON environment variable not set"):
            run_stage("prepare_observations")

    def test_raises_when_config_json_is_empty_string(self):
        os.environ["CONFIG_JSON"] = ""
        with pytest.raises(ValueError, match="CONFIG_JSON environment variable not set"):
            run_stage("prepare_observations")

    def test_raises_on_invalid_json(self):
        os.environ["CONFIG_JSON"] = "not-valid-json{"
        with pytest.raises(Exception):  # json.JSONDecodeError (subclass of ValueError)
            run_stage("prepare_observations")

    def test_raises_on_json_with_invalid_bcsd_fields(self):
        os.environ["CONFIG_JSON"] = json.dumps(
            {"gcm": "X", "variable": "sfcWind", "ensemble_member": "r1i1p1f1"}
        )
        with pytest.raises(Exception):
            run_stage("prepare_observations")

    def test_config_parsed_into_bcsdconfig(self, valid_config_json):
        os.environ["CONFIG_JSON"] = valid_config_json
        captured = {}

        def fake_init(self, config, options):
            captured["config"] = config
            # Prevent real pipeline operations
            raise StopIteration("stop")

        with patch("saidownscale.batch_runner.BCSDPipeline.__init__", fake_init):
            with pytest.raises(StopIteration):
                run_stage("prepare_observations")

        assert isinstance(captured["config"], BCSDConfig)
        assert captured["config"].gcm == "CESM2-WACCM"
        assert captured["config"].variable == "tas"
        assert captured["config"].ensemble_member == "r1i1p1f1"


# ---------------------------------------------------------------------------
# Stage routing
# ---------------------------------------------------------------------------


class TestStageRouting:
    """Each stage name must call the corresponding BCSDPipeline method."""

    def _run_with_mock_pipeline(self, stage: str, valid_config_json: str) -> MagicMock:
        """Helper that runs run_stage with a fully mocked BCSDPipeline."""
        os.environ["CONFIG_JSON"] = valid_config_json
        mock_pipeline = MagicMock()
        mock_pipeline.prepare_observations.return_value = f"{stage}_output_path"
        mock_pipeline.fit_historical.return_value = f"{stage}_output_path"
        mock_pipeline.transform_scenario.return_value = f"{stage}_output_path"

        with patch("saidownscale.batch_runner.BCSDPipeline", return_value=mock_pipeline):
            run_stage(stage)

        return mock_pipeline

    def test_prepare_observations_routes_correctly(self, valid_config_json):
        mock = self._run_with_mock_pipeline("prepare_observations", valid_config_json)
        mock.prepare_observations.assert_called_once()
        mock.fit_historical.assert_not_called()
        mock.transform_scenario.assert_not_called()

    def test_fit_historical_routes_correctly(self, valid_config_json):
        mock = self._run_with_mock_pipeline("fit_historical", valid_config_json)
        mock.fit_historical.assert_called_once()
        mock.prepare_observations.assert_not_called()
        mock.transform_scenario.assert_not_called()

    def test_transform_scenario_routes_correctly(self, valid_config_json):
        mock = self._run_with_mock_pipeline("transform_scenario", valid_config_json)
        mock.transform_scenario.assert_called_once()
        mock.prepare_observations.assert_not_called()
        mock.fit_historical.assert_not_called()

    def test_all_stage_names_are_handled(self, valid_config_json, subtests):
        stage_to_method = {
            "prepare_observations": "prepare_observations",
            "fit_historical": "fit_historical",
            "transform_scenario": "transform_scenario",
        }
        for stage, method in stage_to_method.items():
            with subtests.test(stage=stage):
                mock = self._run_with_mock_pipeline(stage, valid_config_json)
                getattr(mock, method).assert_called_once()

    def test_unknown_stage_raises(self, valid_config_json):
        os.environ["CONFIG_JSON"] = valid_config_json
        mock_pipeline = MagicMock()
        with patch("saidownscale.batch_runner.BCSDPipeline", return_value=mock_pipeline):
            with pytest.raises(ValueError, match="Unknown stage"):
                run_stage("nonexistent_stage")


# ---------------------------------------------------------------------------
# BCSDPipeline construction
# ---------------------------------------------------------------------------


class TestPipelineConstruction:
    def test_pipeline_created_with_parsed_config(self, valid_config_json):
        os.environ["CONFIG_JSON"] = valid_config_json
        mock_pipeline = MagicMock()
        mock_pipeline.prepare_observations.return_value = "obs_path"

        with patch("saidownscale.batch_runner.BCSDPipeline") as MockPipeline:
            MockPipeline.return_value = mock_pipeline
            run_stage("prepare_observations")
            constructor_call = MockPipeline.call_args

        # Should be called with a BCSDConfig instance as the only positional arg
        config_arg = constructor_call.args[0]
        assert isinstance(config_arg, BCSDConfig)
        assert config_arg.gcm == "CESM2-WACCM"

    def test_pipeline_created_exactly_once(self, valid_config_json):
        os.environ["CONFIG_JSON"] = valid_config_json
        mock_pipeline = MagicMock()
        mock_pipeline.prepare_observations.return_value = "obs_path"

        with patch(
            "saidownscale.batch_runner.BCSDPipeline", return_value=mock_pipeline
        ) as MockPipeline:
            run_stage("prepare_observations")

        MockPipeline.assert_called_once()


# ---------------------------------------------------------------------------
# Return value and output
# ---------------------------------------------------------------------------


class TestReturnValue:
    def test_returns_result_path(self, valid_config_json):
        os.environ["CONFIG_JSON"] = valid_config_json
        mock_pipeline = MagicMock()
        mock_pipeline.prepare_observations.return_value = "s3://bucket/obs.icechunk"

    def test_prints_result_path(self, valid_config_json, capsys):
        os.environ["CONFIG_JSON"] = valid_config_json
        mock_pipeline = MagicMock()
        mock_pipeline.prepare_observations.return_value = "s3://bucket/obs.icechunk"

        with patch("saidownscale.batch_runner.BCSDPipeline", return_value=mock_pipeline):
            run_stage("prepare_observations")

        captured = capsys.readouterr()
        assert "s3://bucket/obs.icechunk" in captured.out

    def test_returns_fit_historical_path(self, valid_config_json):
        os.environ["CONFIG_JSON"] = valid_config_json
        mock_pipeline = MagicMock()
        mock_pipeline.fit_historical.return_value = "s3://bucket/hist.icechunk"

        with patch("saidownscale.batch_runner.BCSDPipeline", return_value=mock_pipeline):
            result = run_stage("fit_historical")

        assert result == "s3://bucket/hist.icechunk"

    def test_returns_transform_scenario_path(self, valid_config_json):
        os.environ["CONFIG_JSON"] = valid_config_json
        mock_pipeline = MagicMock()
        mock_pipeline.transform_scenario.return_value = "s3://bucket/ssp245.icechunk"

        with patch("saidownscale.batch_runner.BCSDPipeline", return_value=mock_pipeline):
            result = run_stage("transform_scenario")

        assert result == "s3://bucket/ssp245.icechunk"
