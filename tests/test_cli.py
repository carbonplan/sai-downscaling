"""Tests for CLI helper functions."""

import itertools
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.cli import _validate_predict_periods, app, configs_from_matrix
from srm.validation import CheckResult, CheckStatus


class TestConfigsFromMatrix:
    """Tests for the configs_from_matrix helper function."""

    def test_single_combination_returns_one_config(self):
        configs, _ = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
        )
        assert len(configs) == 1

    def test_cartesian_product_count(self):
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM", "MIROC"],
            variables=["tas", "pr"],
            members=["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
            scenarios=["ssp245", "G6-1pt5k"],
            predict_period_start=2015,
            predict_period_end=2100,
        )
        assert len(configs) == 2 * 2 * 3 * 2  # 24

    def test_returns_bcsd_config_instances(self):
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
        )
        assert all(isinstance(c, BCSDConfig) for c in configs)

    def test_historical_only_scenario_is_none(self):
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1", "r2i1p1f1"],
            scenarios=[None],
        )
        assert len(configs) == 2
        assert all(c.scenario is None for c in configs)

    def test_all_combinations_present(self):
        gcms = ["CESM2-WACCM", "MIROC"]
        variables = ["tas", "pr"]
        members = ["r1i1p1f1", "r2i1p1f1"]
        scenarios = ["ssp245"]
        configs, options = configs_from_matrix(
            gcms=gcms,
            variables=variables,
            members=members,
            scenarios=scenarios,
            predict_period_start=2015,
            predict_period_end=2100,
        )
        actual = {(c.gcm, c.variable, c.ensemble_member, c.scenario) for c in configs}
        expected = set(itertools.product(gcms, variables, members, [s.upper() for s in scenarios]))
        assert actual == expected

    def test_shared_params_applied_to_all_configs(self):
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM", "MIROC"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
            environment="production",
            branch="v2",
            train_period_start=1979,
            train_period_end=2013,
        )
        assert options.environment == "production"
        assert options.branch == "v2"
        assert all(c.train_period_start == 1979 for c in configs)
        assert all(c.train_period_end == 2013 for c in configs)

    def test_subset_bounds_propagated(self):
        bounds = (-35.0, -22.0, 16.0, 33.0)
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
            subset_bounds=bounds,
        )
        assert all(c.subset_bounds == bounds for c in configs)

    def test_scenario_without_predict_period_raises(self):
        """BCSDConfig raises ValidationError when scenario is set but predict periods are missing."""
        with pytest.raises(ValidationError):
            configs_from_matrix(
                gcms=["CESM2-WACCM"],
                variables=["tas"],
                members=["r1i1p1f1"],
                scenarios=["ssp245"],
                # predict_period_start / predict_period_end intentionally omitted
            )

    def test_multiple_scenarios_all_present(self):
        scenarios = ["ssp245", "G6-1pt5k", "G6-termination"]
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=scenarios,
            predict_period_start=2015,
            predict_period_end=2100,
        )
        assert len(configs) == 3
        assert {c.scenario for c in configs} == {s.upper() for s in scenarios}

    def test_empty_members_returns_empty_list(self):
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=[],
            scenarios=[None],
        )
        assert configs == []  # noqa: E711

    def test_fields_assigned_correctly(self):
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["pr"],
            members=["r3i1p1f1"],
            scenarios=["ssp245"],
            predict_period_start=2020,
            predict_period_end=2080,
        )
        cfg = configs[0]
        assert cfg.gcm == "CESM2-WACCM"
        assert cfg.variable == "pr"
        assert cfg.ensemble_member == "r3i1p1f1"
        assert cfg.scenario == "SSP245"
        assert cfg.predict_period_start == 2020
        assert cfg.predict_period_end == 2080
        assert isinstance(options, PipelineOptions)


class TestValidatePredictPeriods:
    """run/run-matrix must reject configs whose predict_period overruns a member's data."""

    def test_truncated_member_overrun_raises(self):
        configs, _ = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tasmax"],
            members=["007"],
            scenarios=["ssp245"],
            predict_period_start=2015,
            predict_period_end=2100,
        )
        with pytest.raises(ValueError, match="2069"):
            _validate_predict_periods(configs)

    def test_truncated_member_within_extent_does_not_raise(self):
        configs, _ = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tasmax"],
            members=["007"],
            scenarios=["ssp245"],
            predict_period_start=2015,
            predict_period_end=2069,
        )
        _validate_predict_periods(configs)  # should not raise


class TestValidateOutputConfigPath:
    """Tests for `bcsd validate-output --config-path`: store discovery + branch default."""

    _CONFIG_YAML = """
gcm: "CESM2-WACCM"
variables: ["tas", "pr"]
ensemble_members: ["001"]
scenarios: ["SSP245"]
train_period_start: 1978
train_period_end: 2014
predict_period_start: 2015
predict_period_end: 2100
output_dir: "s3://bucket/output"
environment: "qa"
branch: "v9"
"""

    def _write_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(self._CONFIG_YAML)
        return config_file

    def test_derives_deduped_store_uri_and_default_branch(self, tmp_path):
        config_file = self._write_config(tmp_path)
        passing_result = CheckResult(
            check_id="lat_valid",
            gcm="CESM2-WACCM",
            scenario="ssp245/tas/001",
            status=CheckStatus.PASS,
        )

        with patch(
            "srm.validation.validate_output_store", return_value=[passing_result]
        ) as mock_validate:
            result = CliRunner().invoke(
                app, ["validate-output", "--config-path", str(config_file), "--no-coiled"]
            )

        assert result.exit_code == 0, result.output
        # Two configs (tas, pr) share the same gcm/subset → one deduped store URI.
        mock_validate.assert_called_once()
        _, kwargs = mock_validate.call_args
        assert kwargs["branch"] == "v9"
        assert kwargs["tag"] is None

    def test_explicit_branch_overrides_config_default(self, tmp_path):
        config_file = self._write_config(tmp_path)
        passing_result = CheckResult(
            check_id="lat_valid",
            gcm="CESM2-WACCM",
            scenario="ssp245/tas/001",
            status=CheckStatus.PASS,
        )

        with patch(
            "srm.validation.validate_output_store", return_value=[passing_result]
        ) as mock_validate:
            result = CliRunner().invoke(
                app,
                [
                    "validate-output",
                    "--config-path",
                    str(config_file),
                    "--branch",
                    "override",
                    "--no-coiled",
                ],
            )

        assert result.exit_code == 0, result.output
        _, kwargs = mock_validate.call_args
        assert kwargs["branch"] == "override"

    def test_store_uris_and_config_path_mutually_exclusive(self, tmp_path):
        config_file = self._write_config(tmp_path)
        result = CliRunner().invoke(
            app,
            ["validate-output", "s3://some/store.icechunk", "--config-path", str(config_file)],
        )
        assert result.exit_code != 0

    def test_neither_store_uris_nor_config_path_errors(self):
        result = CliRunner().invoke(app, ["validate-output"])
        assert result.exit_code != 0
