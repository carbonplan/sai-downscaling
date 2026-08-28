"""Tests for CLI helper functions."""

import itertools
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from srm.bcsd_config import BCSDConfig, PipelineOptions, VariableConfig
from srm.cli import (
    _expand_matrix_config,
    _is_matrix_config,
    _parse_variable_overrides,
    _resolve_variable_config,
    _validate_predict_periods,
    _validate_variable_overrides,
    app,
    configs_from_matrix,
)
from srm.validation import CheckResult, CheckStatus


class TestConfigsFromMatrix:
    """Tests for the configs_from_matrix helper function."""

    def test_single_combination_returns_one_config(self):
        configs, _ = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            downscaling_methods=["BCSD"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
        )
        assert len(configs) == 1

    def test_cartesian_product_count(self):
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM", "MIROC"],
            downscaling_methods=["BCSD"],
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
            downscaling_methods=["BCSD"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
        )
        assert all(isinstance(c, BCSDConfig) for c in configs)

    def test_historical_only_scenario_is_none(self):
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            downscaling_methods=["BCSD"],
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
            downscaling_methods=["BCSD"],
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
            downscaling_methods=["BCSD"],
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
            downscaling_methods=["BCSD"],
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
                downscaling_methods=["BCSD"],
                variables=["tas"],
                members=["r1i1p1f1"],
                scenarios=["ssp245"],
                # predict_period_start / predict_period_end intentionally omitted
            )

    def test_multiple_scenarios_all_present(self):
        scenarios = ["ssp245", "G6-1pt5k", "G6-termination"]
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            downscaling_methods=["BCSD"],
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
            downscaling_methods=["BCSD"],
            variables=["tas"],
            members=[],
            scenarios=[None],
        )
        assert configs == []  # noqa: E711

    def test_fields_assigned_correctly(self):
        configs, options = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            downscaling_methods=["BCSD"],
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
            downscaling_methods=["BCSD"],
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
            downscaling_methods=["BCSD"],
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
downscaling_method: "BCSD"
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


class TestResolveVariableConfig:
    """Three-tier precedence: table default < run-wide < per-variable override."""

    def test_table_default_when_nothing_supplied(self):
        vc = _resolve_variable_config("dtr", "BCSD", None, None)
        assert vc.debias_approach == "nonparametric_hybrid_2sided"
        assert vc.disaggregation_method == "multiplicative"

    def test_run_wide_beats_table_default(self):
        vc = _resolve_variable_config("dtr", "BCSD", {"debias_approach": "parametric"}, None)
        assert vc.debias_approach == "parametric"

    def test_override_beats_run_wide(self):
        vc = _resolve_variable_config(
            "dtr",
            "BCSD",
            {"debias_approach": "parametric"},
            {"dtr": {"debias_approach": "nonparametric"}},
        )
        assert vc.debias_approach == "nonparametric"

    def test_override_for_another_variable_is_ignored(self):
        vc = _resolve_variable_config(
            "tas", "BCSD", None, {"dtr": {"debias_approach": "nonparametric"}}
        )
        assert vc.debias_approach == "nonparametric_hybrid_2sided"

    def test_none_values_in_run_wide_do_not_override(self):
        vc = _resolve_variable_config("tas", "BCSD", {"debias_approach": None}, None)
        assert vc.debias_approach == "nonparametric_hybrid_2sided"

    def test_string_values_are_coerced_and_validated(self):
        vc = _resolve_variable_config("pr", "BCSD", None, {"pr": {"do_windowing": "false"}})
        assert vc.do_windowing is False

    def test_invalid_value_raises(self):
        with pytest.raises(ValidationError):
            _resolve_variable_config(
                "tas", "BCSD", None, {"tas": {"disaggregation_method": "bogus"}}
            )


class TestValidateVariableOverrides:
    def test_unknown_variable_key_raises(self):
        with pytest.raises(ValueError, match="not in variables"):
            _validate_variable_overrides({"dtr": {"debias_approach": "nonparametric"}}, ["tas"])

    def test_unknown_field_raises(self):
        with pytest.raises(ValueError, match="unknown field"):
            _validate_variable_overrides({"tas": {"nonsense": 1}}, ["tas"])

    def test_non_mapping_value_raises(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _validate_variable_overrides({"tas": "nonparametric"}, ["tas"])

    def test_valid_overrides_pass(self):
        _validate_variable_overrides({"tas": {"debias_approach": "parametric"}}, ["tas", "dtr"])


class TestExpandMatrixConfigOverrides:
    def test_per_variable_debias_approach(self):
        configs = _expand_matrix_config(
            {
                "gcm": "CESM2-WACCM",
                "variables": ["tasmax", "dtr"],
                "ensemble_member": "007",
                "scenario": "ssp245",
                "predict_period_start": 2015,
                "predict_period_end": 2100,
                "downscaling_method": "BCSD",
                "variable_overrides": {"dtr": {"debias_approach": "nonparametric"}},
            }
        )
        by_var = {c.variable: c.variable_config.debias_approach for c in configs}
        assert by_var == {
            "tasmax": "nonparametric_hybrid_2sided",
            "dtr": "nonparametric",
        }

    def test_variable_config_with_multiple_variables_still_raises(self):
        with pytest.raises(ValueError, match="variable_overrides"):
            _expand_matrix_config(
                {
                    "gcm": "CESM2-WACCM",
                    "variables": ["tas", "pr"],
                    "ensemble_member": "007",
                    "downscaling_method": "BCSD",
                    "variable_config": {"detrend_data": False},
                }
            )

    def test_single_variable_config_still_allowed(self):
        configs = _expand_matrix_config(
            {
                "gcm": "CESM2-WACCM",
                "variables": ["tas"],
                "ensemble_members": ["007", "008"],
                "downscaling_method": "BCSD",
                "variable_config": VariableConfig.for_variable("tas", "BCSD").model_dump(),
            }
        )
        assert len(configs) == 2

    def test_variable_config_combined_with_overrides_raises(self):
        """An explicit variable_config is used verbatim, so overrides would be dropped."""
        with pytest.raises(ValueError, match="Cannot combine"):
            _expand_matrix_config(
                {
                    "gcm": "CESM2-WACCM",
                    "variables": ["tas"],
                    "ensemble_member": "007",
                    "downscaling_method": "BCSD",
                    "variable_config": VariableConfig.for_variable("tas", "BCSD").model_dump(),
                    "variable_overrides": {"tas": {"debias_approach": "parametric"}},
                }
            )

    def test_variable_overrides_in_non_matrix_config_raises(self):
        """A scalar-only config bypasses matrix expansion, where overrides are resolved.

        Without an explicit rejection, extra="ignore" would drop the key and the run
        would look configured while silently using the defaults.
        """
        with pytest.raises(ValidationError, match="variable_overrides"):
            BCSDConfig(
                gcm="CESM2-WACCM",
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member="007",
                variable_overrides={"tas": {"debias_approach": "parametric"}},
            )


class TestExpandMatrixConfigRunWideDebiasApproach:
    """A top-level ``debias_approach`` in a matrix config applies to the whole run.

    ``debias_approach`` is a ``VariableConfig`` field, so ``BCSDConfig`` rejects it at the
    top level. ``_expand_matrix_config`` pops it first and feeds it to
    ``_resolve_variable_config`` as the run-wide tier, which is how a whole run is switched
    between standard quantile mapping and quantile delta mapping. ``TestResolveVariableConfig``
    covers that resolver directly; these drive the matrix expansion that supplies it.
    """

    @staticmethod
    def _matrix(**extra):
        base = {
            "gcm": "CESM2-WACCM",
            "downscaling_method": "BCSD",
            "variables": ["tasmax", "dtr"],
            "ensemble_member": "007",
            "scenario": "ssp245",
            "predict_period_start": 2015,
            "predict_period_end": 2100,
        }
        base.update(extra)
        return base

    def test_top_level_debias_approach_applies_to_every_variable(self):
        configs = _expand_matrix_config(self._matrix(debias_approach="parametric"))
        by_var = {c.variable: c.variable_config.debias_approach for c in configs}
        assert by_var == {"tasmax": "parametric", "dtr": "parametric"}

    def test_top_level_debias_approach_leaves_other_fields_on_the_table_default(self):
        """Only the one field moves: the rest of the row still comes from the table."""
        configs = _expand_matrix_config(self._matrix(debias_approach="parametric"))
        dtr = next(c for c in configs if c.variable == "dtr")
        table = VariableConfig.for_variable("dtr", "BCSD")
        assert dtr.variable_config.model_dump() == table.model_dump() | {
            "debias_approach": "parametric"
        }

    def test_variable_override_beats_top_level(self):
        configs = _expand_matrix_config(
            self._matrix(
                debias_approach="parametric",
                variable_overrides={"dtr": {"debias_approach": "nonparametric"}},
            )
        )
        by_var = {c.variable: c.variable_config.debias_approach for c in configs}
        assert by_var == {"tasmax": "parametric", "dtr": "nonparametric"}

    def test_variable_config_combined_with_top_level_debias_approach_raises(self):
        """An explicit variable_config is used verbatim, so the top-level key would vanish."""
        with pytest.raises(ValueError, match="Cannot combine 'variable_config' with a top-level"):
            _expand_matrix_config(
                self._matrix(
                    variables=["tas"],
                    debias_approach="parametric",
                    variable_config=VariableConfig.for_variable("tas", "BCSD").model_dump(),
                )
            )

    def test_top_level_qdm_without_qdmsd_raises(self):
        """The cross-field invariant fires through matrix expansion, not just direct construction."""
        with pytest.raises(ValidationError, match="incompatible"):
            _expand_matrix_config(self._matrix(debias_approach="qdm"))

    def test_top_level_qdm_with_qdmsd_expands(self):
        configs = _expand_matrix_config(
            self._matrix(downscaling_method="QDMSD", debias_approach="qdm")
        )
        assert {c.variable_config.debias_approach for c in configs} == {"qdm"}
        assert {c.downscaling_method for c in configs} == {"QDMSD"}

    def test_qdmsd_with_a_non_qdm_top_level_debias_approach_raises(self):
        """The invariant is symmetric: QDMSD overridden away from qdm is rejected too."""
        with pytest.raises(ValidationError, match="incompatible"):
            _expand_matrix_config(
                self._matrix(downscaling_method="QDMSD", debias_approach="nonparametric")
            )


class TestMatrixDownscalingMethodAxis:
    """``downscaling_methods`` expands like the other four matrix axes.

    Unlike them it selects an algorithm rather than a slice of input data, so every
    entry re-derives ``variable_config`` from its own defaults table. The guards below
    exist because ``BCSDConfig`` pins ``debias_approach='qdm'`` to ``'QDMSD'`` and
    forbids it under ``'BCSD'``: a single explicit value contradicts one arm of a
    two-method product no matter which value is chosen.
    """

    @staticmethod
    def _matrix(**extra):
        base = {
            "gcm": "CESM2-WACCM",
            "variables": ["pr"],
            "ensemble_member": "003",
            "scenario": "ssp245",
            "predict_period_start": 2015,
            "predict_period_end": 2100,
            "downscaling_methods": ["BCSD", "QDMSD"],
        }
        base.update(extra)
        return base

    def test_plural_key_alone_makes_it_a_matrix_config(self):
        """The method list is enough, even with every other axis scalar."""
        assert _is_matrix_config(
            {"gcm": "CESM2-WACCM", "variable": "pr", "downscaling_methods": ["BCSD", "QDMSD"]}
        )

    def test_scalar_key_is_still_not_a_matrix_config(self):
        assert not _is_matrix_config(
            {"gcm": "CESM2-WACCM", "variable": "pr", "downscaling_method": "BCSD"}
        )

    def test_expands_one_config_per_method(self):
        configs = _expand_matrix_config(self._matrix())
        assert [c.downscaling_method for c in configs] == ["BCSD", "QDMSD"]

    def test_multiplies_the_other_axes(self):
        d = self._matrix(variables=["tas", "pr"], ensemble_members=["001", "002"])
        del d["ensemble_member"]
        configs = _expand_matrix_config(d)
        assert len(configs) == 2 * 2 * 2

    def test_both_spellings_of_one_axis_raises(self):
        """The singular key would otherwise survive into the constructor kwargs and
        collide with the value the product loop passes."""
        with pytest.raises(ValueError, match="two spellings of the same axis"):
            _expand_matrix_config(self._matrix(downscaling_method="BCSD"))

    def test_scalar_downscaling_method_still_expands_the_other_axes(self):
        """Every existing config keeps the singular key, so it has to keep working."""
        d = self._matrix(variables=["tas", "pr"])
        del d["downscaling_methods"]
        d["downscaling_method"] = "BCSD"
        configs = _expand_matrix_config(d)
        assert len(configs) == 2
        assert {c.downscaling_method for c in configs} == {"BCSD"}

    def test_each_method_gets_its_own_defaults_table(self):
        """The point of the axis: variable_config is re-derived per method, not shared."""
        by_method = {
            c.downscaling_method: c.variable_config for c in _expand_matrix_config(self._matrix())
        }
        for method in ("BCSD", "QDMSD"):
            expected = VariableConfig.for_variable("pr", method).model_dump()
            assert by_method[method].model_dump() == expected
        assert by_method["BCSD"].debias_approach != by_method["QDMSD"].debias_approach

    def test_the_two_configs_hash_differently(self):
        """config_hash omits downscaling_method by design, so variable_config is the only
        thing separating the pair's cache entries."""
        configs = _expand_matrix_config(self._matrix())
        assert configs[0].config_hash != configs[1].config_hash

    def test_run_wide_debias_approach_with_two_methods_raises(self):
        with pytest.raises(ValueError, match="run-wide 'debias_approach'"):
            _expand_matrix_config(self._matrix(debias_approach="qdm"))

    def test_variable_override_debias_approach_with_two_methods_raises(self):
        with pytest.raises(ValueError, match=r"variable_overrides for \['pr'\]"):
            _expand_matrix_config(
                self._matrix(variable_overrides={"pr": {"debias_approach": "qdm"}})
            )

    def test_variable_config_with_two_methods_raises(self):
        with pytest.raises(ValueError, match="multiple 'downscaling_methods'"):
            _expand_matrix_config(
                self._matrix(variable_config=VariableConfig.for_variable("pr", "BCSD").model_dump())
            )

    def test_run_wide_debias_approach_survives_a_single_method(self):
        """The guard keys on the length of the axis, not on the key being present."""
        configs = _expand_matrix_config(
            self._matrix(downscaling_methods=["BCSD"], debias_approach="parametric")
        )
        assert {c.variable_config.debias_approach for c in configs} == {"parametric"}

    def test_overrides_of_other_fields_survive_two_methods(self):
        """Only debias_approach is unsatisfiable across methods; nothing else is blocked."""
        configs = _expand_matrix_config(
            self._matrix(variable_overrides={"pr": {"do_windowing": False}})
        )
        assert len(configs) == 2
        assert all(c.variable_config.do_windowing is False for c in configs)

    def test_configs_from_matrix_expands_over_methods(self):
        configs, _ = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            downscaling_methods=["BCSD", "QDMSD"],
            variables=["pr"],
            members=["003"],
            scenarios=[None],
        )
        assert [c.downscaling_method for c in configs] == ["BCSD", "QDMSD"]

    def test_configs_from_matrix_rejects_run_wide_debias_approach(self):
        with pytest.raises(ValueError, match="run-wide 'debias_approach'"):
            configs_from_matrix(
                gcms=["CESM2-WACCM"],
                downscaling_methods=["BCSD", "QDMSD"],
                variables=["pr"],
                members=["003"],
                scenarios=[None],
                debias_approach="qdm",
            )

    def test_configs_from_matrix_rejects_override_debias_approach(self):
        with pytest.raises(ValueError, match=r"variable_overrides for \['pr'\]"):
            configs_from_matrix(
                gcms=["CESM2-WACCM"],
                downscaling_methods=["BCSD", "QDMSD"],
                variables=["pr"],
                members=["003"],
                scenarios=[None],
                variable_overrides={"pr": {"debias_approach": "qdm"}},
            )

    def test_run_matrix_accepts_a_repeated_method_flag(self):
        with patch("srm.cli._validate_lineage_members"):
            result = CliRunner().invoke(
                app,
                [
                    "run-matrix",
                    "--gcm",
                    "CESM2-WACCM",
                    "--variable",
                    "pr",
                    "--member",
                    "003",
                    "--scenario",
                    "ssp245",
                    "--downscaling-method",
                    "BCSD",
                    "--downscaling-method",
                    "QDMSD",
                    "--predict-period-start",
                    "2015",
                    # CESM member 003 ends in 2099; _validate_predict_periods enforces
                    # that before the --dry-run branch.
                    "--predict-period-end",
                    "2099",
                    "--dry-run",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "BCSD" in result.output and "QDMSD" in result.output


class TestParseVariableOverrides:
    def test_single_override(self):
        assert _parse_variable_overrides(["dtr:debias_approach=nonparametric"]) == {
            "dtr": {"debias_approach": "nonparametric"}
        }

    def test_repeated_flags_accumulate_across_variables(self):
        parsed = _parse_variable_overrides(
            ["dtr:debias_approach=nonparametric", "pr:do_windowing=false"]
        )
        assert parsed == {
            "dtr": {"debias_approach": "nonparametric"},
            "pr": {"do_windowing": "false"},
        }

    def test_repeated_flags_accumulate_within_one_variable(self):
        parsed = _parse_variable_overrides(["pr:do_windowing=false", "pr:running_window_length=15"])
        assert parsed == {"pr": {"do_windowing": "false", "running_window_length": "15"}}

    def test_whitespace_is_stripped(self):
        assert _parse_variable_overrides([" dtr : debias_approach = nonparametric "]) == {
            "dtr": {"debias_approach": "nonparametric"}
        }

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="variable:field=value"):
            _parse_variable_overrides(["debias_approach=nonparametric"])

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="variable:field=value"):
            _parse_variable_overrides(["dtr:debias_approach"])

    def test_empty_variable_raises(self):
        with pytest.raises(ValueError, match="variable:field=value"):
            _parse_variable_overrides([":debias_approach=nonparametric"])


class TestConfigsFromMatrixOverrides:
    def test_per_variable_debias_approach(self):
        configs, _ = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            downscaling_methods=["BCSD"],
            variables=["tasmax", "dtr"],
            members=["007"],
            scenarios=["ssp245"],
            predict_period_start=2015,
            predict_period_end=2100,
            variable_overrides={"dtr": {"debias_approach": "nonparametric"}},
        )
        by_var = {c.variable: c.variable_config.debias_approach for c in configs}
        assert by_var == {"tasmax": "nonparametric_hybrid_2sided", "dtr": "nonparametric"}

    def test_run_wide_flag_still_applies_to_all(self):
        configs, _ = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            downscaling_methods=["BCSD"],
            variables=["tasmax", "dtr"],
            members=["007"],
            scenarios=[None],
            debias_approach="parametric",
        )
        assert {c.variable_config.debias_approach for c in configs} == {"parametric"}

    def test_override_beats_run_wide_flag(self):
        configs, _ = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            downscaling_methods=["BCSD"],
            variables=["tasmax", "dtr"],
            members=["007"],
            scenarios=[None],
            debias_approach="parametric",
            variable_overrides={"dtr": {"debias_approach": "nonparametric"}},
        )
        by_var = {c.variable: c.variable_config.debias_approach for c in configs}
        assert by_var == {"tasmax": "parametric", "dtr": "nonparametric"}

    def test_invalid_override_value_raises(self):
        with pytest.raises(ValidationError):
            configs_from_matrix(
                gcms=["CESM2-WACCM"],
                downscaling_methods=["BCSD"],
                variables=["tas"],
                members=["007"],
                scenarios=[None],
                variable_overrides={"tas": {"disaggregation_method": "bogus"}},
            )

    def test_existing_run_wide_override_is_now_validated(self):
        """Regression: model_copy accepted bad values; validated construction must not."""
        with pytest.raises(ValidationError):
            configs_from_matrix(
                gcms=["CESM2-WACCM"],
                variables=["tas"],
                members=["007"],
                scenarios=[None],
                downscaling_methods=["BCSD"],
                disaggregation_method="bogus",
            )


class TestRunMatrixOverrideFlag:
    def test_dry_run_applies_per_variable_override(self):
        # `_validate_lineage_members` runs before the --dry-run branch and opens each
        # GCM's unified datatree over the network. Patch it out so this test stays hermetic.
        with patch("srm.cli._validate_lineage_members"):
            result = CliRunner().invoke(
                app,
                [
                    "run-matrix",
                    "--downscaling-method",
                    "BCSD",
                    "--gcm",
                    "CESM2-WACCM",
                    "--variable",
                    "tasmax",
                    "--variable",
                    "dtr",
                    "--member",
                    "007",
                    "--scenario",
                    "ssp245",
                    # 007 is a truncated CESM member: its data ends in 2069, which
                    # `_validate_predict_periods` enforces before the --dry-run branch.
                    "--predict-period-start",
                    "2015",
                    "--predict-period-end",
                    "2069",
                    "--variable-override",
                    "dtr:debias_approach=nonparametric",
                    "--dry-run",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "tasmax" in result.output and "dtr" in result.output

    def test_malformed_override_exits_nonzero(self):
        # Parsing happens before any dataset access, so no patch is needed here.
        result = CliRunner().invoke(
            app,
            [
                "run-matrix",
                "--downscaling-method",
                "BCSD",
                "--gcm",
                "CESM2-WACCM",
                "--variable",
                "tas",
                "--member",
                "007",
                "--variable-override",
                "garbage",
                "--dry-run",
            ],
        )
        assert result.exit_code != 0


class TestReleaseCommand:
    """`bcsd release` freezes the stores a config set writes to under an icechunk tag."""

    _CONFIG_YAML = """
gcm: "CESM2-WACCM"
variables: ["tas", "pr"]
ensemble_members: ["001"]
scenarios: ["SSP245"]
train_period_start: 1978
train_period_end: 2014
predict_period_start: 2015
predict_period_end: 2100
downscaling_method: "BCSD"
output_dir: "s3://bucket/output"
environment: "qa"
branch: "v9"
"""

    def _write_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(self._CONFIG_YAML)
        return config_file

    def test_tags_each_distinct_store_once(self, tmp_path):
        # Two configs (tas, pr) share one gcm/obs/subset triple, so they resolve to a
        # single store. Tagging it twice would fail on the second create_tag call.
        config_file = self._write_config(tmp_path)
        with patch("srm.cache.ArtifactCache.release") as mock_release:
            result = CliRunner().invoke(
                app, ["release", "--config-path", str(config_file), "--tag", "snapshot-v1.0.0"]
            )
        assert result.exit_code == 0, result.output
        mock_release.assert_called_once_with("snapshot-v1.0.0")

    def test_branch_option_overrides_the_config_branch(self, tmp_path):
        config_file = self._write_config(tmp_path)
        seen = {}
        with patch(
            "srm.cache.ArtifactCache.release",
            autospec=True,
            side_effect=lambda self, tag: seen.update(branch=self.branch, tag=tag),
        ):
            result = CliRunner().invoke(
                app,
                [
                    "release",
                    "--config-path",
                    str(config_file),
                    "--tag",
                    "snapshot-v1.0.0",
                    "--branch",
                    "v0.12.0",
                ],
            )
        assert result.exit_code == 0, result.output
        assert seen == {"branch": "v0.12.0", "tag": "snapshot-v1.0.0"}

    def test_exits_nonzero_when_the_tag_already_exists(self, tmp_path):
        # icechunk refuses to move an existing tag. Swallowing that would leave the
        # release green while the baseline still points at the previous run.
        config_file = self._write_config(tmp_path)
        with patch("srm.cache.ArtifactCache.release", side_effect=ValueError("tag exists")):
            result = CliRunner().invoke(
                app, ["release", "--config-path", str(config_file), "--tag", "snapshot-v1.0.0"]
            )
        assert result.exit_code == 1, result.output
