"""Tests for CLI helper functions."""

import itertools
from unittest.mock import patch

import pytest
import typer
from pydantic import ValidationError
from typer.testing import CliRunner

from saidownscale.cli import (
    _confirm_cost,
    _expand_matrix_config,
    _is_matrix_config,
    _parse_variable_overrides,
    _render_cost_plan,
    _render_run_environment,
    _resolve_stage,
    _resolve_variable_config,
    _validate_predict_periods,
    _validate_variable_overrides,
    app,
    configs_from_matrix,
    console,
)
from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions, VariableConfig
from saidownscale.validation import CheckResult, CheckStatus

_CONFIG_YAML = """
gcm: "CESM2-WACCM6"
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

_PASS = CheckResult(
    check_id="lat_valid", gcm="CESM2-WACCM6", scenario="ssp245/tas/001", status=CheckStatus.PASS
)


def _cesm(**kwargs):
    return configs_from_matrix(gcms=["CESM2-WACCM6"], **kwargs)


def _matrix(**extra):
    base = {
        "gcm": "CESM2-WACCM6",
        "downscaling_method": "BCSD",
        "variables": ["tasmax", "dtr"],
        "ensemble_member": "007",
        "scenario": "ssp245",
        "predict_period_start": 2015,
        "predict_period_end": 2100,
    }
    return base | extra


def _two_method_matrix(**extra):
    base = _matrix(variables=["pr"], ensemble_member="003", downscaling_methods=["BCSD", "QDMSD"])
    del base["downscaling_method"]
    return base | extra


def _debias_by_var(configs):
    return {c.variable: c.variable_config.debias_approach for c in configs}


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(_CONFIG_YAML)
    return path


@pytest.fixture
def orchestrator(tmp_path):
    from saidownscale.orchestration import DownscalingOrchestrator

    return DownscalingOrchestrator(
        PipelineOptions(
            scratch_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "out"), verbose=False
        )
    )


@pytest.fixture
def ssp245_configs():
    return [
        DownscalingConfig(
            gcm="CESM2-WACCM6",
            downscaling_method="BCSD",
            variable="tas",
            ensemble_member="r1i1p1f1",
            scenario="SSP245",
            predict_period_start=2015,
            predict_period_end=2100,
        )
    ]


def test_configs_from_matrix_expands_the_cartesian_product():
    gcms = ["CESM2-WACCM6", "UKESM1-1-LL"]
    variables = ["tas", "pr"]
    members = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]
    scenarios = ["ssp245", "G6-1pt5k", "G6-termination"]
    bounds = (-35.0, -22.0, 16.0, 33.0)
    configs, options = configs_from_matrix(
        gcms=gcms,
        downscaling_methods=["BCSD"],
        variables=variables,
        members=members,
        scenarios=scenarios,
        predict_period_start=2020,
        predict_period_end=2080,
        environment="production",
        branch="v2",
        train_period_start=1979,
        train_period_end=2013,
        subset_bounds=bounds,
    )
    assert all(isinstance(c, DownscalingConfig) for c in configs)
    assert isinstance(options, PipelineOptions)
    actual = [(c.gcm, c.variable, c.ensemble_member, c.scenario) for c in configs]
    expected = itertools.product(gcms, variables, members, [s.upper() for s in scenarios])
    assert sorted(actual) == sorted(expected)
    assert (options.environment, options.branch) == ("production", "v2")
    assert {
        (c.predict_period_start, c.predict_period_end, c.train_period_start, c.train_period_end)
        for c in configs
    } == {(2020, 2080, 1979, 2013)}
    assert {c.subset_bounds for c in configs} == {bounds}


def test_configs_from_matrix_edge_axes():
    common = dict(downscaling_methods=["BCSD"], variables=["tas"], scenarios=[None])
    historical, _ = _cesm(members=["r1i1p1f1", "r2i1p1f1"], **common)
    assert [c.scenario for c in historical] == [None, None]
    assert _cesm(members=[], **common)[0] == []
    with pytest.raises(ValidationError):
        _cesm(downscaling_methods=["BCSD"], variables=["tas"], members=["r1"], scenarios=["ssp245"])


def test_validate_predict_periods_against_truncated_member():
    def configs(end):
        return _cesm(
            downscaling_methods=["BCSD"],
            variables=["tasmax"],
            members=["007"],
            scenarios=["ssp245"],
            predict_period_start=2015,
            predict_period_end=end,
        )[0]

    with pytest.raises(ValueError, match="2069"):
        _validate_predict_periods(configs(2100))
    _validate_predict_periods(configs(2069))


def test_validate_output_command(subtests, config_file):
    cases = [
        ("config_branch", [], [_PASS], 0, "v9"),
        ("branch_override", ["--branch", "override"], [_PASS], 0, "override"),
        ("empty_store_blocks", [], [], 1, "v9"),
        ("empty_filtered_store_blocks", ["--variable", "tas"], [], 1, "v9"),
    ]
    for name, extra, results, exit_code, branch in cases:
        with subtests.test(name):
            with patch(
                "saidownscale.validation.validate_output_store", return_value=results
            ) as mock_validate:
                result = CliRunner().invoke(
                    app,
                    ["validate-output", "--config-path", str(config_file), *extra, "--no-coiled"],
                )
            assert result.exit_code == exit_code, result.output
            mock_validate.assert_called_once()
            assert mock_validate.call_args.kwargs["branch"] == branch
            assert mock_validate.call_args.kwargs["tag"] is None
    for name, args in [
        ("uris_and_config_path", ["s3://some/store.icechunk", "--config-path", str(config_file)]),
        ("neither", []),
    ]:
        with subtests.test(name):
            assert CliRunner().invoke(app, ["validate-output", *args]).exit_code != 0


def test_resolve_branch_command(subtests, tmp_path):
    cases = [("config", _CONFIG_YAML, "v9"), ("package_default", None, PipelineOptions().branch)]
    for name, text, expected in cases:
        with subtests.test(name):
            path = tmp_path / f"{name}.yaml"
            path.write_text(text or _CONFIG_YAML.replace('branch: "v9"\n', ""))
            result = CliRunner().invoke(app, ["resolve-branch", "--config-path", str(path)])
            assert result.exit_code == 0, result.output
            assert result.output.strip() == expected


def test_resolve_variable_config_precedence(subtests):
    dtr_override = {"dtr": {"debias_approach": "nonparametric"}}
    parametric = {"debias_approach": "parametric"}
    cases = [
        ("table_default", "dtr", None, None, "debias_approach", "nonparametric_hybrid_2sided"),
        ("table_default", "dtr", None, None, "disaggregation_method", "multiplicative"),
        ("run_wide_beats_table", "dtr", parametric, None, "debias_approach", "parametric"),
        ("override_beats_run_wide", "dtr", parametric, dtr_override, "debias_approach", "nonparametric"),
        ("other_var_override_ignored", "tas", None, dtr_override, "debias_approach", "nonparametric_hybrid_2sided"),
        ("none_run_wide_ignored", "tas", {"debias_approach": None}, None, "debias_approach", "nonparametric_hybrid_2sided"),
        ("string_coerced", "pr", None, {"pr": {"do_windowing": "false"}}, "do_windowing", False),
    ]  # fmt: skip
    for name, var, run_wide, overrides, field, expected in cases:
        with subtests.test(name, field=field):
            vc = _resolve_variable_config(var, "BCSD", run_wide, overrides)
            assert getattr(vc, field) == expected
    with pytest.raises(ValidationError):
        _resolve_variable_config("tas", "BCSD", None, {"tas": {"disaggregation_method": "bogus"}})


def test_validate_variable_overrides(subtests):
    cases = [
        ({"dtr": {"debias_approach": "nonparametric"}}, "not in variables"),
        ({"tas": {"nonsense": 1}}, "unknown field"),
        ({"tas": "nonparametric"}, "must be a mapping"),
    ]
    for overrides, match in cases:
        with subtests.test(match):
            with pytest.raises(ValueError, match=match):
                _validate_variable_overrides(overrides, ["tas"])
    _validate_variable_overrides({"tas": {"debias_approach": "parametric"}}, ["tas", "dtr"])


def test_debias_approach_precedence_through_matrix_expansion(subtests):
    dtr_np = {"dtr": {"debias_approach": "nonparametric"}}
    cli_common = dict(downscaling_methods=["BCSD"], variables=["tasmax", "dtr"], members=["007"])
    cases = [
        (
            "per_variable_override",
            lambda: _expand_matrix_config(_matrix(variable_overrides=dtr_np)),
            {"tasmax": "nonparametric_hybrid_2sided", "dtr": "nonparametric"},
        ),
        (
            "top_level_applies_to_all",
            lambda: _expand_matrix_config(_matrix(debias_approach="parametric")),
            {"tasmax": "parametric", "dtr": "parametric"},
        ),
        (
            "override_beats_top_level",
            lambda: _expand_matrix_config(
                _matrix(debias_approach="parametric", variable_overrides=dtr_np)
            ),
            {"tasmax": "parametric", "dtr": "nonparametric"},
        ),
        (
            "top_level_qdm_with_qdmsd",
            lambda: _expand_matrix_config(
                _matrix(downscaling_method="QDMSD", debias_approach="qdm")
            ),
            {"tasmax": "qdm", "dtr": "qdm"},
        ),
        (
            "run_wide_survives_single_method",
            lambda: _expand_matrix_config(
                _two_method_matrix(downscaling_methods=["BCSD"], debias_approach="parametric")
            ),
            {"pr": "parametric"},
        ),
        (
            "configs_from_matrix_override",
            lambda: _cesm(
                **cli_common,
                scenarios=["ssp245"],
                predict_period_start=2015,
                predict_period_end=2100,
                variable_overrides=dtr_np,
            )[0],
            {"tasmax": "nonparametric_hybrid_2sided", "dtr": "nonparametric"},
        ),
        (
            "configs_from_matrix_run_wide",
            lambda: _cesm(**cli_common, scenarios=[None], debias_approach="parametric")[0],
            {"tasmax": "parametric", "dtr": "parametric"},
        ),
        (
            "configs_from_matrix_override_beats_run_wide",
            lambda: _cesm(
                **cli_common,
                scenarios=[None],
                debias_approach="parametric",
                variable_overrides=dtr_np,
            )[0],
            {"tasmax": "parametric", "dtr": "nonparametric"},
        ),
    ]
    for name, build, expected in cases:
        with subtests.test(name):
            assert _debias_by_var(build()) == expected

    dtr = next(
        c
        for c in _expand_matrix_config(_matrix(debias_approach="parametric"))
        if c.variable == "dtr"
    )
    table = VariableConfig.for_variable("dtr", "BCSD").model_dump()
    assert dtr.variable_config.model_dump() == table | {"debias_approach": "parametric"}


def test_matrix_expansion_rejections(subtests):
    tas_vc = VariableConfig.for_variable("tas", "BCSD").model_dump()
    pr_vc = VariableConfig.for_variable("pr", "BCSD").model_dump()
    qdm_pr = {"pr": {"debias_approach": "qdm"}}
    two_methods = dict(
        downscaling_methods=["BCSD", "QDMSD"], variables=["pr"], members=["003"], scenarios=[None]
    )
    one_tas = dict(downscaling_methods=["BCSD"], variables=["tas"], members=["007"])
    cases = [
        (
            "variable_config_with_multiple_variables",
            lambda: _expand_matrix_config(
                {
                    "gcm": "CESM2-WACCM6",
                    "variables": ["tas", "pr"],
                    "ensemble_member": "007",
                    "downscaling_method": "BCSD",
                    "variable_config": {"detrend_data": False},
                }
            ),
            ValueError,
            "variable_overrides",
        ),
        (
            "variable_config_with_overrides",
            lambda: _expand_matrix_config(
                _matrix(
                    variables=["tas"],
                    variable_config=tas_vc,
                    variable_overrides={"tas": {"debias_approach": "parametric"}},
                )
            ),
            ValueError,
            "Cannot combine",
        ),
        (
            "variable_config_with_top_level_debias",
            lambda: _expand_matrix_config(
                _matrix(variables=["tas"], debias_approach="parametric", variable_config=tas_vc)
            ),
            ValueError,
            "Cannot combine 'variable_config' with a top-level",
        ),
        (
            "top_level_qdm_without_qdmsd",
            lambda: _expand_matrix_config(_matrix(debias_approach="qdm")),
            ValidationError,
            "incompatible",
        ),
        (
            "qdmsd_with_non_qdm",
            lambda: _expand_matrix_config(
                _matrix(downscaling_method="QDMSD", debias_approach="nonparametric")
            ),
            ValidationError,
            "incompatible",
        ),
        (
            "both_spellings_of_method_axis",
            lambda: _expand_matrix_config(_two_method_matrix(downscaling_method="BCSD")),
            ValueError,
            "two spellings of the same axis",
        ),
        (
            "run_wide_debias_with_two_methods",
            lambda: _expand_matrix_config(_two_method_matrix(debias_approach="qdm")),
            ValueError,
            "run-wide 'debias_approach'",
        ),
        (
            "override_debias_with_two_methods",
            lambda: _expand_matrix_config(_two_method_matrix(variable_overrides=qdm_pr)),
            ValueError,
            r"variable_overrides for \['pr'\]",
        ),
        (
            "variable_config_with_two_methods",
            lambda: _expand_matrix_config(_two_method_matrix(variable_config=pr_vc)),
            ValueError,
            "multiple 'downscaling_methods'",
        ),
        (
            "configs_from_matrix_run_wide_with_two_methods",
            lambda: _cesm(**two_methods, debias_approach="qdm"),
            ValueError,
            "run-wide 'debias_approach'",
        ),
        (
            "configs_from_matrix_override_with_two_methods",
            lambda: _cesm(**two_methods, variable_overrides=qdm_pr),
            ValueError,
            r"variable_overrides for \['pr'\]",
        ),
        (
            "configs_from_matrix_invalid_override_value",
            lambda: _cesm(
                **one_tas,
                scenarios=[None],
                variable_overrides={"tas": {"disaggregation_method": "bogus"}},
            ),
            ValidationError,
            "disaggregation_method",
        ),
        (
            "configs_from_matrix_invalid_run_wide_value_regression",
            lambda: _cesm(**one_tas, scenarios=[None], disaggregation_method="bogus"),
            ValidationError,
            "disaggregation_method",
        ),
        (
            "variable_overrides_in_non_matrix_config",
            lambda: DownscalingConfig(
                gcm="CESM2-WACCM6",
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member="007",
                variable_overrides={"tas": {"debias_approach": "parametric"}},
            ),
            ValidationError,
            "variable_overrides",
        ),
    ]
    for name, build, exc, match in cases:
        with subtests.test(name):
            with pytest.raises(exc, match=match):
                build()


def test_downscaling_method_axis_expansion():
    assert _is_matrix_config(
        {"gcm": "CESM2-WACCM6", "variable": "pr", "downscaling_methods": ["BCSD", "QDMSD"]}
    )
    assert not _is_matrix_config(
        {"gcm": "CESM2-WACCM6", "variable": "pr", "downscaling_method": "BCSD"}
    )

    configs = _expand_matrix_config(_two_method_matrix())
    assert [c.downscaling_method for c in configs] == ["BCSD", "QDMSD"]
    for c in configs:
        expected = VariableConfig.for_variable("pr", c.downscaling_method).model_dump()
        assert c.variable_config.model_dump() == expected
    assert configs[0].variable_config.debias_approach != configs[1].variable_config.debias_approach
    assert configs[0].config_hash != configs[1].config_hash

    d = _two_method_matrix(variables=["tas", "pr"], ensemble_members=["001", "002"])
    del d["ensemble_member"]
    assert len(_expand_matrix_config(d)) == 8

    scalar = _matrix(variables=["tas", "pr"], ensemble_member="003")
    assert {c.downscaling_method for c in _expand_matrix_config(scalar)} == {"BCSD"}
    assert len(_expand_matrix_config(scalar)) == 2

    single_vc = _matrix(
        variables=["tas"],
        ensemble_members=["007", "008"],
        variable_config=VariableConfig.for_variable("tas", "BCSD").model_dump(),
    )
    del single_vc["ensemble_member"]
    assert len(_expand_matrix_config(single_vc)) == 2

    windowed = _expand_matrix_config(
        _two_method_matrix(variable_overrides={"pr": {"do_windowing": False}})
    )
    assert [c.variable_config.do_windowing for c in windowed] == [False, False]

    cli_configs, _ = _cesm(
        downscaling_methods=["BCSD", "QDMSD"], variables=["pr"], members=["003"], scenarios=[None]
    )
    assert [c.downscaling_method for c in cli_configs] == ["BCSD", "QDMSD"]


def test_parse_variable_overrides(subtests):
    cases = [
        (["dtr:debias_approach=nonparametric"], {"dtr": {"debias_approach": "nonparametric"}}),
        (
            ["dtr:debias_approach=nonparametric", "pr:do_windowing=false"],
            {"dtr": {"debias_approach": "nonparametric"}, "pr": {"do_windowing": "false"}},
        ),
        (
            ["pr:do_windowing=false", "pr:running_window_length=15"],
            {"pr": {"do_windowing": "false", "running_window_length": "15"}},
        ),
        (
            [" dtr : debias_approach = nonparametric "],
            {"dtr": {"debias_approach": "nonparametric"}},
        ),
    ]
    for flags, expected in cases:
        with subtests.test(flags=flags):
            assert _parse_variable_overrides(flags) == expected
    for bad in ("debias_approach=nonparametric", "dtr:debias_approach", ":debias_approach=x"):
        with subtests.test(bad=bad):
            with pytest.raises(ValueError, match="variable:field=value"):
                _parse_variable_overrides([bad])


def test_run_matrix_command(subtests):
    base = ["run-matrix", "--gcm", "CESM2-WACCM6", "--scenario", "ssp245", "--dry-run"]
    cases = [
        (
            "repeated_method_flag",
            ["--variable", "pr", "--member", "003", "--downscaling-method", "BCSD"]
            + ["--downscaling-method", "QDMSD"]
            + ["--predict-period-start", "2015", "--predict-period-end", "2099"],
            ["BCSD", "QDMSD"],
        ),
        (
            "per_variable_override",
            ["--variable", "tasmax", "--variable", "dtr", "--member", "007"]
            + ["--downscaling-method", "BCSD"]
            + ["--predict-period-start", "2015", "--predict-period-end", "2069"]
            + ["--variable-override", "dtr:debias_approach=nonparametric"],
            ["tasmax", "dtr"],
        ),
    ]
    for name, args, expected in cases:
        with subtests.test(name):
            with patch("saidownscale.cli._validate_lineage_members"):
                result = CliRunner().invoke(app, base + args)
            assert result.exit_code == 0, result.output
            assert all(token in result.output for token in expected)
    with subtests.test("malformed_override"):
        malformed = ["run-matrix", "--downscaling-method", "BCSD", "--gcm", "CESM2-WACCM6"]
        malformed += ["--variable", "tas", "--member", "007", "--variable-override", "garbage"]
        assert CliRunner().invoke(app, [*malformed, "--dry-run"]).exit_code != 0


def test_release_command(subtests, config_file):
    release = "saidownscale.cache.ArtifactCache.release"
    base = ["release", "--config-path", str(config_file), "--tag", "snapshot-v1.0.0"]

    with subtests.test("tags_each_distinct_store_once"):
        with patch(release) as mock_release:
            result = CliRunner().invoke(app, base)
        assert result.exit_code == 0, result.output
        mock_release.assert_called_once_with("snapshot-v1.0.0")

    with subtests.test("branch_option_overrides_config"):
        seen = {}
        with patch(
            release,
            autospec=True,
            side_effect=lambda self, tag: seen.update(branch=self.branch, tag=tag),
        ):
            result = CliRunner().invoke(app, [*base, "--branch", "v0.12.0"])
        assert result.exit_code == 0, result.output
        assert seen == {"branch": "v0.12.0", "tag": "snapshot-v1.0.0"}

    with subtests.test("existing_tag_exits_nonzero"):
        with patch(release, side_effect=ValueError("tag exists")):
            result = CliRunner().invoke(app, base)
        assert result.exit_code == 1, result.output


def test_confirm_cost_gate(subtests, orchestrator, ssp245_configs):
    cases = [
        ("local_skips_gate", "local", None, True, False, None, True),
        ("non_interactive_never_prompts", "aws-batch", True, False, False, None, True),
        ("yes_flag_skips_prompt", "aws-batch", True, True, True, None, True),
        ("nothing_to_submit", "aws-batch", False, True, False, None, True),
        ("prompt_refused", "aws-batch", True, True, False, False, False),
        ("prompt_accepted", "aws-batch", True, True, False, True, True),
    ]
    for name, executor, has_work, tty, yes, answer, expected in cases:
        with subtests.test(name):
            with (
                patch("saidownscale.cli._render_cost_plan", return_value=has_work) as mock_render,
                patch("sys.stdin.isatty", return_value=tty),
                patch("typer.confirm", return_value=answer) as mock_confirm,
            ):
                result = _confirm_cost(orchestrator, ssp245_configs, executor, False, yes)
            assert result is expected
            assert mock_render.called is (executor != "local")
            assert mock_confirm.called is (answer is not None)


def test_cost_plan_prices_only_the_selected_stage(subtests, orchestrator, ssp245_configs):
    cases = [
        (None, ["prepare_observations", "fit_historical", "transform_scenario"]),
        ("obs", ["prepare_observations"]),
        ("historical", ["fit_historical"]),
        ("scenario", ["transform_scenario"]),
        ("transform_scenario", ["transform_scenario"]),
    ]
    for stage, expected in cases:
        with subtests.test(stage=stage):
            with patch("saidownscale.cli.estimate_workflow", side_effect=ValueError) as mock_est:
                with pytest.raises(ValueError):
                    _render_cost_plan(orchestrator, ssp245_configs, "aws-batch", False, stage=stage)
            assert [entry[0] for entry in mock_est.call_args.args[0]] == expected


def test_resolve_stage(subtests):
    cases = [
        (None, None),
        ("all", None),
        ("obs", "prepare_observations"),
        ("historical", "fit_historical"),
        ("scenario", "transform_scenario"),
        ("transform_scenario", "transform_scenario"),
    ]
    for given, expected in cases:
        with subtests.test(given=given):
            assert _resolve_stage(given) == expected
    with pytest.raises(typer.BadParameter, match="foo"):
        _resolve_stage("foo")


def _render_environment(orchestrator, executor):
    with console.capture() as cap:
        _render_run_environment(orchestrator, executor)
    return cap.get()


def test_run_environment_table_for_aws_batch(subtests, orchestrator):
    with subtests.test("resolved"):
        resolved = {"job_definition": "srm-downscaling:7", "image": "ecr/img:abc123"}
        with patch.object(orchestrator, "resolve_job_definition", return_value=resolved):
            out = _render_environment(orchestrator, "aws-batch")
        assert all(s in out for s in ("srm-downscaling:7", "abc123", "srm-production"))
    with subtests.test("unresolvable_is_reported"):
        with patch.object(
            orchestrator, "resolve_job_definition", side_effect=RuntimeError("no creds")
        ):
            out = _render_environment(orchestrator, "aws-batch")
        assert "unresolved" in out and "RuntimeError" in out


def test_run_environment_table_for_other_executors(subtests, orchestrator):
    for executor in ("coiled", "local"):
        with subtests.test(executor=executor):
            with patch.object(orchestrator, "resolve_job_definition") as mock_resolve:
                out = _render_environment(orchestrator, executor)
            mock_resolve.assert_not_called()
            assert "job definition" not in out and "job queue" not in out
            assert all(s in out for s in (executor, "branch", "environment"))
