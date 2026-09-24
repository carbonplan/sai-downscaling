"""Tests for downscaling_config.py: VariableConfig, DownscalingConfig, and option classes."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from saidownscale.downscaling_config import (
    CacheConfig,
    DownscalingConfig,
    PipelineOptions,
    RuntimeConfig,
    VariableClipBounds,
    VariableConfig,
    _cache_version,
)

ALL_VARIABLES = ("tas", "tasmax", "tasmin", "pr", "rsds", "dtr", "hurs")
_DROP = object()


def _cfg(**overrides) -> DownscalingConfig:
    fields = {
        "downscaling_method": "BCSD",
        "gcm": "CESM2-WACCM6",
        "variable": "tas",
        "ensemble_member": "r1i1p1f1",
    }
    return DownscalingConfig(**(fields | overrides))


def _scenario_cfg(**overrides) -> DownscalingConfig:
    return _cfg(
        **(
            {"scenario": "ssp245", "predict_period_start": 2015, "predict_period_end": 2100}
            | overrides
        )
    )


def _variable_config(**overrides) -> VariableConfig:
    fields = {
        "detrend_data": True,
        "do_windowing": True,
        "running_window_length": 31,
        "running_window_step_length": 1,
        "disaggregation_method": "additive",
        "disaggregation_clim_method": "fft",
        "disaggregation_tiny_threshold": 0.0,
        "detrend_method": "additive",
        "debias_approach": "nonparametric",
    }
    fields |= overrides
    return VariableConfig(**{k: v for k, v in fields.items() if v is not _DROP})


class TestVariableConfig:
    def test_default_tables(self, subtests):
        """A moved default changes config_hash; rsds is the lone BCSD deviation (#523)."""
        additive = ("additive", True, "additive")
        multiplicative = ("multiplicative", False, "multiplicative")
        method_fields = {
            "tas": additive,
            "tasmax": additive,
            "pr": multiplicative,
            "rsds": multiplicative,
        }
        for variable, (detrend_method, detrend, disagg) in method_fields.items():
            with subtests.test(variable=variable):
                cfg = VariableConfig.for_variable(variable, "BCSD")
                assert (cfg.detrend_method, cfg.detrend_data, cfg.disaggregation_method) == (
                    detrend_method,
                    detrend,
                    disagg,
                )
                assert cfg.do_windowing is True
                assert cfg.disaggregation_clim_method == "fft"
        approaches = {
            "BCSD": {v: "nonparametric_hybrid_2sided" for v in ALL_VARIABLES}
            | {"rsds": "nonparametric"},
            "QDMSD": dict.fromkeys(ALL_VARIABLES, "qdm"),
        }
        for method, table in approaches.items():
            for variable, approach in table.items():
                with subtests.test(method=method, variable=variable):
                    assert VariableConfig.for_variable(variable, method).debias_approach == approach
        bcsd = VariableConfig.for_variable("tas", "BCSD")
        qdmsd = VariableConfig.for_variable("tas", "QDMSD")
        assert (bcsd.running_window_length, bcsd.running_window_step_length) == (31, 1)
        assert (qdmsd.running_window_length, qdmsd.running_window_step_length) == (91, 31)
        assert qdmsd.detrend_data is False

    def test_invalid_inputs_raise(self, subtests):
        cases = [
            (
                "unknown-variable",
                lambda: VariableConfig.for_variable("sfcWind", "BCSD"),
                "Unknown variable",
            ),
            (
                "unknown-method",
                lambda: VariableConfig.for_variable("tas", "BCSDSD"),
                "Unknown downscaling_method",
            ),
            ("bad-literal", lambda: _variable_config(disaggregation_method="multiply"), None),
            (
                "renamed-downscaling_method",
                lambda: _variable_config(
                    disaggregation_method=_DROP, downscaling_method="additive"
                ),
                "renamed to 'disaggregation_method'",
            ),
            (
                "renamed-tiny-threshold-#556",
                lambda: _variable_config(
                    disaggregation_tiny_threshold=_DROP, downscaling_tiny_threshold=1.0e-6
                ),
                "renamed to 'disaggregation_tiny_threshold'",
            ),
        ]
        for case_id, build, match in cases:
            with subtests.test(case=case_id):
                with pytest.raises(ValueError, match=match):
                    build()


class TestDownscalingConfig:
    def test_minimal_config_takes_identity_defaults(self, subtests):
        cfg = _cfg()
        assert (cfg.gcm, cfg.variable, cfg.ensemble_member, cfg.scenario) == (
            "CESM2-WACCM6",
            "tas",
            "r1i1p1f1",
            None,
        )
        assert (cfg.train_period_start, cfg.train_period_end) == (1978, 2014)
        assert isinstance(cfg.variable_config, VariableConfig)
        for method, approach in (("BCSD", "nonparametric_hybrid_2sided"), ("QDMSD", "qdm")):
            with subtests.test(method=method):
                assert _cfg(downscaling_method=method).variable_config.debias_approach == approach
        for gcm in ("CESM2-WACCM6", "UKESM1-1-LL", "SOME-OTHER-GCM"):
            with subtests.test(gcm=gcm):
                assert _cfg(gcm=gcm).gcm == gcm

    def test_variable_params_not_shadowed(self, subtests):
        """Shadowed accessors silently mapped variables to the wrong method (#423)."""
        cfg = _cfg()
        for attr in (
            "detrend_data",
            "detrend_method",
            "do_windowing",
            "running_window_length",
            "running_window_step_length",
            "disaggregation_method",
            "disaggregation_clim_method",
            "debias_approach",
        ):
            with subtests.test(attr=attr):
                assert not hasattr(cfg, attr)
                assert hasattr(cfg.variable_config, attr)

    def test_explicit_variable_config_is_kept(self, subtests, monkeypatch):
        custom = _variable_config(
            detrend_data=False,
            do_windowing=False,
            disaggregation_clim_method="simple",
            debias_approach="nonparametric_hybrid_2sided",
        )
        with subtests.test(source="kwarg"):
            cfg = _scenario_cfg(
                gcm="UKESM1-1-LL", ensemble_member="r2i1p1f2", variable_config=custom
            )
            assert cfg.variable_config == custom
        with subtests.test(source="env"):
            parametric = VariableConfig.for_variable("tas", "BCSD").model_copy(
                update={"debias_approach": "parametric"}
            )
            monkeypatch.setenv("SAIDOWNSCALE_VARIABLE_CONFIG", json.dumps(parametric.model_dump()))
            assert _cfg().variable_config.debias_approach == "parametric"

    def test_config_json_round_trip(self):
        """The nested variable_config must survive the CONFIG_JSON hand-off to batch VMs."""
        cfg = _scenario_cfg(
            variable="dtr",
            variable_config=VariableConfig.for_variable("dtr", "BCSD").model_copy(
                update={"debias_approach": "nonparametric"}
            ),
        )
        computed = set(DownscalingConfig.model_computed_fields.keys())
        restored = DownscalingConfig(**json.loads(json.dumps(cfg.model_dump(exclude=computed))))
        assert restored.variable_config.debias_approach == "nonparametric"
        assert restored.variable_config == cfg.variable_config
        assert restored.config_hash == cfg.config_hash

    def test_invalid_kwargs_raise(self, subtests):
        """Legacy GCM names fail at load, not inside a Batch task (#598)."""
        cases = [
            (
                "top-level-debias_approach",
                {"debias_approach": "nonparametric"},
                "variable_overrides",
            ),
            (
                "renamed-mapping_type",
                {"mapping_type": "parametric"},
                "renamed to 'debias_approach'",
            ),
            (
                "qdm-under-bcsd",
                {"variable_config": VariableConfig.for_variable("tas", "QDMSD")},
                "incompatible with debias_approach",
            ),
            (
                "qdmsd-without-qdm",
                {
                    "downscaling_method": "QDMSD",
                    "variable_config": VariableConfig.for_variable("tas", "BCSD"),
                },
                "incompatible with debias_approach",
            ),
            ("unsupported-variable", {"variable": "sfcWind"}, None),
            ("train-order", {"train_period_start": 2000, "train_period_end": 1990}, None),
            ("lat-order", {"subset_bounds": (20.0, 10.0, 0.0, 30.0)}, "lat_min"),
            ("lon-order", {"subset_bounds": (10.0, 20.0, 50.0, 30.0)}, "lon_min"),
            ("lat-min-range", {"subset_bounds": (-91.0, 0.0, 0.0, 10.0)}, "Latitude"),
            ("lat-max-range", {"subset_bounds": (0.0, 91.0, 0.0, 10.0)}, "Latitude"),
            ("legacy-cesm", {"gcm": "CESM2-WACCM"}, "CESM2-WACCM6"),
            ("legacy-ukesm", {"gcm": "UKESM"}, "UKESM1-1-LL"),
        ]
        for case_id, kwargs, match in cases:
            with subtests.test(case=case_id):
                with pytest.raises(ValueError, match=match):
                    _cfg(**kwargs)
        scenario_cases = [
            ("null-predict-start", {"predict_period_start": None}),
            ("null-predict-end", {"predict_period_end": None}),
            ("predict-order", {"predict_period_start": 2080, "predict_period_end": 2015}),
        ]
        for case_id, kwargs in scenario_cases:
            with subtests.test(case=case_id):
                with pytest.raises(ValidationError):
                    _scenario_cfg(**kwargs)
        with subtests.test(case="missing-downscaling_method"):
            with pytest.raises(ValidationError, match="'downscaling_method' is required"):
                DownscalingConfig(gcm="CESM2-WACCM6", variable="tas", ensemble_member="r1i1p1f1")

    def test_stale_env_vars_raise(self, subtests, monkeypatch):
        """pydantic-settings drops unknown env vars case-insensitively, so each must be rejected."""
        cases = [
            ("SAIDOWNSCALE_DEBIAS_APPROACH", _cfg, "variable_overrides"),
            ("SAIDOWNSCALE_MAPPING_TYPE", _cfg, "variable_overrides"),
            ("saidownscale_debias_approach", _cfg, "variable_overrides"),
            ("BCSD_BRANCH", _cfg, "BCSD_BRANCH -> SAIDOWNSCALE_BRANCH"),
            ("BCSD_BATCH_JOB_QUEUE", PipelineOptions, "SAIDOWNSCALE_BATCH_JOB_QUEUE"),
            ("bcsd_branch", PipelineOptions, "SAIDOWNSCALE_BRANCH"),
        ]
        for env_var, build, match in cases:
            with subtests.test(env_var=env_var), monkeypatch.context() as m:
                m.setenv(env_var, "nonparametric")
                with pytest.raises(ValidationError, match=match):
                    build()

    def test_run_id(self, subtests):
        regional = _scenario_cfg(subset_bounds=(-35.0, -22.0, 16.0, 33.0))
        assert regional.subset_bounds == (-35.0, -22.0, 16.0, 33.0)
        assert "subset" in regional.run_id
        assert _cfg().run_id == "CESM2-WACCM6_tas_r1i1p1f1"
        assert _scenario_cfg().run_id == "CESM2-WACCM6_tas_r1i1p1f1_SSP245"
        for label in ("r12i1p1f2", "01", "r10i1p1f2"):
            with subtests.test(label=label):
                assert f"_{label}" in _cfg(ensemble_member=label).run_id

    def test_config_hash(self):
        minimal = _cfg()
        assert len(minimal.config_hash) == 12
        assert set(minimal.config_hash) <= set("0123456789abcdef")
        assert minimal.config_hash == _cfg().config_hash
        hashes = {
            minimal.config_hash,
            _scenario_cfg().config_hash,
            _scenario_cfg(
                variable="pr", ensemble_member="r2i1p1f1", scenario="G6-1.5K"
            ).config_hash,
        }
        assert len(hashes) == 3

    def test_is_sai_scenario(self, subtests):
        for scenario, expected in (
            (None, False),
            ("ssp245", False),
            ("G6-1.5K", True),
            ("SAI-2050", True),
        ):
            with subtests.test(scenario=scenario):
                cfg = _cfg() if scenario is None else _scenario_cfg(scenario=scenario)
                assert bool(cfg.is_sai_scenario) is expected


class TestOptions:
    def test_unset_options_take_defaults(self, tmp_path):
        opts = PipelineOptions(scratch_dir=str(tmp_path), output_dir=str(tmp_path))
        assert opts.environment == "qa"
        assert opts.branch == _cache_version
        assert "+" not in opts.branch
        assert opts.executor == "coiled"
        assert opts.apply_ocean_mask is False
        assert opts.clip_values is True
        assert (opts.clip_bounds["pr"].min, opts.clip_bounds["pr"].max) == (0.0, None)
        assert (opts.clip_bounds["hurs"].min, opts.clip_bounds["hurs"].max) == (0.0, 105.0)
        cache = CacheConfig()
        assert (cache.environment, cache.branch) == ("qa", _cache_version)
        assert (cache.force_recompute, cache.check_integrity) == (False, True)
        runtime = RuntimeConfig()
        assert (runtime.use_coiled, runtime.coiled_region, runtime.max_parallel_tasks) == (
            True,
            "us-west-2",
            None,
        )

    def test_overrides(self, subtests, monkeypatch):
        with subtests.test(cls="PipelineOptions"):
            opts = PipelineOptions(
                clip_values=False,
                clip_bounds={"pr": VariableClipBounds(min=0.0, max=500.0)},
                branch="v2",
                executor="aws-batch",
            )
            assert opts.clip_values is False
            assert opts.clip_bounds["pr"].max == 500.0
            assert (opts.branch, opts.executor) == ("v2", "aws-batch")
            assert PipelineOptions().model_copy(update={"branch": "v4"}).branch == "v4"
        with subtests.test(cls="PipelineOptions-env"), monkeypatch.context() as m:
            m.setenv("SAIDOWNSCALE_BRANCH", "v3")
            assert PipelineOptions().branch == "v3"
        with subtests.test(cls="CacheConfig"):
            cfg = CacheConfig(environment="production", branch="v3", force_recompute=True)
            assert (cfg.environment, cfg.branch, cfg.force_recompute) == ("production", "v3", True)
        with subtests.test(cls="RuntimeConfig"):
            cfg = RuntimeConfig(use_coiled=False, max_parallel_tasks=4)
            assert (cfg.use_coiled, cfg.max_parallel_tasks) == (False, 4)

    def test_invalid_pipeline_options_raise(self, tmp_path):
        dirs = {"scratch_dir": str(tmp_path), "output_dir": str(tmp_path)}
        with pytest.raises(ValidationError):
            PipelineOptions(**dirs, executor="slurm")
        with pytest.raises(ValueError, match="use_coiled"):
            PipelineOptions(**dirs, use_coiled=True)


def test_usage_examples_set_downscaling_method(subtests):
    """downscaling_method has no default, so a copied example without it raises."""
    import inspect
    import re

    import saidownscale.downscaling_config

    source = inspect.getsource(saidownscale.downscaling_config)
    _, _, block = source.partition("Usage Examples:")
    assert block
    calls = re.findall(r"DownscalingConfig\((.*?)\n\)", block, flags=re.DOTALL)
    assert len(calls) == 4
    for call in calls:
        with subtests.test(example=" ".join(re.findall(r'"([^"]+)"', call)[:3])):
            assert "downscaling_method=" in call
    _, _, yaml_example = block.partition("# Config file: configs/cesm_tas.yaml")
    assert "downscaling_method:" in yaml_example.split('"""')[1]
