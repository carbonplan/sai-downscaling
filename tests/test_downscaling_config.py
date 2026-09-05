"""Tests for downscaling_config.py: VariableConfig, DownscalingConfig, CacheConfig, RuntimeConfig."""

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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config() -> DownscalingConfig:
    """Minimal valid DownscalingConfig for a historical-only run (no scenario)."""
    return DownscalingConfig(
        downscaling_method="BCSD", gcm="CESM2-WACCM6", variable="tas", ensemble_member="r1i1p1f1"
    )


@pytest.fixture
def scenario_config() -> DownscalingConfig:
    """DownscalingConfig with a standard (non-SAI) scenario."""
    return DownscalingConfig(
        gcm="CESM2-WACCM6",
        downscaling_method="BCSD",
        variable="tas",
        ensemble_member="r1i1p1f1",
        scenario="ssp245",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def sai_config() -> DownscalingConfig:
    """DownscalingConfig with a G6-SAI scenario."""
    return DownscalingConfig(
        gcm="CESM2-WACCM6",
        downscaling_method="BCSD",
        variable="pr",
        ensemble_member="r2i1p1f1",
        scenario="G6-1.5K",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def regional_config() -> DownscalingConfig:
    """DownscalingConfig with a spatial subset (South Africa region)."""
    return DownscalingConfig(
        gcm="UKESM1-1-LL",
        downscaling_method="BCSD",
        variable="tasmax",
        ensemble_member="01",
        scenario="ssp245",
        predict_period_start=2015,
        predict_period_end=2100,
        subset_bounds=(-35.0, -22.0, 16.0, 33.0),
    )


# ---------------------------------------------------------------------------
# VariableConfig
# ---------------------------------------------------------------------------


class TestVariableConfig:
    """Tests for VariableConfig and its for_variable factory."""

    _EXPECTED = {
        "tas": {
            "detrend_data": True,
            "detrend_method": "additive",
            "do_windowing": True,
            "disaggregation_method": "additive",
            "disaggregation_clim_method": "fft",
        },
        "tasmax": {
            "detrend_data": True,
            "detrend_method": "additive",
            "do_windowing": True,
            "disaggregation_method": "additive",
            "disaggregation_clim_method": "fft",
        },
        "pr": {
            "detrend_data": False,
            "detrend_method": "multiplicative",
            "do_windowing": True,
            "disaggregation_method": "multiplicative",
            "disaggregation_clim_method": "fft",
        },
        "rsds": {
            "detrend_data": False,
            "detrend_method": "multiplicative",
            "do_windowing": True,
            "disaggregation_method": "multiplicative",
            "disaggregation_clim_method": "fft",
        },
    }

    def test_known_variables_return_config(self, subtests):
        for variable in self._EXPECTED:
            with subtests.test(variable=variable):
                cfg = VariableConfig.for_variable(variable, "BCSD")
                assert isinstance(cfg, VariableConfig)

    def test_variable_settings_are_correct(self, subtests):
        for variable, expected in self._EXPECTED.items():
            with subtests.test(variable=variable):
                cfg = VariableConfig.for_variable(variable, "BCSD")
                assert cfg.detrend_data == expected["detrend_data"]
                assert cfg.detrend_method == expected["detrend_method"]
                assert cfg.do_windowing == expected["do_windowing"]
                assert cfg.disaggregation_method == expected["disaggregation_method"]
                assert cfg.disaggregation_clim_method == expected["disaggregation_clim_method"]

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            VariableConfig.for_variable("sfcWind", "BCSD")

    def test_direct_construction(self):
        cfg = VariableConfig(
            detrend_data=False,
            do_windowing=False,
            running_window_length=31,
            running_window_step_length=1,
            disaggregation_method="multiplicative",
            disaggregation_clim_method="simple",
            disaggregation_tiny_threshold=0.0,
            detrend_method="additive",
            debias_approach="nonparametric",
        )
        assert cfg.detrend_data is False
        assert cfg.disaggregation_method == "multiplicative"

    def test_invalid_disaggregation_method_raises(self):
        with pytest.raises(ValidationError):
            VariableConfig(
                detrend_data=True,
                do_windowing=True,
                running_window_length=31,
                running_window_step_length=1,
                disaggregation_method="multiply",  # not a valid Literal
                disaggregation_clim_method="fft",
                disaggregation_tiny_threshold=0.0,
                detrend_method="additive",
                debias_approach="nonparametric",
            )

    def test_renamed_disaggregation_keys_raise(self):
        """The pre-rename keys must fail loudly, not be dropped by extra="ignore"."""
        with pytest.raises(ValidationError, match="renamed to 'disaggregation_method'"):
            VariableConfig(
                detrend_data=True,
                do_windowing=True,
                running_window_length=31,
                running_window_step_length=1,
                downscaling_method="additive",  # pre-rename name
                disaggregation_clim_method="fft",
                disaggregation_tiny_threshold=0.0,
                detrend_method="additive",
                debias_approach="nonparametric",
            )

    def test_renamed_tiny_threshold_key_raises(self):
        """The #556 spelling of the clipping threshold must be rejected, not dropped."""
        with pytest.raises(ValidationError, match="renamed to 'disaggregation_tiny_threshold'"):
            VariableConfig(
                detrend_data=True,
                do_windowing=True,
                running_window_length=31,
                running_window_step_length=1,
                disaggregation_method="multiplicative",
                disaggregation_clim_method="fft",
                downscaling_tiny_threshold=1.0e-6,  # pre-rename name
                detrend_method="additive",
                debias_approach="nonparametric",
            )

    def test_unknown_downscaling_method_raises(self):
        with pytest.raises(ValueError, match="Unknown downscaling_method"):
            VariableConfig.for_variable("tas", "BCSDSD")

    def test_qdmsd_table_differs_from_bcsd(self):
        """QDMSD rows carry the qdm approach and the wider seasonal window."""
        bcsd = VariableConfig.for_variable("tas", "BCSD")
        qdmsd = VariableConfig.for_variable("tas", "QDMSD")
        assert bcsd.debias_approach == "nonparametric_hybrid_2sided"
        assert qdmsd.debias_approach == "qdm"
        assert (bcsd.running_window_length, bcsd.running_window_step_length) == (31, 1)
        assert (qdmsd.running_window_length, qdmsd.running_window_step_length) == (91, 31)
        assert bcsd.detrend_data is True
        assert qdmsd.detrend_data is False


# ---------------------------------------------------------------------------
# DownscalingConfig – construction & auto-population
# ---------------------------------------------------------------------------


class TestBCSDConfigConstruction:
    """DownscalingConfig is valid for a range of realistic inputs."""

    def test_minimal_historical_config(self, minimal_config):
        assert minimal_config.gcm == "CESM2-WACCM6"
        assert minimal_config.variable == "tas"
        assert minimal_config.ensemble_member == "r1i1p1f1"
        assert minimal_config.scenario is None

    def test_variable_config_auto_populated(self, minimal_config):
        assert minimal_config.variable_config is not None
        assert isinstance(minimal_config.variable_config, VariableConfig)

    def test_variable_params_not_shadowed_on_bcsd_config(self, subtests, minimal_config):
        """VariableConfig-derived params must live only on ``variable_config``.

        Exposing them as computed fields on ``DownscalingConfig`` (issue #423) silently
        mapped variables to the wrong method/attrs when the accessor's hardcoded
        fallback diverged from the variable's real config. They must be reached
        through ``config.variable_config`` so there is a single source of truth.
        """
        shadowed = (
            "detrend_data",
            "detrend_method",
            "do_windowing",
            "running_window_length",
            "running_window_step_length",
            "disaggregation_method",
            "disaggregation_clim_method",
            "debias_approach",
        )
        for attr in shadowed:
            with subtests.test(attr=attr):
                assert not hasattr(minimal_config, attr), (
                    f"DownscalingConfig must not expose {attr!r}; use config.variable_config.{attr}"
                )
                assert hasattr(minimal_config.variable_config, attr)

    def test_default_train_period(self, minimal_config):
        assert minimal_config.train_period_start == 1978
        assert minimal_config.train_period_end == 2014

    def test_default_environment_and_branch(self):
        opts = PipelineOptions()
        assert opts.environment == "qa"
        assert opts.branch == _cache_version

    def test_debias_approach_lives_on_variable_config(self, minimal_config):
        assert minimal_config.variable_config.debias_approach == "nonparametric_hybrid_2sided"

    def test_top_level_debias_approach_key_raises(self):
        """A moved key must fail loudly. extra='ignore' would otherwise drop it silently."""
        with pytest.raises(ValidationError, match="variable_overrides"):
            DownscalingConfig(
                gcm="CESM2-WACCM6",
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member="r1i1p1f1",
                debias_approach="nonparametric",
            )

    def test_variable_config_carries_debias_approach(self):
        cfg = DownscalingConfig(
            gcm="CESM2-WACCM6",
            downscaling_method="BCSD",
            variable="dtr",
            ensemble_member="r1i1p1f1",
            variable_config=VariableConfig.for_variable("dtr", "BCSD").model_copy(
                update={"debias_approach": "nonparametric"}
            ),
        )
        assert cfg.variable_config.debias_approach == "nonparametric"

    def test_qdm_requires_qdmsd_downscaling_method(self):
        """A qdm debias_approach under BCSD would detrend around a trend-carrying method."""
        with pytest.raises(ValidationError, match="incompatible with debias_approach"):
            DownscalingConfig(
                gcm="CESM2-WACCM6",
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member="r1i1p1f1",
                variable_config=VariableConfig.for_variable("tas", "QDMSD"),
            )

    def test_qdmsd_requires_qdm_debias_approach(self):
        """QDMSD without qdm is not quantile delta mapping at all."""
        with pytest.raises(ValidationError, match="incompatible with debias_approach"):
            DownscalingConfig(
                gcm="CESM2-WACCM6",
                downscaling_method="QDMSD",
                variable="tas",
                ensemble_member="r1i1p1f1",
                variable_config=VariableConfig.for_variable("tas", "BCSD"),
            )

    def test_matching_method_and_approach_are_accepted(self, subtests):
        for method, approach in (("BCSD", "nonparametric_hybrid_2sided"), ("QDMSD", "qdm")):
            with subtests.test(method=method):
                cfg = DownscalingConfig(
                    gcm="CESM2-WACCM6",
                    downscaling_method=method,
                    variable="tas",
                    ensemble_member="r1i1p1f1",
                )
                assert cfg.variable_config.debias_approach == approach

    def test_missing_downscaling_method_raises(self):
        """There is no default: a config that omits the method must fail loudly."""
        with pytest.raises(ValidationError, match="'downscaling_method' is required"):
            DownscalingConfig(gcm="CESM2-WACCM6", variable="tas", ensemble_member="r1i1p1f1")

    def test_renamed_mapping_type_key_raises(self):
        """The pre-rename ``mapping_type`` key must fail loudly, not be silently ignored."""
        with pytest.raises(ValidationError, match="renamed to 'debias_approach'"):
            DownscalingConfig(
                gcm="CESM2-WACCM6",
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member="r1i1p1f1",
                mapping_type="parametric",
            )

    @pytest.mark.parametrize("env_var", ["BCSD_DEBIAS_APPROACH", "BCSD_MAPPING_TYPE"])
    def test_moved_key_via_env_raises(self, monkeypatch, env_var):
        """A moved key set through the environment must fail loudly too.

        pydantic-settings filters env vars against the model's fields before any
        validator runs, so without an explicit ``os.environ`` check these would be
        dropped silently, which is exactly the misconfiguration the move removes.
        """
        monkeypatch.setenv(env_var, "nonparametric")
        with pytest.raises(ValidationError, match="variable_overrides"):
            DownscalingConfig(
                downscaling_method="BCSD",
                gcm="CESM2-WACCM6",
                variable="tas",
                ensemble_member="r1i1p1f1",
            )

    def test_moved_key_env_check_is_case_insensitive(self, monkeypatch):
        """pydantic-settings matches env vars case-insensitively; so must the check."""
        monkeypatch.setenv("bcsd_debias_approach", "nonparametric")
        with pytest.raises(ValidationError, match="variable_overrides"):
            DownscalingConfig(
                downscaling_method="BCSD",
                gcm="CESM2-WACCM6",
                variable="tas",
                ensemble_member="r1i1p1f1",
            )

    def test_variable_config_env_override_is_the_supported_path(self, monkeypatch):
        """BCSD_VARIABLE_CONFIG replaces the removed BCSD_DEBIAS_APPROACH env override."""
        monkeypatch.setenv(
            "BCSD_VARIABLE_CONFIG",
            json.dumps(
                VariableConfig.for_variable("tas", "BCSD")
                .model_copy(update={"debias_approach": "parametric"})
                .model_dump()
            ),
        )
        cfg = DownscalingConfig(
            downscaling_method="BCSD",
            gcm="CESM2-WACCM6",
            variable="tas",
            ensemble_member="r1i1p1f1",
        )
        assert cfg.variable_config.debias_approach == "parametric"

    def test_explicit_variable_config_not_overwritten(self):
        """Explicitly supplied variable_config must survive post-init."""
        custom_vc = VariableConfig(
            detrend_data=False,
            do_windowing=False,
            running_window_length=31,
            running_window_step_length=1,
            disaggregation_method="additive",
            disaggregation_clim_method="simple",
            disaggregation_tiny_threshold=0.0,
            detrend_method="additive",
            debias_approach="nonparametric_hybrid_2sided",
        )
        cfg = DownscalingConfig(
            gcm="UKESM1-1-LL",
            downscaling_method="BCSD",
            variable="tas",
            ensemble_member="r2i1p1f2",
            scenario="ssp245",
            predict_period_start=2015,
            predict_period_end=2100,
            variable_config=custom_vc,
        )
        assert cfg.variable_config.detrend_data is False
        assert cfg.variable_config.do_windowing is False

    def test_all_supported_variables_construct(self, subtests):
        for var in ("tas", "tasmax", "pr"):
            with subtests.test(variable=var):
                cfg = DownscalingConfig(
                    downscaling_method="BCSD",
                    gcm="CESM2-WACCM6",
                    variable=var,
                    ensemble_member="r1i1p1f1",
                )
                assert cfg.variable == var

    def test_all_supported_gcms_construct(self, subtests):
        for gcm in ("CESM2-WACCM6", "UKESM1-1-LL"):
            with subtests.test(gcm=gcm):
                cfg = DownscalingConfig(
                    downscaling_method="BCSD", gcm=gcm, variable="tas", ensemble_member="r1i1p1f1"
                )
                assert cfg.gcm == gcm

    def test_apply_ocean_mask_defaults_false(self):
        assert PipelineOptions().apply_ocean_mask is False

    def test_apply_ocean_mask_can_be_disabled(self):
        opts = PipelineOptions(apply_ocean_mask=False)
        assert opts.apply_ocean_mask is False

    def test_clip_values_defaults_true(self):
        assert PipelineOptions().clip_values is True

    def test_clip_values_can_be_disabled(self):
        assert PipelineOptions(clip_values=False).clip_values is False

    def test_clip_bounds_pr_default(self):
        bounds = PipelineOptions().clip_bounds
        assert bounds["pr"].min == 0.0
        assert bounds["pr"].max is None

    def test_clip_bounds_hurs_default(self):
        bounds = PipelineOptions().clip_bounds
        assert bounds["hurs"].min == 0.0
        assert bounds["hurs"].max == 105.0

    def test_clip_bounds_can_be_overridden(self):
        opts = PipelineOptions(clip_bounds={"pr": VariableClipBounds(min=0.0, max=500.0)})
        assert opts.clip_bounds["pr"].max == 500.0

    def test_model_copy_branch_override(self):
        opts = PipelineOptions()
        v2 = opts.model_copy(update={"branch": "v2"})
        assert v2.branch == "v2"
        assert opts.branch == _cache_version


# ---------------------------------------------------------------------------
# debias_approach – per-variable defaults and CONFIG_JSON round-trip
# ---------------------------------------------------------------------------


class TestVariableConfigDebiasDefaults:
    """Every supported variable must carry an explicit debias_approach default.

    ``variable_config`` is hashed into ``config_hash``, so a default that moves
    quietly invalidates every cached artifact and changes every production run.
    Pinning both tables here forces such a move to surface as a reviewable diff.
    """

    def test_all_variables_have_debias_approach(self, subtests):
        expected = {
            "BCSD": {
                "tas": "nonparametric_hybrid_2sided",
                "tasmax": "nonparametric_hybrid_2sided",
                "tasmin": "nonparametric_hybrid_2sided",
                "pr": "nonparametric_hybrid_2sided",
                "dtr": "nonparametric_hybrid_2sided",
                "hurs": "nonparametric_hybrid_2sided",
                # rsds is the one BCSD deviation from the NEX-GDDP hybrid default (#523).
                "rsds": "nonparametric",
            },
            # QDMSD is uniform: the qdm approach must not pick up the BCSD rsds special case.
            "QDMSD": dict.fromkeys(("tas", "tasmax", "tasmin", "pr", "rsds", "dtr", "hurs"), "qdm"),
        }
        for method, table in expected.items():
            for var, approach in table.items():
                with subtests.test(method=method, variable=var):
                    assert VariableConfig.for_variable(var, method).debias_approach == approach


class TestConfigJsonRoundTrip:
    """A resolved config must survive the Coiled hand-off unchanged.

    ``orchestration.py`` dumps the config to CONFIG_JSON and ``batch_runner.py``
    rehydrates it. A per-variable debias_approach only reaches the VM if the nested
    variable_config round-trips intact.
    """

    def test_resolved_debias_approach_survives_round_trip(self):
        cfg = DownscalingConfig(
            gcm="CESM2-WACCM6",
            downscaling_method="BCSD",
            variable="dtr",
            ensemble_member="r1i1p1f1",
            scenario="ssp245",
            predict_period_start=2015,
            predict_period_end=2100,
            variable_config=VariableConfig.for_variable("dtr", "BCSD").model_copy(
                update={"debias_approach": "nonparametric"}
            ),
        )
        computed = set(DownscalingConfig.model_computed_fields.keys())
        payload = json.loads(json.dumps(cfg.model_dump(exclude=computed)))
        restored = DownscalingConfig(**payload)
        assert restored.variable_config.debias_approach == "nonparametric"
        assert restored.variable_config == cfg.variable_config
        assert restored.config_hash == cfg.config_hash


# ---------------------------------------------------------------------------
# DownscalingConfig – computed fields
# ---------------------------------------------------------------------------


class TestBCSDConfigComputedFields:
    """run_id, config_hash, and is_sai_scenario computed fields."""

    def test_run_id_historical_only(self, minimal_config):
        assert minimal_config.run_id == "CESM2-WACCM6_tas_r1i1p1f1"

    def test_run_id_with_scenario(self, scenario_config):
        assert scenario_config.run_id == "CESM2-WACCM6_tas_r1i1p1f1_SSP245"

    def test_run_id_includes_subset_marker(self, regional_config):
        assert "subset" in regional_config.run_id

    def test_run_id_contains_ensemble_label(self, subtests):
        for label in ("r1i1p1f1", "r12i1p1f2", "01", "r10i1p1f2"):
            with subtests.test(label=label):
                cfg = DownscalingConfig(
                    downscaling_method="BCSD",
                    gcm="CESM2-WACCM6",
                    variable="tas",
                    ensemble_member=label,
                )
                assert f"_{label}" in cfg.run_id

    def test_config_hash_is_12_char_hex(self, minimal_config):
        h = minimal_config.config_hash
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_config_hash_is_stable(self):
        cfg_a = DownscalingConfig(
            downscaling_method="BCSD",
            gcm="CESM2-WACCM6",
            variable="tas",
            ensemble_member="r1i1p1f1",
        )
        cfg_b = DownscalingConfig(
            downscaling_method="BCSD",
            gcm="CESM2-WACCM6",
            variable="tas",
            ensemble_member="r1i1p1f1",
        )
        assert cfg_a.config_hash == cfg_b.config_hash

    def test_config_hash_differs_across_configs(
        self, subtests, minimal_config, scenario_config, sai_config
    ):
        pairs = [
            ("minimal_vs_scenario", minimal_config, scenario_config),
            ("scenario_vs_sai", scenario_config, sai_config),
            ("minimal_vs_sai", minimal_config, sai_config),
        ]
        for label, cfg_a, cfg_b in pairs:
            with subtests.test(pair=label):
                assert cfg_a.config_hash != cfg_b.config_hash

    def test_is_sai_false_for_ssp(self, scenario_config):
        assert not scenario_config.is_sai_scenario

    def test_is_sai_false_for_historical_only(self, minimal_config):
        assert not minimal_config.is_sai_scenario

    def test_is_sai_true_for_g6(self, sai_config):
        assert sai_config.is_sai_scenario

    def test_is_sai_true_for_sai_keyword(self):
        cfg = DownscalingConfig(
            gcm="CESM2-WACCM6",
            downscaling_method="BCSD",
            variable="tas",
            ensemble_member="r1i1p1f1",
            scenario="SAI-2050",
            predict_period_start=2015,
            predict_period_end=2100,
        )
        assert cfg.is_sai_scenario

    def test_pr_does_not_detrend(self, sai_config):
        assert sai_config.variable_config.detrend_data is False

    def test_tas_does_detrend(self, scenario_config):
        assert scenario_config.variable_config.detrend_data is True


# ---------------------------------------------------------------------------
# DownscalingConfig – validation
# ---------------------------------------------------------------------------


class TestBCSDConfigValidation:
    """Field validators reject invalid inputs with informative errors."""

    def test_scenario_with_explicit_null_predict_start_raises(self):
        with pytest.raises(ValidationError):
            DownscalingConfig(
                gcm="CESM2-WACCM6",
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member="r1i1p1f1",
                scenario="ssp245",
                predict_period_start=None,  # explicit None triggers the validator
                predict_period_end=2100,
            )

    def test_scenario_with_explicit_null_predict_end_raises(self):
        with pytest.raises(ValidationError):
            DownscalingConfig(
                gcm="CESM2-WACCM6",
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member="r1i1p1f1",
                scenario="ssp245",
                predict_period_start=2015,
                predict_period_end=None,  # explicit None triggers the validator
            )

    def test_train_period_end_before_start_raises(self):
        with pytest.raises(ValidationError):
            DownscalingConfig(
                gcm="CESM2-WACCM6",
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member="r1i1p1f1",
                train_period_start=2000,
                train_period_end=1990,  # end before start
            )

    def test_predict_period_end_before_start_raises(self):
        with pytest.raises(ValidationError):
            DownscalingConfig(
                gcm="CESM2-WACCM6",
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member="r1i1p1f1",
                scenario="ssp245",
                predict_period_start=2080,
                predict_period_end=2015,  # end before start
            )

    def test_unsupported_variable_raises(self):
        with pytest.raises(ValidationError):
            DownscalingConfig(
                downscaling_method="BCSD",
                gcm="CESM2-WACCM6",
                variable="sfcWind",
                ensemble_member="r1i1p1f1",
            )

    def test_subset_bounds_lat_min_ge_max_raises(self):
        with pytest.raises(ValidationError, match="lat_min"):
            DownscalingConfig(
                gcm="CESM2-WACCM6",
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member="r1i1p1f1",
                subset_bounds=(20.0, 10.0, 0.0, 30.0),  # lat_min > lat_max
            )

    def test_subset_bounds_lon_min_ge_max_raises(self):
        with pytest.raises(ValidationError, match="lon_min"):
            DownscalingConfig(
                gcm="CESM2-WACCM6",
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member="r1i1p1f1",
                subset_bounds=(10.0, 20.0, 50.0, 30.0),  # lon_min > lon_max
            )

    def test_subset_bounds_latitude_out_of_range(self, subtests):
        invalid_cases = [
            ((-91.0, 0.0, 0.0, 10.0), "lat_min below -90"),
            ((0.0, 91.0, 0.0, 10.0), "lat_max above 90"),
        ]
        for bounds, desc in invalid_cases:
            with subtests.test(desc=desc):
                with pytest.raises(ValidationError, match="Latitude"):
                    DownscalingConfig(
                        gcm="CESM2-WACCM6",
                        downscaling_method="BCSD",
                        variable="tas",
                        ensemble_member="r1i1p1f1",
                        subset_bounds=bounds,
                    )

    def test_valid_regional_config_accepted(self, regional_config):
        assert regional_config.subset_bounds == (-35.0, -22.0, 16.0, 33.0)


# ---------------------------------------------------------------------------
# PipelineOptions – branch defaulting
# ---------------------------------------------------------------------------


class TestBranchDefaulting:
    """Branch defaults to the installed package version; each release gets a clean slate."""

    def test_default_branch_is_package_version(self):
        assert PipelineOptions().branch == _cache_version

    def test_default_branch_has_no_local_segment(self):
        assert "+" not in PipelineOptions().branch

    def test_explicit_branch_override(self):
        opts = PipelineOptions(branch="v2")
        assert opts.branch == "v2"

    def test_env_var_overrides_branch(self, monkeypatch):
        monkeypatch.setenv("BCSD_BRANCH", "v3")
        opts = PipelineOptions()
        assert opts.branch == "v3"

    def test_cache_config_default_branch_is_package_version(self):
        assert CacheConfig().branch == _cache_version

    def test_pipeline_options_and_cache_config_share_same_default(self):
        assert PipelineOptions().branch == CacheConfig().branch


# ---------------------------------------------------------------------------
# DownscalingConfig – to_legacy_kwargs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CacheConfig
# ---------------------------------------------------------------------------


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.environment == "qa"
        assert cfg.branch == _cache_version
        assert cfg.force_recompute is False
        assert cfg.check_integrity is True

    def test_custom_values(self):
        cfg = CacheConfig(environment="production", branch="v3", force_recompute=True)
        assert cfg.environment == "production"
        assert cfg.branch == "v3"
        assert cfg.force_recompute is True


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------


class TestRuntimeConfig:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.use_coiled is True
        assert cfg.coiled_region == "us-west-2"
        assert cfg.max_parallel_tasks is None

    def test_disable_coiled(self):
        cfg = RuntimeConfig(use_coiled=False)
        assert cfg.use_coiled is False

    def test_max_parallel_tasks_can_be_set(self):
        cfg = RuntimeConfig(max_parallel_tasks=4)
        assert cfg.max_parallel_tasks == 4


# ---------------------------------------------------------------------------
# Module-level usage examples
# ---------------------------------------------------------------------------


class TestUsageExamples:
    """The usage-example block at the foot of downscaling_config.py must stay copy-pasteable.

    ``downscaling_method`` is required and has no default, so an example that omits it
    raises a ValidationError the moment anyone copies it.
    """

    @staticmethod
    def _example_block() -> str:
        import inspect

        import saidownscale.downscaling_config

        source = inspect.getsource(saidownscale.downscaling_config)
        _, _, block = source.partition("Usage Examples:")
        assert block, "usage-example block not found in srm/downscaling_config.py"
        return block

    def test_every_constructor_call_sets_downscaling_method(self, subtests):
        import re

        calls = re.findall(r"DownscalingConfig\((.*?)\n\)", self._example_block(), flags=re.DOTALL)
        assert len(calls) == 4  # examples 1-4; example 5 builds from the YAML block
        for i, call in enumerate(calls, start=1):
            with subtests.test(example=i):
                assert "downscaling_method=" in call

    def test_yaml_example_sets_downscaling_method(self):
        block = self._example_block()
        _, _, yaml_example = block.partition("# Config file: configs/cesm_tas.yaml")
        assert "downscaling_method:" in yaml_example.split('"""')[1]


# ---------------------------------------------------------------------------
# PipelineOptions.executor
# ---------------------------------------------------------------------------


class TestExecutorOption:
    def test_defaults_to_coiled(self, tmp_path):
        options = PipelineOptions(scratch_dir=str(tmp_path), output_dir=str(tmp_path))
        assert options.executor == "coiled"

    def test_accepts_aws_batch(self, tmp_path):
        options = PipelineOptions(
            scratch_dir=str(tmp_path), output_dir=str(tmp_path), executor="aws-batch"
        )
        assert options.executor == "aws-batch"

    def test_rejects_unknown_executor(self, tmp_path):
        with pytest.raises(ValidationError):
            PipelineOptions(scratch_dir=str(tmp_path), output_dir=str(tmp_path), executor="slurm")

    def test_rejects_legacy_use_coiled_key(self, tmp_path):
        with pytest.raises(ValueError, match="use_coiled"):
            PipelineOptions(scratch_dir=str(tmp_path), output_dir=str(tmp_path), use_coiled=True)


class TestLegacyGcmNames:
    """Issue #598: pre-rename model names fail at load, not inside a Batch task."""

    @pytest.mark.parametrize(
        "legacy, replacement",
        [("CESM2-WACCM", "CESM2-WACCM6"), ("UKESM", "UKESM1-1-LL")],
    )
    def test_rejects_legacy_gcm_name(self, legacy, replacement):
        with pytest.raises(ValueError, match=replacement):
            DownscalingConfig(
                downscaling_method="BCSD",
                gcm=legacy,
                variable="tas",
                ensemble_member="r1i1p1f1",
            )

    def test_unknown_name_is_not_rejected_here(self):
        """Only the two legacy spellings are mapped; catalog lookup handles the rest."""
        config = DownscalingConfig(
            downscaling_method="BCSD",
            gcm="SOME-OTHER-GCM",
            variable="tas",
            ensemble_member="r1i1p1f1",
        )
        assert config.gcm == "SOME-OTHER-GCM"
