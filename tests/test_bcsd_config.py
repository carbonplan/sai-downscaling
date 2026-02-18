"""Tests for bcsd_config.py: VariableConfig, BCSDConfig, CacheConfig, RuntimeConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from srm.bcsd_config import BCSDConfig, CacheConfig, RuntimeConfig, VariableConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config() -> BCSDConfig:
    """Minimal valid BCSDConfig for a historical-only run (no scenario)."""
    return BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member=0)


@pytest.fixture
def scenario_config() -> BCSDConfig:
    """BCSDConfig with a standard (non-SAI) scenario."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="tas",
        ensemble_member=0,
        scenario="ssp245",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def sai_config() -> BCSDConfig:
    """BCSDConfig with a G6-SAI scenario."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="pr",
        ensemble_member=1,
        scenario="G6-1.5K",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def regional_config() -> BCSDConfig:
    """BCSDConfig with a spatial subset (South Africa region)."""
    return BCSDConfig(
        gcm="MIROC-ES2H",
        variable="tasmax",
        ensemble_member=0,
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
            "downscaling_method": "additive",
            "downscaling_clim_method": "fft",
        },
        "tasmax": {
            "detrend_data": True,
            "detrend_method": "additive",
            "do_windowing": True,
            "downscaling_method": "additive",
            "downscaling_clim_method": "fft",
        },
        "pr": {
            "detrend_data": False,
            "detrend_method": "multiplicative",
            "do_windowing": True,
            "downscaling_method": "multiplicative",
            "downscaling_clim_method": "simple",
        },
        "rsds": {
            "detrend_data": True,
            "detrend_method": "multiplicative",
            "do_windowing": True,
            "downscaling_method": "multiplicative",
            "downscaling_clim_method": "simple",
        },
    }

    def test_known_variables_return_config(self, subtests):
        for variable in self._EXPECTED:
            with subtests.test(variable=variable):
                cfg = VariableConfig.for_variable(variable)
                assert isinstance(cfg, VariableConfig)

    def test_variable_settings_are_correct(self, subtests):
        for variable, expected in self._EXPECTED.items():
            with subtests.test(variable=variable):
                cfg = VariableConfig.for_variable(variable)
                assert cfg.detrend_data == expected["detrend_data"]
                assert cfg.detrend_method == expected["detrend_method"]
                assert cfg.do_windowing == expected["do_windowing"]
                assert cfg.downscaling_method == expected["downscaling_method"]
                assert cfg.downscaling_clim_method == expected["downscaling_clim_method"]

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            VariableConfig.for_variable("sfcWind")

    def test_direct_construction(self):
        cfg = VariableConfig(
            detrend_data=False,
            do_windowing=False,
            downscaling_method="multiplicative",
            downscaling_clim_method="simple",
        )
        assert cfg.detrend_data is False
        assert cfg.downscaling_method == "multiplicative"

    def test_invalid_downscaling_method_raises(self):
        with pytest.raises(ValidationError):
            VariableConfig(
                detrend_data=True,
                do_windowing=True,
                downscaling_method="multiply",  # not a valid Literal
                downscaling_clim_method="fft",
            )


# ---------------------------------------------------------------------------
# BCSDConfig – construction & auto-population
# ---------------------------------------------------------------------------


class TestBCSDConfigConstruction:
    """BCSDConfig is valid for a range of realistic inputs."""

    def test_minimal_historical_config(self, minimal_config):
        assert minimal_config.gcm == "CESM2-WACCM"
        assert minimal_config.variable == "tas"
        assert minimal_config.ensemble_member == 0
        assert minimal_config.scenario is None

    def test_variable_config_auto_populated(self, minimal_config):
        assert minimal_config.variable_config is not None
        assert isinstance(minimal_config.variable_config, VariableConfig)

    def test_convenience_accessors_match_variable_config(self, subtests, minimal_config):
        vc = minimal_config.variable_config
        checks = {
            "detrend_data": (minimal_config.detrend_data, vc.detrend_data),
            "do_windowing": (minimal_config.do_windowing, vc.do_windowing),
            "downscaling_method": (minimal_config.downscaling_method, vc.downscaling_method),
            "downscaling_clim_method": (
                minimal_config.downscaling_clim_method,
                vc.downscaling_clim_method,
            ),
        }
        for accessor, (actual, expected) in checks.items():
            with subtests.test(accessor=accessor):
                assert actual == expected

    def test_default_train_period(self, minimal_config):
        assert minimal_config.train_period_start == 1978
        assert minimal_config.train_period_end == 2014

    def test_default_environment_and_version(self, minimal_config):
        assert minimal_config.environment == "qa"
        assert minimal_config.version == "v1"

    def test_explicit_variable_config_not_overwritten(self):
        """Explicitly supplied variable_config must survive post-init."""
        custom_vc = VariableConfig(
            detrend_data=False,
            do_windowing=False,
            downscaling_method="additive",
            downscaling_clim_method="simple",
        )
        cfg = BCSDConfig(
            gcm="UKESM",
            variable="tas",
            ensemble_member=2,
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
                cfg = BCSDConfig(gcm="CESM2-WACCM", variable=var, ensemble_member=0)
                assert cfg.variable == var

    def test_all_supported_gcms_construct(self, subtests):
        for gcm in ("CESM2-WACCM", "MIROC-ES2H", "UKESM"):
            with subtests.test(gcm=gcm):
                cfg = BCSDConfig(gcm=gcm, variable="tas", ensemble_member=0)
                assert cfg.gcm == gcm

    def test_model_copy_version_override(self, scenario_config):
        v2 = scenario_config.model_copy(update={"version": "v2"})
        assert v2.version == "v2"
        assert scenario_config.version == "v1"


# ---------------------------------------------------------------------------
# BCSDConfig – computed fields
# ---------------------------------------------------------------------------


class TestBCSDConfigComputedFields:
    """run_id, config_hash, and is_sai_scenario computed fields."""

    def test_run_id_historical_only(self, minimal_config):
        assert minimal_config.run_id == "CESM2-WACCM_tas_000"

    def test_run_id_with_scenario(self, scenario_config):
        assert scenario_config.run_id == "CESM2-WACCM_tas_000_ssp245"

    def test_run_id_includes_subset_marker(self, regional_config):
        assert "subset" in regional_config.run_id

    def test_run_id_ensemble_zero_padding(self, subtests):
        for member, expected_pad in [(0, "000"), (1, "001"), (9, "009"), (10, "010"), (99, "099")]:
            with subtests.test(member=member):
                cfg = BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member=member)
                assert f"_{expected_pad}" in cfg.run_id

    def test_config_hash_is_12_char_hex(self, minimal_config):
        h = minimal_config.config_hash
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_config_hash_is_stable(self):
        cfg_a = BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member=0)
        cfg_b = BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member=0)
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
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member=0,
            scenario="SAI-2050",
            predict_period_start=2015,
            predict_period_end=2100,
        )
        assert cfg.is_sai_scenario

    def test_pr_does_not_detrend(self, sai_config):
        assert sai_config.detrend_data is False

    def test_tas_does_detrend(self, scenario_config):
        assert scenario_config.detrend_data is True


# ---------------------------------------------------------------------------
# BCSDConfig – validation
# ---------------------------------------------------------------------------


class TestBCSDConfigValidation:
    """Field validators reject invalid inputs with informative errors."""

    def test_scenario_with_explicit_null_predict_start_raises(self):
        with pytest.raises(ValidationError):
            BCSDConfig(
                gcm="CESM2-WACCM",
                variable="tas",
                ensemble_member=0,
                scenario="ssp245",
                predict_period_start=None,  # explicit None triggers the validator
                predict_period_end=2100,
            )

    def test_scenario_with_explicit_null_predict_end_raises(self):
        with pytest.raises(ValidationError):
            BCSDConfig(
                gcm="CESM2-WACCM",
                variable="tas",
                ensemble_member=0,
                scenario="ssp245",
                predict_period_start=2015,
                predict_period_end=None,  # explicit None triggers the validator
            )

    def test_train_period_end_before_start_raises(self):
        with pytest.raises(ValidationError):
            BCSDConfig(
                gcm="CESM2-WACCM",
                variable="tas",
                ensemble_member=0,
                train_period_start=2000,
                train_period_end=1990,  # end before start
            )

    def test_predict_period_end_before_start_raises(self):
        with pytest.raises(ValidationError):
            BCSDConfig(
                gcm="CESM2-WACCM",
                variable="tas",
                ensemble_member=0,
                scenario="ssp245",
                predict_period_start=2080,
                predict_period_end=2015,  # end before start
            )

    def test_unsupported_variable_raises(self):
        with pytest.raises(ValidationError):
            BCSDConfig(gcm="CESM2-WACCM", variable="sfcWind", ensemble_member=0)

    def test_negative_ensemble_member_raises(self):
        with pytest.raises(ValidationError):
            BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member=-1)

    def test_subset_bounds_lat_min_ge_max_raises(self):
        with pytest.raises(ValidationError, match="lat_min"):
            BCSDConfig(
                gcm="CESM2-WACCM",
                variable="tas",
                ensemble_member=0,
                subset_bounds=(20.0, 10.0, 0.0, 30.0),  # lat_min > lat_max
            )

    def test_subset_bounds_lon_min_ge_max_raises(self):
        with pytest.raises(ValidationError, match="lon_min"):
            BCSDConfig(
                gcm="CESM2-WACCM",
                variable="tas",
                ensemble_member=0,
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
                    BCSDConfig(
                        gcm="CESM2-WACCM",
                        variable="tas",
                        ensemble_member=0,
                        subset_bounds=bounds,
                    )

    def test_valid_regional_config_accepted(self, regional_config):
        assert regional_config.subset_bounds == (-35.0, -22.0, 16.0, 33.0)


# ---------------------------------------------------------------------------
# BCSDConfig – to_legacy_kwargs
# ---------------------------------------------------------------------------


class TestToLegacyKwargs:
    """Backward-compatibility helper converts fields to the legacy function signature."""

    def test_historical_only_keys(self, minimal_config):
        kwargs = minimal_config.to_legacy_kwargs()
        assert kwargs["gcm"] == "CESM2-WACCM"
        assert kwargs["var_name"] == "tas"
        assert kwargs["train_period_start"] == 1978
        assert kwargs["train_period_end"] == 2014

    def test_historical_only_excludes_predict_period(self, minimal_config):
        kwargs = minimal_config.to_legacy_kwargs()
        assert "predict_period_start" not in kwargs
        assert "predict_period_end" not in kwargs

    def test_scenario_includes_predict_period(self, scenario_config):
        kwargs = scenario_config.to_legacy_kwargs()
        assert kwargs["predict_period_start"] == 2015
        assert kwargs["predict_period_end"] == 2100

    def test_global_run_subset_bounds_is_none(self, minimal_config):
        assert minimal_config.to_legacy_kwargs()["subset_bounds"] is None

    def test_regional_run_subset_bounds_is_list(self, regional_config):
        kwargs = regional_config.to_legacy_kwargs()
        assert isinstance(kwargs["subset_bounds"], list)
        assert kwargs["subset_bounds"] == [-35.0, -22.0, 16.0, 33.0]

    def test_verbose_and_rechunk_present(self, minimal_config):
        kwargs = minimal_config.to_legacy_kwargs()
        assert "verbose" in kwargs
        assert "rechunk_workflow" in kwargs


# ---------------------------------------------------------------------------
# CacheConfig
# ---------------------------------------------------------------------------


class TestCacheConfig:
    def test_defaults(self):
        cfg = CacheConfig()
        assert cfg.environment == "qa"
        assert cfg.version == "v1"
        assert cfg.force_recompute is False
        assert cfg.check_integrity is True

    def test_custom_values(self):
        cfg = CacheConfig(environment="production", version="v3", force_recompute=True)
        assert cfg.environment == "production"
        assert cfg.version == "v3"
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
