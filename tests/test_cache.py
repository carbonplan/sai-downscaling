"""Tests for cache.py: ArtifactCache path generation, existence checks, and cache management."""

from __future__ import annotations

from pathlib import Path

import pytest

from srm.bcsd_config import BCSDConfig
from srm.cache import ArtifactCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def local_cache(tmp_path) -> ArtifactCache:
    """ArtifactCache backed by the local filesystem (avoids S3 in unit tests)."""
    return ArtifactCache(
        base_path=str(tmp_path / "cache"),
        environment="qa",
        version="v1",
    )


@pytest.fixture
def local_cache_with_output(tmp_path) -> ArtifactCache:
    """ArtifactCache with a separate output_dir for final scenario artifacts."""
    return ArtifactCache(
        base_path=str(tmp_path / "cache"),
        environment="qa",
        version="v1",
        output_dir=str(tmp_path / "outputs"),
    )


@pytest.fixture
def base_config() -> BCSDConfig:
    """Standard non-SAI scenario config."""
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
    """SAI G6 scenario config."""
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
    """Config with a spatial subset (South Africa region)."""
    return BCSDConfig(
        gcm="MIROC-ES2H",
        variable="tasmax",
        ensemble_member=2,
        scenario="ssp245",
        predict_period_start=2015,
        predict_period_end=2100,
        subset_bounds=(-35.0, -22.0, 16.0, 33.0),
    )


def make_zarr_store(path: str, marker: str = ".zmetadata") -> None:
    """Create a minimal zarr store directory with the given metadata marker file."""
    store = Path(path)
    store.mkdir(parents=True, exist_ok=True)
    (store / marker).touch()


# ---------------------------------------------------------------------------
# ArtifactCache.__init__
# ---------------------------------------------------------------------------


class TestArtifactCacheInit:
    def test_trailing_slash_stripped_from_base_path(self, tmp_path):
        cache = ArtifactCache(base_path=str(tmp_path) + "/")
        assert not cache.base_path.endswith("/")

    def test_trailing_slash_stripped_from_output_dir(self, tmp_path):
        cache = ArtifactCache(
            base_path=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "outputs") + "/",
        )
        assert not cache.output_dir.endswith("/")

    def test_output_dir_is_none_by_default(self, local_cache):
        assert local_cache.output_dir is None

    def test_environment_and_version_stored(self, subtests):
        for env, ver in [("qa", "v1"), ("staging", "v2"), ("production", "v3")]:
            with subtests.test(environment=env, version=ver):
                cache = ArtifactCache(base_path="/tmp/cache", environment=env, version=ver)
                assert cache.environment == env
                assert cache.version == ver


# ---------------------------------------------------------------------------
# _get_subset_id
# ---------------------------------------------------------------------------


class TestGetSubsetId:
    def test_none_returns_global(self):
        assert ArtifactCache._get_subset_id(None) == "global"

    def test_bounds_produce_readable_id(self, subtests):
        cases = [
            ((-35.0, -22.0, 16.0, 33.0), "lat-35.0to-22.0_lon16.0to33.0"),
            ((0.0, 90.0, -180.0, 180.0), "lat0.0to90.0_lon-180.0to180.0"),
            ((-90.0, 90.0, 0.0, 360.0), "lat-90.0to90.0_lon0.0to360.0"),
        ]
        for bounds, expected in cases:
            with subtests.test(bounds=str(bounds)):
                assert ArtifactCache._get_subset_id(bounds) == expected

    def test_subset_id_contains_no_spaces(self):
        subset_id = ArtifactCache._get_subset_id((-35.0, -22.0, 16.0, 33.0))
        assert " " not in subset_id


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------


class TestObsPath:
    def test_contains_environment_and_version(self, local_cache):
        path = local_cache.get_obs_path("CESM2-WACCM", "tas")
        assert "/qa/" in path
        assert "/v1/" in path

    def test_contains_obs_stage_directory(self, local_cache):
        path = local_cache.get_obs_path("CESM2-WACCM", "tas")
        assert "/obs/" in path

    def test_global_subset_id_in_filename(self, local_cache):
        path = local_cache.get_obs_path("CESM2-WACCM", "tas")
        assert "CESM2-WACCM_tas_global_obs_regridded.zarr" in path

    def test_regional_subset_id_in_filename(self, local_cache):
        path = local_cache.get_obs_path("CESM2-WACCM", "tas", (-35.0, -22.0, 16.0, 33.0))
        assert "lat-35.0to-22.0_lon16.0to33.0" in path
        assert "global" not in path

    def test_paths_differ_per_environment(self, subtests, tmp_path):
        for env in ("qa", "staging", "production"):
            with subtests.test(environment=env):
                cache = ArtifactCache(base_path=str(tmp_path), environment=env, version="v1")
                assert f"/{env}/" in cache.get_obs_path("CESM2-WACCM", "tas")

    def test_paths_differ_per_version(self, subtests, tmp_path):
        for version in ("v1", "v2", "v3"):
            with subtests.test(version=version):
                cache = ArtifactCache(base_path=str(tmp_path), environment="qa", version=version)
                assert f"/{version}/" in cache.get_obs_path("CESM2-WACCM", "tas")


class TestHistoricalPath:
    def test_goes_to_cache_when_no_output_dir(self, local_cache):
        path = local_cache.get_historical_path("CESM2-WACCM", "tas", 0)
        assert local_cache.base_path in path
        assert "/historical/" in path

    def test_goes_to_output_dir_when_specified(self, local_cache_with_output):
        path = local_cache_with_output.get_historical_path("CESM2-WACCM", "tas", 0)
        assert local_cache_with_output.output_dir in path
        assert local_cache_with_output.base_path not in path

    def test_ensemble_member_zero_padded_in_filename(self, subtests, local_cache):
        for member, expected_pad in [(0, "000"), (1, "001"), (9, "009"), (10, "010"), (99, "099")]:
            with subtests.test(member=member):
                path = local_cache.get_historical_path("CESM2-WACCM", "tas", member)
                assert f"_{expected_pad}_" in path

    def test_filename_format(self, local_cache):
        path = local_cache.get_historical_path("CESM2-WACCM", "tas", 0)
        assert "CESM2-WACCM_tas_000_global_historical.zarr" in path


class TestScenarioPath:
    def test_goes_to_cache_scenarios_when_no_output_dir(self, local_cache):
        path = local_cache.get_scenario_path("CESM2-WACCM", "tas", 0, "ssp245")
        assert local_cache.base_path in path
        assert "/scenarios/" in path

    def test_goes_to_output_dir_when_specified(self, local_cache_with_output):
        path = local_cache_with_output.get_scenario_path("CESM2-WACCM", "tas", 0, "ssp245")
        assert local_cache_with_output.output_dir in path
        assert "/scenarios/" not in path

    def test_filename_format(self, local_cache):
        path = local_cache.get_scenario_path("CESM2-WACCM", "tas", 0, "ssp245")
        assert "CESM2-WACCM_tas_000_global_ssp245.zarr" in path

    def test_sai_scenario_name_preserved_in_filename(self, local_cache):
        path = local_cache.get_scenario_path("CESM2-WACCM", "pr", 1, "G6-1.5K")
        assert "G6-1.5K.zarr" in path


# ---------------------------------------------------------------------------
# exists() – local filesystem
# ---------------------------------------------------------------------------


class TestExists:
    def test_nonexistent_path_returns_false(self, local_cache, tmp_path):
        assert local_cache.exists(str(tmp_path / "missing.zarr")) is False

    def test_empty_directory_returns_false(self, local_cache, tmp_path):
        empty_store = tmp_path / "empty.zarr"
        empty_store.mkdir()
        assert local_cache.exists(str(empty_store)) is False

    def test_zarr_v2_zmetadata_detected(self, local_cache, tmp_path):
        store = tmp_path / "v2_meta.zarr"
        make_zarr_store(str(store), ".zmetadata")
        assert local_cache.exists(str(store)) is True

    def test_zarr_v2_zgroup_detected(self, local_cache, tmp_path):
        store = tmp_path / "v2_group.zarr"
        make_zarr_store(str(store), ".zgroup")
        assert local_cache.exists(str(store)) is True

    def test_zarr_v3_zarr_json_detected(self, local_cache, tmp_path):
        store = tmp_path / "v3.zarr"
        make_zarr_store(str(store), "zarr.json")
        assert local_cache.exists(str(store)) is True

    def test_any_valid_marker_is_sufficient(self, subtests, local_cache, tmp_path):
        for marker in (".zmetadata", ".zgroup", "zarr.json"):
            with subtests.test(marker=marker):
                store = tmp_path / f"store_{marker.strip('.')}.zarr"
                make_zarr_store(str(store), marker)
                assert local_cache.exists(str(store)) is True


# ---------------------------------------------------------------------------
# check_dependencies / validate_dependencies
# ---------------------------------------------------------------------------


class TestCheckDependencies:
    def test_prepare_observations_has_no_deps(self, local_cache, base_config):
        deps = local_cache.check_dependencies("prepare_observations", base_config)
        assert deps == {}

    def test_fit_historical_requires_obs_only(self, local_cache, base_config):
        deps = local_cache.check_dependencies("fit_historical", base_config)
        assert set(deps.keys()) == {"obs_regridded"}

    def test_transform_scenario_requires_obs_and_historical(self, local_cache, base_config):
        deps = local_cache.check_dependencies("transform_scenario", base_config)
        assert set(deps.keys()) == {"obs_regridded", "historical"}

    def test_dependency_values_are_bool_path_tuples(self, local_cache, base_config):
        deps = local_cache.check_dependencies("transform_scenario", base_config)
        for name, (exists, path) in deps.items():
            assert isinstance(exists, bool), f"{name}: exists should be bool"
            assert isinstance(path, str), f"{name}: path should be str"

    def test_unknown_stage_raises(self, local_cache, base_config):
        with pytest.raises(ValueError, match="Unknown stage"):
            local_cache.check_dependencies("nonexistent_stage", base_config)

    def test_missing_deps_reported_as_false(self, local_cache, base_config):
        deps = local_cache.check_dependencies("fit_historical", base_config)
        exists, _path = deps["obs_regridded"]
        assert exists is False  # nothing has been written to the local cache yet


class TestValidateDependencies:
    def test_raises_when_deps_missing(self, local_cache, base_config):
        with pytest.raises(ValueError, match="Missing dependencies"):
            local_cache.validate_dependencies("fit_historical", base_config)

    def test_passes_when_obs_store_exists(self, local_cache, base_config):
        obs_path = local_cache.get_obs_path(
            base_config.gcm, base_config.variable, base_config.subset_bounds
        )
        make_zarr_store(obs_path)
        # Should not raise
        local_cache.validate_dependencies("fit_historical", base_config)

    def test_raises_when_only_obs_present_for_scenario_stage(self, local_cache, base_config):
        obs_path = local_cache.get_obs_path(
            base_config.gcm, base_config.variable, base_config.subset_bounds
        )
        make_zarr_store(obs_path)
        with pytest.raises(ValueError, match="Missing dependencies"):
            local_cache.validate_dependencies("transform_scenario", base_config)

    def test_passes_when_all_scenario_deps_present(self, local_cache, base_config):
        obs_path = local_cache.get_obs_path(
            base_config.gcm, base_config.variable, base_config.subset_bounds
        )
        hist_path = local_cache.get_historical_path(
            base_config.gcm,
            base_config.variable,
            base_config.ensemble_member,
            base_config.subset_bounds,
        )
        make_zarr_store(obs_path)
        make_zarr_store(hist_path)
        # Should not raise
        local_cache.validate_dependencies("transform_scenario", base_config)


# ---------------------------------------------------------------------------
# get_output_path
# ---------------------------------------------------------------------------


class TestGetOutputPath:
    def test_prepare_observations_returns_obs_path(self, local_cache, base_config):
        expected = local_cache.get_obs_path(
            base_config.gcm, base_config.variable, base_config.subset_bounds
        )
        assert local_cache.get_output_path("prepare_observations", base_config) == expected

    def test_fit_historical_returns_historical_path(self, local_cache, base_config):
        expected = local_cache.get_historical_path(
            base_config.gcm,
            base_config.variable,
            base_config.ensemble_member,
            base_config.subset_bounds,
        )
        assert local_cache.get_output_path("fit_historical", base_config) == expected

    def test_transform_scenario_returns_scenario_path(self, local_cache, base_config):
        expected = local_cache.get_scenario_path(
            base_config.gcm,
            base_config.variable,
            base_config.ensemble_member,
            base_config.scenario,
            base_config.subset_bounds,
        )
        assert local_cache.get_output_path("transform_scenario", base_config) == expected

    def test_transform_scenario_without_scenario_field_raises(self, local_cache):
        config = BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member=0)
        with pytest.raises(ValueError, match="scenario must be specified"):
            local_cache.get_output_path("transform_scenario", config)

    def test_unknown_stage_raises(self, local_cache, base_config):
        with pytest.raises(ValueError, match="Unknown stage"):
            local_cache.get_output_path("bad_stage", base_config)

    def test_all_valid_stages_return_strings(self, subtests, local_cache, base_config):
        for stage in ("prepare_observations", "fit_historical", "transform_scenario"):
            with subtests.test(stage=stage):
                path = local_cache.get_output_path(stage, base_config)
                assert isinstance(path, str)
                assert path.endswith(".zarr")


# ---------------------------------------------------------------------------
# list_artifacts & clear_cache – integration tests on local filesystem
# ---------------------------------------------------------------------------


class TestListAndClearArtifacts:
    """Integration-style tests that populate a real local directory tree."""

    @pytest.fixture
    def populated_cache(
        self, local_cache, base_config, sai_config, regional_config
    ) -> ArtifactCache:
        """Create one obs/historical/scenario artifact for each of three configs."""
        for cfg in (base_config, sai_config, regional_config):
            make_zarr_store(local_cache.get_obs_path(cfg.gcm, cfg.variable, cfg.subset_bounds))
            make_zarr_store(
                local_cache.get_historical_path(
                    cfg.gcm, cfg.variable, cfg.ensemble_member, cfg.subset_bounds
                )
            )
            make_zarr_store(
                local_cache.get_scenario_path(
                    cfg.gcm, cfg.variable, cfg.ensemble_member, cfg.scenario, cfg.subset_bounds
                )
            )
        return local_cache

    def test_list_all_artifacts_returns_nine(self, populated_cache):
        # 3 configs × 3 stages = 9 artifacts
        assert len(populated_cache.list_artifacts()) == 9

    def test_list_filters_by_stage(self, subtests, populated_cache):
        for stage in ("obs", "historical", "scenarios"):
            with subtests.test(stage=stage):
                artifacts = populated_cache.list_artifacts(stage=stage)
                assert len(artifacts) == 3

    def test_list_filters_by_gcm(self, populated_cache):
        artifacts = populated_cache.list_artifacts(gcm="CESM2-WACCM")
        assert len(artifacts) == 6  # base_config + sai_config, each has 3 stages
        assert all("CESM2-WACCM" in a for a in artifacts)

    def test_list_filters_by_variable(self, populated_cache):
        artifacts = populated_cache.list_artifacts(variable="tas")
        assert len(artifacts) == 3  # only base_config
        for path in artifacts:
            assert Path(path).name.split("_")[1] == "tas"

    def test_list_filters_gcm_and_variable_combined(self, populated_cache):
        artifacts = populated_cache.list_artifacts(gcm="CESM2-WACCM", variable="pr")
        assert len(artifacts) == 3  # only sai_config (CESM2-WACCM + pr)

    def test_list_nonexistent_gcm_returns_empty(self, populated_cache):
        assert populated_cache.list_artifacts(gcm="NONEXISTENT-GCM") == []

    def test_list_is_sorted(self, populated_cache):
        artifacts = populated_cache.list_artifacts()
        assert artifacts == sorted(artifacts)

    def test_clear_all_deletes_nine_artifacts(self, populated_cache):
        deleted = populated_cache.clear_cache()
        assert deleted == 9
        assert populated_cache.list_artifacts() == []

    def test_clear_single_stage_leaves_others_intact(self, populated_cache):
        obs_count = len(populated_cache.list_artifacts(stage="obs"))
        deleted = populated_cache.clear_cache(stage="obs")
        assert deleted == obs_count
        assert populated_cache.list_artifacts(stage="obs") == []
        # historical and scenarios untouched
        assert len(populated_cache.list_artifacts(stage="historical")) == 3
        assert len(populated_cache.list_artifacts(stage="scenarios")) == 3

    def test_clear_filters_by_gcm(self, populated_cache):
        deleted = populated_cache.clear_cache(gcm="MIROC-ES2H")
        assert deleted == 3  # regional_config only
        # CESM2-WACCM artifacts remain
        assert len(populated_cache.list_artifacts(gcm="CESM2-WACCM")) == 6

    def test_clear_returns_zero_for_empty_cache(self, local_cache):
        assert local_cache.clear_cache() == 0
