"""Tests for cache.py: ArtifactCache StoreLocation generation, existence checks, and helpers."""

from __future__ import annotations

import pytest
from conftest import make_icechunk_group

from srm.bcsd_config import BCSDConfig, PipelineOptions, VariableConfig
from srm.cache import ArtifactCache, StoreLocation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def local_cache(tmp_path) -> ArtifactCache:
    """ArtifactCache backed by the local filesystem (avoids S3 in unit tests)."""
    return ArtifactCache(
        scratch_dir=str(tmp_path / "cache"),
        environment="qa",
    )


@pytest.fixture
def local_cache_with_output(tmp_path) -> ArtifactCache:
    """ArtifactCache with a separate output_dir for final scenario artifacts."""
    return ArtifactCache(
        scratch_dir=str(tmp_path / "cache"),
        environment="qa",
        output_dir=str(tmp_path / "outputs"),
    )


@pytest.fixture
def base_config() -> BCSDConfig:
    """Standard SSP245 scenario config."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="tas",
        ensemble_member="r1i1p1f1",
        scenario="SSP245",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def sai_config() -> BCSDConfig:
    """SAI G6 scenario config."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="pr",
        ensemble_member="r2i1p1f1",
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
        ensemble_member="01",
        scenario="SSP245",
        predict_period_start=2015,
        predict_period_end=2100,
        subset_bounds=(-35.0, -22.0, 16.0, 33.0),
    )


@pytest.fixture
def bound_cache(tmp_path, base_config) -> ArtifactCache:
    """ArtifactCache with base_config bound, backed by local filesystem, no output_dir."""
    cache = ArtifactCache(
        scratch_dir=str(tmp_path / "cache"),
        environment="qa",
        output_dir=None,
    )
    cache.config = base_config
    return cache


@pytest.fixture
def bound_cache_with_output(tmp_path, base_config) -> ArtifactCache:
    """ArtifactCache with base_config bound and separate output_dir."""
    return ArtifactCache.from_config(
        base_config,
        PipelineOptions(
            scratch_dir=str(tmp_path / "cache"),
            environment="qa",
            output_dir=str(tmp_path / "outputs"),
        ),
    )


# ---------------------------------------------------------------------------
# StoreLocation
# ---------------------------------------------------------------------------


class TestStoreLocation:
    def test_frozen(self):
        loc = StoreLocation("s3://bucket/path.icechunk", "obs/tas")
        with pytest.raises(Exception):
            loc.store_path = "other"  # type: ignore[misc]

    def test_equality(self):
        a = StoreLocation("s3://bucket/path.icechunk", "obs/tas")
        b = StoreLocation("s3://bucket/path.icechunk", "obs/tas")
        assert a == b

    def test_inequality_on_group(self):
        a = StoreLocation("s3://bucket/path.icechunk", "obs/tas")
        b = StoreLocation("s3://bucket/path.icechunk", "obs/pr")
        assert a != b


# ---------------------------------------------------------------------------
# ArtifactCache.__init__
# ---------------------------------------------------------------------------


class TestArtifactCacheInit:
    def test_trailing_slash_stripped_from_scratch_dir(self, tmp_path):
        cache = ArtifactCache(scratch_dir=str(tmp_path) + "/")
        assert not cache.scratch_dir.endswith("/")

    def test_trailing_slash_stripped_from_output_dir(self, tmp_path):
        cache = ArtifactCache(
            scratch_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "outputs") + "/",
        )
        assert not cache.output_dir.endswith("/")

    def test_output_dir_is_none_by_default(self, local_cache):
        assert local_cache.output_dir is None

    def test_environment_and_branch_stored(self, subtests):
        for env, branch in [("qa", "v1.0"), ("production", "v2.0")]:
            with subtests.test(environment=env, branch=branch):
                cache = ArtifactCache(scratch_dir="/tmp/cache", environment=env, branch=branch)
                assert cache.environment == env
                assert cache.branch == branch


# ---------------------------------------------------------------------------
# from_config / bound location properties
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_config_stored_on_cache(self, base_config):
        cache = ArtifactCache.from_config(base_config, PipelineOptions())
        assert cache.config is base_config

    def test_loc_properties_raise_without_bound_config(self, local_cache):
        with pytest.raises(RuntimeError, match="No config bound"):
            _ = local_cache.obs_loc
        with pytest.raises(RuntimeError, match="No config bound"):
            _ = local_cache.scenario_loc

    def test_config_is_none_on_plain_init(self, local_cache):
        assert local_cache.config is None


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
# Store path formula
# ---------------------------------------------------------------------------


class TestStorePaths:
    def test_scratch_store_ends_with_icechunk(self, bound_cache):
        assert bound_cache._scratch_store.endswith(".icechunk")

    def test_scratch_store_contains_env(self, bound_cache):
        assert "/qa/" in bound_cache._scratch_store

    def test_scratch_store_encodes_gcm_obs_subset(self, bound_cache):
        assert "CESM2-WACCM-ERA5-global.icechunk" in bound_cache._scratch_store

    def test_output_store_uses_output_dir_when_set(self, bound_cache_with_output):
        assert bound_cache_with_output.output_dir in bound_cache_with_output._output_store
        assert bound_cache_with_output.scratch_dir not in bound_cache_with_output._output_store

    def test_output_store_falls_back_to_scratch_when_no_output_dir(self, bound_cache):
        assert bound_cache.scratch_dir in bound_cache._output_store

    def test_regional_subset_id_in_store_path(self, tmp_path, regional_config):
        cache = ArtifactCache.from_config(
            regional_config,
            PipelineOptions(scratch_dir=str(tmp_path / "cache")),
        )
        assert "lat-35.0to-22.0_lon16.0to33.0" in cache._scratch_store

    def test_different_environments_produce_different_stores(self, subtests, tmp_path, base_config):
        for env in ("qa", "production"):
            with subtests.test(environment=env):
                cache = ArtifactCache.from_config(
                    base_config,
                    PipelineOptions(scratch_dir=str(tmp_path), environment=env),
                )
                assert f"/{env}/" in cache._scratch_store


# ---------------------------------------------------------------------------
# obs_loc
# ---------------------------------------------------------------------------


class TestObsLoc:
    def test_store_path_ends_with_icechunk(self, bound_cache):
        assert bound_cache.obs_loc.store_path.endswith(".icechunk")

    def test_store_path_encodes_gcm_obs_subset(self, bound_cache):
        assert "CESM2-WACCM-ERA5-global.icechunk" in bound_cache.obs_loc.store_path

    def test_group_encodes_obs_and_variable(self, bound_cache):
        assert bound_cache.obs_loc.group == "obs/tas"

    def test_store_path_contains_env(self, bound_cache):
        assert "/qa/" in bound_cache.obs_loc.store_path

    def test_regional_subset_id_in_store_path(self, tmp_path, regional_config):
        cache = ArtifactCache.from_config(
            regional_config,
            PipelineOptions(scratch_dir=str(tmp_path / "cache")),
        )
        assert "lat-35.0to-22.0_lon16.0to33.0" in cache.obs_loc.store_path


# ---------------------------------------------------------------------------
# historical_loc
# ---------------------------------------------------------------------------


class TestHistoricalLoc:
    def test_group_encodes_historical_variable_member(self, bound_cache):
        loc = bound_cache.historical_loc("r1i1p1f1")
        assert loc.group == "historical/tas/r1i1p1f1"

    def test_group_changes_with_member(self, bound_cache, subtests):
        for member in ("r1i1p1f1", "r12i1p1f2", "001"):
            with subtests.test(member=member):
                loc = bound_cache.historical_loc(member)
                assert loc.group == f"historical/tas/{member}"

    def test_uses_scratch_store(self, bound_cache):
        loc = bound_cache.historical_loc("r1i1p1f1")
        assert loc.store_path == bound_cache._scratch_store

    def test_store_path_encodes_gcm_obs_subset(self, bound_cache):
        loc = bound_cache.historical_loc("r1i1p1f1")
        assert "CESM2-WACCM-ERA5-global.icechunk" in loc.store_path


# ---------------------------------------------------------------------------
# scenario_loc
# ---------------------------------------------------------------------------


class TestScenarioLoc:
    def test_group_encodes_scenario_variable_member(self, bound_cache):
        assert bound_cache.scenario_loc.group == "ssp245/tas/r1i1p1f1"

    def test_sai_scenario_group(self, tmp_path, sai_config):
        cache = ArtifactCache.from_config(sai_config, PipelineOptions(scratch_dir=str(tmp_path)))
        assert cache.scenario_loc.group == "g6_1p5k/pr/r2i1p1f1"

    def test_uses_output_store_when_set(self, bound_cache_with_output):
        loc = bound_cache_with_output.scenario_loc
        assert bound_cache_with_output.output_dir in loc.store_path

    def test_uses_scratch_store_when_no_output_dir(self, bound_cache):
        loc = bound_cache.scenario_loc
        assert bound_cache.scratch_dir in loc.store_path


# ---------------------------------------------------------------------------
# intermediate locs
# ---------------------------------------------------------------------------


class TestIntermediateLocs:
    def test_detrended_scenario_group(self, bound_cache):
        loc = bound_cache.detrended_scenario_loc()
        assert loc.group == "detrended_scenario/ssp245/tas/r1i1p1f1"

    def test_trend_scenario_group(self, bound_cache):
        loc = bound_cache.trend_scenario_loc()
        assert loc.group == "trend_scenario/ssp245/tas/r1i1p1f1"

    def test_debiased_scenario_group(self, bound_cache):
        loc = bound_cache.debiased_scenario_loc()
        assert loc.group == "debiased_scenario/ssp245/tas/r1i1p1f1"

    def test_all_intermediate_use_scratch_store(self, bound_cache):
        locs = [
            bound_cache.detrended_scenario_loc(),
            bound_cache.trend_scenario_loc(),
            bound_cache.debiased_scenario_loc(),
        ]
        for loc in locs:
            assert loc.store_path == bound_cache._scratch_store

    def test_sai_scenario_intermediate_group(self, tmp_path, sai_config):
        cache = ArtifactCache.from_config(sai_config, PipelineOptions(scratch_dir=str(tmp_path)))
        assert cache.detrended_scenario_loc().group == "detrended_scenario/g6_1p5k/pr/r2i1p1f1"

    def test_intermediate_groups_all_differ(self, bound_cache):
        groups = [
            bound_cache.detrended_scenario_loc().group,
            bound_cache.trend_scenario_loc().group,
            bound_cache.debiased_scenario_loc().group,
        ]
        assert len(groups) == len(set(groups))

    def test_debiased_historical_absent_from_intermediate_prefixes(self):
        assert "debiased_historical/" not in ArtifactCache.INTERMEDIATE_PREFIXES

    def test_debiased_retrended_scenario_absent_from_intermediate_prefixes(self):
        assert "debiased_retrended_scenario/" not in ArtifactCache.INTERMEDIATE_PREFIXES


# ---------------------------------------------------------------------------
# debiased_coarse locs (output store)
# ---------------------------------------------------------------------------


class TestDebiasedCoarseLocs:
    def test_historical_group(self, bound_cache):
        loc = bound_cache.debiased_coarse_historical_loc("r1i1p1f1")
        assert loc.group == "debiased_coarse/historical/tas/r1i1p1f1"

    def test_historical_group_changes_with_member(self, bound_cache, subtests):
        for member in ("r1i1p1f1", "r12i1p1f2", "001"):
            with subtests.test(member=member):
                loc = bound_cache.debiased_coarse_historical_loc(member)
                assert loc.group == f"debiased_coarse/historical/tas/{member}"

    def test_historical_variable_override(self, bound_cache):
        loc = bound_cache.debiased_coarse_historical_loc("r1i1p1f1", variable="dtr")
        assert loc.group == "debiased_coarse/historical/dtr/r1i1p1f1"

    def test_historical_uses_output_store(self, bound_cache_with_output):
        loc = bound_cache_with_output.debiased_coarse_historical_loc("r1i1p1f1")
        assert bound_cache_with_output.output_dir in loc.store_path
        assert bound_cache_with_output.scratch_dir not in loc.store_path

    def test_historical_falls_back_to_scratch_when_no_output_dir(self, bound_cache):
        loc = bound_cache.debiased_coarse_historical_loc("r1i1p1f1")
        assert bound_cache.scratch_dir in loc.store_path

    def test_scenario_group(self, bound_cache):
        loc = bound_cache.debiased_coarse_scenario_loc()
        assert loc.group == "debiased_coarse/ssp245/tas/r1i1p1f1"

    def test_scenario_sai_group(self, tmp_path, sai_config):
        cache = ArtifactCache.from_config(sai_config, PipelineOptions(scratch_dir=str(tmp_path)))
        loc = cache.debiased_coarse_scenario_loc()
        assert loc.group == "debiased_coarse/g6_1p5k/pr/r2i1p1f1"

    def test_scenario_variable_override(self, bound_cache):
        loc = bound_cache.debiased_coarse_scenario_loc(variable="dtr")
        assert loc.group == "debiased_coarse/ssp245/dtr/r1i1p1f1"

    def test_scenario_uses_output_store(self, bound_cache_with_output):
        loc = bound_cache_with_output.debiased_coarse_scenario_loc()
        assert bound_cache_with_output.output_dir in loc.store_path
        assert bound_cache_with_output.scratch_dir not in loc.store_path

    def test_scenario_falls_back_to_scratch_when_no_output_dir(self, bound_cache):
        loc = bound_cache.debiased_coarse_scenario_loc()
        assert bound_cache.scratch_dir in loc.store_path

    def test_historical_and_scenario_groups_differ(self, bound_cache):
        hist = bound_cache.debiased_coarse_historical_loc("r1i1p1f1")
        scen = bound_cache.debiased_coarse_scenario_loc()
        assert hist.group != scen.group


# ---------------------------------------------------------------------------
# _get_varconfig_id – hashing
# ---------------------------------------------------------------------------


class TestVarconfigId:
    def test_hash_length_is_8(self):
        vc = VariableConfig.for_variable("tas")
        assert len(ArtifactCache._get_varconfig_id(vc, "parametric")) == 8

    def test_hash_is_hex_string(self):
        vc = VariableConfig.for_variable("tas")
        h = ArtifactCache._get_varconfig_id(vc, "parametric")
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_is_stable(self):
        vc = VariableConfig.for_variable("tas")
        h1 = ArtifactCache._get_varconfig_id(vc, "parametric")
        h2 = ArtifactCache._get_varconfig_id(vc, "parametric")
        assert h1 == h2

    def test_different_varconfig_produces_different_hash(self):
        vc1 = VariableConfig.for_variable("tas")
        vc2 = vc1.model_copy(update={"do_windowing": False})
        assert ArtifactCache._get_varconfig_id(
            vc1, "parametric"
        ) != ArtifactCache._get_varconfig_id(vc2, "parametric")

    def test_different_debias_approach_produces_different_hash(self):
        vc = VariableConfig.for_variable("tas")
        assert ArtifactCache._get_varconfig_id(vc, "parametric") != ArtifactCache._get_varconfig_id(
            vc, "nonparametric"
        )


# ---------------------------------------------------------------------------
# exists() – local filesystem
# ---------------------------------------------------------------------------


class TestExists:
    def test_nonexistent_store_returns_false(self, bound_cache, tmp_path):
        loc = StoreLocation(str(tmp_path / "missing.icechunk"), "obs/tas")
        assert bound_cache.exists(loc) is False

    def test_store_without_matching_commit_returns_false(self, bound_cache, tmp_path):
        import icechunk

        store_path = str(tmp_path / "empty_repo.icechunk")
        storage = icechunk.local_filesystem_storage(path=store_path)
        icechunk.Repository.open_or_create(storage)
        loc = StoreLocation(store_path, "obs/tas")
        assert bound_cache.exists(loc) is False

    def test_store_with_matching_group_commit_returns_true(self, bound_cache, tmp_path):
        loc = StoreLocation(str(tmp_path / "valid.icechunk"), "obs/tas")
        make_icechunk_group(loc, branch=bound_cache.branch)
        assert bound_cache.exists(loc) is True

    def test_store_with_different_commit_message_returns_false(self, bound_cache, tmp_path):
        loc = StoreLocation(str(tmp_path / "other.icechunk"), "obs/tas")
        wrong_loc = StoreLocation(str(tmp_path / "other.icechunk"), "obs/pr")
        make_icechunk_group(wrong_loc, branch=bound_cache.branch)
        assert bound_cache.exists(loc) is False

    def test_exists_returns_false_on_exception(self, bound_cache, tmp_path, monkeypatch):
        import icechunk

        def raise_error(*a, **kw):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(icechunk.Repository, "open", raise_error)
        loc = StoreLocation(str(tmp_path / "any.icechunk"), "obs/tas")
        assert bound_cache.exists(loc) is False

    def test_exists_via_ancestry_after_write(self, bound_cache, tmp_path):
        loc = StoreLocation(str(tmp_path / "test.icechunk"), "obs/tas")
        assert bound_cache.exists(loc) is False
        make_icechunk_group(loc, branch=bound_cache.branch)
        assert bound_cache.exists(loc) is True

    def test_sibling_group_does_not_satisfy_different_group_check(self, bound_cache, tmp_path):
        obs_loc = StoreLocation(str(tmp_path / "store.icechunk"), "obs/tas")
        hist_loc = StoreLocation(str(tmp_path / "store.icechunk"), "historical/tas/r1i1p1f1")
        make_icechunk_group(obs_loc, branch=bound_cache.branch)
        assert bound_cache.exists(hist_loc) is False


# ---------------------------------------------------------------------------
# check_dependencies / validate_dependencies
# ---------------------------------------------------------------------------


class TestCheckDependencies:
    def test_prepare_observations_has_no_deps(self, bound_cache, base_config):
        deps = bound_cache.check_dependencies("prepare_observations", base_config)
        assert deps == {}

    def test_fit_historical_requires_obs_only(self, bound_cache, base_config):
        deps = bound_cache.check_dependencies("fit_historical", base_config)
        assert set(deps.keys()) == {"obs_regridded"}

    def test_transform_scenario_requires_obs_and_historical(self, bound_cache, base_config):
        deps = bound_cache.check_dependencies("transform_scenario", base_config)
        assert set(deps.keys()) == {"obs_regridded", "historical"}

    def test_dependency_values_are_bool_storelocation_tuples(self, bound_cache, base_config):
        deps = bound_cache.check_dependencies("transform_scenario", base_config)
        for name, (exists, loc) in deps.items():
            assert isinstance(exists, bool), f"{name}: exists should be bool"
            assert isinstance(loc, StoreLocation), f"{name}: loc should be StoreLocation"

    def test_unknown_stage_raises(self, bound_cache, base_config):
        with pytest.raises(ValueError, match="Unknown stage"):
            bound_cache.check_dependencies("nonexistent_stage", base_config)

    def test_missing_deps_reported_as_false(self, bound_cache, base_config):
        deps = bound_cache.check_dependencies("fit_historical", base_config)
        exists, _loc = deps["obs_regridded"]
        assert exists is False


class TestValidateDependencies:
    def test_raises_missing(self, bound_cache, base_config):
        with pytest.raises(ValueError, match="Missing dependencies"):
            bound_cache.validate_dependencies("fit_historical", base_config)

    def test_passes_when_obs_group_exists(self, bound_cache, base_config):
        make_icechunk_group(bound_cache.obs_loc, branch=bound_cache.branch)
        bound_cache.validate_dependencies("fit_historical", base_config)

    def test_raises_when_only_obs_present_for_scenario_stage(self, bound_cache, base_config):
        make_icechunk_group(bound_cache.obs_loc, branch=bound_cache.branch)
        with pytest.raises(ValueError, match="Missing dependencies"):
            bound_cache.validate_dependencies("transform_scenario", base_config)

    def test_passes_when_all_scenario_deps_present(self, bound_cache, base_config):
        make_icechunk_group(bound_cache.obs_loc, branch=bound_cache.branch)
        make_icechunk_group(
            bound_cache.historical_loc(base_config.ensemble_member), branch=bound_cache.branch
        )
        bound_cache.validate_dependencies("transform_scenario", base_config)


# ---------------------------------------------------------------------------
# get_output_path
# ---------------------------------------------------------------------------


class TestGetOutputPath:
    def test_prepare_observations_returns_scratch_store_path(self, bound_cache):
        path = bound_cache.get_output_path("prepare_observations", bound_cache.config)
        assert path == bound_cache.obs_loc.store_path

    def test_fit_historical_returns_scratch_store_path(self, bound_cache):
        path = bound_cache.get_output_path(
            "fit_historical", bound_cache.config, hist_member="r1i1p1f1"
        )
        assert path == bound_cache.historical_loc("r1i1p1f1").store_path

    def test_transform_scenario_returns_output_store_path(self, bound_cache):
        path = bound_cache.get_output_path("transform_scenario", bound_cache.config)
        assert path == bound_cache.scenario_loc.store_path

    def test_transform_scenario_without_scenario_field_raises(self, tmp_path):
        config = BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member="r1i1p1f1")
        cache = ArtifactCache.from_config(config, PipelineOptions(scratch_dir=str(tmp_path)))
        with pytest.raises(ValueError, match="scenario must be specified"):
            cache.get_output_path("transform_scenario", config)

    def test_unknown_stage_raises(self, bound_cache):
        with pytest.raises(ValueError, match="Unknown stage"):
            bound_cache.get_output_path("bad_stage", bound_cache.config)

    def test_all_valid_stages_return_icechunk_strings(self, subtests, bound_cache):
        for stage in ("prepare_observations", "fit_historical", "transform_scenario"):
            with subtests.test(stage=stage):
                path = bound_cache.get_output_path(stage, bound_cache.config)
                assert isinstance(path, str)
                assert path.endswith(".icechunk")
