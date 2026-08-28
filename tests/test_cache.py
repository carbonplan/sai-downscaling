"""Tests for cache.py: ArtifactCache StoreLocation generation, existence checks, and helpers."""

from __future__ import annotations

import pytest
from conftest import make_icechunk_group

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.cache import (
    ArtifactCache,
    CacheCheckError,
    CacheConfigMismatchError,
    StoreLocation,
)

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
        downscaling_method="BCSD",
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
        downscaling_method="BCSD",
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
        gcm="UKESM",
        downscaling_method="BCSD",
        variable="tasmax",
        ensemble_member="01",
        scenario="SSP245",
        predict_period_start=2015,
        predict_period_end=2100,
        subset_bounds=(-35.0, -22.0, 16.0, 33.0),
    )


@pytest.fixture
def dtr_config() -> BCSDConfig:
    """dtr config — bias corrected so tasmin can be derived, never published fine."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="dtr",
        ensemble_member="008",
        scenario="SSP245",
        downscaling_method="BCSD",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def dtr_cache(tmp_path, dtr_config) -> ArtifactCache:
    """ArtifactCache with dtr_config bound and a separate output_dir."""
    return ArtifactCache.from_config(
        dtr_config,
        PipelineOptions(
            scratch_dir=str(tmp_path / "cache"),
            environment="qa",
            output_dir=str(tmp_path / "outputs"),
        ),
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
        assert loc.group == "bcsd/historical/tas/r1i1p1f1"

    def test_group_changes_with_member(self, bound_cache, subtests):
        for member in ("r1i1p1f1", "r12i1p1f2", "001"):
            with subtests.test(member=member):
                loc = bound_cache.historical_loc(member)
                assert loc.group == f"bcsd/historical/tas/{member}"

    def test_uses_output_store_when_set(self, bound_cache_with_output):
        loc = bound_cache_with_output.historical_loc("r1i1p1f1")
        assert bound_cache_with_output.output_dir in loc.store_path
        assert bound_cache_with_output.scratch_dir not in loc.store_path

    def test_uses_scratch_store_when_no_output_dir(self, bound_cache):
        loc = bound_cache.historical_loc("r1i1p1f1")
        assert bound_cache.scratch_dir in loc.store_path

    def test_store_path_encodes_gcm_obs_subset(self, bound_cache):
        loc = bound_cache.historical_loc("r1i1p1f1")
        assert "CESM2-WACCM-ERA5-global.icechunk" in loc.store_path

    def test_variable_override_targets_sibling_group(self, bound_cache):
        # tasmin's swap step reads the sibling fine tasmax output (issue #331).
        loc = bound_cache.historical_loc("r1i1p1f1", variable="tasmax")
        assert loc.group == "bcsd/historical/tasmax/r1i1p1f1"

    def test_no_variable_override_uses_config_variable(self, bound_cache):
        loc = bound_cache.historical_loc("r1i1p1f1")
        assert loc.group == "bcsd/historical/tas/r1i1p1f1"


# ---------------------------------------------------------------------------
# scenario_loc
# ---------------------------------------------------------------------------


class TestScenarioLoc:
    def test_group_encodes_scenario_variable_member(self, bound_cache):
        assert bound_cache.scenario_loc.group == "bcsd/ssp245/tas/r1i1p1f1"

    def test_sai_scenario_group(self, tmp_path, sai_config):
        cache = ArtifactCache.from_config(sai_config, PipelineOptions(scratch_dir=str(tmp_path)))
        assert cache.scenario_loc.group == "bcsd/g6_1p5k/pr/r2i1p1f1"

    def test_uses_output_store_when_set(self, bound_cache_with_output):
        loc = bound_cache_with_output.scenario_loc
        assert bound_cache_with_output.output_dir in loc.store_path

    def test_uses_scratch_store_when_no_output_dir(self, bound_cache):
        loc = bound_cache.scenario_loc
        assert bound_cache.scratch_dir in loc.store_path

    def test_output_loc_variable_override_targets_sibling_group(self, bound_cache):
        # tasmin's swap step reads the sibling fine tasmax scenario output (issue #331).
        loc = bound_cache.scenario_output_loc(variable="tasmax")
        assert loc.group == "bcsd/ssp245/tasmax/r1i1p1f1"

    def test_output_loc_without_override_matches_scenario_loc(self, bound_cache):
        assert bound_cache.scenario_output_loc().group == bound_cache.scenario_loc.group


# ---------------------------------------------------------------------------
# intermediate locs
# ---------------------------------------------------------------------------


class TestIntermediateLocs:
    def test_detrended_scenario_group(self, bound_cache):
        loc = bound_cache.detrended_scenario_loc()
        assert loc.group == "bcsd/detrended_scenario/ssp245/tas/r1i1p1f1"

    def test_trend_scenario_group(self, bound_cache):
        loc = bound_cache.trend_scenario_loc()
        assert loc.group == "bcsd/trend_scenario/ssp245/tas/r1i1p1f1"

    def test_debiased_scenario_group(self, bound_cache):
        loc = bound_cache.debiased_scenario_loc()
        assert loc.group == "bcsd/debiased_scenario/ssp245/tas/r1i1p1f1"

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
        assert cache.detrended_scenario_loc().group == "bcsd/detrended_scenario/g6_1p5k/pr/r2i1p1f1"

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
        assert loc.group == "bcsd/debiased_coarse/historical/tas/r1i1p1f1"

    def test_historical_group_changes_with_member(self, bound_cache, subtests):
        for member in ("r1i1p1f1", "r12i1p1f2", "001"):
            with subtests.test(member=member):
                loc = bound_cache.debiased_coarse_historical_loc(member)
                assert loc.group == f"bcsd/debiased_coarse/historical/tas/{member}"

    def test_historical_variable_override(self, bound_cache):
        loc = bound_cache.debiased_coarse_historical_loc("r1i1p1f1", variable="dtr")
        assert loc.group == "bcsd/debiased_coarse/historical/dtr/r1i1p1f1"

    def test_historical_uses_output_store(self, bound_cache_with_output):
        loc = bound_cache_with_output.debiased_coarse_historical_loc("r1i1p1f1")
        assert bound_cache_with_output.output_dir in loc.store_path
        assert bound_cache_with_output.scratch_dir not in loc.store_path

    def test_historical_falls_back_to_scratch_when_no_output_dir(self, bound_cache):
        loc = bound_cache.debiased_coarse_historical_loc("r1i1p1f1")
        assert bound_cache.scratch_dir in loc.store_path

    def test_scenario_group(self, bound_cache):
        loc = bound_cache.debiased_coarse_scenario_loc()
        assert loc.group == "bcsd/debiased_coarse/ssp245/tas/r1i1p1f1"

    def test_scenario_sai_group(self, tmp_path, sai_config):
        cache = ArtifactCache.from_config(sai_config, PipelineOptions(scratch_dir=str(tmp_path)))
        loc = cache.debiased_coarse_scenario_loc()
        assert loc.group == "bcsd/debiased_coarse/g6_1p5k/pr/r2i1p1f1"

    def test_scenario_variable_override(self, bound_cache):
        loc = bound_cache.debiased_coarse_scenario_loc(variable="dtr")
        assert loc.group == "bcsd/debiased_coarse/ssp245/dtr/r1i1p1f1"

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

    def test_persistent_infra_error_raises_cache_check_error(
        self, bound_cache, tmp_path, monkeypatch
    ):
        """A persistent infrastructure error must surface, not masquerade as a miss.

        Silently returning False here is what caused the v0.8.0 production deploy
        to discard 13 valid, already-committed scenario outputs.
        """
        import srm.cache as cache_module

        monkeypatch.setattr("time.sleep", lambda *a, **kw: None)

        def boom(*a, **kw):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(cache_module.icechunk.Repository, "exists", boom)
        loc = StoreLocation(str(tmp_path / "any.icechunk"), "obs/tas")
        with pytest.raises(CacheCheckError):
            bound_cache.exists(loc)

    def test_transient_error_is_retried_then_succeeds(self, bound_cache, tmp_path, monkeypatch):
        """A transient read error is retried; a real hit is still reported True."""
        import srm.cache as cache_module

        monkeypatch.setattr("time.sleep", lambda *a, **kw: None)
        loc = StoreLocation(str(tmp_path / "valid.icechunk"), "obs/tas")
        make_icechunk_group(loc, branch=bound_cache.branch)

        real_exists = cache_module.icechunk.Repository.exists
        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient network blip")
            return real_exists(*a, **kw)

        monkeypatch.setattr(cache_module.icechunk.Repository, "exists", flaky)
        assert bound_cache.exists(loc) is True
        assert calls["n"] == 2  # failed once, retried, then succeeded

    def test_absent_branch_returns_false_not_error(self, bound_cache, tmp_path):
        """A store that exists but lacks the queried branch is a genuine miss."""
        loc = StoreLocation(str(tmp_path / "store.icechunk"), "obs/tas")
        make_icechunk_group(loc, branch="main")
        other_branch = ArtifactCache(
            scratch_dir=bound_cache.scratch_dir, environment="qa", branch="v9.9.9"
        )
        assert other_branch.exists(loc) is False

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
# stage_loc — which artifact marks a stage complete
# ---------------------------------------------------------------------------


class TestStageLoc:
    """stage_loc is the single answer to 'which artifact means this stage finished'."""

    def test_obs_stage_is_obs_loc(self, bound_cache_with_output, base_config):
        loc = bound_cache_with_output.stage_loc("prepare_observations", base_config)
        assert loc == bound_cache_with_output.obs_loc

    def test_fit_historical_is_fine_loc_for_normal_variable(
        self, bound_cache_with_output, base_config
    ):
        loc = bound_cache_with_output.stage_loc(
            "fit_historical", base_config, hist_member="r3i1p1f1"
        )
        assert loc.group == "bcsd/historical/tas/r3i1p1f1"

    def test_fit_historical_falls_back_to_config_member(self, bound_cache_with_output, base_config):
        loc = bound_cache_with_output.stage_loc("fit_historical", base_config)
        assert loc.group == f"bcsd/historical/tas/{base_config.ensemble_member}"

    def test_transform_scenario_is_fine_loc_for_normal_variable(
        self, bound_cache_with_output, base_config
    ):
        loc = bound_cache_with_output.stage_loc("transform_scenario", base_config)
        assert loc.group == "bcsd/ssp245/tas/r1i1p1f1"

    def test_fit_historical_is_coarse_loc_for_dtr(self, dtr_cache, dtr_config):
        loc = dtr_cache.stage_loc("fit_historical", dtr_config, hist_member="001")
        assert loc.group == "bcsd/debiased_coarse/historical/dtr/001"

    def test_transform_scenario_is_coarse_loc_for_dtr(self, dtr_cache, dtr_config):
        loc = dtr_cache.stage_loc("transform_scenario", dtr_config)
        assert loc.group == "bcsd/debiased_coarse/ssp245/dtr/008"

    def test_dtr_obs_stage_is_still_the_obs_loc(self, dtr_cache, dtr_config):
        """Only the disaggregated outputs are dropped; obs regridding is unaffected."""
        assert dtr_cache.stage_loc("prepare_observations", dtr_config) == dtr_cache.obs_loc

    def test_unknown_stage_raises(self, bound_cache_with_output, base_config):
        with pytest.raises(ValueError, match="Unknown stage"):
            bound_cache_with_output.stage_loc("nope", base_config)

    def test_transform_scenario_without_scenario_raises(self, tmp_path):
        config = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="r1i1p1f1",
            downscaling_method="BCSD",
        )
        cache = ArtifactCache.from_config(config, PipelineOptions(scratch_dir=str(tmp_path)))
        with pytest.raises(ValueError, match="scenario must be specified"):
            cache.stage_loc("transform_scenario", config)

    def test_equal_but_distinct_config_is_accepted(self, bound_cache_with_output, base_config):
        twin = base_config.model_copy(deep=True)
        assert twin is not base_config
        loc = bound_cache_with_output.stage_loc("transform_scenario", twin)
        assert loc.group == "bcsd/ssp245/tas/r1i1p1f1"

    def test_unbound_config_raises_rather_than_mixing(self, subtests, bound_cache_with_output):
        # The location properties read the *bound* config while dispatch reads the passed
        # one, so a mismatch would silently return a path for the wrong artifact.
        for field, value in (
            ("variable", "dtr"),
            ("scenario", "G6-1.5K"),
            ("ensemble_member", "008"),
            ("gcm", "UKESM"),
        ):
            with subtests.test(field=field):
                other = bound_cache_with_output.config.model_copy(update={field: value})
                with pytest.raises(ValueError, match="disagrees with the config bound"):
                    bound_cache_with_output.stage_loc("transform_scenario", other)

    def test_mismatch_message_names_the_offending_field(self, bound_cache_with_output):
        other = bound_cache_with_output.config.model_copy(update={"variable": "dtr"})
        with pytest.raises(ValueError, match="'variable'"):
            bound_cache_with_output.stage_loc("fit_historical", other)

    def test_unbound_cache_raises(self, local_cache, base_config):
        with pytest.raises(RuntimeError, match="No config bound"):
            local_cache.stage_loc("fit_historical", base_config)


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

    def _tasmin_cache(self, tmp_path, scenario="SSP245"):
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            downscaling_method="BCSD",
            variable="tasmin",
            ensemble_member="r1i1p1f1",
            scenario=scenario,
            predict_period_start=2015,
            predict_period_end=2100,
        )
        cache = ArtifactCache(
            scratch_dir=str(tmp_path / "cache"),
            environment="qa",
            output_dir=str(tmp_path / "outputs"),
        )
        cache.config = cfg
        return cache, cfg

    def test_transform_scenario_tasmin_requires_debiased_coarse_dtr_and_tasmax(self, tmp_path):
        # tasmin is reconstructed as debiased_coarse tasmax - dtr, so those sibling
        # stores are hard dependencies of the tasmin scenario stage (issue #363).
        cache, cfg = self._tasmin_cache(tmp_path)
        deps = cache.check_dependencies("transform_scenario", cfg)
        assert "debiased_coarse_dtr" in deps
        assert "debiased_coarse_tasmax" in deps
        assert deps["debiased_coarse_dtr"][1].group == "bcsd/debiased_coarse/ssp245/dtr/r1i1p1f1"
        assert (
            deps["debiased_coarse_tasmax"][1].group == "bcsd/debiased_coarse/ssp245/tasmax/r1i1p1f1"
        )

    def test_fit_historical_tasmin_requires_debiased_coarse_dtr_and_tasmax(self, tmp_path):
        cache, cfg = self._tasmin_cache(tmp_path)
        deps = cache.check_dependencies("fit_historical", cfg)
        assert "debiased_coarse_dtr" in deps
        assert "debiased_coarse_tasmax" in deps
        assert (
            deps["debiased_coarse_dtr"][1].group == "bcsd/debiased_coarse/historical/dtr/r1i1p1f1"
        )
        assert (
            deps["debiased_coarse_tasmax"][1].group
            == "bcsd/debiased_coarse/historical/tasmax/r1i1p1f1"
        )

    def test_non_tasmin_scenario_deps_unchanged(self, bound_cache, base_config):
        # regression: non-derived variables keep the original obs+historical deps only.
        deps = bound_cache.check_dependencies("transform_scenario", base_config)
        assert set(deps.keys()) == {"obs_regridded", "historical"}

    def test_transform_scenario_tasmin_requires_fine_tasmax(self, tmp_path):
        # the tasmax<tasmin swap reads the fine tasmax output, so it is a hard
        # dependency and must fail fast, not deep in the stage (issue #331).
        cache, cfg = self._tasmin_cache(tmp_path)
        deps = cache.check_dependencies("transform_scenario", cfg)
        assert "fine_tasmax" in deps
        assert deps["fine_tasmax"][1].group == "bcsd/ssp245/tasmax/r1i1p1f1"

    def test_fit_historical_tasmin_requires_fine_tasmax(self, tmp_path):
        cache, cfg = self._tasmin_cache(tmp_path)
        deps = cache.check_dependencies("fit_historical", cfg)
        assert "fine_tasmax" in deps
        assert deps["fine_tasmax"][1].group == "bcsd/historical/tasmax/r1i1p1f1"

    def test_mismatched_config_raises_rather_than_mixing(self, bound_cache, base_config):
        other = base_config.model_copy(update={"variable": "dtr"})
        with pytest.raises(ValueError, match="disagrees with the config bound"):
            bound_cache.check_dependencies("fit_historical", other)

    def test_dtr_transform_scenario_gates_on_the_coarse_historical_group(
        self, dtr_cache, dtr_config
    ):
        # dtr writes no fine historical (issue #461), so its scenario gate must be the
        # coarse group or the stage could never start.
        deps = dtr_cache.check_dependencies("transform_scenario", dtr_config, hist_member="001")
        assert deps["historical"][1].group == "bcsd/debiased_coarse/historical/dtr/001"


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

    def test_dtr_scenario_deps_satisfied_by_the_coarse_group_alone(self, dtr_cache, dtr_config):
        make_icechunk_group(dtr_cache.obs_loc, branch=dtr_cache.branch)
        make_icechunk_group(
            dtr_cache.debiased_coarse_historical_loc("001"), branch=dtr_cache.branch
        )
        dtr_cache.validate_dependencies("transform_scenario", dtr_config, hist_member="001")


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
        config = BCSDConfig(
            downscaling_method="BCSD", gcm="CESM2-WACCM", variable="tas", ensemble_member="r1i1p1f1"
        )
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

    def test_delegates_to_stage_loc(self, subtests, bound_cache, base_config):
        # get_output_path must stay a thin view on stage_loc so the two can never
        # disagree about which artifact ends a stage.
        for stage in ("prepare_observations", "fit_historical", "transform_scenario"):
            with subtests.test(stage=stage):
                assert (
                    bound_cache.get_output_path(stage, base_config)
                    == bound_cache.stage_loc(stage, base_config).store_path
                )

    def test_dtr_returns_a_path_for_every_stage(self, subtests, dtr_cache, dtr_config):
        # dtr's coarse and fine locs live in the same store, so only the *group* differs
        # (that distinction is stage_loc's job). get_output_path must still not raise.
        for stage in ("prepare_observations", "fit_historical", "transform_scenario"):
            with subtests.test(stage=stage):
                assert dtr_cache.get_output_path(stage, dtr_config).endswith(".icechunk")


# ---------------------------------------------------------------------------
# exists() – VariableConfig verification on a cache hit
# ---------------------------------------------------------------------------


def _write_artifact_with_attrs(loc: StoreLocation, branch: str, attrs: dict | None) -> None:
    """Commit a group at ``loc.group`` carrying ``attrs``, as the pipeline does.

    ``make_icechunk_group`` writes at the store root, but the provenance the
    verification reads lives on the group itself, so this writes there instead.
    """
    import icechunk
    import numpy as np
    import xarray as xr
    from icechunk.xarray import to_icechunk

    from srm.config import _ensure_root_group, _icechunk_storage_for_path

    storage = _icechunk_storage_for_path(loc.store_path)
    repo = icechunk.Repository.open_or_create(storage)
    root_snapshot_id = _ensure_root_group(repo)
    if branch not in repo.list_branches():
        repo.create_branch(branch, root_snapshot_id)
    session = repo.writable_session(branch)
    ds = xr.Dataset({"dummy": xr.DataArray(np.array([1.0]), dims=["x"])})
    if attrs is not None:
        ds.attrs = attrs
    to_icechunk(ds, session, mode="w", group=loc.group)
    session.commit(loc.group)


def _provenance(config) -> dict:
    """The subset of pipeline provenance attrs that verification reads."""
    return {"srm_downscaling:config_json": config.model_dump_json()}


class TestVariableConfigVerification:
    """A cache hit computed under a different VariableConfig must not be reused."""

    def test_matching_variable_config_is_a_hit(self, bound_cache):
        loc = bound_cache.detrended_scenario_loc()
        _write_artifact_with_attrs(loc, bound_cache.branch, _provenance(bound_cache.config))
        assert bound_cache.exists(loc) is True

    def test_differing_variable_config_raises(self, bound_cache):
        loc = bound_cache.detrended_scenario_loc()
        written_by = bound_cache.config.model_copy(
            update={
                "variable_config": bound_cache.config.variable_config.model_copy(
                    update={"debias_approach": "parametric"}
                )
            }
        )
        _write_artifact_with_attrs(loc, bound_cache.branch, _provenance(written_by))
        with pytest.raises(CacheConfigMismatchError, match="debias_approach"):
            bound_cache.exists(loc)

    def test_mismatch_message_names_both_values(self, bound_cache):
        loc = bound_cache.detrended_scenario_loc()
        written_by = bound_cache.config.model_copy(
            update={
                "variable_config": bound_cache.config.variable_config.model_copy(
                    update={"do_windowing": False}
                )
            }
        )
        _write_artifact_with_attrs(loc, bound_cache.branch, _provenance(written_by))
        with pytest.raises(CacheConfigMismatchError) as excinfo:
            bound_cache.exists(loc)
        assert "cached=False" in str(excinfo.value)
        assert "current=True" in str(excinfo.value)

    def test_artifact_without_provenance_is_allowed(self, bound_cache):
        """Artifacts predating config provenance are unverifiable, not mismatched."""
        loc = bound_cache.detrended_scenario_loc()
        _write_artifact_with_attrs(loc, bound_cache.branch, None)
        assert bound_cache.exists(loc) is True

    def test_obs_artifact_is_never_verified(self, bound_cache):
        """Regridding does not read VariableConfig, so obs stays reusable across runs."""
        loc = bound_cache.obs_loc
        assert loc.config_variable is None
        written_by = bound_cache.config.model_copy(
            update={
                "variable_config": bound_cache.config.variable_config.model_copy(
                    update={"debias_approach": "parametric"}
                )
            }
        )
        _write_artifact_with_attrs(loc, bound_cache.branch, _provenance(written_by))
        assert bound_cache.exists(loc) is True

    def test_sibling_variable_lookup_is_never_verified(self, bound_cache):
        """A sibling's intended config is not knowable from this process."""
        loc = bound_cache.historical_loc("r1i1p1f1", variable="dtr")
        assert loc.config_variable is None

    def test_own_variable_lookup_is_tagged_for_verification(self, bound_cache):
        loc = bound_cache.historical_loc("r1i1p1f1")
        assert loc.config_variable == bound_cache.config.variable

    def test_cache_miss_skips_verification(self, bound_cache, tmp_path):
        loc = StoreLocation(str(tmp_path / "missing.icechunk"), "obs/tas", config_variable="tas")
        assert bound_cache.exists(loc) is False


# ---------------------------------------------------------------------------
# downscaling_method namespacing
# ---------------------------------------------------------------------------


class TestDownscalingMethodNamespacing:
    """Method-dependent groups are namespaced; regridded obs is shared."""

    @staticmethod
    def _cache_for(method: str, tmp_path) -> ArtifactCache:
        return ArtifactCache.from_config(
            BCSDConfig(
                gcm="CESM2-WACCM",
                downscaling_method=method,
                variable="tas",
                ensemble_member="r1i1p1f1",
                scenario="SSP245",
                predict_period_start=2015,
                predict_period_end=2100,
            ),
            PipelineOptions(
                scratch_dir=str(tmp_path / "cache"),
                environment="qa",
                output_dir=str(tmp_path / "outputs"),
            ),
        )

    @staticmethod
    def _method_dependent_groups(cache: ArtifactCache) -> dict[str, str]:
        return {
            "historical": cache.historical_loc("r1i1p1f1").group,
            "scenario": cache.scenario_output_loc().group,
            "debiased_coarse_historical": cache.debiased_coarse_historical_loc("r1i1p1f1").group,
            "debiased_coarse_scenario": cache.debiased_coarse_scenario_loc().group,
            "detrended_scenario": cache.detrended_scenario_loc().group,
            "trend_scenario": cache.trend_scenario_loc().group,
            "debiased_scenario": cache.debiased_scenario_loc().group,
        }

    def test_bcsd_groups_are_namespaced(self, tmp_path):
        groups = self._method_dependent_groups(self._cache_for("BCSD", tmp_path))
        assert groups == {
            "historical": "bcsd/historical/tas/r1i1p1f1",
            "scenario": "bcsd/ssp245/tas/r1i1p1f1",
            "debiased_coarse_historical": "bcsd/debiased_coarse/historical/tas/r1i1p1f1",
            "debiased_coarse_scenario": "bcsd/debiased_coarse/ssp245/tas/r1i1p1f1",
            "detrended_scenario": "bcsd/detrended_scenario/ssp245/tas/r1i1p1f1",
            "trend_scenario": "bcsd/trend_scenario/ssp245/tas/r1i1p1f1",
            "debiased_scenario": "bcsd/debiased_scenario/ssp245/tas/r1i1p1f1",
        }

    def test_qdmsd_groups_are_namespaced(self, tmp_path):
        groups = self._method_dependent_groups(self._cache_for("QDMSD", tmp_path))
        assert groups == {
            "historical": "qdmsd/historical/tas/r1i1p1f1",
            "scenario": "qdmsd/ssp245/tas/r1i1p1f1",
            "debiased_coarse_historical": "qdmsd/debiased_coarse/historical/tas/r1i1p1f1",
            "debiased_coarse_scenario": "qdmsd/debiased_coarse/ssp245/tas/r1i1p1f1",
            "detrended_scenario": "qdmsd/detrended_scenario/ssp245/tas/r1i1p1f1",
            "trend_scenario": "qdmsd/trend_scenario/ssp245/tas/r1i1p1f1",
            "debiased_scenario": "qdmsd/debiased_scenario/ssp245/tas/r1i1p1f1",
        }

    def test_no_group_is_shared_between_methods(self, tmp_path, subtests):
        bcsd = self._method_dependent_groups(self._cache_for("BCSD", tmp_path))
        qdmsd = self._method_dependent_groups(self._cache_for("QDMSD", tmp_path))
        for name in bcsd:
            with subtests.test(artifact=name):
                assert bcsd[name] != qdmsd[name]

    def test_obs_is_shared_between_methods(self, tmp_path):
        bcsd = self._cache_for("BCSD", tmp_path).obs_loc
        qdmsd = self._cache_for("QDMSD", tmp_path).obs_loc
        assert bcsd.group == "obs/tas"
        assert bcsd.group == qdmsd.group
        assert bcsd.store_path == qdmsd.store_path

    def test_method_segment_precedes_debiased_coarse(self, tmp_path):
        """The segment is outermost, so there is one rule rather than a per-group one."""
        cache = self._cache_for("QDMSD", tmp_path)
        assert cache.debiased_coarse_scenario_loc().group.startswith("qdmsd/debiased_coarse/")


class TestIntermediateClassificationAcrossMethods:
    """Intermediate classification must not depend on the bound config's method."""

    def test_strips_the_method_segment(self):
        assert ArtifactCache._strip_method_segment("bcsd/detrended_scenario/ssp245/tas/001") == (
            "detrended_scenario/ssp245/tas/001"
        )
        assert ArtifactCache._strip_method_segment("obs/tas") == "tas"

    def test_classifies_both_methods_in_one_store(self, bound_cache, monkeypatch):
        groups = [
            "obs/tas",
            "bcsd/detrended_scenario/ssp245/tas/001",
            "bcsd/trend_scenario/ssp245/tas/001",
            "bcsd/debiased_scenario/ssp245/tas/001",
            "qdmsd/debiased_scenario/ssp245/tas/001",
            "bcsd/historical/tas/001",
            "qdmsd/ssp245/tas/001",
            "bcsd/debiased_coarse/ssp245/tas/001",
        ]
        monkeypatch.setattr(ArtifactCache, "list_groups_on_branch", lambda self, store: groups)

        result = bound_cache.list_intermediate_groups()
        found = sorted({g for groups_ in result.values() for g in groups_})

        assert found == [
            "bcsd/debiased_scenario/ssp245/tas/001",
            "bcsd/detrended_scenario/ssp245/tas/001",
            "bcsd/trend_scenario/ssp245/tas/001",
            "qdmsd/debiased_scenario/ssp245/tas/001",
        ]

    def test_still_classifies_pre_namespace_paths(self, bound_cache, monkeypatch):
        """Old branches hold unprefixed intermediates; listing them is read-only."""
        groups = ["detrended_scenario/ssp245/tas/001", "historical/tas/001"]
        monkeypatch.setattr(ArtifactCache, "list_groups_on_branch", lambda self, store: groups)

        result = bound_cache.list_intermediate_groups()
        found = sorted({g for groups_ in result.values() for g in groups_})

        assert found == ["detrended_scenario/ssp245/tas/001"]
