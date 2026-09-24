"""Tests for cache.py: ArtifactCache locations, existence checks, and dependency gates."""

from __future__ import annotations

import pytest
from conftest import make_icechunk_group

from saidownscale.cache import (
    ArtifactCache,
    CacheCheckError,
    CacheConfigMismatchError,
    StoreLocation,
)
from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions

STAGES = ("prepare_observations", "fit_historical", "transform_scenario")


def _config(**overrides) -> DownscalingConfig:
    fields = {
        "gcm": "CESM2-WACCM6",
        "downscaling_method": "BCSD",
        "variable": "tas",
        "ensemble_member": "r1i1p1f1",
        "scenario": "SSP245",
        "predict_period_start": 2015,
        "predict_period_end": 2100,
    }
    return DownscalingConfig(**(fields | overrides))


def _bind(config, tmp_path, output_dir=True, environment="qa") -> ArtifactCache:
    cache = ArtifactCache(
        scratch_dir=str(tmp_path / "cache"),
        environment=environment,
        output_dir=str(tmp_path / "outputs") if output_dir else None,
    )
    cache.config = config
    return cache


@pytest.fixture
def local_cache(tmp_path) -> ArtifactCache:
    return ArtifactCache(scratch_dir=str(tmp_path / "cache"), environment="qa")


@pytest.fixture
def base_config() -> DownscalingConfig:
    return _config()


@pytest.fixture
def dtr_config() -> DownscalingConfig:
    return _config(variable="dtr", ensemble_member="008")


@pytest.fixture
def dtr_cache(tmp_path, dtr_config) -> ArtifactCache:
    return _bind(dtr_config, tmp_path)


@pytest.fixture
def bound_cache(tmp_path, base_config) -> ArtifactCache:
    return _bind(base_config, tmp_path, output_dir=False)


@pytest.fixture
def bound_cache_with_output(tmp_path, base_config) -> ArtifactCache:
    return ArtifactCache.from_config(
        base_config,
        PipelineOptions(
            scratch_dir=str(tmp_path / "cache"),
            environment="qa",
            output_dir=str(tmp_path / "outputs"),
        ),
    )


def test_store_location_is_frozen_value_object():
    a = StoreLocation("s3://bucket/path.icechunk", "obs/tas")
    assert a == StoreLocation("s3://bucket/path.icechunk", "obs/tas")
    assert a != StoreLocation("s3://bucket/path.icechunk", "obs/pr")
    with pytest.raises(Exception):
        a.store_path = "other"  # type: ignore[misc]


def test_init_normalizes_dirs_and_stores_settings(tmp_path):
    cache = ArtifactCache(
        scratch_dir=str(tmp_path) + "/",
        environment="production",
        branch="v2.0",
        output_dir=str(tmp_path / "outputs") + "/",
    )
    assert cache.scratch_dir == str(tmp_path)
    assert cache.output_dir == str(tmp_path / "outputs")
    assert (cache.environment, cache.branch, cache.config) == ("production", "v2.0", None)
    assert ArtifactCache(scratch_dir=str(tmp_path)).output_dir is None


def test_from_config_binds_config(base_config):
    assert ArtifactCache.from_config(base_config, PipelineOptions()).config is base_config


def test_unbound_cache_raises(local_cache, base_config):
    with pytest.raises(RuntimeError, match="No config bound"):
        _ = local_cache.obs_loc
    with pytest.raises(RuntimeError, match="No config bound"):
        _ = local_cache.scenario_loc
    with pytest.raises(RuntimeError, match="No config bound"):
        local_cache.stage_loc("fit_historical", base_config)


def test_get_subset_id(subtests):
    cases = [
        (None, "global"),
        ((-35.0, -22.0, 16.0, 33.0), "lat-35.0to-22.0_lon16.0to33.0"),
        ((0.0, 90.0, -180.0, 180.0), "lat0.0to90.0_lon-180.0to180.0"),
    ]
    for bounds, expected in cases:
        with subtests.test(bounds=str(bounds)):
            assert ArtifactCache._get_subset_id(bounds) == expected


def test_store_paths(subtests, tmp_path):
    regional = _config(
        gcm="UKESM1-1-LL",
        variable="tasmax",
        ensemble_member="01",
        subset_bounds=(-35.0, -22.0, 16.0, 33.0),
    )
    cases = [
        ("global-no-output-dir", _config(), "qa", False, "CESM2-WACCM6-ERA5-global.icechunk"),
        ("production", _config(), "production", True, "CESM2-WACCM6-ERA5-global.icechunk"),
        (
            "regional",
            regional,
            "qa",
            True,
            "UKESM1-1-LL-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk",
        ),
    ]
    for case_id, config, env, with_output, name in cases:
        with subtests.test(case=case_id):
            cache = _bind(config, tmp_path, output_dir=with_output, environment=env)
            scratch = f"{tmp_path}/cache/{env}/{name}"
            output = f"{tmp_path}/{'outputs' if with_output else 'cache'}/{env}/{name}"
            assert cache._scratch_store == scratch
            assert cache._output_store == output
            member = config.ensemble_member
            for loc in (
                cache.obs_loc,
                cache.detrended_scenario_loc(),
                cache.trend_scenario_loc(),
                cache.debiased_scenario_loc(),
            ):
                assert loc.store_path == scratch
            for loc in (
                cache.historical_loc(member),
                cache.scenario_loc,
                cache.debiased_coarse_historical_loc(member),
                cache.debiased_coarse_scenario_loc(),
            ):
                assert loc.store_path == output


def test_groups_are_namespaced_by_method_except_obs(subtests, tmp_path):
    for method in ("BCSD", "QDMSD"):
        with subtests.test(method=method):
            cache = _bind(_config(downscaling_method=method), tmp_path)
            m = method.lower()
            groups = {
                "obs": cache.obs_loc.group,
                "historical": cache.historical_loc("r1i1p1f1").group,
                "scenario": cache.scenario_loc.group,
                "scenario_output": cache.scenario_output_loc().group,
                "dc_historical": cache.debiased_coarse_historical_loc("r1i1p1f1").group,
                "dc_scenario": cache.debiased_coarse_scenario_loc().group,
                "detrended": cache.detrended_scenario_loc().group,
                "trend": cache.trend_scenario_loc().group,
                "debiased": cache.debiased_scenario_loc().group,
            }
            assert groups == {
                "obs": "obs/tas",
                "historical": f"{m}/historical/tas/r1i1p1f1",
                "scenario": f"{m}/ssp245/tas/r1i1p1f1",
                "scenario_output": f"{m}/ssp245/tas/r1i1p1f1",
                "dc_historical": f"{m}/debiased_coarse/historical/tas/r1i1p1f1",
                "dc_scenario": f"{m}/debiased_coarse/ssp245/tas/r1i1p1f1",
                "detrended": f"{m}/detrended_scenario/ssp245/tas/r1i1p1f1",
                "trend": f"{m}/trend_scenario/ssp245/tas/r1i1p1f1",
                "debiased": f"{m}/debiased_scenario/ssp245/tas/r1i1p1f1",
            }


def test_group_paths_for_member_variable_and_sai_variants(subtests, tmp_path):
    """Variable overrides let tasmin reach sibling tasmax outputs (#331)."""
    base = _bind(_config(), tmp_path)
    sai = _bind(_config(variable="pr", ensemble_member="r2i1p1f1", scenario="G6-1.5K"), tmp_path)
    cases = [
        ("hist-member", base.historical_loc("001"), "bcsd/historical/tas/001"),
        (
            "hist-override",
            base.historical_loc("r1i1p1f1", variable="tasmax"),
            "bcsd/historical/tasmax/r1i1p1f1",
        ),
        (
            "scenario-override",
            base.scenario_output_loc(variable="tasmax"),
            "bcsd/ssp245/tasmax/r1i1p1f1",
        ),
        (
            "dc-hist-member",
            base.debiased_coarse_historical_loc("001"),
            "bcsd/debiased_coarse/historical/tas/001",
        ),
        (
            "dc-hist-override",
            base.debiased_coarse_historical_loc("r1i1p1f1", variable="dtr"),
            "bcsd/debiased_coarse/historical/dtr/r1i1p1f1",
        ),
        (
            "dc-scenario-override",
            base.debiased_coarse_scenario_loc(variable="dtr"),
            "bcsd/debiased_coarse/ssp245/dtr/r1i1p1f1",
        ),
        ("sai-scenario", sai.scenario_loc, "bcsd/g6_1p5k/pr/r2i1p1f1"),
        (
            "sai-detrended",
            sai.detrended_scenario_loc(),
            "bcsd/detrended_scenario/g6_1p5k/pr/r2i1p1f1",
        ),
        (
            "sai-dc-scenario",
            sai.debiased_coarse_scenario_loc(),
            "bcsd/debiased_coarse/g6_1p5k/pr/r2i1p1f1",
        ),
    ]
    for case_id, loc, expected in cases:
        with subtests.test(case=case_id):
            assert loc.group == expected


class TestExists:
    def test_misses(self, subtests, bound_cache, tmp_path):
        import icechunk

        branch = bound_cache.branch
        empty = str(tmp_path / "empty.icechunk")
        icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(path=empty))
        make_icechunk_group(
            StoreLocation(str(tmp_path / "other.icechunk"), "obs/pr"), branch=branch
        )
        make_icechunk_group(
            StoreLocation(str(tmp_path / "branch.icechunk"), "obs/tas"), branch="main"
        )
        other_branch = ArtifactCache(
            scratch_dir=bound_cache.scratch_dir, environment="qa", branch="v9.9.9"
        )
        cases = [
            ("missing-store", bound_cache, "missing.icechunk"),
            ("empty-repo", bound_cache, "empty.icechunk"),
            ("other-group", bound_cache, "other.icechunk"),
            ("absent-branch", other_branch, "branch.icechunk"),
        ]
        for case_id, cache, store in cases:
            with subtests.test(case=case_id):
                assert cache.exists(StoreLocation(str(tmp_path / store), "obs/tas")) is False

    def test_hit_after_write(self, bound_cache, tmp_path):
        loc = StoreLocation(str(tmp_path / "test.icechunk"), "obs/tas")
        assert bound_cache.exists(loc) is False
        make_icechunk_group(loc, branch=bound_cache.branch)
        assert bound_cache.exists(loc) is True

    def test_persistent_infra_error_raises_cache_check_error(
        self, bound_cache, tmp_path, monkeypatch
    ):
        """Returning False here discarded 13 committed outputs in the v0.8.0 deploy."""
        import saidownscale.cache as cache_module

        monkeypatch.setattr("time.sleep", lambda *a, **kw: None)

        def boom(*a, **kw):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(cache_module.icechunk.Repository, "exists", boom)
        with pytest.raises(CacheCheckError):
            bound_cache.exists(StoreLocation(str(tmp_path / "any.icechunk"), "obs/tas"))

    def test_transient_error_is_retried_then_succeeds(self, bound_cache, tmp_path, monkeypatch):
        import saidownscale.cache as cache_module

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
        assert calls["n"] == 2


class TestStageLoc:
    def test_terminal_artifact_per_stage(self, subtests, bound_cache_with_output, base_config):
        """dtr is never disaggregated, so its terminal artifact is the coarse group (#461)."""
        cases = [
            ("obs", "prepare_observations", {}, "obs/tas"),
            (
                "hist-member",
                "fit_historical",
                {"hist_member": "r3i1p1f1"},
                "bcsd/historical/tas/r3i1p1f1",
            ),
            ("hist-default-member", "fit_historical", {}, "bcsd/historical/tas/r1i1p1f1"),
            ("scenario", "transform_scenario", {}, "bcsd/ssp245/tas/r1i1p1f1"),
        ]
        for case_id, stage, kwargs, expected in cases:
            with subtests.test(case=case_id):
                loc = bound_cache_with_output.stage_loc(stage, base_config, **kwargs)
                assert loc.group == expected
        assert (
            bound_cache_with_output.stage_loc("prepare_observations", base_config)
            == bound_cache_with_output.obs_loc
        )

    def test_dtr_terminal_artifact_is_coarse(self, dtr_cache, dtr_config):
        assert dtr_cache.stage_loc("prepare_observations", dtr_config) == dtr_cache.obs_loc
        assert (
            dtr_cache.stage_loc("fit_historical", dtr_config, hist_member="001").group
            == "bcsd/debiased_coarse/historical/dtr/001"
        )
        assert (
            dtr_cache.stage_loc("transform_scenario", dtr_config).group
            == "bcsd/debiased_coarse/ssp245/dtr/008"
        )

    def test_equal_but_distinct_config_is_accepted(self, bound_cache_with_output, base_config):
        twin = base_config.model_copy(deep=True)
        assert twin is not base_config
        loc = bound_cache_with_output.stage_loc("transform_scenario", twin)
        assert loc.group == "bcsd/ssp245/tas/r1i1p1f1"

    def test_invalid_stage_or_config_raises(self, subtests, tmp_path, bound_cache_with_output):
        no_scenario = DownscalingConfig(
            gcm="CESM2-WACCM6",
            downscaling_method="BCSD",
            variable="tas",
            ensemble_member="r1i1p1f1",
        )
        no_scenario_cache = _bind(no_scenario, tmp_path)
        bound = bound_cache_with_output.config
        with subtests.test(case="unknown-stage"):
            for fn in (bound_cache_with_output.stage_loc, bound_cache_with_output.get_output_path):
                with pytest.raises(ValueError, match="Unknown stage"):
                    fn("nope", bound)
        with subtests.test(case="no-scenario"):
            for fn in (no_scenario_cache.stage_loc, no_scenario_cache.get_output_path):
                with pytest.raises(ValueError, match="scenario must be specified"):
                    fn("transform_scenario", no_scenario)
        for field, value in (
            ("variable", "dtr"),
            ("scenario", "G6-1.5K"),
            ("ensemble_member", "008"),
            ("gcm", "UKESM1-1-LL"),
        ):
            with subtests.test(field=field):
                other = bound.model_copy(update={field: value})
                with pytest.raises(ValueError, match=f"disagrees with the config bound.*'{field}'"):
                    bound_cache_with_output.stage_loc("transform_scenario", other)

    def test_get_output_path_is_stage_loc_store_path(
        self, subtests, bound_cache, dtr_cache, base_config, dtr_config
    ):
        for cache, config in ((bound_cache, base_config), (dtr_cache, dtr_config)):
            for stage in STAGES:
                with subtests.test(variable=config.variable, stage=stage):
                    path = cache.get_output_path(stage, config)
                    assert path.endswith(".icechunk")
                    assert path == cache.stage_loc(stage, config).store_path


class TestCheckDependencies:
    def test_dependency_keys_per_stage(self, bound_cache, base_config):
        assert bound_cache.check_dependencies("prepare_observations", base_config) == {}
        fit = bound_cache.check_dependencies("fit_historical", base_config)
        assert set(fit) == {"obs_regridded"}
        assert fit["obs_regridded"] == (False, bound_cache.obs_loc)
        scen = bound_cache.check_dependencies("transform_scenario", base_config)
        assert set(scen) == {"obs_regridded", "historical"}
        for exists, loc in scen.values():
            assert isinstance(exists, bool)
            assert isinstance(loc, StoreLocation)

    def test_invalid_stage_or_config_raises(self, bound_cache, base_config):
        with pytest.raises(ValueError, match="Unknown stage"):
            bound_cache.check_dependencies("nonexistent_stage", base_config)
        other = base_config.model_copy(update={"variable": "dtr"})
        with pytest.raises(ValueError, match="disagrees with the config bound"):
            bound_cache.check_dependencies("fit_historical", other)

    def test_tasmin_requires_sibling_outputs(self, subtests, tmp_path):
        """tasmin = coarse tasmax - dtr (#363), and its swap reads fine tasmax (#331)."""
        cfg = _config(variable="tasmin")
        cache = _bind(cfg, tmp_path)
        expected = {
            "fit_historical": {
                "debiased_coarse_dtr": "bcsd/debiased_coarse/historical/dtr/r1i1p1f1",
                "debiased_coarse_tasmax": "bcsd/debiased_coarse/historical/tasmax/r1i1p1f1",
                "fine_tasmax": "bcsd/historical/tasmax/r1i1p1f1",
            },
            "transform_scenario": {
                "debiased_coarse_dtr": "bcsd/debiased_coarse/ssp245/dtr/r1i1p1f1",
                "debiased_coarse_tasmax": "bcsd/debiased_coarse/ssp245/tasmax/r1i1p1f1",
                "fine_tasmax": "bcsd/ssp245/tasmax/r1i1p1f1",
            },
        }
        for stage, groups in expected.items():
            with subtests.test(stage=stage):
                deps = cache.check_dependencies(stage, cfg)
                for name, group in groups.items():
                    assert deps[name][1].group == group

    def test_dtr_scenario_gates_on_coarse_historical(self, dtr_cache, dtr_config):
        """dtr writes no fine historical (#461), so the scenario gate is the coarse group."""
        deps = dtr_cache.check_dependencies("transform_scenario", dtr_config, hist_member="001")
        assert deps["historical"][1].group == "bcsd/debiased_coarse/historical/dtr/001"


class TestValidateDependencies:
    def test_fit_historical_raises_until_obs_exists(self, bound_cache, base_config):
        with pytest.raises(ValueError, match="Missing dependencies"):
            bound_cache.validate_dependencies("fit_historical", base_config)
        make_icechunk_group(bound_cache.obs_loc, branch=bound_cache.branch)
        bound_cache.validate_dependencies("fit_historical", base_config)

    def test_transform_scenario_raises_until_obs_and_historical_exist(
        self, bound_cache, base_config
    ):
        make_icechunk_group(bound_cache.obs_loc, branch=bound_cache.branch)
        with pytest.raises(ValueError, match="Missing dependencies"):
            bound_cache.validate_dependencies("transform_scenario", base_config)
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


def _write_artifact_with_attrs(loc: StoreLocation, branch: str, attrs: dict | None) -> None:
    import icechunk
    import numpy as np
    import xarray as xr
    from icechunk.xarray import to_icechunk

    from saidownscale.config import _ensure_root_group, _icechunk_storage_for_path

    repo = icechunk.Repository.open_or_create(_icechunk_storage_for_path(loc.store_path))
    root_snapshot_id = _ensure_root_group(repo)
    if branch not in repo.list_branches():
        repo.create_branch(branch, root_snapshot_id)
    session = repo.writable_session(branch)
    ds = xr.Dataset({"dummy": xr.DataArray(np.array([1.0]), dims=["x"])})
    if attrs is not None:
        ds.attrs = attrs
    to_icechunk(ds, session, mode="w", group=loc.group)
    session.commit(loc.group)


def _provenance(config, **variable_config_updates) -> dict:
    if variable_config_updates:
        config = config.model_copy(
            update={
                "variable_config": config.variable_config.model_copy(update=variable_config_updates)
            }
        )
    return {"srm_downscaling:config_json": config.model_dump_json()}


class TestVariableConfigVerification:
    def test_matching_variable_config_is_a_hit(self, bound_cache):
        loc = bound_cache.detrended_scenario_loc()
        _write_artifact_with_attrs(loc, bound_cache.branch, _provenance(bound_cache.config))
        assert bound_cache.exists(loc) is True

    def test_artifact_without_provenance_is_allowed(self, bound_cache):
        loc = bound_cache.detrended_scenario_loc()
        _write_artifact_with_attrs(loc, bound_cache.branch, None)
        assert bound_cache.exists(loc) is True

    def test_differing_variable_config_raises(self, subtests, bound_cache, tmp_path):
        cases = [
            ("debias_approach", {"debias_approach": "parametric"}, ["debias_approach"]),
            ("do_windowing", {"do_windowing": False}, ["cached=False", "current=True"]),
        ]
        for case_id, update, fragments in cases:
            with subtests.test(field=case_id):
                loc = StoreLocation(
                    str(tmp_path / f"{case_id}.icechunk"),
                    bound_cache.detrended_scenario_loc().group,
                    config_variable=bound_cache.config.variable,
                )
                _write_artifact_with_attrs(
                    loc, bound_cache.branch, _provenance(bound_cache.config, **update)
                )
                with pytest.raises(CacheConfigMismatchError) as excinfo:
                    bound_cache.exists(loc)
                for fragment in fragments:
                    assert fragment in str(excinfo.value)

    def test_obs_and_sibling_lookups_are_never_verified(self, bound_cache):
        """Regridding ignores VariableConfig, and a sibling's intended config is unknowable."""
        assert bound_cache.historical_loc("r1i1p1f1", variable="dtr").config_variable is None
        assert bound_cache.historical_loc("r1i1p1f1").config_variable == "tas"
        loc = bound_cache.obs_loc
        assert loc.config_variable is None
        provenance = _provenance(bound_cache.config, debias_approach="parametric")
        _write_artifact_with_attrs(loc, bound_cache.branch, provenance)
        assert bound_cache.exists(loc) is True

    def test_cache_miss_skips_verification(self, bound_cache, tmp_path):
        loc = StoreLocation(str(tmp_path / "missing.icechunk"), "obs/tas", config_variable="tas")
        assert bound_cache.exists(loc) is False


class TestIntermediateClassification:
    def test_strips_the_method_segment(self):
        assert (
            ArtifactCache._strip_method_segment("bcsd/detrended_scenario/ssp245/tas/001")
            == "detrended_scenario/ssp245/tas/001"
        )
        assert ArtifactCache._strip_method_segment("obs/tas") == "tas"

    def test_classifies_groups(self, subtests, bound_cache, monkeypatch):
        cases = [
            (
                "both-methods",
                [
                    "obs/tas",
                    "bcsd/detrended_scenario/ssp245/tas/001",
                    "bcsd/trend_scenario/ssp245/tas/001",
                    "bcsd/debiased_scenario/ssp245/tas/001",
                    "qdmsd/debiased_scenario/ssp245/tas/001",
                    "bcsd/historical/tas/001",
                    "qdmsd/ssp245/tas/001",
                    "bcsd/debiased_coarse/ssp245/tas/001",
                ],
                [
                    "bcsd/debiased_scenario/ssp245/tas/001",
                    "bcsd/detrended_scenario/ssp245/tas/001",
                    "bcsd/trend_scenario/ssp245/tas/001",
                    "qdmsd/debiased_scenario/ssp245/tas/001",
                ],
            ),
            (
                "pre-namespace",
                ["detrended_scenario/ssp245/tas/001", "historical/tas/001"],
                ["detrended_scenario/ssp245/tas/001"],
            ),
        ]
        for case_id, groups, expected in cases:
            with subtests.test(case=case_id):
                monkeypatch.setattr(
                    ArtifactCache, "list_groups_on_branch", lambda self, store, g=groups: g
                )
                result = bound_cache.list_intermediate_groups()
                assert sorted({g for gs in result.values() for g in gs}) == expected
