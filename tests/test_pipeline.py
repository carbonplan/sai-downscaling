"""Unit tests for DownscalingPipeline with compute and storage mocked out."""

from __future__ import annotations

import logging
import pickle
from contextlib import contextmanager, suppress
from unittest.mock import MagicMock, patch

import dask
import dask.array as dask_array
import dask.system
import numpy as np
import pandas as pd
import pytest
import scipy.stats
import xarray as xr
from conftest import make_icechunk_group as _make_icechunk_group
from ibicus.debias import QuantileDeltaMapping, QuantileMapping
from ibicus.utils import PrecipitationHurdleModelGamma

from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions
from saidownscale.encoding import (
    CHUNK_LAT,
    CHUNK_LON,
    CHUNK_TIME,
    CHUNK_TIME_COARSE,
    COMPRESSOR,
    SHARD_LAT,
    SHARD_LAT_COARSE,
    SHARD_LON,
    SHARD_LON_COARSE,
    SHARD_TIME,
    SHARD_TIME_COARSE,
)
from saidownscale.pipeline import (
    NR_PROCESSES_ENV,
    DownscalingPipeline,
    _assert_stitched_continuity,
    _icechunk_storage_for_path,
    _location_seed,
    _make_debiaser,
    _SeededLocationMixin,
    _SeededQuantileDeltaMapping,
    _SeededQuantileMapping,
    _weibull_min_zero_bounded,
    _WeibullMinZeroBounded,
    calculate_out_of_range_mask,
    debiaser_processes,
    stitch_historical_scenario,
)


@contextmanager
def _mock_prepare_obs_compute():
    with (
        patch("saidownscale.pipeline.get_obs") as mock_get_obs,
        patch("saidownscale.pipeline.get_experiment") as mock_get_exp,
        patch("saidownscale.pipeline.interpolate_fine_to_coarse_grid") as mock_interp,
        patch("saidownscale.pipeline.subset_space") as mock_subset,
        patch("saidownscale.pipeline.rechunk") as mock_rechunk,
        patch.object(DownscalingPipeline, "_write_to_icechunk", return_value="snapshot-abc"),
    ):
        yield mock_get_obs, mock_get_exp, mock_interp, mock_subset, mock_rechunk


@contextmanager
def _mock_fit_historical_compute():
    with (
        patch("saidownscale.pipeline.get_obs"),
        patch("saidownscale.pipeline.get_historical_experiment"),
        patch("saidownscale.pipeline.get_experiment"),
        patch("saidownscale.pipeline.xr.DataArray", return_value=MagicMock()),
        patch("saidownscale.pipeline.rechunk"),
        patch("saidownscale.pipeline.downscale_from_coarse"),
        patch("saidownscale.pipeline.assert_no_nans"),
        patch("saidownscale.pipeline._SeededQuantileMapping") as mock_qm,
        patch("saidownscale.pipeline.dask"),
        patch.object(DownscalingPipeline, "_open_from_icechunk", return_value=MagicMock()),
        patch.object(DownscalingPipeline, "_write_to_icechunk", return_value="snapshot-abc"),
    ):
        mock_qm.from_variable.return_value.apply.return_value = MagicMock()
        yield


@contextmanager
def _mock_transform_scenario_compute():
    with (
        patch("saidownscale.pipeline.get_obs"),
        patch("saidownscale.pipeline.get_historical_experiment"),
        patch("saidownscale.pipeline.get_experiment"),
        patch("saidownscale.pipeline.xr.DataArray", return_value=MagicMock()),
        patch("saidownscale.pipeline.xr.concat", return_value=MagicMock()),
        patch("saidownscale.pipeline.rechunk"),
        patch("saidownscale.pipeline.subset_space"),
        patch("saidownscale.pipeline.calculate_baseline_climatology"),
        patch("saidownscale.pipeline.detrend"),
        patch("saidownscale.pipeline.retrend"),
        patch("saidownscale.pipeline.downscale_from_coarse"),
        patch("saidownscale.pipeline.assert_no_nans"),
        patch("saidownscale.pipeline._SeededQuantileMapping") as mock_qm,
        patch("saidownscale.pipeline.dask"),
        patch.object(DownscalingPipeline, "_open_from_icechunk", return_value=MagicMock()),
        patch.object(DownscalingPipeline, "_build_ocean_mask", return_value=MagicMock()),
        patch.object(
            DownscalingPipeline, "_apply_bias_correction_scenario", return_value=MagicMock()
        ),
        patch.object(DownscalingPipeline, "_write_to_icechunk", return_value="snapshot-abc"),
    ):
        mock_qm.from_variable.return_value.apply.return_value = MagicMock()
        yield


def _cfg(**overrides) -> DownscalingConfig:
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


def _options(root, **overrides) -> PipelineOptions:
    fields = {
        "scratch_dir": str(root / "cache"),
        "output_dir": str(root / "outputs"),
        "verbose": False,
        "rechunk_workflow": False,
    }
    return PipelineOptions(**(fields | overrides))


def _seed(p: DownscalingPipeline, *locs) -> None:
    for loc in locs:
        _make_icechunk_group(loc, branch=p.cache.branch)


def _record_writes(writes: dict):
    def capture(da, loc, **kwargs):
        writes[loc.group] = kwargs
        return "snapshot"

    return capture


@pytest.fixture
def pipeline_options(tmp_path) -> PipelineOptions:
    return _options(tmp_path)


@pytest.fixture
def pr_config() -> DownscalingConfig:
    return _cfg(variable="pr")


@pytest.fixture
def pipeline(pipeline_options) -> DownscalingPipeline:
    return DownscalingPipeline(_cfg(), pipeline_options)


@pytest.fixture
def pipeline_pr(pr_config, pipeline_options) -> DownscalingPipeline:
    return DownscalingPipeline(pr_config, pipeline_options)


@pytest.fixture
def all_deps_present(pipeline) -> DownscalingPipeline:
    _seed(pipeline, pipeline.cache.obs_loc, pipeline.cache.historical_loc(pipeline._hist_member))
    return pipeline


@pytest.fixture
def dtr_pipeline(pipeline_options) -> DownscalingPipeline:
    return DownscalingPipeline(_cfg(variable="dtr", ensemble_member="008"), pipeline_options)


@pytest.fixture
def tasmin_config() -> DownscalingConfig:
    return _cfg(variable="tasmin")


@pytest.fixture
def tasmin_pipeline(tasmin_config, pipeline_options) -> DownscalingPipeline:
    return DownscalingPipeline(tasmin_config, pipeline_options)


def test_cache_is_built_from_options(pipeline):
    cache, options = pipeline.cache, pipeline.options
    assert options.scratch_dir.rstrip("/") in cache.scratch_dir
    assert options.output_dir.rstrip("/") in cache.output_dir
    assert (cache.environment, cache.branch) == (options.environment, options.branch)
    assert pipeline._state == {}


def test_prepare_observations_computes_then_hits_cache(pipeline):
    obs_loc = pipeline.cache.obs_loc
    with _mock_prepare_obs_compute() as (get_obs, get_exp, interp, subset, rechunk):
        assert pipeline.prepare_observations() == obs_loc.store_path
    get_obs.assert_called_once_with(var="tas", dataset_name="ERA5")
    get_exp.assert_called_once_with(gcm="CESM2-WACCM6", scenario="historical", var="tas")
    interp.assert_called_once()
    subset.assert_not_called()
    rechunk.assert_not_called()

    _seed(pipeline, obs_loc)
    with _mock_prepare_obs_compute() as (get_obs, get_exp, interp, *_):
        assert pipeline.prepare_observations() == obs_loc.store_path
    get_obs.assert_not_called()
    get_exp.assert_not_called()
    interp.assert_not_called()

    with _mock_prepare_obs_compute() as (get_obs, *_):
        pipeline.prepare_observations(force=True)
    get_obs.assert_called_once()


def test_prepare_observations_subsets_and_rechunks_when_configured(tmp_path):
    p = DownscalingPipeline(
        _cfg(subset_bounds=(-35.0, -22.0, 16.0, 33.0)), _options(tmp_path, rechunk_workflow=True)
    )
    with _mock_prepare_obs_compute() as (_, _, _, subset, rechunk):
        p.prepare_observations()
    assert subset.call_count == 2
    rechunk.assert_called_once()


def test_build_ocean_mask_uses_catalog_and_descending_lat():
    da = xr.DataArray(
        np.zeros((3, 4, 8)),
        dims=["time", "lat", "lon"],
        coords={"time": range(3), "lat": [-30.0, 0.0, 30.0, 60.0], "lon": list(range(8))},
    )
    captured = {}

    def capture_template(template, *args, **kwargs):
        captured["lat"] = template.coords["lat"].values.tolist()
        return MagicMock()

    with (
        patch("saidownscale.datasets.catalog") as mock_catalog,
        patch.dict("sys.modules", {"xproj": MagicMock()}),
        patch("rasterix.rasterize.geometry_mask", side_effect=capture_template),
    ):
        DownscalingPipeline._build_ocean_mask(da)
    mock_catalog.get.assert_called_once_with("ocean-mask")
    assert captured["lat"] == [60.0, 30.0, 0.0, -30.0]


def test_missing_dependencies_raise_on_the_plain_path(pipeline, subtests):
    for stage in ("fit_historical", "transform_scenario"):
        with subtests.test(stage=stage), patch.object(pipeline, f"{stage}_tasmin") as tasmin:
            with pytest.raises(ValueError, match="Missing dependencies"):
                getattr(pipeline, stage)()
            tasmin.assert_not_called()
    _seed(pipeline, pipeline.cache.obs_loc)
    with subtests.test(stage="transform_scenario with obs only"):
        with pytest.raises(ValueError, match="Missing dependencies"):
            pipeline.transform_scenario()


def test_stages_validate_dependencies(pipeline, subtests):
    cases = {
        "fit_historical": {},
        "transform_scenario": {"hist_member": pipeline._hist_member},
    }
    for stage, kwargs in cases.items():
        with subtests.test(stage=stage):
            with patch.object(pipeline.cache, "validate_dependencies") as validate:
                with suppress(Exception):
                    getattr(pipeline, stage)()
            validate.assert_called_once_with(stage, pipeline.config, **kwargs)


def test_fit_historical_cache_hit_needs_fine_and_coarse(all_deps_present):
    p = all_deps_present
    fine = p.cache.historical_loc(p._hist_member)
    with (
        _mock_fit_historical_compute(),
        patch.object(DownscalingPipeline, "_write_to_icechunk", return_value="snap") as write,
    ):
        assert p.fit_historical() == fine.store_path
    write.assert_called()

    _seed(p, p.cache.debiased_coarse_historical_loc(p._hist_member))
    with patch("saidownscale.pipeline.get_obs") as get_obs:
        assert p.fit_historical() == fine.store_path
    get_obs.assert_not_called()

    with _mock_fit_historical_compute(), patch("saidownscale.pipeline.get_obs") as get_obs:
        p.fit_historical(force=True)
    get_obs.assert_called_once()


def test_transform_scenario_cache_hit_needs_fine_and_coarse(pipeline_pr):
    p = pipeline_pr
    fine = p.cache.scenario_loc
    _seed(p, p.cache.obs_loc, p.cache.historical_loc(p._hist_member), fine)
    with (
        _mock_transform_scenario_compute(),
        patch.object(DownscalingPipeline, "_write_to_icechunk", return_value="snap") as write,
    ):
        assert p.transform_scenario() == fine.store_path
    write.assert_called()

    _seed(p, p.cache.debiased_coarse_scenario_loc())
    with patch("saidownscale.pipeline.get_obs") as get_obs:
        assert p.transform_scenario() == fine.store_path
    get_obs.assert_not_called()

    with _mock_transform_scenario_compute(), patch("saidownscale.pipeline.get_obs") as get_obs:
        p.transform_scenario(force=True)
    get_obs.assert_called_once()


def test_transform_scenario_requires_a_scenario(tmp_path):
    p = DownscalingPipeline(
        DownscalingConfig(
            gcm="CESM2-WACCM6",
            downscaling_method="BCSD",
            variable="tas",
            ensemble_member="r1i1p1f1",
        ),
        _options(tmp_path),
    )
    with pytest.raises(ValueError, match="scenario must be specified"):
        p.transform_scenario()


def test_detrend_runs_only_for_detrended_variables(all_deps_present, pipeline_pr, subtests):
    _seed(pipeline_pr, pipeline_pr.cache.obs_loc)
    _seed(pipeline_pr, pipeline_pr.cache.historical_loc(pipeline_pr._hist_member))
    with subtests.test(variable="pr"):
        with (
            _mock_transform_scenario_compute(),
            patch("saidownscale.pipeline.detrend") as detrend,
        ):
            pipeline_pr.transform_scenario()
        detrend.assert_not_called()
    with subtests.test(variable="tas"):
        with (
            _mock_transform_scenario_compute(),
            patch("saidownscale.pipeline.detrend", return_value=(MagicMock(), None)) as detrend,
            patch("saidownscale.pipeline.stitch_historical_scenario"),
            patch("saidownscale.pipeline.xr"),
            suppress(Exception),
        ):
            all_deps_present.transform_scenario()
        detrend.assert_called()


def test_ocean_mask_follows_the_option(pr_config, tmp_path, subtests):
    for apply in (True, False):
        with subtests.test(apply_ocean_mask=apply):
            p = DownscalingPipeline(
                pr_config, _options(tmp_path / str(apply), apply_ocean_mask=apply)
            )
            _seed(p, p.cache.obs_loc, p.cache.historical_loc(p._hist_member))
            with (
                _mock_transform_scenario_compute(),
                patch.object(
                    DownscalingPipeline, "_build_ocean_mask", return_value=MagicMock()
                ) as mask,
            ):
                p.transform_scenario()
            assert mask.called is apply


def _assert_coarse_encoding(kwargs: dict, variable: str) -> None:
    entry = kwargs["encoding"][variable]
    assert entry["chunks"][0] == CHUNK_TIME_COARSE
    assert entry["shards"][0] == SHARD_TIME_COARSE


def test_transform_scenario_writes_fine_and_coarse(pipeline_pr):
    p = pipeline_pr
    _seed(p, p.cache.obs_loc, p.cache.historical_loc(p._hist_member))
    writes: dict = {}
    with (
        _mock_transform_scenario_compute(),
        patch.object(DownscalingPipeline, "_write_to_icechunk", side_effect=_record_writes(writes)),
    ):
        p.transform_scenario()

    fine = writes[p.cache.scenario_loc.group]["encoding"]["pr"]
    assert fine["chunks"] == (CHUNK_TIME, CHUNK_LAT, CHUNK_LON)
    assert fine["shards"] == (SHARD_TIME, SHARD_LAT, SHARD_LON)
    assert fine["compressors"] == [COMPRESSOR]
    _assert_coarse_encoding(writes[p.cache.debiased_coarse_scenario_loc().group], "pr")


def test_fit_historical_writes_fine_and_coarse(all_deps_present):
    p = all_deps_present
    writes: dict = {}
    with (
        _mock_fit_historical_compute(),
        patch.object(DownscalingPipeline, "_write_to_icechunk", side_effect=_record_writes(writes)),
    ):
        p.fit_historical()

    assert p.cache.historical_loc(p._hist_member).group in writes
    coarse = writes[p.cache.debiased_coarse_historical_loc(p._hist_member).group]
    _assert_coarse_encoding(coarse, "tas")


def test_run_full_pipeline_runs_stages_in_order(pipeline, subtests):
    for kwargs in ({}, {"force": True}):
        with subtests.test(**kwargs):
            calls = []

            def stage(name):
                def run(*, force):
                    calls.append((name, force))
                    return f"{name}_path"

                return run

            with (
                patch.object(pipeline, "prepare_observations", side_effect=stage("obs")),
                patch.object(pipeline, "fit_historical", side_effect=stage("hist")),
                patch.object(pipeline, "transform_scenario", side_effect=stage("scenario")),
            ):
                result = pipeline.run_full_pipeline(**kwargs)
            force = kwargs.get("force", False)
            assert calls == [("obs", force), ("hist", force), ("scenario", force)]
            assert result == "scenario_path"


def test_tasmin_stages_dispatch_to_derive_variants(tasmin_pipeline, subtests):
    """batch_runner calls the stage methods directly, so tasmin dispatches there (#363/#331)."""
    p = tasmin_pipeline
    for stage in ("fit_historical", "transform_scenario"):
        with (
            subtests.test(stage=stage),
            patch.object(p, f"{stage}_tasmin", return_value="tasmin_path") as variant,
        ):
            assert getattr(p, stage)(force=True) == "tasmin_path"
            variant.assert_called_once_with(force=True)
    with subtests.test(stage="run_full_pipeline"):
        with (
            patch.object(p, "prepare_observations", return_value="obs"),
            patch.object(p, "fit_historical_tasmin", return_value="hist") as hist,
            patch.object(p, "transform_scenario_tasmin", return_value="scen") as scen,
        ):
            assert p.run_full_pipeline() == "scen"
        hist.assert_called_once_with(force=False)
        scen.assert_called_once_with(force=False)


_TASMIN_STAGES = {
    "fit_historical_tasmin": ("_load_gcm_obs", 3),
    "transform_scenario_tasmin": ("_load_scenario_data", 5),
}


def _tasmin_locs(p: DownscalingPipeline, stage: str):
    cache = p.cache
    if stage == "fit_historical_tasmin":
        return (
            cache.historical_loc(p._hist_member),
            lambda variable=None: cache.debiased_coarse_historical_loc(
                p._hist_member, **({"variable": variable} if variable else {})
            ),
        )
    return (
        cache.scenario_loc,
        lambda variable=None: cache.debiased_coarse_scenario_loc(
            **({"variable": variable} if variable else {})
        ),
    )


def _seed_tasmin_inputs(p: DownscalingPipeline, stage: str) -> None:
    cache = p.cache
    _, coarse = _tasmin_locs(p, stage)
    _seed(p, cache.obs_loc, coarse("dtr"), coarse("tasmax"))
    if stage == "fit_historical_tasmin":
        _seed(p, cache.historical_loc(p._hist_member, variable="tasmax"))
    else:
        _seed(p, cache.historical_loc(p._hist_member), cache.scenario_output_loc(variable="tasmax"))


def _patch_tasmin_load(stage: str, values=None):
    attr, n = _TASMIN_STAGES[stage]
    values = values if values is not None else tuple(MagicMock() for _ in range(n))
    return patch.object(DownscalingPipeline, attr, return_value=values)


def test_tasmin_disagg_input_is_eager(tasmin_pipeline, subtests):
    """A lazy derived tasmin defers the interp into an all-to-all rechunk that hangs at scale."""
    p = tasmin_pipeline
    coords = {
        "time": xr.date_range("1978-01-01", periods=40, freq="D", use_cftime=True),
        "lat": [0.0, 1.0],
        "lon": [0.0, 1.0],
    }
    ds = xr.Dataset(
        {
            name: xr.DataArray(
                dask_array.full((40, 2, 2), val, chunks=(40, 2, 2), dtype="float32"),
                dims=("time", "lat", "lon"),
                coords=coords,
            )
            for name, val in (("tasmax", 300.0), ("dtr", 10.0))
        }
    )
    obs = ds["tasmax"]
    loads = {
        "fit_historical_tasmin": (obs, obs, obs),
        "transform_scenario_tasmin": (obs, obs, obs, obs, None),
    }
    for stage, values in loads.items():
        with subtests.test(stage=stage):
            captured = {}
            with (
                patch.object(p.cache, "exists", return_value=False),
                patch.object(p.cache, "validate_dependencies"),
                _patch_tasmin_load(stage, values),
                patch.object(p, "_open_from_icechunk", return_value=ds),
                patch.object(p, "_write_to_icechunk", return_value="snap"),
                patch.object(p, "reconcile_temperature_extremes"),
                patch.object(p, "_build_output_attrs", return_value={}),
                patch.object(
                    p,
                    "_apply_spatial_downscaling",
                    side_effect=lambda da, *_: captured.setdefault("input", da),
                ),
            ):
                getattr(p, stage)(force=True)
            assert captured["input"].chunks is None


def _make_daily_da(start: str, end: str, value: float = 1.0) -> xr.DataArray:
    times = pd.date_range(start, end, freq="D")
    return xr.DataArray(np.full(len(times), value), coords={"time": times}, dims=["time"])


def test_stitch_historical_scenario_is_continuous(subtests):
    hist = _make_daily_da("1978-01-01", "2014-12-31")
    ssp = _make_daily_da("2015-01-01", "2034-12-31")
    sai = _make_daily_da("2035-01-01", "2084-12-31")
    cases = {
        "sai": {"model_scenario": sai, "predict_period_start": 2035, "ssp_timeseries": ssp},
        "sai, predict start before sai data (regression)": {
            "model_scenario": sai,
            "predict_period_start": 2015,
            "ssp_timeseries": ssp,
        },
        "non-sai": {
            "model_scenario": _make_daily_da("2015-01-01", "2084-12-31"),
            "predict_period_start": 2015,
        },
    }
    for name, kwargs in cases.items():
        with subtests.test(case=name):
            result = stitch_historical_scenario(model_hist=hist, train_period_end=2014, **kwargs)
            years, counts = np.unique(result["time.year"].values, return_counts=True)
            assert years.tolist() == list(range(1978, 2085))
            assert counts.max() <= 366


def test_stitch_historical_scenario_rejects_empty_scenario():
    with pytest.raises(ValueError, match="no timesteps"):
        stitch_historical_scenario(
            model_hist=_make_daily_da("1978-01-01", "2014-12-31"),
            model_scenario=_make_daily_da("2035-01-01", "2034-12-31"),
            train_period_end=2014,
            predict_period_start=2015,
            ssp_timeseries=_make_daily_da("2015-01-01", "2034-12-31"),
        )


def test_assert_stitched_continuity(subtests):
    missing_day = pd.date_range("2014-01-01", "2014-12-31", freq="D").delete(364)
    accepted = {
        "clean series": _make_daily_da("2000-01-01", "2002-12-31"),
        "one missing day (UKESM1-1-LL quirk)": xr.DataArray(
            np.ones(len(missing_day)), coords={"time": missing_day}, dims=["time"]
        ),
    }
    rejected = {
        "duplicate timestamp": ("2001-06-01", "2002-12-31"),
        "year-level gap": ("2003-01-01", "2004-12-31"),
    }
    for name, da in accepted.items():
        with subtests.test(case=name):
            _assert_stitched_continuity(da)
    for match, (start, end) in rejected.items():
        with subtests.test(case=match), pytest.raises(ValueError, match=match):
            _assert_stitched_continuity(
                xr.concat(
                    [_make_daily_da("2000-01-01", "2001-12-31"), _make_daily_da(start, end)],
                    dim="time",
                )
            )


def test_calculate_out_of_range_mask(subtests):
    model_hist = _make_daily_da("1980-01-01", "1980-12-31", 10.0)
    for value, expect_low, expect_high in (
        (10.0, False, False),
        (20.0, False, True),
        (0.0, True, False),
    ):
        with subtests.test(scenario_value=value):
            result, low, high = calculate_out_of_range_mask(
                model_hist=model_hist,
                scenario_detrended=_make_daily_da("2050-01-01", "2050-12-31", value),
                center_window=31,
            )
            assert bool(result.all()) is (expect_low or expect_high)
            assert bool(result.any()) is (expect_low or expect_high)
            assert bool(low.all()) is expect_low and bool(low.any()) is expect_low
            assert bool(high.all()) is expect_high and bool(high.any()) is expect_high


def test_make_debiaser_forwards_mapping_and_distribution(subtests):
    cases = {
        "tas parametric defaults to norm": ("tas", {}, "parametric", scipy.stats.norm),
        "tas nonparametric": ("tas", {}, "nonparametric", None),
        "pr low tail": (
            "pr",
            {"distribution": _weibull_min_zero_bounded},
            "parametric",
            _weibull_min_zero_bounded,
        ),
        "pr high tail": (
            "pr",
            {"distribution": scipy.stats.gumbel_r},
            "parametric",
            scipy.stats.gumbel_r,
        ),
    }
    for name, (variable, extra, mapping_type, distribution) in cases.items():
        with subtests.test(case=name):
            with patch("saidownscale.pipeline._SeededQuantileMapping") as mock_qm:
                _make_debiaser(variable=variable, mapping_type=mapping_type, **extra)
            kwargs = mock_qm.call_args.kwargs
            assert kwargs["mapping_type"] == mapping_type
            if distribution is not None:
                assert kwargs["distribution"] is distribution


def test_weibull_zero_bounded_pins_loc(subtests):
    """An unpinned 3-parameter Weibull fit drifts loc and yields values like DTR = -300 K."""
    data = scipy.stats.weibull_min.rvs(
        1.5, loc=5, scale=2, size=2000, random_state=np.random.default_rng(0)
    )
    restored = pickle.loads(pickle.dumps(_weibull_min_zero_bounded))
    cases = {
        "override": (_weibull_min_zero_bounded, {}, 0),
        "after pickle round trip": (restored, {}, 0),
        "explicit floc wins": (_weibull_min_zero_bounded, {"floc": 3.0}, 3.0),
    }
    for name, (dist, kwargs, expected) in cases.items():
        with subtests.test(case=name):
            assert dist.fit(data, **kwargs)[1] == expected
    with subtests.test(case="unconstrained fit drifts"):
        assert scipy.stats.weibull_min.fit(data)[1] != 0


def test_pipeline_wires_zero_bounded_weibull_to_low_tail(pipeline_options):
    """pr stands in for rsds, whose BCSD entry is nonparametric since #523."""
    cfg = _cfg(variable="pr")
    assert cfg.variable_config.debias_approach == "nonparametric_hybrid_2sided"
    pipeline = DownscalingPipeline(cfg, pipeline_options)
    coords = {
        "time": pd.date_range("2015-01-01", periods=6),
        "lat": [10.0, 20.0],
        "lon": [0.0, 1.0, 2.0],
    }
    dims = ["time", "lat", "lon"]
    da = xr.DataArray(
        np.abs(np.random.default_rng(1).normal(5, 2, (6, 2, 3))), coords=coords, dims=dims
    )
    mask = xr.DataArray(np.zeros((6, 2, 3), dtype=bool), coords=coords, dims=dims)

    make_debiaser_spy = MagicMock()
    make_debiaser_spy.return_value.apply.return_value = np.zeros((6, 2, 3))
    with (
        patch("saidownscale.pipeline._make_debiaser", make_debiaser_spy),
        patch("saidownscale.pipeline.calculate_out_of_range_mask", return_value=(mask, mask, mask)),
    ):
        pipeline._apply_bias_correction_scenario(da, da, da)

    distributions = [call.kwargs.get("distribution") for call in make_debiaser_spy.call_args_list]
    assert _weibull_min_zero_bounded in distributions
    assert scipy.stats.gumbel_r in distributions


def test_floc_zero_reaches_scipy_through_ibicus():
    weibull_gen = _WeibullMinZeroBounded.__mro__[1]
    real_parent_fit = weibull_gen.fit
    floc_seen: list = []

    def spy_fit(self, data, *args, **kwargs):
        floc_seen.append(kwargs.get("floc"))
        return real_parent_fit(self, data, *args, **kwargs)

    rng = np.random.default_rng(0)
    obs, cm_hist, cm_future = (np.abs(rng.normal(m, 2, (500, 1, 1))) for m in (5, 6, 6))
    quantile_mapping = QuantileMapping(
        distribution=_weibull_min_zero_bounded,
        mapping_type="parametric",
        variable="rsds",
        detrending="no_detrending",
        running_window_mode=False,
    )
    with patch.object(weibull_gen, "fit", spy_fit):
        debiased = quantile_mapping.apply(
            obs, cm_hist, cm_future, progressbar=False, parallel=False
        )

    assert floc_seen
    assert all(floc == 0 for floc in floc_seen)
    assert np.isfinite(debiased).all()


def test_dtr_fit_historical_is_coarse_only(dtr_pipeline):
    """dtr is never published fine (#461), and a stale pre-#461 fine group is not a cache hit."""
    p = dtr_pipeline
    coarse = p.cache.debiased_coarse_historical_loc(p._hist_member)
    _seed(p, p.cache.obs_loc, p.cache.historical_loc(p._hist_member))
    writes: dict = {}
    with (
        _mock_fit_historical_compute(),
        patch.object(DownscalingPipeline, "_write_to_icechunk", side_effect=_record_writes(writes)),
        patch.object(DownscalingPipeline, "_apply_spatial_downscaling") as disagg,
    ):
        result = p.fit_historical()
    assert list(writes) == [coarse.group]
    disagg.assert_not_called()
    assert result == coarse.store_path

    _seed(p, coarse)
    with patch("saidownscale.pipeline.get_obs") as get_obs:
        p.fit_historical()
    get_obs.assert_not_called()


def test_dtr_transform_scenario_is_coarse_only(dtr_pipeline):
    """dtr is never published fine (#461)."""
    p = dtr_pipeline
    coarse = p.cache.debiased_coarse_scenario_loc()
    _seed(p, p.cache.obs_loc, p.cache.debiased_coarse_historical_loc(p._hist_member))
    writes: dict = {}
    with (
        _mock_transform_scenario_compute(),
        patch.object(DownscalingPipeline, "_write_to_icechunk", side_effect=_record_writes(writes)),
        patch.object(DownscalingPipeline, "_apply_spatial_downscaling") as disagg,
    ):
        result = p.transform_scenario()
    assert coarse.group in writes
    assert p.cache.scenario_loc.group not in writes
    disagg.assert_not_called()
    assert result == coarse.store_path

    _seed(p, coarse)
    with patch("saidownscale.pipeline.get_obs") as get_obs:
        p.transform_scenario()
    get_obs.assert_not_called()


def test_tasmin_stages_read_coarse_dtr_and_tasmax(tasmin_pipeline, subtests):
    """Inputs come from the output store's coarse groups (#363); fine tasmin alone is no hit."""
    p = tasmin_pipeline
    for stage in _TASMIN_STAGES:
        with subtests.test(stage=stage):
            _seed_tasmin_inputs(p, stage)
            fine, coarse = _tasmin_locs(p, stage)
            _seed(p, fine)
            opened: list = []

            def capture_open(loc):
                opened.append(loc.group)
                return MagicMock()

            mock_compute = (
                _mock_fit_historical_compute
                if stage == "fit_historical_tasmin"
                else _mock_transform_scenario_compute
            )
            with (
                mock_compute(),
                patch.object(DownscalingPipeline, "_open_from_icechunk", side_effect=capture_open),
                patch.object(DownscalingPipeline, "_write_to_icechunk", return_value="s") as write,
                suppress(Exception),
            ):
                getattr(p, stage)()
            assert coarse("dtr").group in opened
            assert coarse("tasmax").group in opened
            assert p.cache.output_dir in coarse().store_path
            write.assert_called()


def _read_coarse_tasmin(cache, coarse_loc) -> xr.DataArray:
    import icechunk

    repo = icechunk.Repository.open(_icechunk_storage_for_path(coarse_loc.store_path))
    session = repo.readonly_session(branch=cache.branch)
    ds = xr.open_dataset(session.store, engine="zarr", consolidated=False, group=coarse_loc.group)
    return ds["tasmin"]


def _coarse_pair(n_time: int, n_lat: int, n_lon: int, chunks: tuple) -> xr.Dataset:
    coords = {
        "time": pd.date_range("2015-01-01", periods=n_time, freq="D"),
        "lat": np.linspace(-88.0, 88.0, n_lat),
        "lon": np.linspace(0.0, 356.0, n_lon),
    }
    return xr.Dataset(
        {
            name: xr.DataArray(
                dask_array.full((n_time, n_lat, n_lon), val, dtype="float32", chunks=chunks),
                dims=["time", "lat", "lon"],
                coords=coords,
            )
            for name, val in (("tasmax", 300.0), ("dtr", 10.0))
        }
    )


def test_tasmin_coarse_write_survives_misaligned_lat_chunks(tasmin_pipeline, subtests):
    """Production regression CESM2-WACCM6_tasmin_003_G6-1.5K: safe_chunks rejected the write."""
    p = tasmin_pipeline
    fine_tasmin = xr.DataArray(
        np.full((1, 1, 2), 290.0, dtype="float32"),
        dims=["time", "lat", "lon"],
        coords={"time": [np.datetime64("2020-01-01")], "lat": [0.0], "lon": [10.0, 11.0]},
        name="tasmin",
    )
    for stage in _TASMIN_STAGES:
        with subtests.test(stage=stage):
            _seed_tasmin_inputs(p, stage)
            fine, coarse = _tasmin_locs(p, stage)
            with (
                _patch_tasmin_load(stage),
                patch.object(
                    DownscalingPipeline,
                    "_open_from_icechunk",
                    return_value=_coarse_pair(40, 48, 96, (40, 16, 96)),
                ),
                patch.object(
                    DownscalingPipeline, "_apply_spatial_downscaling", return_value=fine_tasmin
                ),
                patch.object(DownscalingPipeline, "_build_ocean_mask", return_value=MagicMock()),
                patch.object(DownscalingPipeline, "reconcile_temperature_extremes"),
            ):
                assert getattr(p, stage)() == fine.store_path
            written = _read_coarse_tasmin(p.cache, coarse())
            assert float(written.isel(time=0, lat=0, lon=0)) == 290.0


def test_tasmin_coarse_write_tiles_time_shards(tasmin_pipeline, subtests):
    """Sub-shard time chunks straddling SHARD_TIME_COARSE failed safe_chunks on axis 0."""
    p = tasmin_pipeline
    for stage in _TASMIN_STAGES:
        with subtests.test(stage=stage):
            _seed_tasmin_inputs(p, stage)
            _, coarse = _tasmin_locs(p, stage)
            captured: dict = {}

            def capture(da, loc, *args, **kwargs):
                if loc.group == coarse().group:
                    captured["da"] = da
                return "snap"

            with (
                _patch_tasmin_load(stage),
                patch.object(
                    DownscalingPipeline,
                    "_open_from_icechunk",
                    return_value=_coarse_pair(SHARD_TIME_COARSE + 1, 24, 48, (600, 24, 48)),
                ),
                patch.object(
                    DownscalingPipeline, "_apply_spatial_downscaling", return_value=MagicMock()
                ),
                patch.object(DownscalingPipeline, "_build_ocean_mask", return_value=MagicMock()),
                patch.object(DownscalingPipeline, "reconcile_temperature_extremes"),
                patch.object(DownscalingPipeline, "_write_to_icechunk", side_effect=capture),
            ):
                getattr(p, stage)()
            for dim, shard in (
                ("time", SHARD_TIME_COARSE),
                ("lat", SHARD_LAT_COARSE),
                ("lon", SHARD_LON_COARSE),
            ):
                chunks = captured["da"].chunksizes[dim]
                assert all(c == shard for c in chunks[:-1]), (dim, chunks, shard)
                assert chunks[-1] <= shard, (dim, chunks, shard)


def test_tasmin_persists_raw_then_reconciles_store_to_store(tasmin_pipeline, subtests):
    """OOM fix: tasmax and the fresh interp graph must never co-reside in memory."""
    p = tasmin_pipeline
    for stage in _TASMIN_STAGES:
        with subtests.test(stage=stage):
            _seed_tasmin_inputs(p, stage)
            fine, _ = _tasmin_locs(p, stage)
            sentinel = MagicMock(name="raw_tasmin")
            writes: list = []
            reconcile: dict = {}

            def capture_write(da, loc, **kwargs):
                writes.append((loc.group, da))
                return "snap"

            def capture_reconcile(tasmin_loc, tasmax_loc, *, tasmin_fine="_unset", force=False):
                reconcile.update(tasmin_fine=tasmin_fine, group=tasmin_loc.group)

            with (
                _patch_tasmin_load(stage),
                patch.object(DownscalingPipeline, "_open_from_icechunk", return_value=MagicMock()),
                patch("saidownscale.pipeline.derive_tasmin", return_value=MagicMock()),
                patch.object(
                    DownscalingPipeline, "_apply_spatial_downscaling", return_value=sentinel
                ),
                patch.object(DownscalingPipeline, "_write_to_icechunk", side_effect=capture_write),
                patch.object(
                    DownscalingPipeline,
                    "reconcile_temperature_extremes",
                    side_effect=capture_reconcile,
                ),
            ):
                getattr(p, stage)()
            assert any(g == fine.group and da is sentinel for g, da in writes)
            assert reconcile == {"tasmin_fine": None, "group": fine.group}


def test_cached_tasmin_served_without_upstream_deps(tasmin_pipeline):
    """A cached tasmin must not require the possibly reaped upstreams (#363 finding #5)."""
    p = tasmin_pipeline
    _seed(p, p.cache.scenario_loc, p.cache.debiased_coarse_scenario_loc())
    with patch.object(DownscalingPipeline, "_write_to_icechunk") as write:
        assert p.transform_scenario_tasmin() == p.cache.scenario_loc.store_path
    write.assert_not_called()


def _fine_pair_with_inversion():
    coords = {"time": [np.datetime64("2020-01-01")], "lat": [0.0], "lon": [10.0, 11.0]}
    dims = ["time", "lat", "lon"]
    tasmax = xr.DataArray([[[300.0, 290.0]]], dims=dims, coords=coords, name="tasmax")
    tasmin = xr.DataArray([[[280.0, 295.0]]], dims=dims, coords=coords, name="tasmin")
    return tasmax, tasmin


def _reconcile(p: DownscalingPipeline, *, force: bool = False):
    tasmax_fine, tasmin_fine = _fine_pair_with_inversion()
    tasmin_loc = p.cache.scenario_loc
    tasmax_loc = p.cache.scenario_output_loc(variable="tasmax")
    _seed(p, tasmax_loc)
    writes: dict = {}

    def capture(da, loc, **kwargs):
        writes[loc.group] = (da, kwargs, dask.config.get("scheduler", None))
        return "snap"

    with (
        patch.object(
            DownscalingPipeline, "_open_from_icechunk", return_value=tasmax_fine.to_dataset()
        ),
        patch.object(DownscalingPipeline, "_write_to_icechunk", side_effect=capture),
    ):
        p.reconcile_temperature_extremes(
            tasmin_loc, tasmax_loc, tasmin_fine=tasmin_fine, force=force
        )
    return writes[tasmax_loc.group], writes[tasmin_loc.group]


def test_reconcile_swaps_inversions_and_writes_both_fields(tasmin_pipeline):
    """Only the tasmin write, made last, may force GC; it lazily reads the old tasmax (#331)."""
    (tasmax, tasmax_kw, tasmax_sched), (tasmin, tasmin_kw, tasmin_sched) = _reconcile(
        tasmin_pipeline, force=True
    )
    assert tasmax.values[0, 0, 1] == 295.0
    assert tasmin.values[0, 0, 1] == 290.0
    assert bool((tasmax >= tasmin).all())
    assert tasmax_kw.get("force") is False
    assert tasmin_kw.get("force") is True
    for da in (tasmax, tasmin):
        assert da.chunks is not None
        assert all(len(c) == 1 for c in da.chunks)
    assert tasmax_sched == tasmin_sched == "synchronous"


def test_reconcile_logs_swap_count_only_in_qa(tasmin_config, tmp_path, caplog, subtests):
    for environment in ("qa", "production"):
        with subtests.test(environment=environment):
            p = DownscalingPipeline(
                tasmin_config, _options(tmp_path / environment, environment=environment)
            )
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="saidownscale.pipeline"):
                _reconcile(p)
            if environment == "qa":
                assert "swapped 1 / 2 valid cells" in caplog.text
                assert "50.0000%" in caplog.text
                assert "max inversion 5.000" in caplog.text
            else:
                assert "swapped" not in caplog.text


def test_reconcile_raises_on_missing_input(tasmin_pipeline, subtests):
    p = tasmin_pipeline
    tasmin_loc = p.cache.scenario_loc
    tasmax_loc = p.cache.scenario_output_loc(variable="tasmax")
    _, tasmin_fine = _fine_pair_with_inversion()
    with subtests.test(missing="tasmax"), pytest.raises(ValueError, match="tasmax"):
        p.reconcile_temperature_extremes(tasmin_loc, tasmax_loc, tasmin_fine=tasmin_fine)
    _seed(p, tasmax_loc)
    with subtests.test(missing="tasmin"), pytest.raises(ValueError, match="tasmin"):
        p.reconcile_temperature_extremes(tasmin_loc, tasmax_loc)


def test_reconcile_store_to_store_reads_shard_aligned(tasmin_pipeline):
    p = tasmin_pipeline
    tasmin_loc = p.cache.scenario_loc
    tasmax_loc = p.cache.scenario_output_loc(variable="tasmax")
    _seed(p, tasmax_loc, tasmin_loc)
    tasmax_fine, tasmin_fine = _fine_pair_with_inversion()
    chunks_seen: list = []
    writes: dict = {}

    def fake_open(loc, chunks="auto"):
        chunks_seen.append(chunks)
        return (tasmin_fine if loc.group == tasmin_loc.group else tasmax_fine).to_dataset()

    def capture(da, loc, **kwargs):
        writes[loc.group] = da
        return "snap"

    with (
        patch.object(DownscalingPipeline, "_open_from_icechunk", side_effect=fake_open),
        patch.object(DownscalingPipeline, "_write_to_icechunk", side_effect=capture),
    ):
        p.reconcile_temperature_extremes(tasmin_loc, tasmax_loc)

    assert writes[tasmax_loc.group].values[0, 0, 1] == 295.0
    assert writes[tasmin_loc.group].values[0, 0, 1] == 290.0
    shard = {"time": SHARD_TIME, "lat": SHARD_LAT, "lon": SHARD_LON}
    assert chunks_seen
    assert all(c == shard for c in chunks_seen)


def test_detrend_off_still_bridges_sai_scenarios(pipeline_options, subtests):
    """dtr and pr skip detrending, but an SAI scenario still needs the SSP245 bridge (#363)."""
    with subtests.test(scenario="G6-1.5K"):
        cfg = _cfg(
            variable="dtr", ensemble_member="003", scenario="G6-1.5K", predict_period_end=2084
        )
        assert cfg.variable_config.detrend_data is False and cfg.is_sai_scenario is True
        ssp = MagicMock(name="ssp_timeseries")
        stitched = MagicMock(name="stitched")
        stitched.sel.return_value = "bridged-predict-slice"
        with patch(
            "saidownscale.pipeline.stitch_historical_scenario", return_value=stitched
        ) as stitch:
            out, trend = DownscalingPipeline(cfg, pipeline_options)._detrend_scenario(
                MagicMock(), MagicMock(), ssp
            )
        stitch.assert_called_once()
        assert stitch.call_args.kwargs["ssp_timeseries"] is ssp
        assert (out, trend) == ("bridged-predict-slice", None)
    with subtests.test(scenario="SSP245"):
        cfg = _cfg(variable="pr", ensemble_member="003", predict_period_end=2099)
        assert cfg.variable_config.detrend_data is False and cfg.is_sai_scenario is False
        model_scenario = MagicMock(name="model_scenario")
        with patch("saidownscale.pipeline.stitch_historical_scenario") as stitch:
            out, trend = DownscalingPipeline(cfg, pipeline_options)._detrend_scenario(
                MagicMock(), model_scenario, None
            )
        stitch.assert_not_called()
        assert out is model_scenario
        assert trend is None


def _spatial_daily_da(start: str, end: str, value: float = 1.0) -> xr.DataArray:
    times = pd.date_range(start, end, freq="D")
    return xr.DataArray(
        np.full((len(times), 2, 2), value),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [0.0, 1.0], "lon": [10.0, 11.0]},
        name="tas",
    )


def test_scenario_model_hist_stops_at_train_period_end(pipeline_options, subtests):
    """#518: slicing through predict_period_start - 1 widened the training pool."""
    cases = {
        2014: _cfg(
            ensemble_member="001",
            scenario="G6-1.5K",
            train_period_end=2014,
            predict_period_start=2035,
            predict_period_end=2060,
        ),
        2008: _cfg(train_period_end=2008, predict_period_end=2060),
    }
    for train_end, cfg in cases.items():
        with subtests.test(train_period_end=train_end):
            p = DownscalingPipeline(cfg, pipeline_options)
            obs = _spatial_daily_da("1978-01-01", "2014-12-31")
            with (
                patch.object(
                    DownscalingPipeline, "_open_from_icechunk", return_value=obs.to_dataset()
                ),
                patch("saidownscale.pipeline.get_obs", return_value=obs),
                patch(
                    "saidownscale.pipeline.get_historical_experiment",
                    return_value=_spatial_daily_da("1978-01-01", "2015-01-16"),
                ),
                patch(
                    "saidownscale.pipeline.get_experiment",
                    return_value=_spatial_daily_da("2015-01-01", "2060-12-31").expand_dims(
                        ensemble_member=[cfg.ensemble_member]
                    ),
                ),
                patch.object(
                    DownscalingPipeline,
                    "_load_ssp245_bridge",
                    return_value=_spatial_daily_da("2015-01-01", "2060-12-31"),
                ),
                patch.object(
                    p.cache,
                    "check_dependencies",
                    return_value={"obs_regridded": (True, MagicMock())},
                ),
            ):
                _, _, scenario_hist, _, _ = p._load_scenario_data()
                _, _, historical_stage_hist = p._load_gcm_obs()
            assert int(scenario_hist["time"].dt.year.max()) == train_end
            assert scenario_hist["time"].max() < np.datetime64(f"{train_end + 1}-01-01")
            np.testing.assert_array_equal(
                scenario_hist["time"].values, historical_stage_hist["time"].values
            )


_RUN_SPECIFIC_ATTRS = (
    "srm_downscaling:downscaling_method",
    "srm_downscaling:bias_correction_method",
    "srm_downscaling:disaggregation_method",
    "srm_downscaling:config_json",
    "srm_downscaling:config_hash",
    "srm_downscaling:scenario",
    "srm_downscaling:ensemble_member",
    "srm_downscaling:historical_ensemble_member",
    "srm_downscaling:ssp245_ensemble_member",
    "srm_downscaling:train_period",
)


def _attrs_pipeline(**overrides) -> DownscalingPipeline:
    return DownscalingPipeline(_cfg(**overrides), PipelineOptions())


def test_obs_attrs_hold_only_shared_provenance(subtests):
    attrs = _attrs_pipeline()._build_obs_attrs()
    for key in _RUN_SPECIFIC_ATTRS:
        with subtests.test(attr=key):
            assert key not in attrs
    with subtests.test(attr="keyed-on fields"):
        assert attrs["srm_downscaling:gcm"] == "CESM2-WACCM6"
        assert attrs["srm_downscaling:gcm_description"] == "CESM2.1.5-WACCM6(TSMLT)"
        assert attrs["srm_downscaling:variable"] == "tas"
        for key in ("observation_dataset", "version", "creation_date"):
            assert f"srm_downscaling:{key}" in attrs
    with subtests.test(attr="identical across methods and members"):
        other = _attrs_pipeline(
            downscaling_method="QDMSD", ensemble_member="r2i1p1f1", scenario="G6-1.5K"
        )._build_obs_attrs()
        volatile = {"history", "srm_downscaling:creation_date"}
        assert {k: v for k, v in attrs.items() if k not in volatile} == {
            k: v for k, v in other.items() if k not in volatile
        }


def test_output_attrs_carry_method_and_gcm_description():
    attrs = _attrs_pipeline(downscaling_method="QDMSD")._build_output_attrs()
    assert attrs["srm_downscaling:downscaling_method"] == "QDMSD"
    assert attrs["srm_downscaling:gcm"] == "CESM2-WACCM6"
    assert attrs["srm_downscaling:gcm_description"] == "CESM2.1.5-WACCM6(TSMLT)"


def test_attrs_omit_description_for_unknown_gcm():
    pipeline = _attrs_pipeline(gcm="SOME-OTHER-GCM")
    assert "srm_downscaling:gcm_description" not in pipeline._build_obs_attrs()
    assert "srm_downscaling:gcm_description" not in pipeline._build_output_attrs()


def _qdm_da(start: str, end: str, value: float = 280.0) -> xr.DataArray:
    return _spatial_daily_da(start, end, value)


def _qdm_pipeline(
    pipeline_options, variable="tas", scenario="SSP245", predict_period_start=2015, **window
):
    cfg = _cfg(
        downscaling_method="QDMSD",
        variable=variable,
        scenario=scenario,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_start + 1,
    )
    assert cfg.variable_config.debias_approach == "qdm"
    for key, value in window.items():
        setattr(cfg.variable_config, key, value)
    return DownscalingPipeline(cfg, pipeline_options)


def _run_qdm(pipe: DownscalingPipeline, stitched: xr.DataArray):
    captured: dict = {}

    def _fake_apply(self, **kwargs):
        captured["debiaser"] = self
        return np.zeros(kwargs["cm_future"].shape)

    hist = _qdm_da("2010-01-01", "2014-12-31")
    scenario = _qdm_da("2015-01-01", "2016-12-31")
    with (
        patch("saidownscale.pipeline.stitch_historical_scenario", return_value=stitched),
        patch.object(QuantileDeltaMapping, "apply", _fake_apply),
    ):
        out = pipe._apply_bias_correction_scenario(
            hist, hist, scenario, model_scenario_for_qdm=scenario
        )
    return out, captured.get("debiaser")


def test_qdm_debiaser_is_seeded_and_windowed_from_variable_config(pipeline_options, subtests):
    """The years window must stay at the ibicus default: pad_years is derived from it."""
    for variable in ("pr", "rsds", "dtr", "tas"):
        with subtests.test(variable=variable):
            pipe = _qdm_pipeline(
                pipeline_options,
                variable,
                running_window_length=45,
                running_window_step_length=7,
            )
            _, debiaser = _run_qdm(pipe, _qdm_da("1990-01-01", "2014-12-31"))
            assert isinstance(debiaser, _SeededQuantileDeltaMapping)
            assert debiaser.running_window_length == 45
            assert debiaser.running_window_step_length == 7
            assert debiaser.running_window_mode is True
            assert debiaser.running_window_mode_over_years_of_cm_future is True
            assert debiaser.running_window_over_years_of_cm_future_length == 31


def test_qdm_lead_in_pad_must_be_complete(pipeline_options, subtests):
    pipe = _qdm_pipeline(pipeline_options)
    with subtests.test(pad="full"):
        out, _ = _run_qdm(pipe, _qdm_da("2000-01-01", "2014-12-31"))
        assert out.sizes["time"] == 731
    with subtests.test(pad="short"):
        with pytest.raises(ValueError) as excinfo:
            _run_qdm(pipe, _qdm_da("2010-01-01", "2014-12-31"))
        message = str(excinfo.value)
        for fragment in ("2000 to 2014", "2010 to 2014", "2009", "2010-01-01"):
            assert fragment in message
    with subtests.test(pad="empty"), pytest.raises(ValueError, match="no years"):
        _run_qdm(pipe, _qdm_da("2015-01-01", "2016-12-31"))


def test_qdm_rejects_missing_lead_in_source(pipeline_options, subtests):
    da = _qdm_da("2010-01-01", "2014-12-31")
    with subtests.test(missing="model_scenario_for_qdm"):
        with pytest.raises(ValueError, match="model_scenario_for_qdm"):
            _qdm_pipeline(pipeline_options)._apply_bias_correction_scenario(da, da, da)
    with subtests.test(missing="ssp_timeseries_for_qdm"):
        pipe = _qdm_pipeline(pipeline_options, scenario="G6-1.5K", predict_period_start=2035)
        assert pipe.config.is_sai_scenario
        with pytest.raises(ValueError, match="ssp_timeseries_for_qdm"):
            pipe._apply_bias_correction_scenario(da, da, da, model_scenario_for_qdm=da)


def _precip_arrays(nt: int = 200):
    rng = np.random.default_rng(0)
    shape = (nt, 2, 2)

    def draw(scale: float) -> np.ndarray:
        values = rng.gamma(0.6, scale, size=shape)
        values[rng.random(shape) < 0.7] = 0.0
        return values

    times = pd.date_range("2000-01-01", periods=nt, freq="D").values
    return draw(3e-5), draw(4e-5), draw(4.5e-5), times


def _apply_debiaser(debiaser, *, parallel: bool, nr_processes: int = 1) -> np.ndarray:
    obs, cm_hist, cm_future, times = _precip_arrays()
    return debiaser.apply(
        obs=obs,
        cm_hist=cm_hist,
        cm_future=cm_future,
        time_obs=times,
        time_cm_hist=times,
        time_cm_future=times,
        parallel=parallel,
        nr_processes=nr_processes,
        progressbar=False,
        failsafe=True,
    )


def _qdm_debiaser(cls_):
    return cls_.for_precipitation(
        running_window_mode=True, running_window_length=31, running_window_step_length=31
    )


def _qm_debiaser(cls_):
    return cls_(
        variable="pr",
        distribution=PrecipitationHurdleModelGamma,
        mapping_type="parametric",
        running_window_mode=True,
        running_window_length=31,
        running_window_step_length=31,
    )


_DEBIASERS = {
    "qdm": (_qdm_debiaser, _SeededQuantileDeltaMapping, QuantileDeltaMapping),
    "qm": (_qm_debiaser, _SeededQuantileMapping, QuantileMapping),
}


def test_location_seed(subtests):
    obs, cm_hist, cm_future, _ = _precip_arrays()
    seeds = [
        _location_seed(obs[:, i, j], cm_hist[:, i, j], cm_future[:, i, j])
        for i in range(2)
        for j in range(2)
    ]
    with subtests.test("deterministic"):
        assert _location_seed(obs[:, 0, 0], cm_hist[:, 0, 0], cm_future[:, 0, 0]) == seeds[0]
    with subtests.test("distinct per location"):
        assert len(set(seeds)) == 4
    with subtests.test("accepted by numpy"):
        assert all(0 <= seed < 2**32 for seed in seeds)
        np.random.seed(seeds[0])


def test_seeded_debiaser_preconditions(subtests):
    qdm = _qdm_debiaser(_SeededQuantileDeltaMapping)
    qm = _make_debiaser(variable="tas", mapping_type="parametric")
    with subtests.test("qdm survives pickling for pool workers"):
        assert isinstance(pickle.loads(pickle.dumps(qdm)), _SeededQuantileDeltaMapping)
    with subtests.test("_make_debiaser builds a picklable seeded qm"):
        assert isinstance(qm, _SeededQuantileMapping)
        assert isinstance(pickle.loads(pickle.dumps(qm)), _SeededQuantileMapping)
    with subtests.test("mixin precedes ibicus in the MRO"):
        mro = _SeededQuantileMapping.__mro__
        assert mro.index(_SeededLocationMixin) < mro.index(QuantileMapping)
    with subtests.test("qdm censors precipitation"):
        assert qdm.censor_values_to_zero
    with subtests.test("hurdle model randomizes"):
        assert PrecipitationHurdleModelGamma.cdf_randomization


def test_only_seeded_output_ignores_ambient_rng_state(subtests):
    """#627: stock ibicus draws from the process-global RNG; the seeded subclasses must not."""
    for method, (build, seeded, stock) in _DEBIASERS.items():
        for cls_, should_match in ((stock, False), (seeded, True)):
            with subtests.test(method=method, debiaser=cls_.__name__):
                runs = []
                for seed in (42, 43):
                    np.random.seed(seed)
                    runs.append(_apply_debiaser(build(cls_), parallel=False))
                assert np.array_equal(*runs) is should_match


@pytest.mark.parametrize("method", list(_DEBIASERS))
def test_seeded_parallel_matches_serial(method):
    """Pool scheduling and worker count must not change the answer (#627)."""
    build, seeded, _ = _DEBIASERS[method]
    parallel = _apply_debiaser(build(seeded), parallel=True, nr_processes=2)
    serial = _apply_debiaser(build(seeded), parallel=False)
    np.testing.assert_array_equal(parallel, serial)


def test_debiaser_processes(monkeypatch, subtests):
    """A malformed value must fall back rather than kill a multi-hour task."""
    cases = {"16": 16, None: dask.system.CPU_COUNT}
    cases |= dict.fromkeys(("", "not-a-number", "0", "-4"), dask.system.CPU_COUNT)
    for value, expected in cases.items():
        with subtests.test(value=value):
            if value is None:
                monkeypatch.delenv(NR_PROCESSES_ENV, raising=False)
            else:
                monkeypatch.setenv(NR_PROCESSES_ENV, value)
            assert debiaser_processes() == expected
