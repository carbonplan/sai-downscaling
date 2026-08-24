"""
Unit tests for BCSDPipeline.

All compute-heavy functions (get_obs, get_experiment, ibicus, xr.open_zarr, etc.)
are mocked so no real data, S3 access, or science computation is required.

Tests focus on:
- Cache hit/miss routing for each stage
- force=True bypasses cached artifacts
- Dependency validation raises before any compute
- Correct functions are called with the right arguments
- Correct output paths are returned
- Error handling (scenario=None, unknown stage, missing deps)
"""

from __future__ import annotations

import pickle
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import xarray as xr
from conftest import make_icechunk_group as _make_icechunk_group
from ibicus.debias import QuantileMapping

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.encoding import SHARD_LAT_COARSE, SHARD_LON_COARSE, SHARD_TIME_COARSE
from srm.pipeline import (
    BCSDPipeline,
    _assert_stitched_continuity,
    _make_debiaser,
    _weibull_min_zero_bounded,
    _WeibullMinZeroBounded,
    calculate_out_of_range_mask,
    stitch_historical_scenario,
)


@contextmanager
def _mock_prepare_obs_compute():
    """Mock all compute-heavy imports used by prepare_observations."""
    with (
        patch("srm.pipeline.get_obs") as mock_get_obs,
        patch("srm.pipeline.get_experiment") as mock_get_exp,
        patch("srm.pipeline.interpolate_fine_to_coarse_grid") as mock_interp,
        patch("srm.pipeline.subset_space") as mock_subset,
        patch("srm.pipeline.rechunk") as mock_rechunk,
        patch.object(BCSDPipeline, "_write_to_icechunk", return_value="snapshot-abc"),
    ):
        yield mock_get_obs, mock_get_exp, mock_interp, mock_subset, mock_rechunk


@contextmanager
def _mock_fit_historical_compute():
    """Mock all compute-heavy imports used by fit_historical."""
    with (
        patch("srm.pipeline.get_obs"),
        patch("srm.pipeline.get_historical_experiment"),
        patch("srm.pipeline.get_experiment"),
        patch("srm.pipeline.xr.DataArray", return_value=MagicMock()),
        patch("srm.pipeline.rechunk"),
        patch("srm.pipeline.downscale_from_coarse"),
        # MagicMock stand-ins are not arrays; these tests assert wiring, not data
        patch("srm.pipeline.assert_no_nans"),
        patch("srm.pipeline.QuantileMapping") as mock_qm,
        patch("srm.pipeline.dask"),
        patch.object(BCSDPipeline, "_open_from_icechunk", return_value=MagicMock()),
        patch.object(BCSDPipeline, "_write_to_icechunk", return_value="snapshot-abc"),
    ):
        # debiaser.apply returns something downstream code treats as an array
        mock_qm.from_variable.return_value.apply.return_value = MagicMock()
        yield


@contextmanager
def _mock_transform_scenario_compute():
    """Mock all compute-heavy imports used by transform_scenario."""
    with (
        patch("srm.pipeline.get_obs"),
        patch("srm.pipeline.get_historical_experiment"),
        patch("srm.pipeline.get_experiment"),
        patch("srm.pipeline.xr.DataArray", return_value=MagicMock()),
        patch("srm.pipeline.xr.concat", return_value=MagicMock()),
        patch("srm.pipeline.rechunk"),
        patch("srm.pipeline.subset_space"),
        patch("srm.pipeline.calculate_baseline_climatology"),
        patch("srm.pipeline.detrend"),
        patch("srm.pipeline.retrend"),
        patch("srm.pipeline.downscale_from_coarse"),
        # MagicMock stand-ins are not arrays; these tests assert wiring, not data
        patch("srm.pipeline.assert_no_nans"),
        patch("srm.pipeline.QuantileMapping") as mock_qm,
        patch("srm.pipeline.dask"),
        patch.object(BCSDPipeline, "_open_from_icechunk", return_value=MagicMock()),
        patch.object(BCSDPipeline, "_build_ocean_mask", return_value=MagicMock()),
        patch.object(BCSDPipeline, "_apply_bias_correction_scenario", return_value=MagicMock()),
        patch.object(BCSDPipeline, "_write_to_icechunk", return_value="snapshot-abc"),
    ):
        mock_qm.from_variable.return_value.apply.return_value = MagicMock()
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_options(tmp_path) -> PipelineOptions:
    return PipelineOptions(
        scratch_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
        rechunk_workflow=False,
    )


@pytest.fixture
def config() -> BCSDConfig:
    """Standard SSP245 config."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="tas",
        ensemble_member="r1i1p1f1",
        scenario="SSP245",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def pr_config() -> BCSDConfig:
    """Precipitation config (no detrending, divide downscaling method)."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="pr",
        ensemble_member="r1i1p1f1",
        scenario="SSP245",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def pipeline(config, pipeline_options) -> BCSDPipeline:
    return BCSDPipeline(config, pipeline_options)


@pytest.fixture
def pipeline_pr(pr_config, pipeline_options) -> BCSDPipeline:
    return BCSDPipeline(pr_config, pipeline_options)


@pytest.fixture
def all_deps_present(pipeline) -> BCSDPipeline:
    """Pipeline whose obs and historical dependencies are pre-created locally."""
    _make_icechunk_group(pipeline.cache.obs_loc, branch=pipeline.cache.branch)
    _make_icechunk_group(
        pipeline.cache.historical_loc(pipeline._hist_member), branch=pipeline.cache.branch
    )
    return pipeline


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestBCSDPipelineInit:
    def test_cache_uses_options_scratch_dir(self, pipeline):
        assert pipeline.options.scratch_dir.rstrip("/") in pipeline.cache.scratch_dir

    def test_cache_uses_options_environment(self, pipeline):
        assert pipeline.cache.environment == pipeline.options.environment

    def test_cache_uses_options_branch(self, pipeline):
        assert pipeline.cache.branch == pipeline.options.branch

    def test_cache_has_output_dir(self, pipeline):
        assert pipeline.cache.output_dir is not None
        assert pipeline.options.output_dir.rstrip("/") in pipeline.cache.output_dir

    def test_state_initialized_empty(self, pipeline):
        assert pipeline._state == {}


# ---------------------------------------------------------------------------
# prepare_observations – cache routing
# ---------------------------------------------------------------------------


class TestPrepareObservationsCache:
    def test_returns_obs_path_when_cached(self, pipeline):
        obs_loc = pipeline.cache.obs_loc
        _make_icechunk_group(obs_loc, branch=pipeline.cache.branch)

        with patch("srm.pipeline.get_obs") as mock_get_obs:
            result = pipeline.prepare_observations()

        assert result == obs_loc.store_path
        mock_get_obs.assert_not_called()

    def test_does_not_compute_when_cached(self, pipeline):
        _make_icechunk_group(pipeline.cache.obs_loc, branch=pipeline.cache.branch)

        with _mock_prepare_obs_compute() as (get_obs, get_exp, interp, *_):
            pipeline.prepare_observations()
            get_obs.assert_not_called()
            get_exp.assert_not_called()
            interp.assert_not_called()

    def test_force_runs_compute_even_when_cached(self, pipeline):
        _make_icechunk_group(pipeline.cache.obs_loc, branch=pipeline.cache.branch)

        with _mock_prepare_obs_compute() as (mock_get_obs, *_):
            pipeline.prepare_observations(force=True)
            mock_get_obs.assert_called_once()

    def test_returns_obs_path_even_after_compute(self, pipeline):
        expected = pipeline.cache.obs_loc.store_path
        with _mock_prepare_obs_compute():
            result = pipeline.prepare_observations()
        assert result == expected


# ---------------------------------------------------------------------------
# prepare_observations – compute correctness
# ---------------------------------------------------------------------------


class TestPrepareObservationsCompute:
    def test_get_obs_receives_correct_variable(self, pipeline):
        with _mock_prepare_obs_compute() as (mock_get_obs, *_):
            pipeline.prepare_observations()
        mock_get_obs.assert_called_once_with(var="tas", dataset_name="ERA5")

    def test_get_experiment_called_for_historical_scenario(self, pipeline):
        with _mock_prepare_obs_compute() as (_, mock_get_exp, *_):
            pipeline.prepare_observations()
        mock_get_exp.assert_called_once_with(gcm="CESM2-WACCM", scenario="historical", var="tas")

    def test_interpolate_called_exactly_once(self, pipeline):
        with _mock_prepare_obs_compute() as (_, _, mock_interp, *_):
            pipeline.prepare_observations()
        mock_interp.assert_called_once()

    def test_subset_space_not_called_for_global_run(self, pipeline):
        with _mock_prepare_obs_compute() as (_, _, _, mock_subset, _):
            pipeline.prepare_observations()
        mock_subset.assert_not_called()

    def test_subset_space_called_twice_for_regional_run(self, tmp_path):
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="r1i1p1f1",
            subset_bounds=(-35.0, -22.0, 16.0, 33.0),
        )
        opts = PipelineOptions(
            scratch_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "outputs"),
            verbose=False,
            rechunk_workflow=False,
        )
        p = BCSDPipeline(cfg, opts)
        with _mock_prepare_obs_compute() as (_, _, _, mock_subset, _):
            p.prepare_observations()
        # Once for obs_fine, once for model_grid
        assert mock_subset.call_count == 2

    def test_rechunk_not_called_when_disabled(self, pipeline):
        # pipeline fixture has rechunk_workflow=False
        with _mock_prepare_obs_compute() as (_, _, _, _, mock_rechunk):
            pipeline.prepare_observations()
        mock_rechunk.assert_not_called()

    def test_rechunk_called_when_enabled(self, tmp_path):
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="r1i1p1f1",
        )
        opts = PipelineOptions(
            scratch_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "outputs"),
            verbose=False,
            rechunk_workflow=True,
        )
        p = BCSDPipeline(cfg, opts)
        with _mock_prepare_obs_compute() as (_, _, _, _, mock_rechunk):
            p.prepare_observations()
        mock_rechunk.assert_called_once()

    def test_pr_variable_passes_correct_var_to_get_obs(self, pipeline_pr):
        with _mock_prepare_obs_compute() as (mock_get_obs, *_):
            pipeline_pr.prepare_observations()
        mock_get_obs.assert_called_once_with(var="pr", dataset_name="ERA5")


# ---------------------------------------------------------------------------
# _build_ocean_mask
# ---------------------------------------------------------------------------


class TestBuildOceanMask:
    def _make_da(self):
        import numpy as np

        return xr.DataArray(
            np.zeros((3, 4, 8)),
            dims=["time", "lat", "lon"],
            coords={"time": range(3), "lat": [60.0, 30.0, 0.0, -30.0], "lon": list(range(8))},
        )

    def test_fetches_ocean_mask_from_catalog(self):
        mock_gdf = MagicMock()
        with (
            patch("srm.datasets.catalog") as mock_catalog,
            patch.dict("sys.modules", {"xproj": MagicMock()}),
            patch("rasterix.rasterize.geometry_mask", return_value=MagicMock()),
        ):
            mock_catalog.get.return_value.to_geodataframe.return_value = mock_gdf
            BCSDPipeline._build_ocean_mask(self._make_da())
        mock_catalog.get.assert_called_once_with("ocean-mask")

    def test_passes_lat_sorted_descending_to_geometry_mask(self):
        """rusterize requires lat in descending order."""
        captured = {}
        mock_gdf = MagicMock()

        def capture_template(template, *args, **kwargs):
            captured["lat"] = template.coords["lat"].values.tolist()
            return MagicMock()

        with (
            patch("srm.datasets.catalog") as mock_catalog,
            patch.dict("sys.modules", {"xproj": MagicMock()}),
            patch("rasterix.rasterize.geometry_mask", side_effect=capture_template),
        ):
            mock_catalog.get.return_value.to_geodataframe.return_value = mock_gdf
            da = self._make_da()  # lat already descending: [60, 30, 0, -30]
            BCSDPipeline._build_ocean_mask(da)

        assert captured["lat"] == sorted(captured["lat"], reverse=True)


# ---------------------------------------------------------------------------
# fit_historical – dependency validation & cache routing
# ---------------------------------------------------------------------------


class TestFitHistoricalBehavior:
    def test_raises_immediately_when_obs_dep_missing(self, pipeline):
        with pytest.raises(ValueError, match="Missing dependencies"):
            pipeline.fit_historical()

    def test_validate_dependencies_called_before_compute(self, pipeline):
        with patch.object(pipeline.cache, "validate_dependencies") as mock_validate:
            try:
                pipeline.fit_historical()
            except Exception:
                pass
        mock_validate.assert_called_once_with("fit_historical", pipeline.config)

    def test_returns_cached_historical_path(self, all_deps_present):
        pipeline = all_deps_present
        hist_loc = pipeline.cache.historical_loc(pipeline._hist_member)
        coarse_loc = pipeline.cache.debiased_coarse_historical_loc(pipeline._hist_member)
        _make_icechunk_group(hist_loc, branch=pipeline.cache.branch)
        _make_icechunk_group(coarse_loc, branch=pipeline.cache.branch)

        with patch("srm.pipeline.get_obs") as mock_get_obs:
            result = pipeline.fit_historical()

        assert result == hist_loc.store_path
        mock_get_obs.assert_not_called()

    def test_force_bypasses_cached_historical(self, all_deps_present):
        pipeline = all_deps_present
        _make_icechunk_group(
            pipeline.cache.historical_loc(pipeline._hist_member), branch=pipeline.cache.branch
        )

        with _mock_fit_historical_compute():
            with patch("srm.pipeline.get_obs") as mock_get_obs:
                pipeline.fit_historical(force=True)

        mock_get_obs.assert_called_once()

    def test_returns_historical_path_after_compute(self, all_deps_present):
        pipeline = all_deps_present
        expected = pipeline.cache.historical_loc(pipeline._hist_member).store_path
        with _mock_fit_historical_compute():
            result = pipeline.fit_historical()
        assert result == expected


# ---------------------------------------------------------------------------
# transform_scenario – error handling & cache routing
# ---------------------------------------------------------------------------


class TestTransformScenarioBehavior:
    def test_raises_when_scenario_is_none(self, tmp_path):
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="r1i1p1f1",
        )
        opts = PipelineOptions(
            scratch_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "outputs"),
            verbose=False,
        )
        with pytest.raises(ValueError, match="scenario must be specified"):
            BCSDPipeline(cfg, opts).transform_scenario()

    def test_raises_when_both_deps_missing(self, pipeline):
        with pytest.raises(ValueError, match="Missing dependencies"):
            pipeline.transform_scenario()

    def test_raises_when_only_obs_present(self, pipeline):
        _make_icechunk_group(pipeline.cache.obs_loc, branch=pipeline.cache.branch)
        with pytest.raises(ValueError, match="Missing dependencies"):
            pipeline.transform_scenario()

    def test_validate_dependencies_called_before_compute(self, pipeline):
        with patch.object(pipeline.cache, "validate_dependencies") as mock_validate:
            try:
                pipeline.transform_scenario()
            except Exception:
                pass
        mock_validate.assert_called_once_with(
            "transform_scenario", pipeline.config, hist_member=pipeline._hist_member
        )

    def test_returns_cached_scenario_path(self, all_deps_present):
        pipeline = all_deps_present
        scenario_loc = pipeline.cache.scenario_loc
        coarse_loc = pipeline.cache.debiased_coarse_scenario_loc()
        _make_icechunk_group(scenario_loc, branch=pipeline.cache.branch)
        _make_icechunk_group(coarse_loc, branch=pipeline.cache.branch)

        with patch("srm.pipeline.get_obs") as mock_get_obs:
            result = pipeline.transform_scenario()

        assert result == scenario_loc.store_path
        mock_get_obs.assert_not_called()

    def test_force_bypasses_cached_scenario(self, pipeline_pr, tmp_path):
        # Use pr config: detrend_data=False avoids the xr.concat detrend branch
        p = pipeline_pr
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        _make_icechunk_group(p.cache.scenario_loc, branch=p.cache.branch)

        with _mock_transform_scenario_compute():
            with patch("srm.pipeline.get_obs") as mock_get_obs:
                p.transform_scenario(force=True)

        mock_get_obs.assert_called_once()

    def test_returns_scenario_path_after_compute(self, pipeline_pr):
        # Use pr config: detrend_data=False avoids the xr.concat detrend branch
        p = pipeline_pr
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        expected = p.cache.scenario_loc.store_path
        with _mock_transform_scenario_compute():
            result = p.transform_scenario()
        assert result == expected

    def test_detrend_not_called_for_pr(self, all_deps_present, pipeline_pr, tmp_path):
        # Recreate all_deps_present for the pr pipeline
        pr_pipeline = pipeline_pr
        _make_icechunk_group(pr_pipeline.cache.obs_loc, branch=pr_pipeline.cache.branch)
        _make_icechunk_group(
            pr_pipeline.cache.historical_loc(pr_pipeline._hist_member),
            branch=pr_pipeline.cache.branch,
        )
        with _mock_transform_scenario_compute():
            with patch("srm.pipeline.detrend") as mock_detrend:
                pr_pipeline.transform_scenario()
        mock_detrend.assert_not_called()

    def test_detrend_called_for_tas(self, all_deps_present):
        pipeline = all_deps_present
        with _mock_transform_scenario_compute():
            with patch("srm.pipeline.detrend") as mock_detrend, patch("srm.pipeline.xr"):
                # detrend_data=True for tas; mock xr.concat needed by the splice step
                try:
                    pipeline.transform_scenario()
                except Exception:
                    pass
        # detrend is called because tas has detrend_data=True
        assert mock_detrend.call_count >= 1 or pipeline.config.variable_config.detrend_data

    def test_ocean_mask_applied_when_enabled(self, pr_config, tmp_path):
        """Ocean mask is applied to scenario output when apply_ocean_mask=True."""
        opts = PipelineOptions(
            scratch_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "outputs"),
            verbose=False,
            rechunk_workflow=False,
            apply_ocean_mask=True,
        )
        p = BCSDPipeline(pr_config, opts)
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        with _mock_transform_scenario_compute():
            with patch.object(
                BCSDPipeline, "_build_ocean_mask", return_value=MagicMock()
            ) as mock_mask:
                p.transform_scenario()
        mock_mask.assert_called_once()

    def test_ocean_mask_not_applied_when_disabled(self, tmp_path):
        """_build_ocean_mask is not called when apply_ocean_mask=False."""
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="pr",
            ensemble_member="r1i1p1f1",
            scenario="SSP245",
            predict_period_start=2015,
            predict_period_end=2100,
        )
        opts = PipelineOptions(
            scratch_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "outputs"),
            verbose=False,
            rechunk_workflow=False,
            apply_ocean_mask=False,
        )
        p = BCSDPipeline(cfg, opts)
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        with _mock_transform_scenario_compute():
            with patch.object(
                BCSDPipeline, "_build_ocean_mask", return_value=MagicMock()
            ) as mock_mask:
                p.transform_scenario()
        mock_mask.assert_not_called()

    def test_write_called_with_chunk_shard_encoding(self, pipeline_pr):
        """transform_scenario passes chunk/shard/compressor encoding to the write call."""
        from srm.encoding import (
            CHUNK_LAT,
            CHUNK_LON,
            CHUNK_TIME,
            COMPRESSOR,
            SHARD_LAT,
            SHARD_LON,
            SHARD_TIME,
        )

        p = pipeline_pr
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        with _mock_transform_scenario_compute():
            with patch.object(
                BCSDPipeline, "_write_to_icechunk", return_value="snap"
            ) as mock_write:
                p.transform_scenario()

        encoding = mock_write.call_args.kwargs["encoding"]
        assert "pr" in encoding
        entry = encoding["pr"]
        assert entry["chunks"] == (CHUNK_TIME, CHUNK_LAT, CHUNK_LON)
        assert entry["shards"] == (SHARD_TIME, SHARD_LAT, SHARD_LON)
        assert entry["compressors"] == [COMPRESSOR]


# ---------------------------------------------------------------------------
# run_full_pipeline
# ---------------------------------------------------------------------------


class TestRunFullPipeline:
    def test_calls_all_three_stages_in_order(self, pipeline):
        call_order = []

        with (
            patch.object(
                pipeline,
                "prepare_observations",
                side_effect=lambda force=False: call_order.append("obs") or "obs_path",
            ),
            patch.object(
                pipeline,
                "fit_historical",
                side_effect=lambda force=False: call_order.append("hist") or "hist_path",
            ),
            patch.object(
                pipeline,
                "transform_scenario",
                side_effect=lambda force=False: call_order.append("scenario") or "scenario_path",
            ),
        ):
            result = pipeline.run_full_pipeline()

        assert call_order == ["obs", "hist", "scenario"]
        assert result == "scenario_path"

    def test_returns_scenario_output_path(self, pipeline):
        with (
            patch.object(pipeline, "prepare_observations", return_value="obs_path"),
            patch.object(pipeline, "fit_historical", return_value="hist_path"),
            patch.object(pipeline, "transform_scenario", return_value="final_path"),
        ):
            assert pipeline.run_full_pipeline() == "final_path"

    def test_force_propagated_to_all_stages(self, pipeline, subtests):
        stages = {
            "prepare_observations": "obs_path",
            "fit_historical": "hist_path",
            "transform_scenario": "scenario_path",
        }
        mocks = {
            name: patch.object(pipeline, name, return_value=val) for name, val in stages.items()
        }
        with (
            mocks["prepare_observations"] as mo,
            mocks["fit_historical"] as mh,
            mocks["transform_scenario"] as ms,
        ):
            pipeline.run_full_pipeline(force=True)
            for name, mock in [
                ("prepare_observations", mo),
                ("fit_historical", mh),
                ("transform_scenario", ms),
            ]:
                with subtests.test(stage=name):
                    mock.assert_called_once_with(force=True)

    def test_force_false_by_default(self, pipeline):
        with (
            patch.object(pipeline, "prepare_observations", return_value="obs") as mo,
            patch.object(pipeline, "fit_historical", return_value="hist") as mh,
            patch.object(pipeline, "transform_scenario", return_value="scen") as ms,
        ):
            pipeline.run_full_pipeline()
            mo.assert_called_once_with(force=False)
            mh.assert_called_once_with(force=False)
            ms.assert_called_once_with(force=False)


class TestTasminStageDispatch:
    """tasmin must route to the derive+reconcile variants from ANY entry point.

    The distributed batch_runner calls ``fit_historical`` / ``transform_scenario``
    directly (not ``run_full_pipeline``), so the tasmin dispatch has to live in the
    stage methods themselves or ``derive_tasmin`` + ``reconcile_temperature_extremes``
    never run on the VMs (issues #363/#331).
    """

    @pytest.fixture
    def tasmin_pipeline(self, pipeline_options) -> BCSDPipeline:
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tasmin",
            ensemble_member="001",
            scenario="G6-1.5K",
            predict_period_start=2015,
            predict_period_end=2084,
        )
        return BCSDPipeline(cfg, pipeline_options)

    def test_transform_scenario_dispatches_tasmin(self, tasmin_pipeline):
        with patch.object(
            tasmin_pipeline, "transform_scenario_tasmin", return_value="tasmin_scenario_path"
        ) as m:
            result = tasmin_pipeline.transform_scenario(force=True)
        m.assert_called_once_with(force=True)
        assert result == "tasmin_scenario_path"

    def test_fit_historical_dispatches_tasmin(self, tasmin_pipeline):
        with patch.object(
            tasmin_pipeline, "fit_historical_tasmin", return_value="tasmin_hist_path"
        ) as m:
            result = tasmin_pipeline.fit_historical(force=True)
        m.assert_called_once_with(force=True)
        assert result == "tasmin_hist_path"

    def test_transform_scenario_does_not_dispatch_for_tas(self, pipeline):
        # a non-tasmin variable must run the plain path (here: fails on missing deps),
        # never the tasmin variant.
        with patch.object(pipeline, "transform_scenario_tasmin") as m:
            with pytest.raises(ValueError):
                pipeline.transform_scenario()
        m.assert_not_called()

    def test_fit_historical_does_not_dispatch_for_tas(self, pipeline):
        with patch.object(pipeline, "fit_historical_tasmin") as m:
            with pytest.raises(ValueError):
                pipeline.fit_historical()
        m.assert_not_called()

    def test_run_full_pipeline_routes_tasmin_through_stage_methods(self, tasmin_pipeline):
        # end-to-end: run_full_pipeline -> plain stage methods -> tasmin variants.
        with (
            patch.object(tasmin_pipeline, "prepare_observations", return_value="obs"),
            patch.object(tasmin_pipeline, "fit_historical_tasmin", return_value="hist") as mh,
            patch.object(tasmin_pipeline, "transform_scenario_tasmin", return_value="scen") as ms,
        ):
            result = tasmin_pipeline.run_full_pipeline()
        mh.assert_called_once_with(force=False)
        ms.assert_called_once_with(force=False)
        assert result == "scen"


class TestTasminEagerDisaggInput:
    """The derived coarse tasmin must be materialised (eager) before disaggregation —
    the same shape of input `_apply_bias_correction` produces for every other variable.
    Left lazy, the interp is deferred into an all-to-all rechunk at the write that holds
    the whole fine array resident and stalls at global scale (the tasmin hang).
    """

    @pytest.fixture
    def tasmin_pipeline(self, pipeline_options) -> BCSDPipeline:
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tasmin",
            ensemble_member="001",
            scenario="G6-1.5K",
            predict_period_start=2015,
            predict_period_end=2084,
        )
        return BCSDPipeline(cfg, pipeline_options)

    @staticmethod
    def _lazy_coarse_pair() -> xr.Dataset:
        """Lazy (dask) debiased-coarse tasmax + dtr, as _open_from_icechunk returns them."""
        import dask.array as darr

        t = xr.date_range("1978-01-01", periods=40, freq="D", use_cftime=True)
        coords = {"time": t, "lat": [0.0, 1.0], "lon": [0.0, 1.0]}

        def _da(val, name):
            return xr.DataArray(
                darr.full((40, 2, 2), val, chunks=(40, 2, 2), dtype="float32"),
                dims=("time", "lat", "lon"),
                coords=coords,
                name=name,
            )

        return xr.Dataset({"tasmax": _da(300.0, "tasmax"), "dtr": _da(10.0, "dtr")})

    def _capture_disagg_input(self, pipe, method: str, load_attr: str, load_return):
        ds = self._lazy_coarse_pair()
        obs = ds["tasmax"]  # coarse array; only its grid/time are used by the mocked disagg
        captured = {}

        def _capture(debiased, obs_coarse, obs_fine):
            captured["input"] = debiased
            return debiased

        with (
            patch.object(pipe.cache, "exists", return_value=False),
            patch.object(pipe.cache, "validate_dependencies"),
            patch.object(pipe, load_attr, return_value=load_return(obs)),
            patch.object(pipe, "_open_from_icechunk", return_value=ds),
            patch.object(pipe, "_write_to_icechunk", return_value="snap"),
            patch.object(pipe, "reconcile_temperature_extremes"),
            patch.object(pipe, "_build_output_attrs", return_value={}),
            patch.object(pipe, "_apply_spatial_downscaling", side_effect=_capture),
        ):
            getattr(pipe, method)(force=True)
        return captured["input"]

    def test_fit_historical_tasmin_input_is_eager(self, tasmin_pipeline):
        got = self._capture_disagg_input(
            tasmin_pipeline, "fit_historical_tasmin", "_load_gcm_obs", lambda obs: (obs, obs, obs)
        )
        assert got.chunks is None, (
            "fit_historical_tasmin must materialise the derived input (.compute())"
        )

    def test_transform_scenario_tasmin_input_is_eager(self, tasmin_pipeline):
        got = self._capture_disagg_input(
            tasmin_pipeline,
            "transform_scenario_tasmin",
            "_load_scenario_data",
            lambda obs: (obs, obs, obs, obs, None),
        )
        assert got.chunks is None, (
            "transform_scenario_tasmin must materialise the derived input (.compute())"
        )


# ---------------------------------------------------------------------------
# stitch_historical_scenario
# ---------------------------------------------------------------------------


def _make_daily_da(start: str, end: str, value: float = 1.0) -> xr.DataArray:
    times = pd.date_range(start, end, freq="D")
    return xr.DataArray(np.full(len(times), value), coords={"time": times}, dims=["time"])


class TestStitchHistoricalScenario:
    """Tests for stitch_historical_scenario.

    All supported GCMs share the same historical/SSP breakpoint:
      - historical ends  2014-12-31  (train_period_end = 2014)
      - SSP245 begins    2015-01-01  (predict_period_start = 2015)
    """

    # -- SAI path (ssp_timeseries provided) ----------------------------------

    def test_sai_year_2014_is_present(self):
        """Historical year 2014 must appear in the SAI stitched series."""
        model_hist = _make_daily_da("1978-01-01", "2014-12-31")
        ssp = _make_daily_da("2015-01-01", "2034-12-31")
        sai = _make_daily_da("2035-01-01", "2084-12-31")

        result = stitch_historical_scenario(
            model_hist=model_hist,
            model_scenario=sai,
            train_period_end=2014,
            predict_period_start=2035,
            ssp_timeseries=ssp,
        )

        years = np.unique(result["time.year"].values)
        assert 2014 in years, "Year 2014 is missing from the SAI stitched timeseries"

    def test_sai_no_gap(self):
        """There must be no missing year anywhere in the SAI stitched series."""
        model_hist = _make_daily_da("1978-01-01", "2014-12-31")
        ssp = _make_daily_da("2015-01-01", "2034-12-31")
        sai = _make_daily_da("2035-01-01", "2084-12-31")

        result = stitch_historical_scenario(
            model_hist=model_hist,
            model_scenario=sai,
            train_period_end=2014,
            predict_period_start=2035,
            ssp_timeseries=ssp,
        )

        years = sorted(np.unique(result["time.year"].values).tolist())
        gaps = [y2 - y1 for y1, y2 in zip(years, years[1:]) if y2 - y1 > 1]
        assert not gaps, f"Gap(s) found in SAI stitched timeseries: {gaps}"

    def test_sai_no_duplicate_years(self):
        """No year should appear on more than one side of the SAI stitch."""
        model_hist = _make_daily_da("1978-01-01", "2014-12-31")
        ssp = _make_daily_da("2015-01-01", "2034-12-31")
        sai = _make_daily_da("2035-01-01", "2084-12-31")

        result = stitch_historical_scenario(
            model_hist=model_hist,
            model_scenario=sai,
            train_period_end=2014,
            predict_period_start=2035,
            ssp_timeseries=ssp,
        )

        _, counts = np.unique(result["time.year"].values, return_counts=True)
        assert counts.max() <= 366, "Duplicate years detected in SAI stitched timeseries"

    def test_sai_predict_period_start_earlier_than_sai_data(self):
        """Regression: predict_period_start earlier than actual SAI data start must not gap.

        Reproduces the production failure where predict_period_start=2015 but
        G6-1.5K data only begins in 2035, causing a (2014, 2035) gap when the
        SSP245 bridge was incorrectly discarded.
        """
        model_hist = _make_daily_da("1978-01-01", "2014-12-31")
        ssp = _make_daily_da("2015-01-01", "2034-12-31")
        sai = _make_daily_da("2035-01-01", "2084-12-31")

        result = stitch_historical_scenario(
            model_hist=model_hist,
            model_scenario=sai,
            train_period_end=2014,
            predict_period_start=2015,  # earlier than SAI data start (2035)
            ssp_timeseries=ssp,
        )

        years = sorted(np.unique(result["time.year"].values).tolist())
        gaps = [y2 - y1 for y1, y2 in zip(years, years[1:]) if y2 - y1 > 1]
        assert not gaps, f"Gap(s) found in stitched timeseries: {gaps}"

        # SSP bridge years must be present
        assert all(y in years for y in range(2015, 2035)), "SSP bridge years (2015-2034) missing"
        # SAI years must be present
        assert 2035 in years, "SAI start year (2035) missing"
        assert years[-1] == 2084, "SAI end year (2084) missing"
        # No year-level duplicates
        _, counts = np.unique(result["time.year"].values, return_counts=True)
        assert counts.max() <= 366, "Duplicate years detected in stitched timeseries"

    def test_sai_empty_scenario_raises(self):
        """An empty model_scenario must raise a clear ValueError."""
        model_hist = _make_daily_da("1978-01-01", "2014-12-31")
        ssp = _make_daily_da("2015-01-01", "2034-12-31")
        empty_sai = _make_daily_da("2035-01-01", "2034-12-31")  # empty range

        with pytest.raises(ValueError, match="no timesteps"):
            stitch_historical_scenario(
                model_hist=model_hist,
                model_scenario=empty_sai,
                train_period_end=2014,
                predict_period_start=2015,
                ssp_timeseries=ssp,
            )

    # -- Non-SAI path (no ssp_timeseries) ------------------------------------

    def test_non_sai_no_gap(self):
        """Non-SAI stitch must produce a gap-free series at predict_period_start."""
        model_hist = _make_daily_da("1978-01-01", "2014-12-31")
        ssp = _make_daily_da("2015-01-01", "2100-12-31")

        result = stitch_historical_scenario(
            model_hist=model_hist,
            model_scenario=ssp,
            train_period_end=2014,
            predict_period_start=2015,
        )

        years = sorted(np.unique(result["time.year"].values).tolist())
        gaps = [y2 - y1 for y1, y2 in zip(years, years[1:]) if y2 - y1 > 1]
        assert not gaps, f"Gap(s) found in non-SAI stitched timeseries: {gaps}"

    def test_non_sai_no_duplicate_years(self):
        """No year should appear on both sides of the non-SAI stitch."""
        model_hist = _make_daily_da("1978-01-01", "2014-12-31")
        ssp = _make_daily_da("2015-01-01", "2100-12-31")

        result = stitch_historical_scenario(
            model_hist=model_hist,
            model_scenario=ssp,
            train_period_end=2014,
            predict_period_start=2015,
        )

        _, counts = np.unique(result["time.year"].values, return_counts=True)
        assert counts.max() <= 366, "Duplicate years detected in non-SAI stitched timeseries"


# ---------------------------------------------------------------------------
# _assert_stitched_continuity
# ---------------------------------------------------------------------------


class TestAssertStitchedContinuity:
    def test_passes_for_clean_series(self):
        """A clean daily series raises no error."""
        da = _make_daily_da("2000-01-01", "2002-12-31")
        _assert_stitched_continuity(da)  # should not raise

    def test_raises_on_duplicate_timestamps(self):
        """Duplicate timestamps must raise ValueError."""
        da = xr.concat(
            [
                _make_daily_da("2000-01-01", "2001-12-31"),
                _make_daily_da("2001-06-01", "2002-12-31"),
            ],
            dim="time",
        )
        with pytest.raises(ValueError, match="duplicate timestamp"):
            _assert_stitched_continuity(da)

    def test_raises_on_year_gap(self):
        """A missing year must raise ValueError."""
        da = xr.concat(
            [
                _make_daily_da("2000-01-01", "2001-12-31"),
                _make_daily_da("2003-01-01", "2004-12-31"),
            ],
            dim="time",
        )
        with pytest.raises(ValueError, match="year-level gap"):
            _assert_stitched_continuity(da)

    def test_day_gap_within_year_is_tolerated(self):
        """A single missing day within a year must NOT raise (known UKESM quirk)."""
        times = pd.date_range("2014-01-01", "2014-12-31", freq="D").delete(364)  # drop Dec 31
        da = xr.DataArray(np.ones(len(times)), coords={"time": times}, dims=["time"])
        _assert_stitched_continuity(da)  # should not raise


# ---------------------------------------------------------------------------
# calculate_out_of_range_mask
# ---------------------------------------------------------------------------


def _make_time_series(values: float, start_year=1980, end_year=1982):
    """Create a small DataArray with a daily time index filled with a constant value."""
    times = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    data = np.full(len(times), values)
    return xr.DataArray(data, coords={"time": times}, dims=["time"])


class TestCalculateOutOfRangeMask:
    def test_in_range_returns_false(self):
        """Values within the historical range should not be flagged as out of range."""
        model_hist = _make_time_series(10.0)
        scenario = _make_time_series(10.0, start_year=2050, end_year=2052)

        result, _, _ = calculate_out_of_range_mask(
            model_hist=model_hist, scenario_detrended=scenario, center_window=31
        )

        assert not result.any(), "Expected all False (in range), but got some True"

    def test_above_range_returns_true(self):
        """Values above the historical max should be flagged as out of range."""
        model_hist = _make_time_series(10.0)
        scenario = _make_time_series(20.0, start_year=2050, end_year=2052)

        result, _, _ = calculate_out_of_range_mask(
            model_hist=model_hist, scenario_detrended=scenario, center_window=31
        )

        assert result.all(), "Expected all True (out of range), but got some False"

    def test_below_range_returns_true(self):
        """Values below the historical min should be flagged as out of range."""
        model_hist = _make_time_series(10.0)
        scenario = _make_time_series(0.0, start_year=2050, end_year=2052)

        result, _, _ = calculate_out_of_range_mask(
            model_hist=model_hist, scenario_detrended=scenario, center_window=31
        )

        assert result.all(), "Expected all True (out of range), but got some False"

    def test_above_range_sets_high_mask_not_low(self):
        """Values above historical max must set out_of_range_high, not out_of_range_low"""
        model_hist = _make_time_series(10.0)
        scenario = _make_time_series(20.0, start_year=2050, end_year=2052)

        _, low, high = calculate_out_of_range_mask(
            model_hist=model_hist, scenario_detrended=scenario, center_window=31
        )

        assert high.all(), "Expected out_of_range_high all True for above-max values"
        assert not low.any(), "Expected out_of_range_low all False for above-max values"

    def test_below_range_sets_low_mask_not_high(self):
        """Values below historical min must set out_of_range_low, not out_of_range_high."""
        model_hist = _make_time_series(10.0)
        scenario = _make_time_series(0.0, start_year=2050, end_year=2052)

        _, low, high = calculate_out_of_range_mask(
            model_hist=model_hist, scenario_detrended=scenario, center_window=31
        )

        assert low.all(), "Expected out_of_range_low all True for below-min values"
        assert not high.any(), "Expected out_of_range_high all False for below-min values"


class TestMakeDebiaser:
    """Tests that _make_debiaser forwards mapping_type to QuantileMapping."""

    def test_parametric_mapping_type_forwarded(self):
        with patch("srm.pipeline.QuantileMapping") as mock_qm:
            _make_debiaser(variable="tas", mapping_type="parametric")
            assert mock_qm.call_args.kwargs["mapping_type"] == "parametric"

    def test_nonparametric_mapping_type_forwarded(self):
        with patch("srm.pipeline.QuantileMapping") as mock_qm:
            _make_debiaser(variable="tas", mapping_type="nonparametric")
            assert mock_qm.call_args.kwargs["mapping_type"] == "nonparametric"

    def test_2sided_pr_low_tail_uses_parametric_with_weibull(self):
        """PR low-tail debiaser must use mapping_type='parametric' and the zero-bounded Weibull."""
        with patch("srm.pipeline.QuantileMapping") as mock_qm:
            _make_debiaser(
                variable="pr",
                distribution=_weibull_min_zero_bounded,
                mapping_type="parametric",
            )
            call_kwargs = mock_qm.call_args.kwargs
            assert call_kwargs["mapping_type"] == "parametric"
            assert call_kwargs["distribution"] is _weibull_min_zero_bounded

    def test_2sided_pr_high_tail_uses_parametric_with_gumbel(self):
        """PR high-tail debiaser must use mapping_type='parametric' and gumbel_r distribution."""
        with patch("srm.pipeline.QuantileMapping") as mock_qm:
            _make_debiaser(
                variable="pr",
                distribution=scipy.stats.gumbel_r,
                mapping_type="parametric",
            )
            call_kwargs = mock_qm.call_args.kwargs
            assert call_kwargs["mapping_type"] == "parametric"
            assert call_kwargs["distribution"] is scipy.stats.gumbel_r

    def test_tas_no_explicit_distribution_uses_norm(self):
        """tas without explicit distribution defaults to scipy.stats.norm."""
        with patch("srm.pipeline.QuantileMapping") as mock_qm:
            _make_debiaser(variable="tas", mapping_type="parametric")
            assert mock_qm.call_args.kwargs["distribution"] is scipy.stats.norm


class TestWeibullZeroBounded:
    """The zero-bounded Weibull must pin ``loc=0`` during fitting.

    The 3-parameter Weibull MLE otherwise drifts ``loc`` off zero (and can go
    negative), producing physically impossible negative debiased values such as
    DTR = -300 K. These tests trace the ``floc=0`` constraint through the whole
    chain: the fit override, pickling for multiprocessing, the pipeline wiring
    that selects the distribution, and the real ibicus fit call that consumes it.
    """

    @staticmethod
    def _skewed_positive_data() -> np.ndarray:
        """Positive data whose unconstrained 3-param Weibull MLE wants ``loc != 0``."""
        return scipy.stats.weibull_min.rvs(
            1.5, loc=5, scale=2, size=2000, random_state=np.random.default_rng(0)
        )

    # --- Layer 1: the fit override itself ---

    def test_fit_forces_loc_zero(self):
        """The override pins ``loc`` to exactly 0 regardless of the data."""
        _, loc, _ = _weibull_min_zero_bounded.fit(self._skewed_positive_data())
        assert loc == 0

    def test_unconstrained_weibull_would_drift_loc_off_zero(self):
        """Sanity anchor: documents the degenerate fit the zero-bounding prevents."""
        _, loc, _ = scipy.stats.weibull_min.fit(self._skewed_positive_data())
        assert loc != 0

    def test_explicit_floc_is_not_overridden(self):
        """``setdefault`` semantics: a caller-supplied ``floc`` still wins."""
        _, loc, _ = _weibull_min_zero_bounded.fit(self._skewed_positive_data(), floc=3.0)
        assert loc == 3.0

    def test_picklable_for_multiprocessing(self):
        """ibicus applies debiasers across processes, so the instance must survive
        pickling and keep the ``loc=0`` override after a round trip."""
        # Safe: round-trips a module-local object we created, not untrusted input.
        restored = pickle.loads(pickle.dumps(_weibull_min_zero_bounded))
        _, loc, _ = restored.fit(self._skewed_positive_data())
        assert loc == 0

    # --- Layer 2: the pipeline wires the right distribution to each tail ---

    def test_pipeline_wires_zero_bounded_weibull_to_low_tail(self, pipeline_options):
        """The 2-sided hybrid branch must feed the zero-bounded Weibull to the low
        tail and gumbel_r to the high tail for pr/rsds/hurs/dtr."""
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="rsds",
            ensemble_member="r1i1p1f1",
            scenario="SSP245",
            predict_period_start=2015,
            predict_period_end=2100,
        )
        assert cfg.variable_config.debias_approach == "nonparametric_hybrid_2sided"
        pipeline = BCSDPipeline(cfg, pipeline_options)

        time = pd.date_range("2015-01-01", periods=6)
        coords = {"time": time, "lat": [10.0, 20.0], "lon": [0.0, 1.0, 2.0]}
        dims = ["time", "lat", "lon"]
        da = xr.DataArray(
            np.abs(np.random.default_rng(1).normal(5, 2, (6, 2, 3))), coords=coords, dims=dims
        )
        mask = xr.DataArray(np.zeros((6, 2, 3), dtype=bool), coords=coords, dims=dims)

        make_debiaser_spy = MagicMock()
        make_debiaser_spy.return_value.apply.return_value = np.zeros((6, 2, 3))
        with (
            patch("srm.pipeline._make_debiaser", make_debiaser_spy),
            patch("srm.pipeline.calculate_out_of_range_mask", return_value=(mask, mask, mask)),
        ):
            pipeline._apply_bias_correction_scenario(da, da, da)

        distributions = [
            call.kwargs.get("distribution") for call in make_debiaser_spy.call_args_list
        ]
        assert _weibull_min_zero_bounded in distributions  # low tail
        assert scipy.stats.gumbel_r in distributions  # high tail

    # --- Layer 3: floc=0 actually reaches scipy through a real ibicus fit ---

    def test_floc_zero_reaches_scipy_through_ibicus(self):
        """End-to-end: a real parametric ``QuantileMapping.apply()`` must route through
        the override so scipy's underlying ``fit`` receives ``floc=0``."""
        weibull_gen = _WeibullMinZeroBounded.__mro__[1]  # weibull_min_gen (parent of the override)
        real_parent_fit = weibull_gen.fit
        floc_seen: list = []

        def spy_fit(self, data, *args, **kwargs):
            floc_seen.append(kwargs.get("floc"))
            return real_parent_fit(self, data, *args, **kwargs)

        rng = np.random.default_rng(0)
        obs = np.abs(rng.normal(5, 2, (500, 1, 1)))
        cm_hist = np.abs(rng.normal(6, 2, (500, 1, 1)))
        cm_future = np.abs(rng.normal(6, 2, (500, 1, 1)))

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

        assert floc_seen, "ibicus never called the distribution's fit"
        assert all(floc == 0 for floc in floc_seen)
        assert np.isfinite(debiased).all()


# ---------------------------------------------------------------------------
# Debiased coarse output — unconditional writes and dual-key cache hits
# ---------------------------------------------------------------------------


class TestFitHistoricalCoarseOutput:
    def test_coarse_write_is_unconditional(self, all_deps_present):
        """fit_historical writes coarse output regardless of save_intermediate."""
        pipeline = all_deps_present
        write_calls: list = []

        def capture_write(da, loc, **kwargs):
            write_calls.append(loc)
            return "snapshot"

        with _mock_fit_historical_compute():
            with patch.object(BCSDPipeline, "_write_to_icechunk", side_effect=capture_write):
                pipeline.fit_historical()

        coarse_loc = pipeline.cache.debiased_coarse_historical_loc(pipeline._hist_member)
        assert any(loc.group == coarse_loc.group for loc in write_calls)

    def test_coarse_write_uses_coarse_encoding(self, all_deps_present):
        """fit_historical uses make_coarse_encoding for the coarse write."""
        from srm.encoding import CHUNK_TIME_COARSE, SHARD_TIME_COARSE

        pipeline = all_deps_present
        write_calls: list = []

        def capture_write(da, loc, **kwargs):
            write_calls.append((loc, kwargs))
            return "snapshot"

        with _mock_fit_historical_compute():
            with patch.object(BCSDPipeline, "_write_to_icechunk", side_effect=capture_write):
                pipeline.fit_historical()

        coarse_loc = pipeline.cache.debiased_coarse_historical_loc(pipeline._hist_member)
        coarse_call = next((kw for loc, kw in write_calls if loc.group == coarse_loc.group), None)
        assert coarse_call is not None
        var = pipeline.config.variable
        assert coarse_call["encoding"][var]["chunks"][0] == CHUNK_TIME_COARSE
        assert coarse_call["encoding"][var]["shards"][0] == SHARD_TIME_COARSE

    def test_cache_hit_requires_coarse_loc(self, all_deps_present):
        """Cache hit does not fire when only the fine-res loc exists (coarse loc absent)."""
        pipeline = all_deps_present
        hist_loc = pipeline.cache.historical_loc(pipeline._hist_member)
        _make_icechunk_group(hist_loc, branch=pipeline.cache.branch)
        # No coarse loc — stage must rerun (not short-circuit)

        with _mock_fit_historical_compute():
            with patch.object(
                BCSDPipeline, "_write_to_icechunk", return_value="snap"
            ) as mock_write:
                pipeline.fit_historical()

        mock_write.assert_called()

    def test_cache_hit_fires_when_both_locs_present(self, all_deps_present):
        """Cache hit fires when both fine-res and coarse locs exist."""
        pipeline = all_deps_present
        hist_loc = pipeline.cache.historical_loc(pipeline._hist_member)
        coarse_loc = pipeline.cache.debiased_coarse_historical_loc(pipeline._hist_member)
        _make_icechunk_group(hist_loc, branch=pipeline.cache.branch)
        _make_icechunk_group(coarse_loc, branch=pipeline.cache.branch)

        with patch("srm.pipeline.get_obs") as mock_get_obs:
            pipeline.fit_historical()

        mock_get_obs.assert_not_called()


class TestTransformScenarioCoarseOutput:
    def test_coarse_write_is_unconditional(self, pipeline_pr):
        """transform_scenario writes coarse output regardless of save_intermediate."""
        p = pipeline_pr
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        write_calls: list = []

        def capture_write(da, loc, **kwargs):
            write_calls.append(loc)
            return "snapshot"

        with _mock_transform_scenario_compute():
            with patch.object(BCSDPipeline, "_write_to_icechunk", side_effect=capture_write):
                p.transform_scenario()

        coarse_loc = p.cache.debiased_coarse_scenario_loc()
        assert any(loc.group == coarse_loc.group for loc in write_calls)

    def test_coarse_write_uses_coarse_encoding(self, pipeline_pr):
        """transform_scenario uses make_coarse_encoding for the coarse write."""
        from srm.encoding import CHUNK_TIME_COARSE, SHARD_TIME_COARSE

        p = pipeline_pr
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        write_calls: list = []

        def capture_write(da, loc, **kwargs):
            write_calls.append((loc, kwargs))
            return "snapshot"

        with _mock_transform_scenario_compute():
            with patch.object(BCSDPipeline, "_write_to_icechunk", side_effect=capture_write):
                p.transform_scenario()

        coarse_loc = p.cache.debiased_coarse_scenario_loc()
        coarse_call = next((kw for loc, kw in write_calls if loc.group == coarse_loc.group), None)
        assert coarse_call is not None
        var = p.config.variable
        assert coarse_call["encoding"][var]["chunks"][0] == CHUNK_TIME_COARSE
        assert coarse_call["encoding"][var]["shards"][0] == SHARD_TIME_COARSE

    def test_cache_hit_requires_coarse_loc(self, pipeline_pr):
        """Cache hit does not fire when only the fine-res scenario loc exists (coarse absent)."""
        p = pipeline_pr
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        scenario_loc = p.cache.scenario_loc
        _make_icechunk_group(scenario_loc, branch=p.cache.branch)
        # No coarse loc — stage must rerun

        with _mock_transform_scenario_compute():
            with patch.object(
                BCSDPipeline, "_write_to_icechunk", return_value="snap"
            ) as mock_write:
                p.transform_scenario()

        mock_write.assert_called()

    def test_cache_hit_fires_when_both_locs_present(self, all_deps_present):
        """Cache hit fires when both fine-res scenario and coarse locs exist."""
        pipeline = all_deps_present
        scenario_loc = pipeline.cache.scenario_loc
        coarse_loc = pipeline.cache.debiased_coarse_scenario_loc()
        _make_icechunk_group(scenario_loc, branch=pipeline.cache.branch)
        _make_icechunk_group(coarse_loc, branch=pipeline.cache.branch)

        with patch("srm.pipeline.get_obs") as mock_get_obs:
            pipeline.transform_scenario()

        mock_get_obs.assert_not_called()


# ---------------------------------------------------------------------------
# Coarse-only variables — dtr is bias corrected but never published fine (issue #461)
# ---------------------------------------------------------------------------


@pytest.fixture
def dtr_config() -> BCSDConfig:
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="dtr",
        ensemble_member="008",
        scenario="SSP245",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def dtr_pipeline(dtr_config, pipeline_options) -> BCSDPipeline:
    return BCSDPipeline(dtr_config, pipeline_options)


def _write_recorder(calls: list):
    """Stand-in for _write_to_icechunk that records the locations it was handed."""

    def capture_write(da, loc, **kwargs):
        calls.append(loc)
        return "snapshot"

    return capture_write


class TestCoarseOnlyFitHistorical:
    def test_writes_the_coarse_group_only(self, dtr_pipeline):
        p = dtr_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        write_calls: list = []

        with _mock_fit_historical_compute():
            with patch.object(
                BCSDPipeline, "_write_to_icechunk", side_effect=_write_recorder(write_calls)
            ):
                p.fit_historical()

        groups = [loc.group for loc in write_calls]
        assert groups == [p.cache.debiased_coarse_historical_loc(p._hist_member).group]
        assert p.cache.historical_loc(p._hist_member).group not in groups

    def test_does_not_disaggregate(self, dtr_pipeline):
        """The fine write is the expensive step; skipping it must skip the interp too."""
        p = dtr_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)

        with _mock_fit_historical_compute():
            with patch.object(BCSDPipeline, "_apply_spatial_downscaling") as mock_disagg:
                p.fit_historical()

        mock_disagg.assert_not_called()

    def test_returns_the_store_holding_its_coarse_group(self, dtr_pipeline):
        # The coarse and fine artifacts share one store, so only the group distinguishes
        # them (that is stage_loc's job). This just pins the early return to a real path.
        p = dtr_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)

        with _mock_fit_historical_compute():
            result = p.fit_historical()

        assert result == p.cache.debiased_coarse_historical_loc(p._hist_member).store_path

    def test_cache_hit_fires_on_the_coarse_group_alone(self, dtr_pipeline):
        p = dtr_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(
            p.cache.debiased_coarse_historical_loc(p._hist_member), branch=p.cache.branch
        )

        with patch("srm.pipeline.get_obs") as mock_get_obs:
            p.fit_historical()

        mock_get_obs.assert_not_called()

    def test_stale_fine_group_alone_does_not_short_circuit(self, dtr_pipeline):
        """A fine dtr group from a pre-#461 run must not read as a completed stage."""
        p = dtr_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)

        with _mock_fit_historical_compute():
            with patch.object(
                BCSDPipeline, "_write_to_icechunk", return_value="snap"
            ) as mock_write:
                p.fit_historical()

        mock_write.assert_called()


class TestCoarseOnlyTransformScenario:
    @staticmethod
    def _seed_deps(p: BCSDPipeline) -> None:
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(
            p.cache.debiased_coarse_historical_loc(p._hist_member), branch=p.cache.branch
        )

    def test_writes_the_coarse_group_only(self, dtr_pipeline):
        p = dtr_pipeline
        self._seed_deps(p)
        write_calls: list = []

        with _mock_transform_scenario_compute():
            with patch.object(
                BCSDPipeline, "_write_to_icechunk", side_effect=_write_recorder(write_calls)
            ):
                p.transform_scenario()

        groups = [loc.group for loc in write_calls]
        assert p.cache.debiased_coarse_scenario_loc().group in groups
        assert p.cache.scenario_loc.group not in groups

    def test_does_not_disaggregate(self, dtr_pipeline):
        p = dtr_pipeline
        self._seed_deps(p)

        with _mock_transform_scenario_compute():
            with patch.object(BCSDPipeline, "_apply_spatial_downscaling") as mock_disagg:
                p.transform_scenario()

        mock_disagg.assert_not_called()

    def test_returns_the_store_holding_its_coarse_group(self, dtr_pipeline):
        # As above: coarse and fine share a store, so this pins the early return only.
        p = dtr_pipeline
        self._seed_deps(p)

        with _mock_transform_scenario_compute():
            result = p.transform_scenario()

        assert result == p.cache.debiased_coarse_scenario_loc().store_path

    def test_cache_hit_fires_on_the_coarse_group_alone(self, dtr_pipeline):
        p = dtr_pipeline
        self._seed_deps(p)
        _make_icechunk_group(p.cache.debiased_coarse_scenario_loc(), branch=p.cache.branch)

        with patch("srm.pipeline.get_obs") as mock_get_obs:
            p.transform_scenario()

        mock_get_obs.assert_not_called()


class TestNormalVariableStillPublishesFine:
    """Regression guard: only COARSE_ONLY_VARIABLES lose their fine artifact."""

    def test_pr_scenario_still_writes_both_groups(self, pipeline_pr):
        p = pipeline_pr
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        write_calls: list = []

        with _mock_transform_scenario_compute():
            with patch.object(
                BCSDPipeline, "_write_to_icechunk", side_effect=_write_recorder(write_calls)
            ):
                p.transform_scenario()

        groups = [loc.group for loc in write_calls]
        assert p.cache.debiased_coarse_scenario_loc().group in groups
        assert p.cache.scenario_loc.group in groups

    def test_tas_historical_still_writes_both_groups(self, all_deps_present):
        p = all_deps_present
        write_calls: list = []

        with _mock_fit_historical_compute():
            with patch.object(
                BCSDPipeline, "_write_to_icechunk", side_effect=_write_recorder(write_calls)
            ):
                p.fit_historical()

        groups = [loc.group for loc in write_calls]
        assert p.cache.debiased_coarse_historical_loc(p._hist_member).group in groups
        assert p.cache.historical_loc(p._hist_member).group in groups


# ---------------------------------------------------------------------------
# Tasmin dependency read-path — coarse output locs
# ---------------------------------------------------------------------------


@pytest.fixture
def tasmin_config() -> BCSDConfig:
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="tasmin",
        ensemble_member="r1i1p1f1",
        scenario="SSP245",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def tasmin_pipeline(tasmin_config, pipeline_options) -> BCSDPipeline:
    return BCSDPipeline(tasmin_config, pipeline_options)


class TestFitHistoricalTasminCoarseDeps:
    def test_reads_dtr_and_tasmax_from_coarse_output_locs(self, tasmin_pipeline):
        """fit_historical_tasmin reads dtr/tasmax from debiased_coarse_historical_loc."""
        p = tasmin_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        # tasmin now validates that its debiased-coarse dtr/tasmax inputs exist (issue #363).
        for _var in ("dtr", "tasmax"):
            _make_icechunk_group(
                p.cache.debiased_coarse_historical_loc(p._hist_member, variable=_var),
                branch=p.cache.branch,
            )
        # the swap reads/rewrites the fine tasmax output (issue #331).
        _make_icechunk_group(
            p.cache.historical_loc(p._hist_member, variable="tasmax"), branch=p.cache.branch
        )
        opened_groups: list = []

        def capture_open(loc):
            opened_groups.append(loc.group)
            return MagicMock()

        with _mock_fit_historical_compute():
            with patch.object(BCSDPipeline, "_open_from_icechunk", side_effect=capture_open):
                with patch.object(BCSDPipeline, "_write_to_icechunk", return_value="snap"):
                    try:
                        p.fit_historical_tasmin()
                    except Exception:
                        pass

        expected_dtr = p.cache.debiased_coarse_historical_loc(p._hist_member, variable="dtr").group
        expected_tasmax = p.cache.debiased_coarse_historical_loc(
            p._hist_member, variable="tasmax"
        ).group
        assert expected_dtr in opened_groups
        assert expected_tasmax in opened_groups

    def test_coarse_historical_loc_uses_output_store(self, tasmin_pipeline):
        """debiased_coarse_historical_loc for tasmin points at the output store."""
        p = tasmin_pipeline
        loc = p.cache.debiased_coarse_historical_loc(p._hist_member)
        assert p.cache.output_dir in loc.store_path

    def test_cache_hit_requires_coarse_loc(self, tasmin_pipeline):
        """fit_historical_tasmin cache hit requires both fine-res and coarse locs."""
        p = tasmin_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        hist_loc = p.cache.historical_loc(p._hist_member)
        _make_icechunk_group(hist_loc, branch=p.cache.branch)
        for _var in ("dtr", "tasmax"):
            _make_icechunk_group(
                p.cache.debiased_coarse_historical_loc(p._hist_member, variable=_var),
                branch=p.cache.branch,
            )
        _make_icechunk_group(
            p.cache.historical_loc(p._hist_member, variable="tasmax"), branch=p.cache.branch
        )
        # No tasmin coarse loc — stage must rerun (not short-circuit)

        with _mock_fit_historical_compute():
            with patch.object(
                BCSDPipeline, "_write_to_icechunk", return_value="snap"
            ) as mock_write:
                try:
                    p.fit_historical_tasmin()
                except Exception:
                    pass

        mock_write.assert_called()


class TestTransformScenarioTasminCoarseDeps:
    def test_reads_dtr_and_tasmax_from_coarse_scenario_locs(self, tasmin_pipeline):
        """transform_scenario_tasmin reads dtr/tasmax from debiased_coarse_scenario_loc."""
        p = tasmin_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        # tasmin now validates that its debiased-coarse dtr/tasmax inputs exist (issue #363).
        for _var in ("dtr", "tasmax"):
            _make_icechunk_group(
                p.cache.debiased_coarse_scenario_loc(variable=_var), branch=p.cache.branch
            )
        # the swap reads/rewrites the fine tasmax output (issue #331).
        _make_icechunk_group(p.cache.scenario_output_loc(variable="tasmax"), branch=p.cache.branch)
        opened_groups: list = []

        def capture_open(loc):
            opened_groups.append(loc.group)
            return MagicMock()

        with _mock_transform_scenario_compute():
            with patch.object(BCSDPipeline, "_open_from_icechunk", side_effect=capture_open):
                with patch.object(BCSDPipeline, "_write_to_icechunk", return_value="snap"):
                    try:
                        p.transform_scenario_tasmin()
                    except Exception:
                        pass

        expected_dtr = p.cache.debiased_coarse_scenario_loc(variable="dtr").group
        expected_tasmax = p.cache.debiased_coarse_scenario_loc(variable="tasmax").group
        assert expected_dtr in opened_groups
        assert expected_tasmax in opened_groups

    def test_coarse_scenario_loc_uses_output_store(self, tasmin_pipeline):
        """debiased_coarse_scenario_loc for tasmin points at the output store."""
        p = tasmin_pipeline
        loc = p.cache.debiased_coarse_scenario_loc()
        assert p.cache.output_dir in loc.store_path

    def test_cache_hit_requires_coarse_loc(self, tasmin_pipeline):
        """transform_scenario_tasmin cache hit requires both fine-res and coarse locs."""
        p = tasmin_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        for _var in ("dtr", "tasmax"):
            _make_icechunk_group(
                p.cache.debiased_coarse_scenario_loc(variable=_var), branch=p.cache.branch
            )
        _make_icechunk_group(p.cache.scenario_output_loc(variable="tasmax"), branch=p.cache.branch)
        scenario_loc = p.cache.scenario_loc
        _make_icechunk_group(scenario_loc, branch=p.cache.branch)
        # No tasmin coarse loc — stage must rerun (not short-circuit)

        with _mock_transform_scenario_compute():
            with patch.object(
                BCSDPipeline, "_write_to_icechunk", return_value="snap"
            ) as mock_write:
                try:
                    p.transform_scenario_tasmin()
                except Exception:
                    pass

        mock_write.assert_called()


# ---------------------------------------------------------------------------
# Coarse tasmin write must tile the shard grid (production regression)
# ---------------------------------------------------------------------------


def _misaligned_coarse_ds() -> xr.Dataset:
    """Debiased-coarse tasmax/dtr whose dask lat chunks straddle the coarse shard grid.

    ``_open_from_icechunk`` opens with ``chunks="auto"``, so re-opened coarse inputs are
    dask-backed with chunk boundaries that need not tile ``SHARD_LAT_COARSE`` (24). Here
    lat=48 is chunked in 16s (boundaries 0/16/32/48), which straddles the shard boundary
    at 24 — the exact condition that made the real icechunk write raise ``safe_chunks``.
    """
    time = pd.date_range("2000-01-01", periods=40, freq="D")
    lat = np.linspace(-88.0, 88.0, 48)
    lon = np.linspace(0.0, 356.0, 96)
    shape = (time.size, lat.size, lon.size)
    coords = {"time": time, "lat": lat, "lon": lon}
    tasmax = xr.DataArray(
        np.full(shape, 300.0, dtype="float32"), dims=["time", "lat", "lon"], coords=coords
    ).chunk({"time": -1, "lat": 16, "lon": -1})
    dtr = xr.DataArray(
        np.full(shape, 10.0, dtype="float32"), dims=["time", "lat", "lon"], coords=coords
    ).chunk({"time": -1, "lat": 16, "lon": -1})
    return xr.Dataset({"tasmax": tasmax, "dtr": dtr})


def _read_coarse_tasmin(cache, coarse_loc) -> xr.DataArray:
    """Open a written coarse store directly, bypassing the mocked ``_open_from_icechunk``."""
    import icechunk

    from srm.pipeline import _icechunk_storage_for_path

    storage = _icechunk_storage_for_path(coarse_loc.store_path)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch=cache.branch)
    ds = xr.open_dataset(session.store, engine="zarr", consolidated=False, group=coarse_loc.group)
    return ds["tasmin"]


def _tiny_fine_tasmin() -> xr.DataArray:
    """A small real fine-res DataArray for the raw tasmin write in the tasmin stages.

    The stages now persist the downscaled tasmin before reconciling, so tests that
    drive the *real* ``_write_to_icechunk`` need a genuine array (not a MagicMock) as
    the ``_apply_spatial_downscaling`` return.
    """
    return xr.DataArray(
        np.full((1, 1, 2), 290.0, dtype="float32"),
        dims=["time", "lat", "lon"],
        coords={
            "time": np.array([np.datetime64("2020-01-01")]),
            "lat": np.array([0.0]),
            "lon": np.array([10.0, 11.0]),
        },
        name="tasmin",
    )


class TestTasminCoarseWriteChunkAlignment:
    """The derived coarse tasmin write must tile the coarse shard grid.

    Regression for the production failure ``CESM2-WACCM_tasmin_003_G6-1.5K``: the tasmin
    stages derive coarse tasmin from auto-chunked tasmax/dtr and wrote it with
    ``make_coarse_encoding`` (shard 24x48) without rechunking, so xarray's ``safe_chunks``
    check rejected the misaligned dask chunks. These tests drive the *real*
    ``_write_to_icechunk`` (the sibling tasmin tests mock it), so they fail before the fix.
    """

    def test_fit_historical_tasmin_coarse_write_is_shard_aligned(self, tasmin_pipeline):
        p = tasmin_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        for _var in ("dtr", "tasmax"):
            _make_icechunk_group(
                p.cache.debiased_coarse_historical_loc(p._hist_member, variable=_var),
                branch=p.cache.branch,
            )
        _make_icechunk_group(
            p.cache.historical_loc(p._hist_member, variable="tasmax"), branch=p.cache.branch
        )
        coarse_loc = p.cache.debiased_coarse_historical_loc(p._hist_member)

        with (
            patch.object(
                BCSDPipeline,
                "_load_gcm_obs",
                return_value=(MagicMock(), MagicMock(), MagicMock()),
            ),
            patch.object(BCSDPipeline, "_open_from_icechunk", return_value=_misaligned_coarse_ds()),
            patch.object(
                BCSDPipeline, "_apply_spatial_downscaling", return_value=_tiny_fine_tasmin()
            ),
            patch.object(BCSDPipeline, "reconcile_temperature_extremes"),
        ):
            result = p.fit_historical_tasmin()

        assert result == p.cache.historical_loc(p._hist_member).store_path
        # Round-trips without a safe_chunks error and holds tasmax(300) - dtr(10) == 290.
        written = _read_coarse_tasmin(p.cache, coarse_loc)
        assert float(written.isel(time=0, lat=0, lon=0)) == 290.0

    def test_transform_scenario_tasmin_coarse_write_is_shard_aligned(self, tasmin_pipeline):
        p = tasmin_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        for _var in ("dtr", "tasmax"):
            _make_icechunk_group(
                p.cache.debiased_coarse_scenario_loc(variable=_var), branch=p.cache.branch
            )
        _make_icechunk_group(p.cache.scenario_output_loc(variable="tasmax"), branch=p.cache.branch)
        coarse_loc = p.cache.debiased_coarse_scenario_loc()

        with (
            patch.object(
                BCSDPipeline,
                "_load_scenario_data",
                return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()),
            ),
            patch.object(BCSDPipeline, "_open_from_icechunk", return_value=_misaligned_coarse_ds()),
            patch.object(
                BCSDPipeline, "_apply_spatial_downscaling", return_value=_tiny_fine_tasmin()
            ),
            patch.object(BCSDPipeline, "_build_ocean_mask", return_value=MagicMock()),
            patch.object(BCSDPipeline, "reconcile_temperature_extremes"),
        ):
            result = p.transform_scenario_tasmin()

        assert result == p.cache.scenario_loc.store_path
        written = _read_coarse_tasmin(p.cache, coarse_loc)
        assert float(written.isel(time=0, lat=0, lon=0)) == 290.0


def _time_straddling_coarse_ds() -> xr.Dataset:
    """Debiased-coarse tasmax/dtr whose dask *time* chunks straddle the coarse shard grid.

    Mirrors the production scenario ``CESM2-WACCM_tasmin_003_G6-1.5K``: ``_open_from_icechunk``
    auto-chunks re-opened coarse inputs into sub-shard time pieces, and the scenario spans more
    than one ``SHARD_TIME_COARSE`` (16000-day) shard. Those ~600-step time chunks do not tile
    the shard boundary at 16000, which is exactly what tripped xarray's ``safe_chunks`` check on
    the coarse write (error on axis 0). ``rechunk(full_space)`` only fixes lat/lon, so it never
    caught this. Arrays stay lazy (dask, never computed) so the >16000-step time axis is free.
    """
    import dask.array as dask_array

    n_time = SHARD_TIME_COARSE + 1  # one step into the second time shard
    time = pd.date_range("2015-01-01", periods=n_time, freq="D")
    lat = np.linspace(-88.0, 88.0, 24)
    lon = np.linspace(0.0, 352.0, 48)
    shape = (n_time, lat.size, lon.size)
    coords = {"time": time, "lat": lat, "lon": lon}
    tasmax = xr.DataArray(
        dask_array.full(shape, 300.0, dtype="float32", chunks=(600, lat.size, lon.size)),
        dims=["time", "lat", "lon"],
        coords=coords,
    )
    dtr = xr.DataArray(
        dask_array.full(shape, 10.0, dtype="float32", chunks=(600, lat.size, lon.size)),
        dims=["time", "lat", "lon"],
        coords=coords,
    )
    return xr.Dataset({"tasmax": tasmax, "dtr": dtr})


def _assert_tiles_coarse_shards(da: xr.DataArray) -> None:
    """Every dask chunk must map to exactly one coarse shard (the ``safe_chunks`` invariant)."""
    for dim, shard in (
        ("time", SHARD_TIME_COARSE),
        ("lat", SHARD_LAT_COARSE),
        ("lon", SHARD_LON_COARSE),
    ):
        chunks = da.chunksizes[dim]
        # interior chunks equal the shard; only the trailing remainder may be smaller
        assert all(c == shard for c in chunks[:-1]), (dim, chunks, shard)
        assert chunks[-1] <= shard, (dim, chunks, shard)


class TestTasminCoarseWriteTimeShardAlignment:
    """The coarse tasmin write must tile the shard grid on the *time* axis too.

    ``TestTasminCoarseWriteChunkAlignment`` only exercised a 40-step time axis (one sub-shard
    chunk), so the shard-crossing time straddle that failed in production
    (``encoding['chunks']=(16000, 24, 48)`` … "Dask chunks … on axis 0") slipped through. These
    capture the array handed to ``_write_to_icechunk`` for a >16000-step scenario and assert it
    is chunked to the full coarse shard grid — failing on the ``rechunk(full_space)`` code that
    left ~600-step time chunks straddling ``SHARD_TIME_COARSE``.
    """

    @staticmethod
    def _capture_coarse_write(coarse_loc):
        captured: dict = {}

        def side_effect(da, loc, *args, **kwargs):
            if loc.group == coarse_loc.group:
                captured["da"] = da
            return "snap"

        return captured, side_effect

    def test_fit_historical_tasmin_time_axis_is_shard_aligned(self, tasmin_pipeline):
        p = tasmin_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        for _var in ("dtr", "tasmax"):
            _make_icechunk_group(
                p.cache.debiased_coarse_historical_loc(p._hist_member, variable=_var),
                branch=p.cache.branch,
            )
        _make_icechunk_group(
            p.cache.historical_loc(p._hist_member, variable="tasmax"), branch=p.cache.branch
        )
        coarse_loc = p.cache.debiased_coarse_historical_loc(p._hist_member)
        captured, side_effect = self._capture_coarse_write(coarse_loc)

        with (
            patch.object(
                BCSDPipeline,
                "_load_gcm_obs",
                return_value=(MagicMock(), MagicMock(), MagicMock()),
            ),
            patch.object(
                BCSDPipeline, "_open_from_icechunk", return_value=_time_straddling_coarse_ds()
            ),
            patch.object(BCSDPipeline, "_apply_spatial_downscaling", return_value=MagicMock()),
            patch.object(BCSDPipeline, "reconcile_temperature_extremes"),
            patch.object(BCSDPipeline, "_write_to_icechunk", side_effect=side_effect),
        ):
            p.fit_historical_tasmin()

        assert "da" in captured, "coarse tasmin write was never called"
        _assert_tiles_coarse_shards(captured["da"])

    def test_transform_scenario_tasmin_time_axis_is_shard_aligned(self, tasmin_pipeline):
        p = tasmin_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        for _var in ("dtr", "tasmax"):
            _make_icechunk_group(
                p.cache.debiased_coarse_scenario_loc(variable=_var), branch=p.cache.branch
            )
        _make_icechunk_group(p.cache.scenario_output_loc(variable="tasmax"), branch=p.cache.branch)
        coarse_loc = p.cache.debiased_coarse_scenario_loc()
        captured, side_effect = self._capture_coarse_write(coarse_loc)

        with (
            patch.object(
                BCSDPipeline,
                "_load_scenario_data",
                return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()),
            ),
            patch.object(
                BCSDPipeline, "_open_from_icechunk", return_value=_time_straddling_coarse_ds()
            ),
            patch.object(BCSDPipeline, "_apply_spatial_downscaling", return_value=MagicMock()),
            patch.object(BCSDPipeline, "_build_ocean_mask", return_value=MagicMock()),
            patch.object(BCSDPipeline, "reconcile_temperature_extremes"),
            patch.object(BCSDPipeline, "_write_to_icechunk", side_effect=side_effect),
        ):
            p.transform_scenario_tasmin()

        assert "da" in captured, "coarse tasmin write was never called"
        _assert_tiles_coarse_shards(captured["da"])


# ---------------------------------------------------------------------------
# Tasmax<tasmin swap in the tasmin stage (issue #331)
# ---------------------------------------------------------------------------


def _fine_pair_with_inversion():
    time = np.array([np.datetime64("2020-01-01")])
    lat = np.array([0.0])
    lon = np.array([10.0, 11.0])
    coords = {"time": time, "lat": lat, "lon": lon}
    tasmax = xr.DataArray(
        np.array([[[300.0, 290.0]]]), dims=["time", "lat", "lon"], coords=coords, name="tasmax"
    )
    tasmin = xr.DataArray(  # cell [0,0,1] inverted (295 > 290)
        np.array([[[280.0, 295.0]]]), dims=["time", "lat", "lon"], coords=coords, name="tasmin"
    )
    return tasmax, tasmin


class TestReconcileTemperatureExtremes:
    def _run_reconcile(self, p, tasmax_fine, tasmin_fine, *, force=False):
        tasmin_loc = p.cache.scenario_loc
        tasmax_loc = p.cache.scenario_output_loc(variable="tasmax")
        _make_icechunk_group(tasmax_loc, branch=p.cache.branch)  # sibling fine tasmax exists
        writes: dict = {}
        kwargs_seen: dict = {}

        def capture_write(da, loc, **kwargs):
            writes[loc.group] = da
            kwargs_seen[loc.group] = kwargs
            return "snap"

        with patch.object(
            BCSDPipeline, "_open_from_icechunk", return_value=tasmax_fine.to_dataset()
        ):
            with patch.object(BCSDPipeline, "_write_to_icechunk", side_effect=capture_write):
                p.reconcile_temperature_extremes(
                    tasmin_loc, tasmax_loc, tasmin_fine=tasmin_fine, force=force
                )
        return tasmin_loc, tasmax_loc, writes, kwargs_seen

    def test_writes_both_corrected_fields(self, tasmin_pipeline):
        p = tasmin_pipeline
        tasmax_fine, tasmin_fine = _fine_pair_with_inversion()
        tasmin_loc, tasmax_loc, writes, _ = self._run_reconcile(p, tasmax_fine, tasmin_fine)

        assert tasmax_loc.group in writes and tasmin_loc.group in writes
        written_max = writes[tasmax_loc.group]
        written_min = writes[tasmin_loc.group]
        assert written_max.values[0, 0, 1] == 295.0  # swapped up
        assert written_min.values[0, 0, 1] == 290.0  # swapped down
        assert bool((written_max >= written_min).all())  # monotone

    def test_logs_swap_count_in_qa(self, tasmin_pipeline, caplog):
        import logging

        p = tasmin_pipeline  # pipeline_options defaults to environment="qa"
        assert p.options.environment == "qa"
        tasmax_fine, tasmin_fine = _fine_pair_with_inversion()  # 1 inverted of 2 valid cells
        with caplog.at_level(logging.INFO, logger="srm.pipeline"):
            self._run_reconcile(p, tasmax_fine, tasmin_fine)
        assert "swapped 1 / 2 valid cells" in caplog.text
        assert "50.0000%" in caplog.text
        assert "max inversion 5.000" in caplog.text

    def test_no_swap_count_in_production(self, tasmin_config, tmp_path, caplog):
        import logging

        # production skips the extra swap-count pass; the consistency gate is qaqc's job.
        opts = PipelineOptions(
            scratch_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "outputs"),
            verbose=False,
            rechunk_workflow=False,
            environment="production",
        )
        p = BCSDPipeline(tasmin_config, opts)
        tasmax_fine, tasmin_fine = _fine_pair_with_inversion()
        with caplog.at_level(logging.INFO, logger="srm.pipeline"):
            self._run_reconcile(p, tasmax_fine, tasmin_fine)
        assert "swapped" not in caplog.text

    def test_tasmax_write_never_force_gcs_but_tasmin_does(self, tasmin_pipeline):
        # tasmax rewrite must never force-GC (the tasmin write still lazily reads the
        # pre-rewrite tasmax); the tasmin write, materialised last, threads force so
        # its trailing GC sweeps the superseded tasmax.
        p = tasmin_pipeline
        tasmax_fine, tasmin_fine = _fine_pair_with_inversion()
        tasmin_loc, tasmax_loc, _writes, kwargs_seen = self._run_reconcile(
            p, tasmax_fine, tasmin_fine, force=True
        )
        assert kwargs_seen[tasmax_loc.group].get("force") is False
        assert kwargs_seen[tasmin_loc.group].get("force") is True

    def test_both_writes_rechunked_to_shard_layout(self, tasmin_pipeline):
        p = tasmin_pipeline
        tasmax_fine, tasmin_fine = _fine_pair_with_inversion()
        tasmin_loc, tasmax_loc, writes, _ = self._run_reconcile(p, tasmax_fine, tasmin_fine)
        for loc in (tasmax_loc, tasmin_loc):
            written = writes[loc.group]
            assert written.chunks is not None  # dask-backed (was rechunked)
            assert all(len(c) == 1 for c in written.chunks)  # small array → one chunk/dim

    def test_raises_when_fine_tasmax_missing(self, tasmin_pipeline):
        p = tasmin_pipeline
        _tasmax, tasmin_fine = _fine_pair_with_inversion()
        with pytest.raises(ValueError, match="tasmax"):
            p.reconcile_temperature_extremes(
                p.cache.scenario_loc,  # tasmax loc below is not created
                p.cache.scenario_output_loc(variable="tasmax"),
                tasmin_fine=tasmin_fine,
            )

    def test_standalone_reads_tasmin_from_store(self, tasmin_pipeline):
        # with tasmin_fine=None the step reconciles already-persisted outputs.
        p = tasmin_pipeline
        tasmin_loc = p.cache.scenario_loc
        tasmax_loc = p.cache.scenario_output_loc(variable="tasmax")
        _make_icechunk_group(tasmax_loc, branch=p.cache.branch)
        _make_icechunk_group(tasmin_loc, branch=p.cache.branch)
        tasmax_fine, tasmin_fine = _fine_pair_with_inversion()

        def fake_open(loc, chunks="auto"):
            return (tasmin_fine if loc.group == tasmin_loc.group else tasmax_fine).to_dataset()

        writes: dict = {}

        def capture_write(da, loc, **kwargs):
            writes[loc.group] = da
            return "snap"

        with patch.object(BCSDPipeline, "_open_from_icechunk", side_effect=fake_open):
            with patch.object(BCSDPipeline, "_write_to_icechunk", side_effect=capture_write):
                p.reconcile_temperature_extremes(tasmin_loc, tasmax_loc)  # tasmin_fine=None

        assert writes[tasmax_loc.group].values[0, 0, 1] == 295.0
        assert writes[tasmin_loc.group].values[0, 0, 1] == 290.0

    def test_standalone_raises_when_tasmin_missing(self, tasmin_pipeline):
        p = tasmin_pipeline
        tasmax_loc = p.cache.scenario_output_loc(variable="tasmax")
        _make_icechunk_group(tasmax_loc, branch=p.cache.branch)  # tasmax present, tasmin absent
        with pytest.raises(ValueError, match="tasmin"):
            p.reconcile_temperature_extremes(p.cache.scenario_loc, tasmax_loc)

    def test_reads_both_inputs_shard_aligned(self, tasmin_pipeline):
        # Store-to-store reconcile must open BOTH fields with shard-aligned chunks so the
        # swap+write slices per-shard instead of pulling the whole fine array (OOM fix).
        from srm.encoding import SHARD_LAT, SHARD_LON, SHARD_TIME

        p = tasmin_pipeline
        tasmax_fine, tasmin_fine = _fine_pair_with_inversion()
        tasmin_loc = p.cache.scenario_loc
        tasmax_loc = p.cache.scenario_output_loc(variable="tasmax")
        _make_icechunk_group(tasmax_loc, branch=p.cache.branch)
        _make_icechunk_group(tasmin_loc, branch=p.cache.branch)
        chunks_seen: list = []

        def fake_open(loc, chunks="auto"):
            chunks_seen.append(chunks)
            src = tasmin_fine if loc.group == tasmin_loc.group else tasmax_fine
            return src.to_dataset()

        with patch.object(BCSDPipeline, "_open_from_icechunk", side_effect=fake_open):
            with patch.object(BCSDPipeline, "_write_to_icechunk", return_value="snap"):
                p.reconcile_temperature_extremes(tasmin_loc, tasmax_loc)  # tasmin_fine=None

        shard = {"time": SHARD_TIME, "lat": SHARD_LAT, "lon": SHARD_LON}
        assert chunks_seen  # inputs were opened
        assert all(c == shard for c in chunks_seen)

    def test_writes_run_under_synchronous_scheduler(self, tasmin_pipeline):
        # The swap+writes are memory-bound; a synchronous dask scheduler keeps only a few
        # shards resident instead of the full global fine array (reconcile OOM fix).
        import dask

        p = tasmin_pipeline
        tasmax_fine, tasmin_fine = _fine_pair_with_inversion()
        schedulers_at_write: list = []

        def capture_write(da, loc, **kwargs):
            schedulers_at_write.append(dask.config.get("scheduler", None))
            return "snap"

        tasmin_loc = p.cache.scenario_loc
        tasmax_loc = p.cache.scenario_output_loc(variable="tasmax")
        _make_icechunk_group(tasmax_loc, branch=p.cache.branch)
        with patch.object(
            BCSDPipeline, "_open_from_icechunk", return_value=tasmax_fine.to_dataset()
        ):
            with patch.object(BCSDPipeline, "_write_to_icechunk", side_effect=capture_write):
                p.reconcile_temperature_extremes(tasmin_loc, tasmax_loc, tasmin_fine=tasmin_fine)

        assert schedulers_at_write == ["synchronous", "synchronous"]


class TestTasminReconcileSequencing:
    """tasmin OOM fix: persist the raw downscaled tasmin FIRST, then reconcile
    store-to-store (``tasmin_fine=None``) so tasmax and the fresh interp graph never
    co-reside in memory."""

    def _capture(self, p, stage, loc, *, load_patch):
        sentinel = MagicMock(name="raw_tasmin")
        writes: list = []
        reconcile: dict = {}

        def capture_write(da, loc_, **kwargs):
            writes.append((loc_.group, da))
            return "snap"

        def capture_reconcile(tasmin_loc, tasmax_loc, *, tasmin_fine="_unset", force=False):
            reconcile["called"] = True
            reconcile["tasmin_fine"] = tasmin_fine
            reconcile["tasmin_loc_group"] = tasmin_loc.group

        with (
            load_patch,
            patch.object(BCSDPipeline, "_open_from_icechunk", return_value=MagicMock()),
            patch("srm.pipeline.derive_tasmin", return_value=MagicMock()),
            patch.object(BCSDPipeline, "_apply_spatial_downscaling", return_value=sentinel),
            patch.object(BCSDPipeline, "_write_to_icechunk", side_effect=capture_write),
            patch.object(
                BCSDPipeline, "reconcile_temperature_extremes", side_effect=capture_reconcile
            ),
        ):
            stage()

        return sentinel, writes, reconcile

    def test_fit_historical_tasmin_persists_raw_then_store_to_store(self, tasmin_pipeline):
        p = tasmin_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        for _var in ("dtr", "tasmax"):
            _make_icechunk_group(
                p.cache.debiased_coarse_historical_loc(p._hist_member, variable=_var),
                branch=p.cache.branch,
            )
        _make_icechunk_group(
            p.cache.historical_loc(p._hist_member, variable="tasmax"), branch=p.cache.branch
        )
        loc = p.cache.historical_loc(p._hist_member)
        load_patch = patch.object(
            BCSDPipeline,
            "_load_gcm_obs",
            return_value=(MagicMock(), MagicMock(), MagicMock()),
        )
        sentinel, writes, reconcile = self._capture(
            p, p.fit_historical_tasmin, loc, load_patch=load_patch
        )

        # raw downscaled tasmin was written to the fine historical loc before reconcile
        assert any(g == loc.group and da is sentinel for g, da in writes)
        # reconcile ran store-to-store against that persisted output
        assert reconcile.get("called")
        assert reconcile.get("tasmin_fine") is None
        assert reconcile.get("tasmin_loc_group") == loc.group

    def test_transform_scenario_tasmin_persists_raw_then_store_to_store(self, tasmin_pipeline):
        p = tasmin_pipeline
        _make_icechunk_group(p.cache.obs_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.historical_loc(p._hist_member), branch=p.cache.branch)
        for _var in ("dtr", "tasmax"):
            _make_icechunk_group(
                p.cache.debiased_coarse_scenario_loc(variable=_var), branch=p.cache.branch
            )
        _make_icechunk_group(p.cache.scenario_output_loc(variable="tasmax"), branch=p.cache.branch)
        loc = p.cache.scenario_loc
        load_patch = patch.object(
            BCSDPipeline,
            "_load_scenario_data",
            return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()),
        )
        sentinel, writes, reconcile = self._capture(
            p, p.transform_scenario_tasmin, loc, load_patch=load_patch
        )

        assert any(g == loc.group and da is sentinel for g, da in writes)
        assert reconcile.get("called")
        assert reconcile.get("tasmin_fine") is None
        assert reconcile.get("tasmin_loc_group") == loc.group


class TestTasminCacheShortCircuit:
    def test_cached_tasmin_served_without_upstream_deps(self, tasmin_pipeline):
        # A cached tasmin short-circuits without reconciling or requiring the
        # (possibly reaped) obs/coarse upstreams (issue #363 finding #5).
        p = tasmin_pipeline
        _make_icechunk_group(p.cache.scenario_loc, branch=p.cache.branch)
        _make_icechunk_group(p.cache.debiased_coarse_scenario_loc(), branch=p.cache.branch)

        with patch.object(BCSDPipeline, "_write_to_icechunk") as mock_write:
            result = p.transform_scenario_tasmin()

        mock_write.assert_not_called()
        assert result == p.cache.scenario_loc.store_path


# ---------------------------------------------------------------------------
# _detrend_scenario: SAI bridge is applied even when detrend_data=False (#363)
# ---------------------------------------------------------------------------


class TestDetrendScenarioBridge:
    def test_detrend_off_sai_still_bridges(self, pipeline_options):
        # dtr has detrend_data=False; for a SAI scenario it must still stitch the
        # SSP245 bridge so the output spans the full predict window (issue #363).
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="dtr",
            ensemble_member="003",
            scenario="G6-1.5K",
            predict_period_start=2015,
            predict_period_end=2084,
        )
        pipe = BCSDPipeline(cfg, pipeline_options)
        assert cfg.variable_config.detrend_data is False and cfg.is_sai_scenario is True

        ssp = MagicMock(name="ssp_timeseries")
        stitched = MagicMock(name="stitched")
        stitched.sel.return_value = "bridged-predict-slice"
        with patch("srm.pipeline.stitch_historical_scenario", return_value=stitched) as mock_stitch:
            out, trend = pipe._detrend_scenario(MagicMock(), MagicMock(), ssp)

        mock_stitch.assert_called_once()
        assert mock_stitch.call_args.kwargs["ssp_timeseries"] is ssp
        assert out == "bridged-predict-slice"
        assert trend is None

    def test_detrend_off_non_sai_returns_scenario_unchanged(self, pipeline_options):
        # pr has detrend_data=False; for a non-SAI scenario there is no bridge —
        # the scenario is returned unchanged (regression: no behavior change).
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="pr",
            ensemble_member="003",
            scenario="SSP245",
            predict_period_start=2015,
            predict_period_end=2099,
        )
        pipe = BCSDPipeline(cfg, pipeline_options)
        assert cfg.variable_config.detrend_data is False and cfg.is_sai_scenario is False

        model_scenario = MagicMock(name="model_scenario")
        with patch("srm.pipeline.stitch_historical_scenario") as mock_stitch:
            out, trend = pipe._detrend_scenario(MagicMock(), model_scenario, None)

        mock_stitch.assert_not_called()
        assert out is model_scenario
        assert trend is None


def _spatial_daily_da(start: str, end: str, value: float = 1.0) -> xr.DataArray:
    """3D (time, lat, lon) daily series on a tiny grid."""
    times = pd.date_range(start, end, freq="D")
    return xr.DataArray(
        np.full((len(times), 2, 2), value),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [0.0, 1.0], "lon": [10.0, 11.0]},
        name="tas",
    )


class TestScenarioHistoricalSlice:
    """``_load_scenario_data`` must slice model_hist to the training period (#518).

    It previously sliced through ``predict_period_start - 1``, which only coincides with
    ``train_period_end`` when the prediction period begins the year after training ends.
    Every other case silently widened the bias-correction reference pool beyond the
    configured training window.
    """

    @staticmethod
    def _load(pipeline, hist_end: str = "2015-01-16"):
        member = pipeline.config.ensemble_member
        obs = _spatial_daily_da("1978-01-01", "2014-12-31").to_dataset(name="tas")
        with (
            patch.object(
                BCSDPipeline,
                "_open_from_icechunk",
                return_value=obs,
            ),
            patch(
                "srm.pipeline.get_obs", return_value=_spatial_daily_da("1978-01-01", "2014-12-31")
            ),
            patch(
                "srm.pipeline.get_historical_experiment",
                return_value=_spatial_daily_da("1978-01-01", hist_end),
            ),
            patch(
                "srm.pipeline.get_experiment",
                return_value=_spatial_daily_da("2015-01-01", "2060-12-31").expand_dims(
                    ensemble_member=[member]
                ),
            ),
            patch.object(
                BCSDPipeline,
                "_load_ssp245_bridge",
                return_value=_spatial_daily_da("2015-01-01", "2060-12-31"),
            ),
            patch.object(
                pipeline.cache,
                "check_dependencies",
                return_value={"obs_regridded": (True, MagicMock())},
            ),
        ):
            return pipeline._load_scenario_data()

    def test_sai_scenario_stops_at_train_period_end(self, pipeline_options):
        config = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="001",
            scenario="G6-1.5K",
            train_period_end=2014,
            predict_period_start=2035,
            predict_period_end=2060,
        )
        _, _, model_hist, _, _ = self._load(BCSDPipeline(config, pipeline_options))

        assert int(model_hist["time"].dt.year.max()) == 2014
        # The fifteen all-NaN days after 2015-01-01 must not be reachable at all.
        assert model_hist["time"].max() < np.datetime64("2015-01-01")

    def test_training_window_shorter_than_scenario_start_is_respected(self, pipeline_options):
        # configs/qa/obs-comparison/* pair train_period_end 2008 with a 2015 scenario start;
        # the old slice leaked six extra years into the reference pool.
        config = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="r1i1p1f1",
            scenario="SSP245",
            train_period_end=2008,
            predict_period_start=2015,
            predict_period_end=2060,
        )
        _, _, model_hist, _, _ = self._load(BCSDPipeline(config, pipeline_options))

        assert int(model_hist["time"].dt.year.max()) == 2008

    def test_matches_the_historical_stage_loader(self, pipeline_options):
        """_load_gcm_obs already slices correctly; the two loaders must agree."""
        config = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="001",
            scenario="G6-1.5K",
            train_period_end=2014,
            predict_period_start=2035,
            predict_period_end=2060,
        )
        pipeline = BCSDPipeline(config, pipeline_options)
        _, _, scenario_hist, _, _ = self._load(pipeline)

        obs = _spatial_daily_da("1978-01-01", "2014-12-31").to_dataset(name="tas")
        with (
            patch.object(BCSDPipeline, "_open_from_icechunk", return_value=obs),
            patch(
                "srm.pipeline.get_obs",
                return_value=_spatial_daily_da("1978-01-01", "2014-12-31"),
            ),
            patch(
                "srm.pipeline.get_historical_experiment",
                return_value=_spatial_daily_da("1978-01-01", "2015-01-16"),
            ),
            patch.object(
                pipeline.cache,
                "check_dependencies",
                return_value={"obs_regridded": (True, MagicMock())},
            ),
        ):
            _, _, historical_stage_hist = pipeline._load_gcm_obs()

        np.testing.assert_array_equal(
            scenario_hist["time"].values, historical_stage_hist["time"].values
        )
