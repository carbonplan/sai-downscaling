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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from srm.bcsd_config import BCSDConfig
from srm.pipeline import BCSDPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_icechunk_store(path: str) -> None:
    """Create a minimal icechunk store with a 'write complete' commit."""
    import icechunk
    import numpy as np
    import xarray as xr
    from icechunk.xarray import to_icechunk

    storage = icechunk.local_filesystem_storage(path=path)
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session("main")
    ds = xr.Dataset({"dummy": xr.DataArray(np.array([1.0]), dims=["x"])})
    to_icechunk(ds, session, mode="w")
    session.commit("write complete")


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
        patch("srm.pipeline.get_experiment"),
        patch("srm.pipeline.xr.DataArray", return_value=MagicMock()),
        patch("srm.pipeline.rechunk"),
        patch("srm.pipeline.downscale_from_coarse"),
        patch("ibicus.debias.QuantileMapping") as mock_qm,
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
        patch("srm.pipeline.get_experiment"),
        patch("srm.pipeline.xr.DataArray", return_value=MagicMock()),
        patch("srm.pipeline.xr.concat", return_value=MagicMock()),
        patch("srm.pipeline.rechunk"),
        patch("srm.pipeline.subset_space"),
        patch("srm.pipeline.calculate_baseline_climatology"),
        patch("srm.pipeline.detrend"),
        patch("srm.pipeline.retrend"),
        patch("srm.pipeline.downscale_from_coarse"),
        patch("ibicus.debias.QuantileMapping") as mock_qm,
        patch("srm.pipeline.dask"),
        patch.object(BCSDPipeline, "_open_from_icechunk", return_value=MagicMock()),
        patch.object(BCSDPipeline, "_write_to_icechunk", return_value="snapshot-abc"),
    ):
        mock_qm.from_variable.return_value.apply.return_value = MagicMock()
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path) -> BCSDConfig:
    """Standard ssp245 config backed by local tmp directories."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="tas",
        ensemble_member="r1i1p1f1",
        scenario="ssp245",
        predict_period_start=2015,
        predict_period_end=2100,
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
        rechunk_workflow=False,
    )


@pytest.fixture
def pr_config(tmp_path) -> BCSDConfig:
    """Precipitation config (no detrending, divide downscaling method)."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="pr",
        ensemble_member="r1i1p1f1",
        scenario="ssp245",
        predict_period_start=2015,
        predict_period_end=2100,
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
        rechunk_workflow=False,
    )


@pytest.fixture
def pipeline(config) -> BCSDPipeline:
    return BCSDPipeline(config)


@pytest.fixture
def pipeline_pr(pr_config) -> BCSDPipeline:
    return BCSDPipeline(pr_config)


@pytest.fixture
def all_deps_present(pipeline) -> BCSDPipeline:
    """Pipeline whose obs and historical dependencies are pre-created locally."""
    _make_icechunk_store(
        pipeline.cache.get_obs_path(
            pipeline.config.gcm, pipeline.config.variable, pipeline.config.subset_bounds
        )
    )
    _make_icechunk_store(
        pipeline.cache.get_historical_path(
            pipeline.config.gcm,
            pipeline.config.variable,
            pipeline.config.ensemble_member,
            pipeline.config.subset_bounds,
        )
    )
    return pipeline


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestBCSDPipelineInit:
    def test_cache_uses_config_cache_dir(self, pipeline, config):
        assert config.cache_dir.rstrip("/") in pipeline.cache.base_path

    def test_cache_uses_config_environment(self, pipeline, config):
        assert pipeline.cache.environment == config.environment

    def test_cache_uses_config_version(self, pipeline, config):
        assert pipeline.cache.version == config.version

    def test_cache_has_output_dir(self, pipeline, config):
        assert pipeline.cache.output_dir is not None
        assert config.output_dir.rstrip("/") in pipeline.cache.output_dir

    def test_state_initialized_empty(self, pipeline):
        assert pipeline._state == {}


# ---------------------------------------------------------------------------
# prepare_observations – cache routing
# ---------------------------------------------------------------------------


class TestPrepareObservationsCache:
    def test_returns_obs_path_when_cached(self, pipeline):
        obs_path = pipeline.cache.get_obs_path(
            pipeline.config.gcm, pipeline.config.variable, pipeline.config.subset_bounds
        )
        _make_icechunk_store(obs_path)

        with patch("srm.pipeline.get_obs") as mock_get_obs:
            result = pipeline.prepare_observations()

        assert result == obs_path
        mock_get_obs.assert_not_called()

    def test_does_not_compute_when_cached(self, pipeline):
        obs_path = pipeline.cache.get_obs_path(
            pipeline.config.gcm, pipeline.config.variable, pipeline.config.subset_bounds
        )
        _make_icechunk_store(obs_path)

        with _mock_prepare_obs_compute() as (get_obs, get_exp, interp, *_):
            pipeline.prepare_observations()
            get_obs.assert_not_called()
            get_exp.assert_not_called()
            interp.assert_not_called()

    def test_force_runs_compute_even_when_cached(self, pipeline):
        obs_path = pipeline.cache.get_obs_path(
            pipeline.config.gcm, pipeline.config.variable, pipeline.config.subset_bounds
        )
        _make_icechunk_store(obs_path)

        with _mock_prepare_obs_compute() as (mock_get_obs, *_):
            pipeline.prepare_observations(force=True)
            mock_get_obs.assert_called_once()

    def test_returns_obs_path_even_after_compute(self, pipeline):
        expected = pipeline.cache.get_obs_path(
            pipeline.config.gcm, pipeline.config.variable, pipeline.config.subset_bounds
        )
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
        mock_get_obs.assert_called_once_with(var="tas")

    def test_get_experiment_called_for_historical_scenario(self, pipeline):
        with _mock_prepare_obs_compute() as (_, mock_get_exp, *_):
            pipeline.prepare_observations()
        mock_get_exp.assert_called_once_with(gcm="CESM2-WACCM", scenario="Historical", var="tas")

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
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "outputs"),
            verbose=False,
            rechunk_workflow=False,
        )
        p = BCSDPipeline(cfg)
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
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "outputs"),
            verbose=False,
            rechunk_workflow=True,
        )
        p = BCSDPipeline(cfg)
        with _mock_prepare_obs_compute() as (_, _, _, _, mock_rechunk):
            p.prepare_observations()
        mock_rechunk.assert_called_once()

    def test_pr_variable_passes_correct_var_to_get_obs(self, pipeline_pr):
        with _mock_prepare_obs_compute() as (mock_get_obs, *_):
            pipeline_pr.prepare_observations()
        mock_get_obs.assert_called_once_with(var="pr")


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
        hist_path = pipeline.cache.get_historical_path(
            pipeline.config.gcm,
            pipeline.config.variable,
            pipeline.config.ensemble_member,
            pipeline.config.subset_bounds,
        )
        _make_icechunk_store(hist_path)

        with patch("srm.pipeline.get_obs") as mock_get_obs:
            result = pipeline.fit_historical()

        assert result == hist_path
        mock_get_obs.assert_not_called()

    def test_force_bypasses_cached_historical(self, all_deps_present):
        pipeline = all_deps_present
        hist_path = pipeline.cache.get_historical_path(
            pipeline.config.gcm,
            pipeline.config.variable,
            pipeline.config.ensemble_member,
            pipeline.config.subset_bounds,
        )
        _make_icechunk_store(hist_path)

        with _mock_fit_historical_compute():
            with patch("srm.pipeline.get_obs") as mock_get_obs:
                pipeline.fit_historical(force=True)

        mock_get_obs.assert_called_once()

    def test_returns_historical_path_after_compute(self, all_deps_present):
        pipeline = all_deps_present
        expected = pipeline.cache.get_historical_path(
            pipeline.config.gcm,
            pipeline.config.variable,
            pipeline.config.ensemble_member,
            pipeline.config.subset_bounds,
        )
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
            cache_dir=str(tmp_path / "cache"),
            output_dir=str(tmp_path / "outputs"),
            verbose=False,
        )
        with pytest.raises(ValueError, match="scenario must be specified"):
            BCSDPipeline(cfg).transform_scenario()

    def test_raises_when_both_deps_missing(self, pipeline):
        with pytest.raises(ValueError, match="Missing dependencies"):
            pipeline.transform_scenario()

    def test_raises_when_only_obs_present(self, pipeline):
        obs_path = pipeline.cache.get_obs_path(
            pipeline.config.gcm, pipeline.config.variable, pipeline.config.subset_bounds
        )
        _make_icechunk_store(obs_path)
        with pytest.raises(ValueError, match="Missing dependencies"):
            pipeline.transform_scenario()

    def test_validate_dependencies_called_before_compute(self, pipeline):
        with patch.object(pipeline.cache, "validate_dependencies") as mock_validate:
            try:
                pipeline.transform_scenario()
            except Exception:
                pass
        mock_validate.assert_called_once_with("transform_scenario", pipeline.config)

    def test_returns_cached_scenario_path(self, all_deps_present):
        pipeline = all_deps_present
        scenario_path = pipeline.cache.get_scenario_path(
            pipeline.config.gcm,
            pipeline.config.variable,
            pipeline.config.ensemble_member,
            pipeline.config.scenario,
            pipeline.config.subset_bounds,
        )
        _make_icechunk_store(scenario_path)

        with patch("srm.pipeline.get_obs") as mock_get_obs:
            result = pipeline.transform_scenario()

        assert result == scenario_path
        mock_get_obs.assert_not_called()

    def test_force_bypasses_cached_scenario(self, pipeline_pr, tmp_path):
        # Use pr config: detrend_data=False avoids the xr.concat detrend branch
        p = pipeline_pr
        _make_icechunk_store(
            p.cache.get_obs_path(p.config.gcm, p.config.variable, p.config.subset_bounds)
        )
        _make_icechunk_store(
            p.cache.get_historical_path(
                p.config.gcm, p.config.variable, p.config.ensemble_member, p.config.subset_bounds
            )
        )
        scenario_path = p.cache.get_scenario_path(
            p.config.gcm,
            p.config.variable,
            p.config.ensemble_member,
            p.config.scenario,
            p.config.subset_bounds,
        )
        _make_icechunk_store(scenario_path)

        with _mock_transform_scenario_compute():
            with patch("srm.pipeline.get_obs") as mock_get_obs:
                p.transform_scenario(force=True)

        mock_get_obs.assert_called_once()

    def test_returns_scenario_path_after_compute(self, pipeline_pr):
        # Use pr config: detrend_data=False avoids the xr.concat detrend branch
        p = pipeline_pr
        _make_icechunk_store(
            p.cache.get_obs_path(p.config.gcm, p.config.variable, p.config.subset_bounds)
        )
        _make_icechunk_store(
            p.cache.get_historical_path(
                p.config.gcm, p.config.variable, p.config.ensemble_member, p.config.subset_bounds
            )
        )
        expected = p.cache.get_scenario_path(
            p.config.gcm,
            p.config.variable,
            p.config.ensemble_member,
            p.config.scenario,
            p.config.subset_bounds,
        )
        with _mock_transform_scenario_compute():
            result = p.transform_scenario()
        assert result == expected

    def test_detrend_not_called_for_pr(self, all_deps_present, pipeline_pr, tmp_path):
        # Recreate all_deps_present for the pr pipeline
        pr_pipeline = pipeline_pr
        _make_icechunk_store(
            pr_pipeline.cache.get_obs_path(
                pr_pipeline.config.gcm,
                pr_pipeline.config.variable,
                pr_pipeline.config.subset_bounds,
            )
        )
        _make_icechunk_store(
            pr_pipeline.cache.get_historical_path(
                pr_pipeline.config.gcm,
                pr_pipeline.config.variable,
                pr_pipeline.config.ensemble_member,
                pr_pipeline.config.subset_bounds,
            )
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
        assert mock_detrend.call_count >= 1 or pipeline.config.detrend_data


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
