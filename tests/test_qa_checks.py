"""Tests for the hard NaN assertions used during pipeline execution (#517)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions
from saidownscale.pipeline import DownscalingPipeline
from saidownscale.qa_checks import NaNCheckError, assert_no_nans


def _daily_da(values: np.ndarray, start: str = "2015-01-01") -> xr.DataArray:
    n_time, n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.date_range(start, periods=n_time, freq="D"),
            "lat": np.linspace(-10.0, 10.0, n_lat),
            "lon": np.linspace(20.0, 40.0, n_lon),
        },
    )


def _with_nans(shape, *cells) -> np.ndarray:
    values = np.ones(shape)
    for cell in cells:
        values[cell] = np.nan
    return values


def _edge_band(n_time: int, *interior) -> tuple[xr.DataArray, dict]:
    """NaN first-lat band excluded by a (lat, lon) `where` mask, plus optional interior NaNs."""
    da = _daily_da(_with_nans((n_time, 3, 4), (slice(None), 0, slice(None)), *interior))
    mask = xr.DataArray(
        np.array([[False] * 4, [True] * 4, [True] * 4]),
        dims=["lat", "lon"],
        coords={"lat": da["lat"], "lon": da["lon"]},
    )
    return da, {"where": mask}


def _full_shape_mask_case():
    mask = np.ones((200, 3, 4), dtype=bool)
    mask[10, 0, 0] = False
    return _daily_da(_with_nans((200, 3, 4), (10, 0, 0), (150, 0, 0))), {"where": mask}


def _transposed_mask_case():
    da = xr.DataArray(
        _with_nans((4, 3, 5), (2, 1, 4)),
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.date_range("2015-01-01", periods=4, freq="D"),
            "lat": [0.0, 1.0, 2.0],
            "lon": [10.0, 11.0, 12.0, 13.0, 14.0],
        },
    )
    return da, {"where": xr.DataArray(np.ones((5, 3), dtype=bool), dims=["lon", "lat"])}


def _non_trailing_mask_case():
    da = xr.DataArray(
        _with_nans((3, 3, 3), (1, 2, 0)),
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.date_range("2015-01-01", periods=3, freq="D"),
            "lat": [0.0, 1.0, 2.0],
            "lon": [10.0, 11.0, 12.0],
        },
    )
    return da, {"where": xr.DataArray(np.ones((3, 3), dtype=bool), dims=["time", "lat"])}


def test_clean_or_masked_arrays_pass(subtests):
    cases = {
        "numpy": lambda: (np.ones((4, 3)), {}),
        "dataarray": lambda: (_daily_da(np.ones((5, 3, 4))), {}),
        "masked_edge_band": lambda: _edge_band(4),
        "masked_edge_band_many_slabs": lambda: _edge_band(200),
    }
    for case, build in cases.items():
        with subtests.test(case=case):
            data, kwargs = build()
            assert_no_nans(data, name="obs", **kwargs)


def test_nans_raise_with_diagnostic_message(subtests):
    cases = {
        "counts_nans": (lambda: (_with_nans((4, 3), (2, 1), (3, 0)), {}), r"x.*2 NaN"),
        "all_nan": (lambda: (np.full((3, 3), np.nan), {}), "100"),
        "timestamps": (lambda: (_daily_da(_with_nans((5, 3, 4), 1)), {}), "2015-01-02"),
        "context": (
            lambda: (np.full((2, 2), np.nan), {"context": {"gcm": "CESM2-WACCM6"}}),
            "gcm=CESM2-WACCM6",
        ),
        "dask": (
            lambda: (_daily_da(_with_nans((5, 3, 4), (0, 0, 0))).chunk({"time": 2}), {}),
            "1 NaN",
        ),
        "masked_interior": (
            lambda: _edge_band(4, (2, 1, 1)),
            r"1 NaN",
        ),
        "slab_offset": (
            lambda: (_daily_da(_with_nans((200, 3, 4), (150, 1, 1))), {}),
            "2015-05-31",
        ),
        "multi_slab_in_order": (
            lambda: (_daily_da(_with_nans((200, 3, 4), (10, 0, 0), (70, 0, 0), (199, 0, 0))), {}),
            r"(?s)3 NaN.*2015-01-11, 2015-03-12, 2015-07-19",
        ),
        "checked_total_counts_skipped_slabs": (
            lambda: (_daily_da(_with_nans((200, 5, 5), (150, 2, 2))), {}),
            r"0\.02% of 5000 checked cells",
        ),
        "masked_later_slab": (
            lambda: _edge_band(200, (150, 1, 1)),
            r"1 NaN",
        ),
        "full_shape_mask_per_slab": (_full_shape_mask_case, r"1 NaN"),
        "transposed_mask_realigned": (_transposed_mask_case, r"1 NaN"),
    }
    for case, (build, match) in cases.items():
        with subtests.test(case=case):
            data, kwargs = build()
            with pytest.raises(NaNCheckError, match=match):
                assert_no_nans(data, name="x", **kwargs)


def test_misaligned_masks_are_rejected(subtests):
    cases = {
        "non_trailing_dims": (_non_trailing_mask_case, "trailing"),
        "unknown_dim": (
            lambda: (
                _daily_da(np.ones((4, 3, 4))),
                {"where": xr.DataArray(np.ones((3, 4), dtype=bool), dims=["y", "x"])},
            ),
            "dimensions",
        ),
        "ambiguous_numpy_shape": (
            lambda: (np.ones((100, 4, 4)), {"where": np.ones((1, 4, 4), dtype=bool)}),
            "shape",
        ),
    }
    for case, (build, match) in cases.items():
        with subtests.test(case=case):
            data, kwargs = build()
            with pytest.raises(ValueError, match=match):
                assert_no_nans(data, name="residuals_fine", **kwargs)


@pytest.fixture
def pipeline_options(tmp_path) -> PipelineOptions:
    return PipelineOptions(
        scratch_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
        rechunk_workflow=False,
    )


@pytest.fixture
def historical_pipeline(pipeline_options) -> DownscalingPipeline:
    config = DownscalingConfig(
        downscaling_method="BCSD", gcm="CESM2-WACCM6", variable="tas", ensemble_member="r1i1p1f1"
    )
    return DownscalingPipeline(config, pipeline_options)


@pytest.fixture
def scenario_pipeline(pipeline_options) -> DownscalingPipeline:
    config = DownscalingConfig(
        gcm="CESM2-WACCM6",
        downscaling_method="BCSD",
        variable="tas",
        ensemble_member="001",
        scenario="G6-1.5K",
        predict_period_start=2035,
        predict_period_end=2040,
    )
    return DownscalingPipeline(config, pipeline_options)


def test_nan_input_aborts_bias_correction(historical_pipeline, scenario_pipeline, subtests):
    """Debiaser inputs must be NaN-free; the all-NaN day mirrors #514/#518."""
    clean = _daily_da(np.ones((10, 3, 4)))
    nan_day = _daily_da(_with_nans((10, 3, 4), 4))
    nan_cell = _daily_da(_with_nans((10, 3, 4), (slice(None), 0, 0)))
    scen_clean = _daily_da(np.ones((10, 3, 4)), start="2035-01-01")
    scen_nan = _daily_da(_with_nans((10, 3, 4), 7), start="2035-01-01")
    cases = {
        "hist_cm_hist": (
            lambda: historical_pipeline._apply_bias_correction(clean, nan_day),
            "cm_hist",
        ),
        "hist_obs": (lambda: historical_pipeline._apply_bias_correction(nan_cell, clean), "obs"),
        "scen_cm_hist": (
            lambda: scenario_pipeline._apply_bias_correction_scenario(clean, nan_day, scen_clean),
            "cm_hist",
        ),
        "scen_cm_future": (
            lambda: scenario_pipeline._apply_bias_correction_scenario(clean, clean, scen_nan),
            "cm_future",
        ),
    }
    for case, (call, match) in cases.items():
        with subtests.test(case=case):
            with pytest.raises(NaNCheckError, match=match):
                call()


def test_debiaser_output_is_gated_for_failsafe_nans(historical_pipeline):
    """ibicus failsafe=True writes NaN for cells it cannot debias, so the output needs a gate."""
    debiaser = MagicMock()
    clean = _daily_da(np.ones((10, 3, 4)))
    with patch("saidownscale.pipeline._make_debiaser", return_value=debiaser):
        debiaser.apply.return_value = np.ones((10, 3, 4))
        result = historical_pipeline._apply_bias_correction(clean, clean)
        assert result.dims == ("time", "lat", "lon")
        assert not bool(result.isnull().any())

        debiaser.apply.return_value = _with_nans((10, 3, 4), (2, 1, 1))
        with pytest.raises(NaNCheckError, match="debiased_coarse"):
            historical_pipeline._apply_bias_correction(clean, clean)
