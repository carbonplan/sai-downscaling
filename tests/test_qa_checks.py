"""Tests for the hard NaN assertions used during pipeline execution (issue #517)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.pipeline import BCSDPipeline
from srm.qa_checks import NaNCheckError, assert_no_nans


def _daily_da(values: np.ndarray, start: str = "2015-01-01") -> xr.DataArray:
    """3D (time, lat, lon) DataArray with real daily timestamps."""
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


class TestAssertNoNans:
    def test_clean_numpy_array_passes(self):
        assert_no_nans(np.ones((4, 3)), name="obs")

    def test_clean_dataarray_passes(self):
        assert_no_nans(_daily_da(np.ones((5, 3, 4))), name="obs")

    def test_raises_on_nan_in_numpy_array(self):
        arr = np.ones((4, 3))
        arr[2, 1] = np.nan

        with pytest.raises(NaNCheckError):
            assert_no_nans(arr, name="cm_hist")

    def test_error_names_the_array_and_counts_the_nans(self):
        arr = np.ones((4, 3))
        arr[2, 1] = np.nan
        arr[3, 0] = np.nan

        with pytest.raises(NaNCheckError, match=r"cm_hist.*2 NaN"):
            assert_no_nans(arr, name="cm_hist")

    def test_error_reports_offending_timestamps(self):
        values = np.ones((5, 3, 4))
        values[1] = np.nan
        da = _daily_da(values)

        with pytest.raises(NaNCheckError, match="2015-01-02"):
            assert_no_nans(da, name="cm_hist")

    def test_error_includes_run_context(self):
        arr = np.full((2, 2), np.nan)

        with pytest.raises(NaNCheckError, match="gcm=CESM2-WACCM"):
            assert_no_nans(arr, name="obs", context={"gcm": "CESM2-WACCM", "variable": "pr"})

    def test_raises_on_dask_backed_array(self):
        values = np.ones((5, 3, 4))
        values[0, 0, 0] = np.nan
        da = _daily_da(values).chunk({"time": 2})

        with pytest.raises(NaNCheckError, match="residuals_fine"):
            assert_no_nans(da, name="residuals_fine")

    def test_where_mask_excludes_nans_outside_it(self):
        values = np.ones((4, 3, 4))
        values[:, 0, :] = np.nan  # a NaN edge band
        da = _daily_da(values)
        mask = xr.DataArray(
            np.array([[False] * 4, [True] * 4, [True] * 4]),
            dims=["lat", "lon"],
            coords={"lat": da["lat"], "lon": da["lon"]},
        )

        assert_no_nans(da, name="residuals_fine", where=mask)

    def test_where_mask_still_catches_nans_inside_it(self):
        values = np.ones((4, 3, 4))
        values[:, 0, :] = np.nan  # excluded edge band
        values[2, 1, 1] = np.nan  # interior NaN — must be caught
        da = _daily_da(values)
        mask = xr.DataArray(
            np.array([[False] * 4, [True] * 4, [True] * 4]),
            dims=["lat", "lon"],
            coords={"lat": da["lat"], "lon": da["lon"]},
        )

        with pytest.raises(NaNCheckError, match=r"1 NaN"):
            assert_no_nans(da, name="residuals_fine", where=mask)

    def test_all_nan_array_raises(self):
        with pytest.raises(NaNCheckError, match="100"):
            assert_no_nans(np.full((3, 3), np.nan), name="obs")


@pytest.fixture
def pipeline_options(tmp_path) -> PipelineOptions:
    return PipelineOptions(
        scratch_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
        rechunk_workflow=False,
    )


@pytest.fixture
def historical_pipeline(pipeline_options) -> BCSDPipeline:
    config = BCSDConfig(
        downscaling_method="BCSD", gcm="CESM2-WACCM", variable="tas", ensemble_member="r1i1p1f1"
    )
    return BCSDPipeline(config, pipeline_options)


@pytest.fixture
def scenario_pipeline(pipeline_options) -> BCSDPipeline:
    config = BCSDConfig(
        gcm="CESM2-WACCM",
        downscaling_method="BCSD",
        variable="tas",
        ensemble_member="001",
        scenario="G6-1.5K",
        predict_period_start=2035,
        predict_period_end=2040,
    )
    return BCSDPipeline(config, pipeline_options)


class TestBiasCorrectionInputChecks:
    """The debiaser must never be handed NaN input (issue #517).

    The all-NaN day mirrors a real defect in the input data. The CESM2-WACCM historical
    store ends at 2015-01-16 with fifteen fully-NaN days, and every g6 config slices
    historical through ``predict_period_start - 1 == 2034``, so those days reach
    ``cm_hist`` (issues #514 and #518).
    """

    def test_nan_in_model_hist_aborts_historical_bias_correction(self, historical_pipeline):
        obs_coarse = _daily_da(np.ones((10, 3, 4)))
        values = np.ones((10, 3, 4))
        values[4] = np.nan
        model_hist = _daily_da(values)

        with pytest.raises(NaNCheckError, match="cm_hist"):
            historical_pipeline._apply_bias_correction(obs_coarse, model_hist)

    def test_nan_in_obs_aborts_historical_bias_correction(self, historical_pipeline):
        values = np.ones((10, 3, 4))
        values[:, 0, 0] = np.nan
        obs_coarse = _daily_da(values)
        model_hist = _daily_da(np.ones((10, 3, 4)))

        with pytest.raises(NaNCheckError, match="obs"):
            historical_pipeline._apply_bias_correction(obs_coarse, model_hist)

    def test_nan_in_model_hist_aborts_scenario_bias_correction(self, scenario_pipeline):
        obs_coarse = _daily_da(np.ones((10, 3, 4)))
        values = np.ones((10, 3, 4))
        values[4] = np.nan
        model_hist = _daily_da(values)
        scenario_detrended = _daily_da(np.ones((10, 3, 4)), start="2035-01-01")

        with pytest.raises(NaNCheckError, match="cm_hist"):
            scenario_pipeline._apply_bias_correction_scenario(
                obs_coarse, model_hist, scenario_detrended
            )

    def test_nan_in_scenario_aborts_scenario_bias_correction(self, scenario_pipeline):
        obs_coarse = _daily_da(np.ones((10, 3, 4)))
        model_hist = _daily_da(np.ones((10, 3, 4)))
        values = np.ones((10, 3, 4))
        values[7] = np.nan
        scenario_detrended = _daily_da(values, start="2035-01-01")

        with pytest.raises(NaNCheckError, match="cm_future"):
            scenario_pipeline._apply_bias_correction_scenario(
                obs_coarse, model_hist, scenario_detrended
            )


class TestSlabBoundaries:
    """The scan walks the leading axis in ``_NAN_SCAN_BLOCK``-sized slabs.

    Production arrays span hundreds of slabs and clean ones are skipped without being
    reduced, so the per-slab bookkeeping (global offender offsets, and the checked-cell
    total that the reported percentage divides by) has to survive that skip. Every other
    test here fits inside a single slab and would not notice if it did not.
    """

    def test_offender_index_is_offset_by_the_slab_it_falls_in(self):
        # 200 steps spans four slabs at block size 64; day 150 lands in the fourth,
        # preceded by clean slabs that the scan skips over.
        values = np.ones((200, 3, 4))
        values[150, 1, 1] = np.nan

        with pytest.raises(NaNCheckError, match="2015-05-31"):
            assert_no_nans(_daily_da(values), name="residuals_fine")

    def test_offenders_span_multiple_slabs_in_order(self):
        values = np.ones((200, 3, 4))
        values[10, 0, 0] = np.nan  # first slab
        values[70, 0, 0] = np.nan  # second
        values[199, 0, 0] = np.nan  # last, a short slab

        with pytest.raises(NaNCheckError) as excinfo:
            assert_no_nans(_daily_da(values), name="residuals_fine")

        message = str(excinfo.value)
        assert "3 NaN" in message
        assert "2015-01-11, 2015-03-12, 2015-07-19" in message

    def test_checked_cell_total_includes_skipped_clean_slabs(self):
        # 200 * 5 * 5 = 5000 cells; a single NaN is 0.02%. If clean slabs were skipped
        # before their cells were counted, the denominator would collapse to one slab.
        values = np.ones((200, 5, 5))
        values[150, 2, 2] = np.nan

        with pytest.raises(NaNCheckError, match=r"0\.02% of 5000 checked cells"):
            assert_no_nans(_daily_da(values), name="residuals_fine")

    def test_masked_out_nans_pass_across_many_slabs(self):
        values = np.ones((200, 3, 4))
        values[:, 0, :] = np.nan  # excluded edge band, present in every slab
        da = _daily_da(values)
        mask = xr.DataArray(
            np.array([[False] * 4, [True] * 4, [True] * 4]),
            dims=["lat", "lon"],
            coords={"lat": da["lat"], "lon": da["lon"]},
        )

        assert_no_nans(da, name="residuals_fine", where=mask)

    def test_masked_scan_still_catches_a_nan_in_a_later_slab(self):
        values = np.ones((200, 3, 4))
        values[:, 0, :] = np.nan  # excluded edge band
        values[150, 1, 1] = np.nan  # interior NaN in the fourth slab
        da = _daily_da(values)
        mask = xr.DataArray(
            np.array([[False] * 4, [True] * 4, [True] * 4]),
            dims=["lat", "lon"],
            coords={"lat": da["lat"], "lon": da["lon"]},
        )

        with pytest.raises(NaNCheckError, match=r"1 NaN"):
            assert_no_nans(da, name="residuals_fine", where=mask)

    def test_full_shape_mask_tracks_its_own_slab(self):
        # A full-shape mask is sliced per slab rather than broadcast; an excluded cell
        # in one slab must not excuse the same cell in another.
        values = np.ones((200, 3, 4))
        values[10, 0, 0] = np.nan
        values[150, 0, 0] = np.nan
        mask = np.ones((200, 3, 4), dtype=bool)
        mask[10, 0, 0] = False  # excuse the first, not the second

        with pytest.raises(NaNCheckError, match=r"1 NaN"):
            assert_no_nans(_daily_da(values), name="residuals_fine", where=mask)


class TestMaskAlignmentGuards:
    """A ``where`` mask must never silently check the wrong cells.

    The mask is reduced to a bare numpy array before scanning, so shape alone cannot
    distinguish a (lat, lon) mask from a (time, lat) one when the sizes coincide.
    Alignment is therefore validated against dimension names up front.
    """

    def test_mask_over_non_trailing_dims_is_rejected(self):
        values = np.ones((3, 3, 3))
        values[1, 2, 0] = np.nan
        da = xr.DataArray(
            values,
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2015-01-01", periods=3, freq="D"),
                "lat": [0.0, 1.0, 2.0],
                "lon": [10.0, 11.0, 12.0],
            },
        )
        # Same shape as a (lat, lon) mask, but over the wrong axes.
        mask = xr.DataArray(np.ones((3, 3), dtype=bool), dims=["time", "lat"])

        with pytest.raises(ValueError, match="trailing"):
            assert_no_nans(da, name="residuals_fine", where=mask)

    def test_mask_with_unknown_dim_is_rejected(self):
        da = _daily_da(np.ones((4, 3, 4)))
        mask = xr.DataArray(np.ones((3, 4), dtype=bool), dims=["y", "x"])

        with pytest.raises(ValueError, match="dimensions"):
            assert_no_nans(da, name="residuals_fine", where=mask)

    def test_transposed_mask_is_realigned_not_misapplied(self):
        values = np.ones((4, 3, 5))
        values[2, 1, 4] = np.nan
        da = xr.DataArray(
            values,
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2015-01-01", periods=4, freq="D"),
                "lat": [0.0, 1.0, 2.0],
                "lon": [10.0, 11.0, 12.0, 13.0, 14.0],
            },
        )
        # Author-order (lon, lat) rather than (lat, lon); must be transposed, not misread.
        mask = xr.DataArray(np.ones((5, 3), dtype=bool), dims=["lon", "lat"])

        with pytest.raises(NaNCheckError, match=r"1 NaN"):
            assert_no_nans(da, name="residuals_fine", where=mask)

    def test_ambiguous_numpy_mask_shape_is_rejected(self):
        # Same ndim as the data but not equal to it, and not trailing-aligned.
        with pytest.raises(ValueError, match="shape"):
            assert_no_nans(
                np.ones((100, 4, 4)), name="residuals_fine", where=np.ones((1, 4, 4), dtype=bool)
            )


class TestDebiaserOutputCheck:
    """``failsafe=True`` lets ibicus write NaN for any cell it cannot debias.

    Clean inputs are therefore not sufficient to guarantee a clean result, and the
    debiased coarse array is written to the store before disaggregation would catch
    it. The output needs its own gate.
    """

    def test_failsafe_nan_in_historical_output_aborts(self, historical_pipeline):
        failsafe_output = np.ones((10, 3, 4))
        failsafe_output[2, 1, 1] = np.nan  # what failsafe=True writes for a failed cell
        debiaser = MagicMock()
        debiaser.apply.return_value = failsafe_output

        with patch("srm.pipeline._make_debiaser", return_value=debiaser):
            with pytest.raises(NaNCheckError, match="debiased_coarse"):
                historical_pipeline._apply_bias_correction(
                    _daily_da(np.ones((10, 3, 4))), _daily_da(np.ones((10, 3, 4)))
                )

    def test_clean_debiaser_output_is_not_aborted(self, historical_pipeline):
        debiaser = MagicMock()
        debiaser.apply.return_value = np.ones((10, 3, 4))

        with patch("srm.pipeline._make_debiaser", return_value=debiaser):
            result = historical_pipeline._apply_bias_correction(
                _daily_da(np.ones((10, 3, 4))), _daily_da(np.ones((10, 3, 4)))
            )

        assert result.dims == ("time", "lat", "lon")
        assert not bool(result.isnull().any())
