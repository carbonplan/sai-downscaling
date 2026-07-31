from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from srm import downscaling_utils
from srm.downscaling_utils import (
    calculate_baseline_climatology,
    coarse_domain_mask,
    derive_tasmin,
    detrend,
    downscale_from_coarse,
    fft_smooth_3harmonics,
    get_historical_experiment,
    interpolate_coarse_to_fine_grid,
    rechunk,
    retrend,
    subset_space,
    swap_temperature_extremes,
)
from srm.qa_checks import NaNCheckError


@pytest.fixture
def sample_da() -> xr.DataArray:
    lat = np.array([0.0, 1.0, 2.0, 3.0])
    lon = np.array([10.0, 11.0, 12.0, 13.0])
    data = np.arange(lat.size * lon.size).reshape(lat.size, lon.size)
    return xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})


@pytest.fixture
def sample_3d_da() -> xr.DataArray:
    time = np.arange(10)
    lat = np.array([0.0, 1.0, 2.0, 3.0])
    lon = np.array([10.0, 11.0, 12.0, 13.0])
    data = np.arange(time.size * lat.size * lon.size).reshape(time.size, lat.size, lon.size)
    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_subset_space_legacy_bounds_order_is_lat_then_lon(sample_da):
    result = subset_space(sample_da, coord_bounds_list=[1.0, 2.0, 11.0, 12.0])

    np.testing.assert_array_equal(result["lat"].values, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(result["lon"].values, np.array([11.0, 12.0]))
    assert result.shape == (2, 2)


def test_subset_space_named_bounds(sample_da):
    result = subset_space(sample_da, lat_bounds=(1.0, 2.0), lon_bounds=(11.0, 12.0))

    np.testing.assert_array_equal(result["lat"].values, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(result["lon"].values, np.array([11.0, 12.0]))
    assert result.shape == (2, 2)


def test_subset_space_raises_for_short_legacy_bounds(sample_da):
    with pytest.raises(ValueError, match="must contain four values"):
        subset_space(sample_da, coord_bounds_list=[1.0, 2.0, 11.0])


def test_subset_space_raises_for_invalid_lat_bound_order(sample_da):
    with pytest.raises(ValueError, match="lat_bounds must be"):
        subset_space(sample_da, lat_bounds=(2.0, 1.0), lon_bounds=(11.0, 12.0))


def test_subset_space_raises_for_invalid_lon_bound_order(sample_da):
    with pytest.raises(ValueError, match="lon_bounds must be"):
        subset_space(sample_da, lat_bounds=(1.0, 2.0), lon_bounds=(12.0, 11.0))


def test_subset_space_raises_when_mixing_legacy_and_named_bounds(sample_da):
    with pytest.raises(ValueError, match="not both"):
        subset_space(
            sample_da,
            coord_bounds_list=[1.0, 2.0, 11.0, 12.0],
            lat_bounds=(1.0, 2.0),
            lon_bounds=(11.0, 12.0),
        )


def test_rechunk_full_space_chunks_over_time_only(sample_3d_da, monkeypatch):
    monkeypatch.setattr(downscaling_utils, "_TARGET_CHUNK_BYTES", 128)

    result = rechunk(sample_3d_da, pattern="full_space")

    assert len(result.chunksizes["time"]) > 1
    assert len(result.chunksizes["lat"]) == 1
    assert len(result.chunksizes["lon"]) == 1


def test_rechunk_full_time_chunks_over_space_only(sample_3d_da, monkeypatch):
    monkeypatch.setattr(downscaling_utils, "_TARGET_CHUNK_BYTES", 128)

    result = rechunk(sample_3d_da, pattern="full_time")

    assert len(result.chunksizes["time"]) == 1
    assert len(result.chunksizes["lat"]) > 1
    assert len(result.chunksizes["lon"]) > 1


def test_rechunk_full_space_noop_when_already_chunked(sample_3d_da):
    already_chunked = sample_3d_da.chunk({"time": 1, "lat": -1, "lon": -1})

    result = rechunk(already_chunked, pattern="full_space")

    assert result is already_chunked


def test_rechunk_full_time_noop_when_already_chunked(sample_3d_da):
    already_chunked = sample_3d_da.chunk({"time": -1, "lat": 1, "lon": 1})

    result = rechunk(already_chunked, pattern="full_time")

    assert result is already_chunked


def test_rechunk_full_space_forces_rechunk_when_last_chunk_larger_than_first():
    """Regression: concat-inherited chunks (e.g. ssp-bridge 7305 + g6 18250) must be
    rechunked even though lat/lon are full-extent, because zarr requires last ≤ first."""
    time_ssp = pd.date_range("2015-01-01", "2034-12-31", freq="D")
    time_g6 = pd.date_range("2035-01-01", "2084-12-31", freq="D")
    lat = np.array([0.0, 1.0])
    lon = np.array([10.0, 11.0])

    def _make(times):
        return xr.DataArray(
            np.ones((len(times), 2, 2), dtype=np.float32),
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": lat, "lon": lon},
        ).chunk({"time": -1, "lat": -1, "lon": -1})

    # Simulate what concat produces: last chunk (g6) is larger than first (ssp)
    concat_da = xr.concat([_make(time_ssp), _make(time_g6)], dim="time")
    assert concat_da.chunksizes["time"][-1] > concat_da.chunksizes["time"][0]

    result = rechunk(concat_da, pattern="full_space")

    # After rechunking, last chunk must be ≤ first (zarr-valid)
    result_chunks = result.chunksizes["time"]
    assert result_chunks[-1] <= result_chunks[0]


def test_calculate_baseline_climatology_preserves_input_dtype():
    time = np.arange(np.datetime64("2000-01-01"), np.datetime64("2002-01-01"))
    lat = np.array([0.0, 1.0])
    lon = np.array([10.0, 11.0])
    data = np.arange(time.size * lat.size * lon.size, dtype=np.float32).reshape(
        time.size, lat.size, lon.size
    )
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )

    clim = calculate_baseline_climatology(
        da,
        baseline_period_start=2000,
        baseline_period_end=2001,
    )

    assert clim.dtype == da.dtype
    np.testing.assert_array_equal(np.sort(clim["month"].values), np.arange(1, 13))


def _make_constant_daily_da(value: float, dtype: np.dtype = np.float32) -> xr.DataArray:
    # Use a decade so each calendar month group has enough points for the
    # 9-year rolling window used in detrend().
    time = np.arange(np.datetime64("2001-01-01"), np.datetime64("2011-01-01"))
    lat = np.array([0.0, 1.0])
    lon = np.array([10.0, 11.0])
    data = np.full((time.size, lat.size, lon.size), value, dtype=dtype)
    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )


def _make_baseline_clim(value: float, lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    months = np.arange(1, 13)
    data = np.full((months.size, lat.size, lon.size), value, dtype=np.float32)
    return xr.DataArray(
        data,
        dims=["month", "lat", "lon"],
        coords={"month": months, "lat": lat, "lon": lon},
    )


def test_detrend_additive_returns_expected_values_and_alignment():
    da = _make_constant_daily_da(10.0, dtype=np.float32)
    baseline = _make_baseline_clim(0.0, lat=da["lat"].values, lon=da["lon"].values)

    detrended, trend_on_daily_timestep = detrend(
        da=da,
        da_baseline_clim=baseline,
        detrend_method="additive",
    )

    np.testing.assert_array_equal(trend_on_daily_timestep["time"].values, da["time"].values)
    assert trend_on_daily_timestep.shape == da.shape
    assert detrended.shape == da.shape
    assert trend_on_daily_timestep.dtype == da.dtype
    assert detrended.dtype == da.dtype
    np.testing.assert_allclose(trend_on_daily_timestep.values, 10.0)
    np.testing.assert_allclose(detrended.values, 0.0)


def test_detrend_multiplicative_returns_expected_values_and_alignment():
    da = _make_constant_daily_da(10.0, dtype=np.float32)
    baseline = _make_baseline_clim(2.0, lat=da["lat"].values, lon=da["lon"].values)

    detrended, trend_on_daily_timestep = detrend(
        da=da,
        da_baseline_clim=baseline,
        detrend_method="multiplicative",
    )

    np.testing.assert_array_equal(trend_on_daily_timestep["time"].values, da["time"].values)
    assert trend_on_daily_timestep.shape == da.shape
    assert detrended.shape == da.shape
    assert trend_on_daily_timestep.dtype == da.dtype
    assert detrended.dtype == da.dtype
    np.testing.assert_allclose(trend_on_daily_timestep.values, 5.0)
    np.testing.assert_allclose(detrended.values, 2.0)


def _make_retrend_inputs() -> tuple[xr.DataArray, xr.DataArray]:
    time = np.arange(np.datetime64("2001-01-01"), np.datetime64("2001-01-11"))
    lat = np.array([0.0, 1.0])
    lon = np.array([10.0, 11.0])

    detrended = xr.DataArray(
        np.full((time.size, lat.size, lon.size), 2.0, dtype=np.float32),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )
    trend = xr.DataArray(
        np.full((time.size, lat.size, lon.size), 5.0, dtype=np.float32),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return detrended, trend


def test_retrend_additive_returns_expected_values():
    detrended, trend = _make_retrend_inputs()

    result = retrend(
        bias_corrected_detrended=detrended,
        trend_on_daily_timestep=trend,
        detrend_method="additive",
    )

    np.testing.assert_allclose(result.values, 7.0)
    np.testing.assert_array_equal(result["time"].values, detrended["time"].values)


def test_retrend_multiplicative_returns_expected_values():
    detrended, trend = _make_retrend_inputs()

    result = retrend(
        bias_corrected_detrended=detrended,
        trend_on_daily_timestep=trend,
        detrend_method="multiplicative",
    )

    np.testing.assert_allclose(result.values, 10.0)
    np.testing.assert_array_equal(result["time"].values, detrended["time"].values)


def test_retrend_raises_for_invalid_method():
    detrended, trend = _make_retrend_inputs()

    with pytest.raises(ValueError, match="currently not supported"):
        retrend(
            bias_corrected_detrended=detrended,
            trend_on_daily_timestep=trend,
            detrend_method="invalid",  # type: ignore[arg-type]
        )


def test_detrend_multiplicative_with_zero_climatology_produces_non_finite_trend():
    da = _make_constant_daily_da(10.0, dtype=np.float32)
    baseline = _make_baseline_clim(0.0, lat=da["lat"].values, lon=da["lon"].values)

    detrended, trend_on_daily_timestep = detrend(
        da=da,
        da_baseline_clim=baseline,
        detrend_method="multiplicative",
    )

    # Current behavior: divide-by-zero in trend calculation leads to inf trend values.
    assert np.isinf(trend_on_daily_timestep.values).any()
    # Then da / inf -> 0 for positive da values.
    np.testing.assert_allclose(detrended.values, 0.0)


def test_detrend_multiplicative_with_partial_zero_climatology_affects_zero_months_only():
    da = _make_constant_daily_da(10.0, dtype=np.float32)
    baseline = _make_baseline_clim(2.0, lat=da["lat"].values, lon=da["lon"].values)
    baseline.loc[dict(month=1)] = 0.0

    detrended, trend_on_daily_timestep = detrend(
        da=da,
        da_baseline_clim=baseline,
        detrend_method="multiplicative",
    )

    jan_mask = trend_on_daily_timestep["time"].dt.month == 1
    feb_mask = trend_on_daily_timestep["time"].dt.month == 2

    jan_trend = trend_on_daily_timestep.sel(time=jan_mask)
    feb_trend = trend_on_daily_timestep.sel(time=feb_mask)
    jan_detrended = detrended.sel(time=jan_mask)
    feb_detrended = detrended.sel(time=feb_mask)

    assert np.isinf(jan_trend.values).any()
    assert np.isfinite(feb_trend.values).all()
    np.testing.assert_allclose(jan_detrended.values, 0.0)
    np.testing.assert_allclose(feb_detrended.values, 2.0)


def _make_demo_daily_data() -> xr.DataArray:
    """Synthetic daily signal with trend + seasonal cycle (same shape as demo script)."""
    time = np.arange(np.datetime64("2001-01-01"), np.datetime64("2013-01-01"))
    lat = np.array([0.0])
    lon = np.array([10.0])

    t = np.arange(time.size, dtype=np.float32)
    annual_cycle = 2.0 * np.sin(2.0 * np.pi * t / 365.25)
    signal = (10.0 + 0.002 * t + annual_cycle).astype(np.float32)

    return xr.DataArray(
        signal[:, None, None],
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="synthetic_daily",
    )


def _make_zero_baseline(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    """Monthly climatology with zeros so additive trend equals smoothed monthly signal."""
    return xr.DataArray(
        np.zeros((12, lat.size, lon.size), dtype=np.float32),
        dims=["month", "lat", "lon"],
        coords={"month": np.arange(1, 13), "lat": lat, "lon": lon},
        name="baseline_clim",
    )


def test_grouped_rolling_window_counts_match_expected_edge_pattern():
    years = np.arange(2001, 2013)  # 12 years
    monthly_times = np.array([np.datetime64(f"{year}-01-01") for year in years])
    jan_values = xr.DataArray(
        np.arange(1, len(years) + 1, dtype=np.float32),
        dims=["time"],
        coords={"time": monthly_times},
        name="jan_signal",
    )

    rolled_count = jan_values.rolling(time=9, center=True, min_periods=1).count()

    expected = np.array([5, 6, 7, 8, 9, 9, 9, 9, 8, 7, 6, 5])
    np.testing.assert_array_equal(rolled_count.values.astype(int), expected)


def test_detrend_additive_matches_manual_grouped_rolling_for_january():
    da = _make_demo_daily_data()
    baseline = _make_zero_baseline(da["lat"].values, da["lon"].values)

    _, trend_daily = detrend(
        da=da,
        da_baseline_clim=baseline,
        detrend_method="additive",
    )

    # Extract one January value per year from detrend() output.
    jan_from_detrend = (
        trend_daily.sel(time=trend_daily.time.dt.month == 1)
        .resample(time="YS")
        .first()
        .squeeze(drop=True)
    )

    # Build the same January trend manually from monthly means + 9-year centered rolling.
    da_mon = da.resample(time="1MS").mean("time")
    jan_mon = da_mon.sel(time=da_mon.time.dt.month == 1).squeeze(drop=True)
    jan_manual = jan_mon.rolling(time=9, center=True, min_periods=1).mean()

    # Show edge-window behavior explicitly.
    jan_counts = jan_mon.rolling(time=9, center=True, min_periods=1).count()
    expected_counts = np.array([5, 6, 7, 8, 9, 9, 9, 9, 8, 7, 6, 5])
    np.testing.assert_array_equal(jan_counts.values.astype(int), expected_counts)

    np.testing.assert_array_equal(jan_from_detrend["time"].values, jan_manual["time"].values)
    np.testing.assert_allclose(jan_from_detrend.values, jan_manual.values, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# fft_smooth_3harmonics
# ---------------------------------------------------------------------------

N_DOY = 365  # standard year length used across fft tests


def _make_3harmonic_signal(n: int = N_DOY) -> np.ndarray:
    """Return a signal that is exactly DC + 3 harmonics (float64)."""
    t = np.arange(n)
    return (
        5.0
        + 3.0 * np.cos(2 * np.pi * 1 * t / n + 0.50)
        + 2.0 * np.cos(2 * np.pi * 2 * t / n + 1.00)
        + 1.0 * np.cos(2 * np.pi * 3 * t / n + 1.50)
    )


def test_fft_smooth_3harmonics_all_nan_returns_input_unchanged():
    data = np.full(N_DOY, np.nan)
    result = fft_smooth_3harmonics(data)
    np.testing.assert_array_equal(result, data)


def test_fft_smooth_3harmonics_preserves_constant_signal():
    # A constant (DC-only) signal has no harmonics; the filter must reproduce it exactly.
    data = np.full(N_DOY, 7.5)
    result = fft_smooth_3harmonics(data)
    np.testing.assert_allclose(result, data, atol=1e-10)


def test_fft_smooth_3harmonics_exact_3harmonic_signal_is_lossless():
    # A signal built from exactly DC + 3 harmonics must survive the filter unchanged
    # (up to floating-point rounding, ~machine epsilon).
    data = _make_3harmonic_signal()
    result = fft_smooth_3harmonics(data)
    np.testing.assert_allclose(result, data, atol=1e-10)


def test_fft_smooth_3harmonics_preserves_output_shape():
    data = _make_3harmonic_signal()
    result = fft_smooth_3harmonics(data)
    assert result.shape == data.shape


def test_fft_smooth_3harmonics_preserves_dtype_float32():
    data = _make_3harmonic_signal().astype(np.float32)
    result = fft_smooth_3harmonics(data)
    assert result.dtype == np.float32


def test_fft_smooth_3harmonics_preserves_dtype_float64():
    data = _make_3harmonic_signal().astype(np.float64)
    result = fft_smooth_3harmonics(data)
    assert result.dtype == np.float64


def test_fft_smooth_3harmonics_preserves_seasonal_peak_position():
    # A single annual cosine cycle peaks at a known index.  The filter must not
    # invert or shift the day-of-year axis.
    t = np.arange(N_DOY)
    seasonal = np.cos(2 * np.pi * t / N_DOY)  # peak at index 0
    result = fft_smooth_3harmonics(seasonal)
    assert np.argmax(result) == np.argmax(seasonal)


def test_fft_smooth_3harmonics_attenuates_high_frequency_content():
    # A spike at a single day has energy spread across all frequencies.
    # After retaining only the first 3 harmonics, the RMS of the output
    # must be substantially smaller than that of the original spike signal.
    data = np.zeros(N_DOY)
    data[0] = 1.0
    result = fft_smooth_3harmonics(data)
    assert np.sqrt(np.mean(result**2)) < np.sqrt(np.mean(data**2))


def _make_global_coarse_da(dtype: str = "float64") -> xr.DataArray:
    """Global coarse grid in the -180..180 convention (no point at exactly +180),
    holding a smooth periodic function of longitude."""
    lon = np.arange(-180.0, 180.0, 45.0)  # [-180, -135, ..., 135]
    lat = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    data = np.sin(np.deg2rad(lon))[np.newaxis, :] + 0.1 * np.cos(np.deg2rad(lat))[:, np.newaxis]
    return xr.DataArray(data.astype(dtype), dims=["lat", "lon"], coords={"lat": lat, "lon": lon})


def _make_fine_grid() -> xr.DataArray:
    """Fine target grid with points close to the +/-180 antimeridian."""
    lon = np.array([-179.9, -170.0, -90.0, 0.0, 90.0, 170.0, 179.9])
    lat = np.array([-55.0, 0.0, 55.0])
    return xr.DataArray(
        np.zeros((lat.size, lon.size)),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )


def test_interpolate_coarse_to_fine_grid_wraps_periodically():
    # Fine points between the last coarse lon (135) and +180 must interpolate
    # between lon=135 and the wrapped lon=-180 point, not return NaN.
    coarse = _make_global_coarse_da()
    fine = _make_fine_grid()
    result = interpolate_coarse_to_fine_grid(coarse, fine)

    w = (179.9 - 135.0) / 45.0
    expected_lon_term = (1 - w) * np.sin(np.deg2rad(135.0)) + w * np.sin(np.deg2rad(-180.0))
    expected = expected_lon_term + 0.1 * np.cos(np.deg2rad(0.0))
    np.testing.assert_allclose(result.sel(lat=0.0, lon=179.9).item(), expected, atol=1e-6)
    # And the interpolated field should approximate the true periodic function
    # everywhere near the seam (linear-interp error over 45 deg spacing < 0.1).
    truth = (
        np.sin(np.deg2rad(fine["lon"].values))[np.newaxis, :]
        + 0.1 * np.cos(np.deg2rad(fine["lat"].values))[:, np.newaxis]
    )
    np.testing.assert_allclose(result.values, truth, atol=0.1)


def test_interpolate_coarse_to_fine_grid_interior_matches_plain_interp():
    # Away from the antimeridian, results must be identical to a plain
    # (non-padded) interp: padding only fills the seam, never perturbs interior.
    coarse = _make_global_coarse_da()
    fine = _make_fine_grid()
    result = interpolate_coarse_to_fine_grid(coarse, fine)
    plain = coarse.interp(lon=fine["lon"], lat=fine["lat"], method="slinear")
    interior = ~plain.isnull()
    np.testing.assert_array_equal(result.values[interior.values], plain.values[interior.values])


def test_interpolate_coarse_to_fine_grid_preserves_dtype():
    coarse = _make_global_coarse_da(dtype="float32")
    fine = _make_fine_grid()
    result = interpolate_coarse_to_fine_grid(coarse, fine)
    assert result.dtype == np.float32


def test_interpolate_coarse_to_fine_grid_dask_backed_no_nan():
    # Mirrors the production call site: time-dependent residuals, dask-backed,
    # chunked along lon.
    coarse = _make_global_coarse_da().expand_dims(time=pd.date_range("2000-01-01", periods=3))
    coarse = coarse.chunk({"time": 1, "lat": -1, "lon": 4})
    fine = _make_fine_grid()
    result = interpolate_coarse_to_fine_grid(coarse, fine).compute()
    assert not result.isnull().any()


def test_interpolate_coarse_to_fine_grid_unsorted_lon_handled():
    # Coarse lon handed in descending / shuffled order must be sorted internally
    # so the periodic-padding math and slinear interp stay correct.
    coarse = _make_global_coarse_da()
    fine = _make_fine_grid()
    expected = interpolate_coarse_to_fine_grid(coarse, fine)
    shuffled = coarse.isel(lon=np.array([3, 0, 7, 1, 5, 2, 6, 4]))
    result = interpolate_coarse_to_fine_grid(shuffled, fine)
    assert not result.isnull().any()
    np.testing.assert_allclose(result.values, expected.values, atol=1e-12)


def test_interpolate_coarse_to_fine_grid_regional_edge_stays_nan():
    # Regional (non-global) domain: fine cells beyond the coarse lon edges must
    # stay NaN exactly as plain interp leaves them — periodic padding must not
    # wrap a regional domain's east edge around to its west edge.
    lon = np.arange(16.25, 32.6, 1.25)  # South-Africa-like subset, centers only
    lat = np.arange(-34.5, -22.0, 1.0)
    data = np.outer(np.cos(np.deg2rad(lat)), np.sin(np.deg2rad(lon)))
    coarse = xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
    fine = xr.DataArray(
        np.zeros((3, 5)),
        dims=["lat", "lon"],
        coords={"lat": [-30.0, -28.0, -26.0], "lon": [16.0, 20.0, 25.0, 32.75, 33.0]},
    )
    result = interpolate_coarse_to_fine_grid(coarse, fine)
    plain = coarse.interp(lon=fine["lon"], lat=fine["lat"], method="slinear")
    assert plain.isnull().any()  # sanity: this setup does have out-of-hull cells
    np.testing.assert_array_equal(result.values, plain.values)


@pytest.mark.parametrize(
    "gcm, member",
    [
        ("CESM2-WACCM", "r1i1p1f1"),
        ("CESM2-WACCM", "001"),
        ("MIROC-ES2H", "r1i1p4f2"),
        ("UKESM", "r2i1p1f2"),
    ],
)
def test_get_historical_experiment_uses_unified_store(gcm: str, member: str):
    """get_historical_experiment always reads from the unified 'historical' group."""
    mock_da = MagicMock(spec=xr.DataArray)
    mock_da.sel.return_value = mock_da

    mock_ds = MagicMock()
    mock_ds.proj.assign_crs.return_value = mock_ds

    mock_node = MagicMock()
    mock_node.to_dataset.return_value = mock_ds

    mock_dt = MagicMock()
    mock_dt.__getitem__ = MagicMock(return_value=mock_node)

    with patch("srm.downscaling_utils._gcm_datatree", return_value=mock_dt) as mock_fn:
        with patch("srm.downscaling_utils.get_variable", return_value=mock_da):
            get_historical_experiment(gcm, member, "tas")
            mock_fn.assert_called_once_with(gcm)
            mock_dt.__getitem__.assert_called_once_with("historical")


# ---------------------------------------------------------------------------
# swap_temperature_extremes (issue #331)
# ---------------------------------------------------------------------------


def _temp_pair():
    """(tasmax, tasmin) with one inversion, one monotone cell, one NaN cell."""
    lat = np.array([0.0, 1.0])
    lon = np.array([10.0, 11.0])
    tasmax = xr.DataArray(
        np.array([[300.0, 290.0], [np.nan, 305.0]]),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        name="tasmax",
    )
    tasmin = xr.DataArray(
        # [0,0] monotone (280<300); [0,1] INVERSION (295>290);
        # [1,0] NaN tasmax; [1,1] monotone (300<305)
        np.array([[280.0, 295.0], [285.0, 300.0]]),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        name="tasmin",
    )
    return tasmax, tasmin


class TestSwapTemperatureExtremes:
    def test_inverted_cell_is_swapped(self):
        tasmax, tasmin = _temp_pair()
        new_max, new_min = swap_temperature_extremes(tasmax, tasmin)
        # inverted cell [0,1]: max/min exchanged
        assert new_max.values[0, 1] == 295.0
        assert new_min.values[0, 1] == 290.0

    def test_monotone_cells_unchanged(self):
        tasmax, tasmin = _temp_pair()
        new_max, new_min = swap_temperature_extremes(tasmax, tasmin)
        assert new_max.values[0, 0] == 300.0
        assert new_min.values[0, 0] == 280.0
        assert new_max.values[1, 1] == 305.0
        assert new_min.values[1, 1] == 300.0

    def test_nan_cells_left_untouched(self):
        tasmax, tasmin = _temp_pair()
        new_max, new_min = swap_temperature_extremes(tasmax, tasmin)
        assert np.isnan(new_max.values[1, 0])
        assert new_min.values[1, 0] == 285.0

    def test_result_is_monotone_everywhere(self):
        tasmax, tasmin = _temp_pair()
        new_max, new_min = swap_temperature_extremes(tasmax, tasmin)
        finite = np.isfinite(new_max.values) & np.isfinite(new_min.values)
        assert np.all(new_max.values[finite] >= new_min.values[finite])

    def test_names_preserved(self):
        tasmax, tasmin = _temp_pair()
        new_max, new_min = swap_temperature_extremes(tasmax, tasmin)
        assert new_max.name == "tasmax"
        assert new_min.name == "tasmin"

    def test_3d_dask_backed_stays_lazy_and_monotone(self):
        # Production calls this with (time, lat, lon) dask arrays read back from
        # icechunk; make sure the op stays lazy and the invariant holds.
        rng = np.random.default_rng(0)
        time = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-06"))
        lat = np.array([0.0, 1.0])
        lon = np.array([10.0, 11.0])
        coords = {"time": time, "lat": lat, "lon": lon}
        tasmax = xr.DataArray(
            rng.normal(300.0, 3.0, (5, 2, 2)),
            dims=["time", "lat", "lon"],
            coords=coords,
            name="tasmax",
        ).chunk({"time": 2})
        tasmin = xr.DataArray(
            rng.normal(300.0, 3.0, (5, 2, 2)),
            dims=["time", "lat", "lon"],
            coords=coords,
            name="tasmin",
        ).chunk({"time": 2})

        new_max, new_min = swap_temperature_extremes(tasmax, tasmin)

        assert new_max.chunks is not None and new_min.chunks is not None  # never computed
        a, b = new_max.compute().values, new_min.compute().values
        finite = np.isfinite(a) & np.isfinite(b)
        assert np.all(a[finite] >= b[finite])

    def test_raises_on_misaligned_coords(self):
        # A silent inner-join would drop/NaN cells instead of swapping; require exact coords.
        lat = np.array([0.0])
        tasmax = xr.DataArray(
            [[300.0, 290.0]],
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": [10.0, 11.0]},
            name="tasmax",
        )
        tasmin = xr.DataArray(
            [[280.0, 295.0]],
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": [10.001, 11.001]},
            name="tasmin",
        )
        with pytest.raises(ValueError):
            swap_temperature_extremes(tasmax, tasmin)

    def test_handles_unnamed_tasmin(self):
        lat, lon = np.array([0.0]), np.array([10.0, 11.0])
        tasmax = xr.DataArray(
            [[300.0, 290.0]], dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name="tasmax"
        )
        tasmin = xr.DataArray(  # no name — the fresh downscaled tasmin may be unnamed
            [[280.0, 295.0]], dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
        )
        new_max, new_min = swap_temperature_extremes(tasmax, tasmin)
        assert new_max.name == "tasmax"
        assert new_max.values[0, 1] == 295.0


# ---------------------------------------------------------------------------
# derive_tasmin — fail loud on mismatched tasmax/dtr time axes (issue #363)
# ---------------------------------------------------------------------------


def _temp_series(times, value, name):
    return xr.DataArray(
        np.full((len(times), 1, 1), value, dtype="float32"),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [0.0], "lon": [0.0]},
        name=name,
    )


class TestDeriveTasmin:
    def test_matching_axes_subtracts(self):
        t = np.arange("2015-01-01", "2015-01-05", dtype="datetime64[D]")
        out = derive_tasmin(_temp_series(t, 300.0, "tasmax"), _temp_series(t, 10.0, "dtr"))
        assert out.name == "tasmin"
        assert float(out.isel(time=0, lat=0, lon=0)) == 290.0

    def test_mismatched_start_raises(self):
        t1 = np.arange("2015-01-01", "2015-01-05", dtype="datetime64[D]")
        t2 = np.arange("2035-01-01", "2035-01-05", dtype="datetime64[D]")
        with pytest.raises(ValueError, match="#363"):
            derive_tasmin(_temp_series(t1, 300.0, "tasmax"), _temp_series(t2, 10.0, "dtr"))

    def test_different_length_raises(self):
        t1 = np.arange("2015-01-01", "2020-01-01", dtype="datetime64[D]")
        t2 = np.arange("2015-01-01", "2018-01-01", dtype="datetime64[D]")
        with pytest.raises(ValueError, match="time ax"):
            derive_tasmin(_temp_series(t1, 300.0, "tasmax"), _temp_series(t2, 10.0, "dtr"))


# ---------------------------------------------------------------------------
# NaN guards in spatial disaggregation (issue #517)
# ---------------------------------------------------------------------------


def _regional_grids(n_time: int = 40) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Coarse simulation, coarse obs, and fine obs on a regional (non-global) domain.

    The fine grid deliberately extends past the outermost coarse cell centres, which is
    what production subset runs do. Those out-of-domain fine cells interpolate to NaN,
    and the guards must tolerate exactly that frame and nothing more.
    """
    time = pd.date_range("2015-01-01", periods=n_time, freq="D")
    coarse_lat = np.array([-30.0, -20.0, -10.0])
    coarse_lon = np.array([20.0, 30.0, 40.0])
    fine_lat = np.arange(-35.0, -4.9, 5.0)
    fine_lon = np.arange(15.0, 45.1, 5.0)

    def _da(lat, lon, offset):
        shape = (time.size, lat.size, lon.size)
        values = offset + np.arange(np.prod(shape), dtype="float64").reshape(shape) % 7
        return xr.DataArray(
            values,
            dims=["time", "lat", "lon"],
            coords={"time": time, "lat": lat, "lon": lon},
            name="tas",
        )

    return (
        _da(coarse_lat, coarse_lon, 300.0),
        _da(coarse_lat, coarse_lon, 299.0),
        _da(fine_lat, fine_lon, 298.0),
    )


def _global_grids() -> tuple[xr.DataArray, xr.DataArray]:
    coarse = _make_global_coarse_da()
    fine_lat = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    fine_lon = np.arange(-180.0, 180.0, 10.0)
    fine = xr.DataArray(
        np.zeros((fine_lat.size, fine_lon.size)),
        dims=["lat", "lon"],
        coords={"lat": fine_lat, "lon": fine_lon},
    )
    coarse = coarse.assign_coords(lat=np.array([-90.0, -45.0, 0.0, 45.0, 90.0]))
    return coarse, fine


class TestCoarseDomainMask:
    def test_global_coarse_grid_covers_every_fine_cell(self):
        coarse, fine = _global_grids()

        mask = coarse_domain_mask(coarse, fine)

        assert bool(mask.all())

    def test_mask_matches_where_interpolation_produces_values_globally(self):
        coarse, fine = _global_grids()

        mask = coarse_domain_mask(coarse, fine)
        interpolated = interpolate_coarse_to_fine_grid(coarse, fine)

        np.testing.assert_array_equal(mask.values, ~np.isnan(interpolated.values))

    def test_mask_matches_where_interpolation_produces_values_regionally(self):
        coarse_sim, _, obs_fine = _regional_grids()
        coarse = coarse_sim.isel(time=0, drop=True)
        fine = obs_fine.isel(time=0, drop=True)

        mask = coarse_domain_mask(coarse, fine)
        interpolated = interpolate_coarse_to_fine_grid(coarse, fine)

        np.testing.assert_array_equal(mask.values, ~np.isnan(interpolated.values))

    def test_regional_mask_excludes_the_out_of_domain_frame(self):
        coarse_sim, _, obs_fine = _regional_grids()

        mask = coarse_domain_mask(
            coarse_sim.isel(time=0, drop=True), obs_fine.isel(time=0, drop=True)
        )

        assert not bool(mask.isel(lat=0).any())  # fine lat -35 is below coarse lat -30
        assert not bool(mask.isel(lon=0).any())  # fine lon 15 is west of coarse lon 20
        assert bool(mask.isel(lat=3, lon=3))  # interior stays valid


class TestDownscaleFromCoarseNanGuards:
    def test_regional_edge_nans_do_not_abort_the_run(self):
        """configs/qa/ subsets legitimately produce a NaN frame — that must still run."""
        coarse_sim, obs_coarse, obs_fine = _regional_grids()

        result = downscale_from_coarse(da=coarse_sim, obs_coarse=obs_coarse, obs_fine=obs_fine)

        # The legitimate out-of-domain frame is preserved, not silently filled.
        assert bool(np.isnan(result.isel(time=0, lat=0)).all())
        assert not bool(np.isnan(result.isel(time=0, lat=3, lon=3)))

    def test_nan_in_debiased_input_aborts(self):
        coarse_sim, obs_coarse, obs_fine = _regional_grids()
        coarse_sim = coarse_sim.copy()
        coarse_sim[5, :, :] = np.nan

        with pytest.raises(NaNCheckError, match="residuals"):
            downscale_from_coarse(da=coarse_sim, obs_coarse=obs_coarse, obs_fine=obs_fine)

    def test_nan_in_fine_observations_aborts(self):
        coarse_sim, obs_coarse, obs_fine = _regional_grids()
        obs_fine = obs_fine.copy()
        obs_fine[:, 3, 3] = np.nan

        with pytest.raises(NaNCheckError, match="doy_means"):
            downscale_from_coarse(da=coarse_sim, obs_coarse=obs_coarse, obs_fine=obs_fine)


class TestEnforceConservationNanGuard:
    """The conservation correction runs after the residual gates, so it needs its own.

    Its nearest-neighbour spreading of a coarse correction can push NaN into interior
    fine cells that the earlier checks already certified clean. The parameter is public
    and documented, so the guarantee has to hold when a caller turns it on.
    """

    def test_clean_conservation_run_is_not_aborted(self):
        coarse_sim, obs_coarse, obs_fine = _regional_grids()

        result = downscale_from_coarse(
            da=coarse_sim,
            obs_coarse=obs_coarse,
            obs_fine=obs_fine,
            enforce_conservation=True,
        )

        assert not bool(np.isnan(result.isel(time=0, lat=3, lon=3)))

    def test_nan_introduced_by_the_correction_aborts(self):
        coarse_sim, obs_coarse, obs_fine = _regional_grids()
        real = downscaling_utils.interpolate_fine_to_coarse_grid
        calls = []

        def _recoarsen_returns_nan(*args, **kwargs):
            out = real(*args, **kwargs)
            calls.append(1)
            # Second call is the recoarsen step inside the conservation branch.
            return out.where(False) if len(calls) == 2 else out

        with patch.object(
            downscaling_utils,
            "interpolate_fine_to_coarse_grid",
            side_effect=_recoarsen_returns_nan,
        ):
            with pytest.raises(NaNCheckError, match="downscaled"):
                downscale_from_coarse(
                    da=coarse_sim,
                    obs_coarse=obs_coarse,
                    obs_fine=obs_fine,
                    enforce_conservation=True,
                )
