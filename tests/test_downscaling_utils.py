import numpy as np
import pytest
import xarray as xr

from srm import downscaling_utils
from srm.downscaling_utils import calculate_baseline_climatology, detrend, rechunk, subset_space


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
