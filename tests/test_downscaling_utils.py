import numpy as np
import pytest
import xarray as xr

from srm import downscaling_utils
from srm.downscaling_utils import calculate_baseline_climatology, rechunk, subset_space


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
