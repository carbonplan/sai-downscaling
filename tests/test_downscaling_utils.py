import numpy as np
import pytest
import xarray as xr

from srm.downscaling_utils import subset_space


@pytest.fixture
def sample_da() -> xr.DataArray:
    lat = np.array([0.0, 1.0, 2.0, 3.0])
    lon = np.array([10.0, 11.0, 12.0, 13.0])
    data = np.arange(lat.size * lon.size).reshape(lat.size, lon.size)
    return xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})


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
