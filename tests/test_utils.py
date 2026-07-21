import numpy as np
import xarray as xr

from srm.qaqc import DatasetChecker as DatasetValidator
from srm.utils import lon_to_180, to_proleptic_gregorian


class TestLonTo180:
    def test_monotonic(self, ds_monotonic):
        result = lon_to_180(ds_monotonic)
        validator = DatasetValidator(result)
        validation_result = validator.validate_lon(check_monotonic=True)
        assert validation_result, f"{validation_result.issues}"

    def test_non_monotonic(self, ds_non_monotonic):
        result = lon_to_180(ds_non_monotonic)
        validator = DatasetValidator(result)
        validation_result = validator.validate_lon(check_monotonic=True)
        assert validation_result, f"{validation_result.issues}"

        expected_sorted = [-180, -90, -60, -10, -5, 0, 5, 10]
        np.testing.assert_array_almost_equal(validator.ds["lon"].values, expected_sorted)

    def test_0_360_range_fails(self, ds_0_360):
        validator = DatasetValidator(ds_0_360)
        validation_result = validator.validate_lon(check_monotonic=True)

        assert not validation_result, "expected validation to fail for 0-360 longitude range"
        assert any("outside" in issue for issue in validation_result.issues)


def _cftime_ds(start: str, end: str, calendar: str, freq: str = "D") -> xr.Dataset:
    times = xr.date_range(start, end, freq=freq, use_cftime=True, calendar=calendar)
    da = xr.DataArray(np.arange(len(times), dtype="float64"), dims=["time"], coords={"time": times})
    return xr.Dataset({"tas": da})


class TestToProlepticGregorian:
    def test_noleap_interpolates_feb29(self):
        ds = _cftime_ds("2020-01-01", "2020-12-31", "noleap")
        result = to_proleptic_gregorian(ds)

        assert isinstance(result.time.values[0], np.datetime64)
        assert not result["tas"].isnull().any()
        assert np.datetime64("2020-02-29") in result.time.values

    def test_360_day_interpolates_and_realigns(self):
        ds = _cftime_ds("2020-01-01", "2020-12-30", "360_day")
        result = to_proleptic_gregorian(ds)

        assert isinstance(result.time.values[0], np.datetime64)
        assert not result["tas"].isnull().any()
        assert result.time.values[-1] <= np.datetime64("2020-12-31")

    def test_standard_calendar_is_passthrough(self):
        ds = _cftime_ds("2020-01-01", "2020-12-31", "standard")
        result = to_proleptic_gregorian(ds)

        assert isinstance(result.time.values[0], np.datetime64)
        np.testing.assert_array_equal(result["tas"].values, ds["tas"].values)
