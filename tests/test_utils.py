import numpy as np
import pytest
import xarray as xr

from saidownscale.qaqc import DatasetChecker as DatasetValidator
from saidownscale.utils import decode_time_from_bounds, lon_to_180, to_proleptic_gregorian


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


def _bounded_ds(lower: list[str], upper: list[str], values: list[float], stamps: list[str]):
    """A dataset carrying CF time bounds, stamped however the caller asks."""
    bounds = np.stack(
        [np.array(lower, dtype="datetime64[ns]"), np.array(upper, dtype="datetime64[ns]")], axis=1
    )
    ds = xr.Dataset(
        {
            "rsds": ("time", np.array(values, dtype="float64")),
            "time_bnds": (("time", "nbnd"), bounds),
        },
        coords={"time": np.array(stamps, dtype="datetime64[ns]")},
    )
    ds.time.attrs["bounds"] = "time_bnds"
    return ds


def _cam_like_ds():
    """A miniature CAM history file (issue #521).

    Daily means for 2015-01-01 through 2015-01-03, each stamped at the END of its averaging
    interval, preceded by the zero-width record holding the instantaneous initial state.
    """
    return _bounded_ds(
        lower=["2015-01-01", "2015-01-01", "2015-01-02", "2015-01-03"],
        upper=["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
        values=[-999.0, 1.0, 2.0, 3.0],
        stamps=["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
    )


class TestDecodeTimeFromBounds:
    def test_drops_zero_width_initialisation_record(self):
        result = decode_time_from_bounds(_cam_like_ds())

        assert result.sizes["time"] == 3
        assert -999.0 not in result["rsds"].values

    def test_restamps_intervals_to_their_start(self):
        result = decode_time_from_bounds(_cam_like_ds())

        np.testing.assert_array_equal(
            result.time.values,
            np.array(["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]"),
        )

    def test_returns_unchanged_when_no_bounds_declared(self):
        ds = _cam_like_ds().drop_vars("time_bnds")
        del ds.time.attrs["bounds"]

        xr.testing.assert_identical(decode_time_from_bounds(ds), ds)

    def test_keeps_each_value_with_the_day_it_covers(self):
        """The restamp must move labels, never reorder or reassign data."""
        result = decode_time_from_bounds(_cam_like_ds())

        np.testing.assert_array_equal(result["rsds"].values, [1.0, 2.0, 3.0])
        assert float(result["rsds"].sel(time="2015-01-01")) == 1.0

    def test_rejects_bounds_carrying_an_extra_dimension(self):
        """Decoding a bounds variable that also spans members would silently mis-stamp it.

        Reachable by calling this after ``expand_dims``/``combine_by_coords``, so it must
        fail loudly rather than pick an arbitrary dimension.
        """
        ds = _cam_like_ds().expand_dims({"ensemble_member": ["001", "002"]})

        with pytest.raises(ValueError, match="ambiguous"):
            decode_time_from_bounds(ds)

    def test_midpoint_stamped_data_stays_on_the_same_day(self):
        """UKESM and MIROC stamp at 12:00, so decoding must be a no-op by calendar day."""
        ds = _bounded_ds(
            lower=["2015-01-01", "2015-01-02"],
            upper=["2015-01-02", "2015-01-03"],
            values=[1.0, 2.0],
            stamps=["2015-01-01T12:00", "2015-01-02T12:00"],
        )

        result = decode_time_from_bounds(ds)

        assert result.sizes["time"] == 2
        np.testing.assert_array_equal(
            result.time.values.astype("datetime64[D]"),
            ds.time.values.astype("datetime64[D]"),
        )
