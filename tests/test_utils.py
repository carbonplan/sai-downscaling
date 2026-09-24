import numpy as np
import pytest
import xarray as xr

from saidownscale.qaqc import DatasetChecker
from saidownscale.utils import decode_time_from_bounds, lon_to_180, to_proleptic_gregorian


def test_lon_to_180_yields_valid_sorted_longitudes(ds_monotonic, ds_non_monotonic, ds_0_360):
    for ds in (ds_monotonic, ds_non_monotonic):
        result = DatasetChecker(lon_to_180(ds)).validate_lon(check_monotonic=True)
        assert result, f"{result.issues}"
    np.testing.assert_array_almost_equal(
        lon_to_180(ds_non_monotonic)["lon"].values, [-180, -90, -60, -10, -5, 0, 5, 10]
    )

    result = DatasetChecker(ds_0_360).validate_lon(check_monotonic=True)
    assert not result
    assert any("outside" in issue for issue in result.issues)


def _cftime_ds(start: str, end: str, calendar: str) -> xr.Dataset:
    times = xr.date_range(start, end, freq="D", use_cftime=True, calendar=calendar)
    da = xr.DataArray(np.arange(len(times), dtype="float64"), dims=["time"], coords={"time": times})
    return xr.Dataset({"tas": da})


def test_to_proleptic_gregorian(subtests):
    for calendar, end in {
        "noleap": "2020-12-31",
        "360_day": "2020-12-30",
        "standard": "2020-12-31",
    }.items():
        with subtests.test(calendar=calendar):
            ds = _cftime_ds("2020-01-01", end, calendar)
            result = to_proleptic_gregorian(ds)
            assert isinstance(result.time.values[0], np.datetime64)
            assert not result["tas"].isnull().any()
            assert result.time.values[-1] <= np.datetime64("2020-12-31")
            if calendar == "noleap":
                assert np.datetime64("2020-02-29") in result.time.values
            if calendar == "standard":
                np.testing.assert_array_equal(result["tas"].values, ds["tas"].values)


def _bounded_ds(lower: list[str], upper: list[str], values: list[float], stamps: list[str]):
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
    """End-of-interval daily means preceded by a zero-width initial-state record (#521)."""
    return _bounded_ds(
        lower=["2015-01-01", "2015-01-01", "2015-01-02", "2015-01-03"],
        upper=["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
        values=[-999.0, 1.0, 2.0, 3.0],
        stamps=["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
    )


def test_decode_time_from_bounds_restamps_cam_intervals_to_their_start():
    """#521: drop the initial-state record and move labels, never reassign data."""
    result = decode_time_from_bounds(_cam_like_ds())

    np.testing.assert_array_equal(
        result.time.values,
        np.array(["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(result["rsds"].values, [1.0, 2.0, 3.0])


def test_decode_time_from_bounds_edge_cases(subtests):
    with subtests.test(case="no_bounds_is_passthrough"):
        ds = _cam_like_ds().drop_vars("time_bnds")
        del ds.time.attrs["bounds"]
        xr.testing.assert_identical(decode_time_from_bounds(ds), ds)

    with subtests.test(case="bounds_with_extra_dim_rejected"):
        ds = _cam_like_ds().expand_dims({"ensemble_member": ["001", "002"]})
        with pytest.raises(ValueError, match="ambiguous"):
            decode_time_from_bounds(ds)

    with subtests.test(case="midpoint_stamps_stay_on_their_day"):
        ds = _bounded_ds(
            lower=["2015-01-01", "2015-01-02"],
            upper=["2015-01-02", "2015-01-03"],
            values=[1.0, 2.0],
            stamps=["2015-01-01T12:00", "2015-01-02T12:00"],
        )
        result = decode_time_from_bounds(ds)
        np.testing.assert_array_equal(
            result.time.values.astype("datetime64[D]"), ds.time.values.astype("datetime64[D]")
        )
