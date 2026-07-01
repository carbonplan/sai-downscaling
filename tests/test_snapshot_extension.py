import numpy as np
import xarray as xr

from srm.snapshot.extension import SrmXarraySnapshotExtension


def _ds(values, name="tas"):
    return xr.Dataset({name: xr.DataArray(np.asarray(values, dtype="float64"), dims=["x"])})


def _ext():
    # Bypass syrupy's framework constructor; matches/diff_lines need no instance state.
    return SrmXarraySnapshotExtension.__new__(SrmXarraySnapshotExtension)


def test_matches_true_when_within_tolerance():
    a = _ds([1.0, 2.0, 3.0])
    assert _ext().matches(serialized_data=a, snapshot_data=a) is True


def test_matches_false_when_out_of_tolerance():
    a = _ds([1.0, 2.0, 3.0])
    b = _ds([1.0, 2.0, 9.0])
    assert _ext().matches(serialized_data=b, snapshot_data=a) is False


def test_matches_false_when_snapshot_missing():
    a = _ds([1.0, 2.0, 3.0])
    assert _ext().matches(serialized_data=a, snapshot_data=None) is False


def test_diff_lines_mentions_failing_leaf():
    a = _ds([1.0, 2.0, 3.0])
    b = _ds([1.0, 2.0, 9.0])
    lines = list(_ext().diff_lines(b, a))
    assert any("tas" in line for line in lines)
