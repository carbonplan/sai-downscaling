import numpy as np
import pytest
import xarray as xr

from saidownscale.snapshot.baselines import CESM2_WACCM_GLOBAL
from saidownscale.snapshot.compare import (
    DiffReport,
    InvariantCheck,
    LeafDiff,
    _compare_dataarray,
    compare,
)
from saidownscale.snapshot.tolerances import TOLERANCES


def _da(values):
    return xr.DataArray(np.asarray(values, dtype="float64"), dims=["x"])


def _ds(values, name="tas"):
    return xr.Dataset({name: _da(values)})


def _grid(values, dims=("lat", "lon"), coords=None):
    arr = np.asarray(values, dtype="float64")
    coords = coords or {d: np.arange(s, dtype="float64") for d, s in zip(dims, arr.shape)}
    return xr.DataArray(arr, dims=dims, coords=coords)


def test_compare_dataarray_verdicts(subtests):
    square = [[5.0, 5.0], [5.0, 5.0]]
    cases = {
        "identical": (
            _da([1.0, 2.0, 3.0]),
            _da([1.0, 2.0, 3.0]),
            {},
            {
                "within_tol": True,
                "max_abs_diff": 0.0,
                "frac_over_tol": 0.0,
                "nan_mismatch_count": 0,
                "shape_mismatch": False,
            },
        ),
        "tiny_change_fails_without_a_floor": (
            _da([1.0, 2.0, 3.0 + 5e-4]),
            _da([1.0, 2.0, 3.0]),
            {},
            {"within_tol": False, "frac_over_tol": 1 / 3, "max_abs_diff": 5e-4},
        ),
        "change_fails": (
            _da([1.0, 2.0, 3.5]),
            _da([1.0, 2.0, 3.0]),
            {},
            {"within_tol": False, "frac_over_tol": 1 / 3, "max_abs_diff": 0.5},
        ),
        "explicit_atol_band": (
            _da([1.0, 2.0, 3.0 + 5e-4]),
            _da([1.0, 2.0, 3.0]),
            {"rtol": 0.0, "atol": 1e-3},
            {"within_tol": True, "frac_over_tol": 0.0},
        ),
        "shape_mismatch": (
            _da([1.0, 2.0]),
            _da([1.0, 2.0, 3.0]),
            {},
            {"shape_mismatch": True, "within_tol": False},
        ),
        "new_nan": (
            _da([1.0, np.nan, 3.0]),
            _da([1.0, 2.0, 3.0]),
            {},
            {"nan_mismatch_count": 1, "within_tol": False},
        ),
        "shared_nan": (
            _da([1.0, np.nan, 3.0]),
            _da([1.0, np.nan, 3.0]),
            {},
            {"nan_mismatch_count": 0, "within_tol": True},
        ),
        "partial_coordinate_overlap": (
            xr.DataArray(np.full(3, 5.0), dims="x", coords={"x": [0, 1, 2]}),
            xr.DataArray(np.full(3, 5.0), dims="x", coords={"x": [1, 2, 3]}),
            {},
            {"within_tol": False},
        ),
        "inf_in_snapshot_invisible_to_frac_but_still_fails": (
            _da([1.0, 5.0, 3.0]),
            _da([1.0, np.inf, 3.0]),
            {},
            {"within_tol": False, "frac_over_tol": 0.0},
        ),
        "opposite_infinities": (_da([-np.inf]), _da([np.inf]), {}, {"within_tol": False}),
        "matching_infinities": (_da([1.0, np.inf]), _da([1.0, np.inf]), {}, {"within_tol": True}),
        "renamed_dims": (
            _grid(square, dims=("lat", "lon")),
            _grid(square, dims=("y", "x")),
            {},
            {"shape_mismatch": True, "within_tol": False},
        ),
        "shifted_coordinates": (
            _grid([[1.0, 2.0]]),
            _grid([[1.0, 2.0]], coords={"lat": [9.0], "lon": [8.0, 9.0]}),
            {},
            {"shape_mismatch": True, "within_tol": False},
        ),
    }
    for case, (candidate, snapshot, kwargs, expected) in cases.items():
        with subtests.test(case=case):
            d = _compare_dataarray(candidate, snapshot, path="g/v", variable="tas", **kwargs)
            assert isinstance(d, LeafDiff)
            assert {k: getattr(d, k) for k in expected} == pytest.approx(expected)


def test_compare_dataset_tolerances(subtests):
    zeros, bumped = [0.0, 0.0, 0.0], [0.0, 0.0, 1e-3]
    cases = {
        "drift": ([1.0, 2.0, 9.0], [1.0, 2.0, 3.0], "tas", None, False),
        "default_exact_pr": (bumped, zeros, "pr", None, False),
        "default_exact_tas": (bumped, zeros, "tas", None, False),
        "table_keeps_pr_tight": (bumped, zeros, "pr", TOLERANCES, False),
        "table_bands_tas": (bumped, zeros, "tas", TOLERANCES, True),
        "table_still_catches_inf": ([1.0, 5.0], [1.0, np.inf], "tas", TOLERANCES, False),
        "partial_mapping_leaves_tas_exact": (bumped, zeros, "tas", {"pr": TOLERANCES["pr"]}, False),
        "empty_mapping_is_exact": (bumped, zeros, "tas", {}, False),
    }
    for case, (cand, snap, name, tolerances, within) in cases.items():
        with subtests.test(case=case):
            report = compare(_ds(cand, name), _ds(snap, name), tolerances=tolerances)
            assert report.within_tolerance is within


def test_compare_reports_leaves_and_plain_text_lines():
    a = _ds([1.0, 2.0, 3.0])
    clean = compare(a, a)
    assert isinstance(clean, DiffReport)
    assert clean.within_tolerance is True
    assert [leaf.variable for leaf in clean.leaves] == ["tas"]

    lines = compare(_ds([1.0, 2.0, 9.0]), a).to_lines()
    assert any("tas" in line for line in lines)
    assert all(isinstance(line, str) for line in lines)


def test_compare_datatree_walks_leaves():
    a = xr.DataTree.from_dict(
        {"g6_1p5k/tas": _ds([1.0, 2.0], name="tas"), "g6_1p5k/pr": _ds([0.0, 0.0], name="pr")}
    )
    b = xr.DataTree.from_dict(
        {"g6_1p5k/tas": _ds([1.0, 2.0], name="tas"), "g6_1p5k/pr": _ds([0.0, 5.0], name="pr")}
    )
    report = compare(b, a)
    assert {leaf.variable: leaf.within_tol for leaf in report.leaves} == {"tas": True, "pr": False}
    assert report.within_tolerance is False


def test_invariant_checks_gate_the_report_verdict(subtests):
    passing_leaf = LeafDiff("g/tas", "tas", 0.0, 0.0, 0.0, 0, False, True)
    path = "ssp245/tasmax_ge_tasmin/008"
    cases = {
        "no_checks": ([], True),
        "check_holds": ([InvariantCheck(path, True, "")], True),
        "check_violated": ([InvariantCheck(path, False, "tasmax < tasmin at 2 point(s)")], False),
    }
    for case, (checks, passed) in cases.items():
        with subtests.test(case=case):
            report = DiffReport(leaves=[passing_leaf], invariant_checks=checks)
            assert report.within_tolerance is True
            assert report.invariants_hold is passed
            assert report.passed is passed


def test_global_baseline_pointer_is_well_formed():
    """The branch must never be "main": that is an empty anchor commit, not a run."""
    assert CESM2_WACCM_GLOBAL.uri.startswith(
        "s3://us-west-2.opendata.source.coop/carbonplan/srm-downscaling/output/production/"
    )
    assert CESM2_WACCM_GLOBAL.uri.endswith(".icechunk")
    assert CESM2_WACCM_GLOBAL.branch
    assert CESM2_WACCM_GLOBAL.branch != "main"
