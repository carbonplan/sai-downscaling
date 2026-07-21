import numpy as np
import xarray as xr

from srm.snapshot.compare import DiffReport, LeafDiff, _compare_dataarray, compare


def _da(values):
    return xr.DataArray(np.asarray(values, dtype="float64"), dims=["x"])


def test_identical_arrays_within_tol():
    a = _da([1.0, 2.0, 3.0])
    d = _compare_dataarray(a, a, path="g/v", variable="tas", rtol=0.0, atol=0.0)
    assert isinstance(d, LeafDiff)
    assert d.within_tol is True
    assert d.max_abs_diff == 0.0
    assert d.frac_over_tol == 0.0
    assert d.nan_mismatch_count == 0
    assert d.shape_mismatch is False


def test_small_change_inside_atol_passes():
    a = _da([1.0, 2.0, 3.0])
    b = _da([1.0, 2.0, 3.0 + 5e-4])
    d = _compare_dataarray(b, a, path="g/v", variable="tas", rtol=0.0, atol=1e-3)
    assert d.within_tol is True
    assert d.frac_over_tol == 0.0
    assert 4e-4 < d.max_abs_diff < 6e-4


def test_change_exceeding_atol_fails():
    a = _da([1.0, 2.0, 3.0])
    b = _da([1.0, 2.0, 3.5])
    d = _compare_dataarray(b, a, path="g/v", variable="tas", rtol=0.0, atol=1e-3)
    assert d.within_tol is False
    assert d.frac_over_tol > 0.0
    assert abs(d.max_abs_diff - 0.5) < 1e-9


def test_shape_mismatch_flagged():
    a = _da([1.0, 2.0, 3.0])
    b = _da([1.0, 2.0])
    d = _compare_dataarray(b, a, path="g/v", variable="tas", rtol=0.0, atol=1e-3)
    assert d.shape_mismatch is True
    assert d.within_tol is False


def test_new_nan_is_a_mismatch():
    a = _da([1.0, 2.0, 3.0])
    b = _da([1.0, np.nan, 3.0])
    d = _compare_dataarray(b, a, path="g/v", variable="tas", rtol=0.0, atol=1.0)
    assert d.nan_mismatch_count == 1
    assert d.within_tol is False


def _ds(values, name="tas"):
    return xr.Dataset({name: xr.DataArray(np.asarray(values, dtype="float64"), dims=["x"])})


def test_compare_dataset_clean():
    a = _ds([1.0, 2.0, 3.0])
    report = compare(a, a)
    assert isinstance(report, DiffReport)
    assert report.within_tolerance is True
    assert len(report.leaves) == 1
    assert report.leaves[0].variable == "tas"


def test_compare_dataset_detects_drift():
    a = _ds([1.0, 2.0, 3.0])
    b = _ds([1.0, 2.0, 9.0])
    report = compare(b, a)
    assert report.within_tolerance is False


def test_compare_uses_per_variable_tolerance():
    # pr has a tight near-zero floor (atol=1e-10); a 1e-3 change must fail for pr...
    a = _ds([0.0, 0.0, 0.0], name="pr")
    b = _ds([0.0, 0.0, 1e-3], name="pr")
    assert compare(b, a).within_tolerance is False
    # ...but the same change passes for tas (atol=1e-3).
    a2 = _ds([0.0, 0.0, 0.0], name="tas")
    b2 = _ds([0.0, 0.0, 1e-3], name="tas")
    assert compare(b2, a2).within_tolerance is True


def test_compare_datatree_walks_leaves():
    a = xr.DataTree.from_dict(
        {
            "g6_1p5k/tas": _ds([1.0, 2.0], name="tas"),
            "g6_1p5k/pr": _ds([0.0, 0.0], name="pr"),
        }
    )
    b = xr.DataTree.from_dict(
        {
            "g6_1p5k/tas": _ds([1.0, 2.0], name="tas"),
            "g6_1p5k/pr": _ds([0.0, 5.0], name="pr"),
        }
    )
    report = compare(b, a)
    assert {leaf.variable for leaf in report.leaves} == {"tas", "pr"}
    assert report.within_tolerance is False
    pr_leaf = next(leaf for leaf in report.leaves if leaf.variable == "pr")
    assert pr_leaf.within_tol is False


def test_to_lines_is_plain_text():
    a = _ds([1.0, 2.0, 3.0])
    b = _ds([1.0, 2.0, 9.0])
    lines = compare(b, a).to_lines()
    assert any("tas" in line for line in lines)
    assert all(isinstance(line, str) for line in lines)


def _passing_leaf():
    return LeafDiff("g/tas", "tas", 0.0, 0.0, 0.0, 0, False, True)


def test_report_with_no_invariants_passes_on_leaves_alone():
    from srm.snapshot.compare import DiffReport

    report = DiffReport(leaves=[_passing_leaf()])
    assert report.within_tolerance is True
    assert report.invariants_hold is True  # vacuously true when no checks ran
    assert report.passed is True


def test_invariant_violation_fails_passed_even_when_leaves_within_tol():
    from srm.snapshot.compare import DiffReport, InvariantCheck

    report = DiffReport(
        leaves=[_passing_leaf()],
        invariant_checks=[
            InvariantCheck(
                "ssp245/tasmax_ge_tasmin/008", False, "tasmax < tasmin at 2 grid point(s)"
            )
        ],
    )
    assert report.within_tolerance is True  # leaves are fine on their own
    assert report.invariants_hold is False
    assert report.passed is False  # the invariant violation gates the overall verdict


def test_report_passes_when_invariant_holds():
    from srm.snapshot.compare import DiffReport, InvariantCheck

    report = DiffReport(
        leaves=[_passing_leaf()],
        invariant_checks=[InvariantCheck("ssp245/tasmax_ge_tasmin/008", True, "")],
    )
    assert report.passed is True


def test_global_baseline_pointer_is_well_formed():
    from srm.snapshot.baselines import CESM2_WACCM_GLOBAL

    assert CESM2_WACCM_GLOBAL.uri.startswith("s3://carbonplan-srm/")
    assert CESM2_WACCM_GLOBAL.uri.endswith(".icechunk")
    assert CESM2_WACCM_GLOBAL.branch


def test_partial_coordinate_overlap_not_within_tol():
    # Same shape, equal values on the overlap, but coordinates only partially overlap.
    # A shape-only verdict inner-joins to x=[1, 2] and calls this within tolerance;
    # aligning on coordinates (assert_allclose) must not.
    a = xr.DataArray(np.full(3, 5.0), dims="x", coords={"x": [0, 1, 2]})
    b = xr.DataArray(np.full(3, 5.0), dims="x", coords={"x": [1, 2, 3]})
    d = _compare_dataarray(a, b, path="g/v", variable="tas", rtol=1e-5, atol=1e-8)
    assert d.within_tol is False
