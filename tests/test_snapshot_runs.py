"""Tests for srm.snapshot.runs._compare_datatrees (the tree-level comparison).

compare_runs() itself is a thin S3 wrapper over this function; the logic — per-leaf
spatial alignment, all-leaves diff, and one-sided-leaf reporting — lives here and is
tested with tiny synthetic datatrees (no S3).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from srm.snapshot.runs import _compare_datatrees


def _leaf(lats, lons, val, name="tas"):
    da = xr.DataArray(
        np.full((len(lats), len(lons)), float(val)),
        dims=("lat", "lon"),
        coords={"lat": [float(x) for x in lats], "lon": [float(x) for x in lons]},
        name=name,
    )
    return da.to_dataset()


def test_subset_aligns_global_snapshot_to_candidate():
    snap = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([0, 1, 2, 3], [0, 1, 2, 3], 5.0)})
    cand = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0)})
    report = _compare_datatrees(cand, snap, subset_to_candidate=True)
    leaf = next(lf for lf in report.leaves if lf.path == "ssp245/tas/008/tas")
    assert not leaf.shape_mismatch
    assert leaf.within_tol


def test_no_subset_leaves_shape_mismatch():
    snap = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([0, 1, 2, 3], [0, 1, 2, 3], 5.0)})
    cand = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0)})
    report = _compare_datatrees(cand, snap, subset_to_candidate=False)
    leaf = next(lf for lf in report.leaves if lf.path == "ssp245/tas/008/tas")
    assert leaf.shape_mismatch


def test_detects_out_of_tolerance_difference():
    snap = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0)})
    cand = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([1, 2], [1, 2], 9.0)})
    report = _compare_datatrees(cand, snap, subset_to_candidate=True)
    assert not report.within_tolerance


def test_snapshot_only_leaf_is_reported():
    snap = xr.DataTree.from_dict(
        {
            "/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0),
            "/ssp245/pr/008": _leaf([1, 2], [1, 2], 1.0, name="pr"),
        }
    )
    cand = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0)})
    report = _compare_datatrees(cand, snap, subset_to_candidate=True)
    pr = next(lf for lf in report.leaves if lf.path == "ssp245/pr/008/pr")
    assert pr.shape_mismatch and not pr.within_tol


def test_variable_filter():
    snap = xr.DataTree.from_dict(
        {
            "/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0),
            "/ssp245/pr/008": _leaf([1, 2], [1, 2], 1.0, name="pr"),
        }
    )
    report = _compare_datatrees(snap, snap, subset_to_candidate=True, variables=["tas"])
    assert {lf.variable for lf in report.leaves} == {"tas"}


def test_scenario_filter():
    snap = xr.DataTree.from_dict(
        {
            "/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0),
            "/g6_1p5k/tas/003": _leaf([1, 2], [1, 2], 5.0),
        }
    )
    report = _compare_datatrees(snap, snap, subset_to_candidate=True, scenarios=["g6_1p5k"])
    assert {lf.path.split("/")[0] for lf in report.leaves} == {"g6_1p5k"}
