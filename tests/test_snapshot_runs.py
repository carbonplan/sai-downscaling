"""Tests for srm.snapshot.runs._compare_datatrees (the tree-level comparison).

compare_runs() is a thin S3 wrapper over this function; the diff + one-sided-leaf
reporting + filtering live here and are tested with tiny synthetic datatrees (no S3).
There is no coordinate alignment: trees are compared as-is.
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


def test_identical_trees_within_tolerance():
    tree = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0)})
    report = _compare_datatrees(tree, tree)
    assert report.within_tolerance


def test_detects_out_of_tolerance_difference():
    snap = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0)})
    cand = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([1, 2], [1, 2], 9.0)})
    assert not _compare_datatrees(cand, snap).within_tolerance


def test_snapshot_only_leaf_reported():
    snap = xr.DataTree.from_dict(
        {
            "/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0),
            "/ssp245/pr/008": _leaf([1, 2], [1, 2], 1.0, name="pr"),
        }
    )
    cand = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0)})
    report = _compare_datatrees(cand, snap)
    pr = next(lf for lf in report.leaves if lf.path == "ssp245/pr/008/pr")
    assert pr.shape_mismatch and not pr.within_tol


def test_candidate_only_leaf_flagged():
    snap = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0)})
    cand = xr.DataTree.from_dict(
        {
            "/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0),
            "/ssp245/pr/008": _leaf([1, 2], [1, 2], 1.0, name="pr"),
        }
    )
    report = _compare_datatrees(cand, snap)
    pr = next(lf for lf in report.leaves if lf.path == "ssp245/pr/008/pr")
    assert pr.shape_mismatch


def test_variable_filter():
    tree = xr.DataTree.from_dict(
        {
            "/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0),
            "/ssp245/pr/008": _leaf([1, 2], [1, 2], 1.0, name="pr"),
        }
    )
    report = _compare_datatrees(tree, tree, variables=["tas"])
    assert {lf.variable for lf in report.leaves} == {"tas"}


def test_scenario_filter():
    tree = xr.DataTree.from_dict(
        {
            "/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0),
            "/g6_1p5k/tas/003": _leaf([1, 2], [1, 2], 5.0),
        }
    )
    report = _compare_datatrees(tree, tree, scenarios=["g6_1p5k"])
    assert {lf.path.split("/")[0] for lf in report.leaves} == {"g6_1p5k"}
