"""Tests for saidownscale.snapshot.runs._compare_datatrees (the tree-level comparison).

compare_runs() is a thin S3 wrapper over this function; the diff + one-sided-leaf
reporting + filtering live here and are tested with tiny synthetic datatrees (no S3).
There is no coordinate alignment: trees are compared as-is.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from saidownscale.snapshot.runs import _check_tasmax_ge_tasmin, _compare_datatrees


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


def _temp_pair(tasmax_val, tasmin_val, method=None):
    """A tasmax/tasmin pair, under the method-namespaced layout when ``method`` is set."""
    prefix = f"/{method}" if method else ""
    return xr.DataTree.from_dict(
        {
            f"{prefix}/ssp245/tasmax/008": _leaf([1, 2], [1, 2], tasmax_val, name="tasmax"),
            f"{prefix}/ssp245/tasmin/008": _leaf([1, 2], [1, 2], tasmin_val, name="tasmin"),
        }
    )


def test_invariant_holds_when_tasmax_ge_tasmin():
    cand = _temp_pair(tasmax_val=300.0, tasmin_val=290.0)
    report = _compare_datatrees(cand, cand)
    assert report.invariant_checks  # the tasmax>=tasmin pair was actually checked
    assert report.invariants_hold
    assert report.passed


def test_invariant_violation_fails_report_though_leaves_match():
    # candidate == snapshot, so every leaf is within tolerance, but the candidate
    # itself violates tasmax >= tasmin — the exact gap the invariant closes.
    cand = _temp_pair(tasmax_val=290.0, tasmin_val=300.0)
    report = _compare_datatrees(cand, cand)
    assert report.within_tolerance is True
    assert report.invariants_hold is False
    assert report.passed is False
    viol = next(c for c in report.invariant_checks if not c.holds)
    assert viol.path == "ssp245/tasmax_ge_tasmin/008"
    assert "tasmax < tasmin" in viol.detail


def test_no_invariant_check_without_both_temperature_extremes():
    tree = xr.DataTree.from_dict({"/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0)})
    report = _compare_datatrees(tree, tree)
    assert report.invariant_checks == []
    assert report.invariants_hold  # vacuously true
    assert report.passed


def test_invariant_check_skipped_when_filtering_to_unrelated_variable():
    # Filtering to variables that exclude tasmax/tasmin drops the cross-variable
    # invariant too: it isn't under review, so a violation must not gate this report.
    cand = xr.DataTree.from_dict(
        {
            "/ssp245/tasmax/008": _leaf([1, 2], [1, 2], 290.0, name="tasmax"),
            "/ssp245/tasmin/008": _leaf([1, 2], [1, 2], 300.0, name="tasmin"),  # would violate
            "/ssp245/pr/008": _leaf([1, 2], [1, 2], 1.0, name="pr"),
        }
    )
    report = _compare_datatrees(cand, cand, variables=["pr"])
    assert {lf.variable for lf in report.leaves} == {"pr"}
    assert report.invariant_checks == []
    assert report.passed  # only pr under review, which matches; invariant not gated


class TestInvariantAcrossStoreLayouts:
    """The tasmax >= tasmin gate must fire on both store layouts.

    Method-dependent groups are namespaced under a leading downscaling-method segment,
    so a depth-blind walk finds only method segments at the top level, reads their
    children as variables, and returns no checks at all: a silent pass on the one gate
    that exists to stop a reconcile-skip shipping (issue #331).
    """

    @pytest.mark.parametrize("method", [None, "bcsd", "qdmsd"])
    def test_gate_fires_and_holds(self, method):
        cand = _temp_pair(tasmax_val=300.0, tasmin_val=290.0, method=method)
        checks = _check_tasmax_ge_tasmin(cand)
        assert [c.path for c in checks] == ["ssp245/tasmax_ge_tasmin/008"]
        assert all(c.holds for c in checks)

    @pytest.mark.parametrize("method", [None, "bcsd"])
    def test_gate_catches_inversion(self, method):
        cand = _temp_pair(tasmax_val=290.0, tasmin_val=300.0, method=method)
        report = _compare_datatrees(cand, cand)
        assert report.within_tolerance is True
        assert report.invariants_hold is False
        assert report.passed is False
        viol = next(c for c in report.invariant_checks if not c.holds)
        assert viol.path == "ssp245/tasmax_ge_tasmin/008"

    def test_debiased_coarse_subtree_still_skipped(self):
        """debiased_coarse is pre-reconcile, and sits beside the scenario groups."""
        cand = xr.DataTree.from_dict(
            {
                "/bcsd/ssp245/tasmax/008": _leaf([1, 2], [1, 2], 300.0, name="tasmax"),
                "/bcsd/ssp245/tasmin/008": _leaf([1, 2], [1, 2], 290.0, name="tasmin"),
                # Inverted on purpose: pre-reconcile coarse output may hold tasmin > tasmax.
                "/bcsd/debiased_coarse/ssp245/tasmax/008": _leaf(
                    [1, 2], [1, 2], 290.0, name="tasmax"
                ),
                "/bcsd/debiased_coarse/ssp245/tasmin/008": _leaf(
                    [1, 2], [1, 2], 300.0, name="tasmin"
                ),
            }
        )
        checks = _check_tasmax_ge_tasmin(cand)
        assert [c.path for c in checks] == ["ssp245/tasmax_ge_tasmin/008"]
        assert all(c.holds for c in checks)

    def test_mixed_layout_store_checks_both(self):
        """A store holding a method segment beside pre-namespace groups checks both."""
        cand = xr.DataTree.from_dict(
            {
                "/bcsd/ssp245/tasmax/008": _leaf([1, 2], [1, 2], 300.0, name="tasmax"),
                "/bcsd/ssp245/tasmin/008": _leaf([1, 2], [1, 2], 290.0, name="tasmin"),
                "/g6_1p5k/tasmax/003": _leaf([1, 2], [1, 2], 290.0, name="tasmax"),
                "/g6_1p5k/tasmin/003": _leaf([1, 2], [1, 2], 300.0, name="tasmin"),
            }
        )
        checks = {c.path: c.holds for c in _check_tasmax_ge_tasmin(cand)}
        assert checks == {
            "ssp245/tasmax_ge_tasmin/008": True,
            "g6_1p5k/tasmax_ge_tasmin/003": False,
        }


class TestScenarioFilterAcrossStoreLayouts:
    """``scenarios=`` must select the scenario group under either store layout.

    Leaf paths are full node paths, so on a method-namespaced store segment 0 is the
    downscaling method and a filter written against scenario groups matches nothing. The
    requested comparison is then silently skipped: the report holds no leaves at all
    rather than the leaves that were asked for.
    """

    @staticmethod
    def _tree(prefix="", **leaves):
        return xr.DataTree.from_dict(
            {f"{prefix}/{path}": ds for path, ds in leaves.items()},
        )

    @pytest.mark.parametrize("prefix", ["", "/bcsd", "/qdmsd"])
    def test_filter_selects_the_requested_scenario(self, prefix):
        tree = self._tree(
            prefix,
            **{
                "ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0),
                "g6_1p5k/tas/003": _leaf([1, 2], [1, 2], 5.0),
            },
        )
        report = _compare_datatrees(tree, tree, scenarios=["g6_1p5k"])
        assert [lf.path for lf in report.leaves] == [
            f"{prefix.lstrip('/')}/g6_1p5k/tas/003/tas".lstrip("/")
        ]

    @pytest.mark.parametrize("prefix", ["", "/bcsd"])
    def test_filter_still_catches_a_mismatch(self, prefix):
        """The end-to-end failure: a filtered comparison must not skip its own leaves."""
        snap = self._tree(prefix, **{"ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0)})
        cand = self._tree(prefix, **{"ssp245/tas/008": _leaf([1, 2], [1, 2], 9.0)})
        report = _compare_datatrees(cand, snap, scenarios=["ssp245"])
        assert report.leaves  # something was actually compared
        assert not report.within_tolerance

    def test_filter_on_mixed_layout_store(self):
        """One store holding both layouts filters on the scenario group in either."""
        tree = xr.DataTree.from_dict(
            {
                "/bcsd/ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0),
                "/g6_1p5k/tas/003": _leaf([1, 2], [1, 2], 5.0),
            }
        )
        assert [lf.path for lf in _compare_datatrees(tree, tree, scenarios=["ssp245"]).leaves] == [
            "bcsd/ssp245/tas/008/tas"
        ]
        assert [lf.path for lf in _compare_datatrees(tree, tree, scenarios=["g6_1p5k"]).leaves] == [
            "g6_1p5k/tas/003/tas"
        ]

    @pytest.mark.parametrize("prefix", ["", "/bcsd"])
    def test_filter_keeps_the_invariant_check(self, prefix):
        """Invariant paths stay unprefixed, so the same filter must keep matching them.

        This is the reason ``InvariantCheck.path`` names the scenario group only. The
        filter tolerates both, so the issue #331 gate still fires under a scenario filter
        on a namespaced store.
        """
        cand = _temp_pair(tasmax_val=290.0, tasmin_val=300.0, method=prefix.lstrip("/") or None)
        report = _compare_datatrees(cand, cand, scenarios=["ssp245"])
        assert [c.path for c in report.invariant_checks] == ["ssp245/tasmax_ge_tasmin/008"]
        assert report.passed is False

    @pytest.mark.parametrize("prefix", ["", "/bcsd"])
    def test_unrequested_scenario_is_still_excluded(self, prefix):
        """Widening the filter must not turn it into a no-op."""
        tree = self._tree(
            prefix,
            **{
                "ssp245/tas/008": _leaf([1, 2], [1, 2], 5.0),
                "g6_1p5k/tas/003": _leaf([1, 2], [1, 2], 5.0),
            },
        )
        report = _compare_datatrees(tree, tree, scenarios=["ssp245"])
        assert all("g6_1p5k" not in lf.path for lf in report.leaves)
        assert len(report.leaves) == 1
