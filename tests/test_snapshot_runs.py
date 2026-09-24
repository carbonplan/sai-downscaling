"""Tests for the tree-level snapshot comparison, on tiny synthetic datatrees (no S3)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from saidownscale.snapshot.runs import _check_tasmax_ge_tasmin, _compare_datatrees

_INVARIANT = "ssp245/tasmax_ge_tasmin/008"
_LAYOUTS = {"pre_namespace": "", "bcsd": "/bcsd", "qdmsd": "/qdmsd"}


def _leaf(val, name="tas"):
    da = xr.DataArray(
        np.full((2, 2), float(val)),
        dims=("lat", "lon"),
        coords={"lat": [1.0, 2.0], "lon": [1.0, 2.0]},
        name=name,
    )
    return da.to_dataset()


def _tree(prefix="", **leaves):
    return xr.DataTree.from_dict({f"{prefix}/{path}": ds for path, ds in leaves.items()})


def _temp_pair(tasmax_val, tasmin_val, prefix=""):
    return _tree(
        prefix,
        **{
            "ssp245/tasmax/008": _leaf(tasmax_val, name="tasmax"),
            "ssp245/tasmin/008": _leaf(tasmin_val, name="tasmin"),
        },
    )


def test_compare_datatrees_leaf_verdicts(subtests):
    tas = {"ssp245/tas/008": _leaf(5.0)}
    tas_pr = {**tas, "ssp245/pr/008": _leaf(1.0, name="pr")}

    with subtests.test(case="identical_without_temperature_pair"):
        report = _compare_datatrees(_tree(**tas), _tree(**tas))
        assert report.within_tolerance
        assert report.invariant_checks == []
        assert report.passed

    with subtests.test(case="value_difference"):
        cand = _tree(**{"ssp245/tas/008": _leaf(9.0)})
        assert not _compare_datatrees(cand, _tree(**tas)).within_tolerance

    for case, (cand, snap) in {
        "snapshot_only": (tas, tas_pr),
        "candidate_only": (tas_pr, tas),
    }.items():
        with subtests.test(case=case):
            report = _compare_datatrees(_tree(**cand), _tree(**snap))
            pr = next(lf for lf in report.leaves if lf.path == "ssp245/pr/008/pr")
            assert pr.shape_mismatch and not pr.within_tol


def test_variable_filter_drops_unrelated_invariant():
    cand = _tree(
        **{
            "ssp245/tasmax/008": _leaf(290.0, name="tasmax"),
            "ssp245/tasmin/008": _leaf(300.0, name="tasmin"),
            "ssp245/pr/008": _leaf(1.0, name="pr"),
        }
    )
    report = _compare_datatrees(cand, cand, variables=["pr"])
    assert {lf.variable for lf in report.leaves} == {"pr"}
    assert report.invariant_checks == []
    assert report.passed


def test_tasmax_ge_tasmin_gate_fires_on_every_store_layout(subtests):
    """#331: a depth-blind walk over method-namespaced stores returned no checks at all."""
    for layout, prefix in _LAYOUTS.items():
        with subtests.test(layout=layout):
            ok = _temp_pair(300.0, 290.0, prefix)
            report = _compare_datatrees(ok, ok)
            assert [c.path for c in report.invariant_checks] == [_INVARIANT]
            assert report.invariants_hold and report.passed

            inverted = _temp_pair(290.0, 300.0, prefix)
            report = _compare_datatrees(inverted, inverted)
            assert report.within_tolerance is True
            assert report.invariants_hold is False
            assert report.passed is False
            (viol,) = report.invariant_checks
            assert viol.path == _INVARIANT
            assert "tasmax < tasmin" in viol.detail


def test_tasmax_ge_tasmin_gate_walks_nested_and_mixed_layouts(subtests):
    with subtests.test(case="debiased_coarse_is_pre_reconcile_and_skipped"):
        cand = _tree(
            "/bcsd",
            **{
                "ssp245/tasmax/008": _leaf(300.0, name="tasmax"),
                "ssp245/tasmin/008": _leaf(290.0, name="tasmin"),
                "debiased_coarse/ssp245/tasmax/008": _leaf(290.0, name="tasmax"),
                "debiased_coarse/ssp245/tasmin/008": _leaf(300.0, name="tasmin"),
            },
        )
        assert {c.path: c.holds for c in _check_tasmax_ge_tasmin(cand)} == {_INVARIANT: True}

    with subtests.test(case="mixed_layout_checks_both"):
        cand = _tree(
            **{
                "bcsd/ssp245/tasmax/008": _leaf(300.0, name="tasmax"),
                "bcsd/ssp245/tasmin/008": _leaf(290.0, name="tasmin"),
                "g6_1p5k/tasmax/003": _leaf(290.0, name="tasmax"),
                "g6_1p5k/tasmin/003": _leaf(300.0, name="tasmin"),
            }
        )
        assert {c.path: c.holds for c in _check_tasmax_ge_tasmin(cand)} == {
            _INVARIANT: True,
            "g6_1p5k/tasmax_ge_tasmin/003": False,
        }


def test_scenario_filter_selects_the_scenario_group_on_every_store_layout(subtests):
    """#331: a filter matched against segment 0 silently selected nothing on namespaced stores."""
    for layout, prefix in _LAYOUTS.items():
        with subtests.test(layout=layout):
            head = prefix.lstrip("/")
            both = _tree(prefix, **{"ssp245/tas/008": _leaf(5.0), "g6_1p5k/tas/003": _leaf(5.0)})
            for scenario, path in {
                "g6_1p5k": "g6_1p5k/tas/003",
                "ssp245": "ssp245/tas/008",
            }.items():
                report = _compare_datatrees(both, both, scenarios=[scenario])
                assert [lf.path for lf in report.leaves] == [f"{head}/{path}/tas".lstrip("/")]

            snap = _tree(prefix, **{"ssp245/tas/008": _leaf(5.0)})
            cand = _tree(prefix, **{"ssp245/tas/008": _leaf(9.0)})
            report = _compare_datatrees(cand, snap, scenarios=["ssp245"])
            assert report.leaves and not report.within_tolerance

            inverted = _temp_pair(290.0, 300.0, prefix)
            report = _compare_datatrees(inverted, inverted, scenarios=["ssp245"])
            assert [c.path for c in report.invariant_checks] == [_INVARIANT]
            assert report.passed is False

    with subtests.test(layout="mixed"):
        tree = _tree(**{"bcsd/ssp245/tas/008": _leaf(5.0), "g6_1p5k/tas/003": _leaf(5.0)})
        for scenario, path in {
            "ssp245": "bcsd/ssp245/tas/008/tas",
            "g6_1p5k": "g6_1p5k/tas/003/tas",
        }.items():
            report = _compare_datatrees(tree, tree, scenarios=[scenario])
            assert [lf.path for lf in report.leaves] == [path]
