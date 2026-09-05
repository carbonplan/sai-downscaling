"""Compare a candidate BCSD run against the snapshot.

:func:`compare_runs` opens two output stores on the same icechunk branch and diffs
them leaf-by-leaf, by default for exact equality, reporting leaves that appear on
only one side. Pass ``tolerances`` to compare under an ``atol``/``rtol`` band instead
(see :func:`saidownscale.snapshot.compare.compare`). It performs **no coordinate alignment**:
the two stores are compared as-is, so a grid mismatch surfaces as a mismatch /
shape-mismatch leaf rather than being silently reconciled. To compare a regional
candidate against the global snapshot, subset the snapshot to the candidate's extent
first (see the comparison notebook).
"""

from __future__ import annotations

import xarray as xr

from saidownscale.bcsd_config import METHOD_SEGMENTS
from saidownscale.qaqc import DatasetChecker
from saidownscale.snapshot.compare import (
    DiffReport,
    InvariantCheck,
    LeafDiff,
    _iter_leaf_datasets,
    _select_matching_branch,
    compare,
)
from saidownscale.snapshot.tolerances import Tolerance
from saidownscale.validation import _open_output_datatree


def _scenario_segment(path: str) -> str:
    """Scenario group of a report path, under either store layout.

    Leaf paths come from :func:`~saidownscale.snapshot.compare._iter_leaf_datasets`, which yields
    full relative node paths, so segment 0 is the scenario group on a pre-namespace store
    but the downscaling method on a method-namespaced one
    (``{method}/{group}/{variable}/{member}/{variable}``). Dropping a leading method
    segment normalizes both, in the same way the walks in
    :func:`_check_tasmax_ge_tasmin` and :func:`~saidownscale.validation.validate_output_store`
    unwrap it, and each path is classified on its own so a store mixing the two layouts
    filters correctly.

    Invariant paths are already unprefixed by design (see :func:`_check_tasmax_ge_tasmin`),
    and no scenario group shares a name with a downscaling method, so passing them through
    here is a no-op that keeps one filter for both kinds of path.
    """
    segments = path.split("/")
    if segments and segments[0] in METHOD_SEGMENTS:
        segments = segments[1:]
    return segments[0] if segments else ""


def _check_tasmax_ge_tasmin(candidate) -> list[InvariantCheck]:
    """Verify ``tasmax >= tasmin`` per (scenario group, member) on the candidate tree.

    The per-variable leaf diff compares tasmax and tasmin independently, so a broken
    reconcile that leaves each field matching the snapshot yet inverts the pair would pass
    every leaf. This pairs the sibling ``{group}/tasmax/{member}`` and
    ``{group}/tasmin/{member}`` final-product leaves and runs the same NaN-safe gate
    ``bcsd validate`` uses (:meth:`~saidownscale.qaqc.DatasetChecker.validate_tasmax_ge_tasmin`,
    issue #331), so the snapshot comparison catches the inversion end-to-end.

    tasmin is derived (``tasmax - dtr``) then reconciled against tasmax, so this is the
    physical constraint that the whole derive-and-reconcile path must preserve. Only the
    final products are checked: the ``debiased_coarse`` subtree is pre-reconcile (its
    ``tasmin = tasmax - dtr`` is not swapped), and it is skipped naturally because its
    children are scenario groups, not the ``tasmax``/``tasmin`` variable nodes. That
    holds under either store layout, since ``debiased_coarse`` sits at the same depth as
    the scenario groups it is compared against.

    Both layouts are walked. Method-dependent groups are namespaced under a leading
    downscaling-method segment (``{method}/{group}/{variable}/{member}``), so a top-level
    method segment is unwrapped to reach the scenario groups; stores written before the
    namespacing already start at the scenario group. A store may hold both forms, so each
    top-level child is classified on its own. The reported path names the scenario group
    only, so it stays comparable with the ``scenarios`` filter in either layout.
    """
    if not isinstance(candidate, xr.DataTree):
        return []
    checks: list[InvariantCheck] = []
    scenario_nodes = [
        sn
        for top in candidate.children.values()
        for sn in (top.children.values() if top.name in METHOD_SEGMENTS else [top])
    ]
    for node in scenario_nodes:
        group = node.name
        svars = node.children
        if "tasmax" not in svars or "tasmin" not in svars:
            continue
        tmax = {leaf.name: leaf for leaf in svars["tasmax"].leaves}
        tmin = {leaf.name: leaf for leaf in svars["tasmin"].leaves}
        for member in sorted(set(tmax) & set(tmin)):
            paired = xr.merge(
                [tmax[member].to_dataset(), tmin[member].to_dataset()],
                compat="override",
                join="inner",
            )
            result = DatasetChecker(paired).validate_tasmax_ge_tasmin()
            checks.append(
                InvariantCheck(
                    path=f"{group}/tasmax_ge_tasmin/{member}",
                    holds=bool(result),
                    detail="" if result else "; ".join(result.issues),
                )
            )
    return checks


def _compare_datatrees(
    candidate,
    snapshot,
    *,
    scenarios: list[str] | None = None,
    variables: list[str] | None = None,
    tolerances: dict[str, Tolerance] | None = None,
) -> DiffReport:
    """Diff ``candidate`` against ``snapshot`` across all leaves (no alignment).

    Runs :func:`~saidownscale.snapshot.compare.compare` (which flags candidate leaves absent
    from the snapshot), then appends leaves present only in the snapshot so nothing is
    silently skipped. ``scenarios`` / ``variables`` restrict the reported leaves;
    ``scenarios`` matches the scenario group under either store layout, via
    :func:`_scenario_segment`.
    ``tolerances`` is passed straight through to :func:`~saidownscale.snapshot.compare.compare`
    (default: exact equality).
    """
    report = compare(candidate, snapshot, tolerances=tolerances)
    leaves = list(report.leaves)

    # compare() diffs the candidate's bcsd branch against a pre-namespacing snapshot
    # when the layouts don't already match (see _select_matching_branch); the
    # snapshot-only leaves added below must be detected against those same paths, or
    # every snapshot leaf would look candidate-missing on top of what compare() reported.
    matched_candidate = _select_matching_branch(candidate, snapshot)
    cand_pairs = {
        (p, str(v)) for p, ds in _iter_leaf_datasets(matched_candidate) for v in ds.data_vars
    }
    for path, ds in _iter_leaf_datasets(snapshot):
        for v in ds.data_vars:
            if (path, str(v)) not in cand_pairs:
                leaf_path = f"{path}/{v}" if path else str(v)
                leaves.append(
                    LeafDiff(leaf_path, str(v), float("nan"), float("nan"), 1.0, -1, True, False)
                )

    # Cross-variable gate: tasmax >= tasmin on the candidate (issue #331). Filtered to
    # match the reported leaves — restricted to the requested scenarios, and dropped
    # entirely when ``variables`` excludes both tasmax and tasmin (the invariant is not
    # under review, so it must not gate the verdict).
    invariant_checks = _check_tasmax_ge_tasmin(candidate)
    if variables and not ({"tasmax", "tasmin"} & set(variables)):
        invariant_checks = []

    if scenarios:
        scen = set(scenarios)
        leaves = [lf for lf in leaves if _scenario_segment(lf.path) in scen]
        invariant_checks = [c for c in invariant_checks if _scenario_segment(c.path) in scen]
    if variables:
        var = set(variables)
        leaves = [lf for lf in leaves if lf.variable in var]
    return DiffReport(leaves=leaves, invariant_checks=invariant_checks)


def compare_runs(
    candidate_uri: str,
    snapshot_uri: str,
    *,
    candidate_branch: str,
    snapshot_branch: str,
    scenarios: list[str] | None = None,
    variables: list[str] | None = None,
    tolerances: dict[str, Tolerance] | None = None,
) -> DiffReport:
    """Open two output stores and compare them as-is.

    ``candidate_branch`` and ``snapshot_branch`` are independent: the candidate and the
    snapshot are separate stores, so each is read on its own icechunk branch (the
    candidate need not be on the snapshot's branch). The stores must be on the same grid
    (see the module docstring); use the comparison notebook to subset the global
    snapshot to a regional candidate first. ``tolerances`` defaults to ``None`` (exact
    equality); pass :data:`saidownscale.snapshot.tolerances.TOLERANCES` to compare under a
    tolerance band instead.
    """
    candidate = _open_output_datatree(candidate_uri, branch=candidate_branch)
    snapshot = _open_output_datatree(snapshot_uri, branch=snapshot_branch)
    return _compare_datatrees(
        candidate, snapshot, scenarios=scenarios, variables=variables, tolerances=tolerances
    )
