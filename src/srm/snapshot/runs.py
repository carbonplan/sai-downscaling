"""Compare a candidate BCSD run against the global snapshot.

:func:`compare_runs` opens two output stores and diffs them leaf-by-leaf under the
per-variable tolerances. For the cheap South Africa pre-merge check (issue #410) the
global snapshot is spatially aligned to the candidate's extent (per leaf, using each
leaf's own ``lat``/``lon``), so a regional subset can be compared against the global
run. Leaves present on only one side are reported rather than silently skipped.
"""

from __future__ import annotations

import xarray as xr

from srm.snapshot.compare import DiffReport, LeafDiff, _iter_leaf_datasets, compare
from srm.validation import _open_output_datatree


def _tree_key(path: str) -> str:
    """Map an :func:`_iter_leaf_datasets` group path to a ``DataTree.from_dict`` key."""
    if not path:
        return "/"
    return path if path.startswith("/") else f"/{path}"


def _align_latlon_to(snapshot: xr.DataTree, candidate: xr.DataTree) -> xr.DataTree:
    """Return ``snapshot`` with each leaf's ``lat``/``lon`` aligned to the matching
    candidate leaf.

    For every snapshot leaf that also exists in ``candidate``, select the nearest
    cells to the candidate's ``lat``/``lon`` and overwrite the coordinate labels with
    the candidate's exact values, so the subsequent element-wise diff aligns cell for
    cell even under tiny floating-point drift. Snapshot-only leaves are passed through
    unchanged (they are reported as one-sided later).
    """
    cand_leaves = dict(_iter_leaf_datasets(candidate))
    aligned: dict[str, xr.Dataset] = {}
    for path, snap_ds in _iter_leaf_datasets(snapshot):
        cand_ds = cand_leaves.get(path)
        if cand_ds is not None:
            dims = [d for d in ("lat", "lon") if d in cand_ds.coords and d in snap_ds.coords]
            if dims:
                snap_ds = snap_ds.sel({d: cand_ds[d] for d in dims}, method="nearest")
                snap_ds = snap_ds.assign_coords({d: cand_ds[d] for d in dims})
        aligned[_tree_key(path)] = snap_ds
    return xr.DataTree.from_dict(aligned)


def _compare_datatrees(
    candidate: xr.DataTree,
    snapshot: xr.DataTree,
    *,
    subset_to_candidate: bool = True,
    scenarios: list[str] | None = None,
    variables: list[str] | None = None,
) -> DiffReport:
    """Diff ``candidate`` against ``snapshot`` across all leaves.

    Parameters
    ----------
    candidate, snapshot : xr.DataTree
        Output datatrees to compare.
    subset_to_candidate : bool
        If True, align the snapshot to the candidate's spatial extent per leaf before
        comparing (the cheap South Africa check). If False, compare as-is (global run
        vs global snapshot).
    scenarios, variables : list of str, optional
        Restrict the reported leaves to these scenario groups / variables.

    Returns
    -------
    DiffReport
        One :class:`~srm.snapshot.compare.LeafDiff` per leaf, including leaves present
        on only one side (marked ``shape_mismatch``).
    """
    if subset_to_candidate:
        snapshot = _align_latlon_to(snapshot, candidate)

    report = compare(candidate, snapshot)
    leaves = list(report.leaves)

    cand_pairs = {(p, str(v)) for p, ds in _iter_leaf_datasets(candidate) for v in ds.data_vars}
    for path, ds in _iter_leaf_datasets(snapshot):
        for v in ds.data_vars:
            if (path, str(v)) not in cand_pairs:
                leaf_path = f"{path}/{v}" if path else str(v)
                leaves.append(
                    LeafDiff(leaf_path, str(v), float("nan"), float("nan"), 1.0, -1, True, False)
                )

    if scenarios:
        scen = set(scenarios)
        leaves = [lf for lf in leaves if lf.path.split("/")[0] in scen]
    if variables:
        var = set(variables)
        leaves = [lf for lf in leaves if lf.variable in var]
    return DiffReport(leaves=leaves)


def compare_runs(
    candidate_uri: str,
    snapshot_uri: str,
    *,
    branch: str,
    subset_to_candidate: bool = True,
    scenarios: list[str] | None = None,
    variables: list[str] | None = None,
) -> DiffReport:
    """Open two output stores and compare them (see :func:`_compare_datatrees`).

    Both stores are read on ``branch``. Use ``subset_to_candidate=True`` (default) to
    compare a regional candidate against the global snapshot, or ``False`` for a
    global-vs-global comparison.
    """
    candidate = _open_output_datatree(candidate_uri, branch=branch)
    snapshot = _open_output_datatree(snapshot_uri, branch=branch)
    return _compare_datatrees(
        candidate,
        snapshot,
        subset_to_candidate=subset_to_candidate,
        scenarios=scenarios,
        variables=variables,
    )
