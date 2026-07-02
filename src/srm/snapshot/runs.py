"""Compare a candidate BCSD run against the global snapshot.

:func:`compare_runs` opens two output stores on the same icechunk branch and diffs
them leaf-by-leaf under the per-variable tolerances, reporting leaves that appear on
only one side. It performs **no coordinate alignment**: the two stores are compared
as-is, so a grid mismatch surfaces as an out-of-tolerance / shape-mismatch leaf rather
than being silently reconciled. To compare a regional candidate against the global
snapshot, subset the snapshot to the candidate's extent first (see the comparison
notebook).
"""

from __future__ import annotations

from srm.snapshot.compare import DiffReport, LeafDiff, _iter_leaf_datasets, compare
from srm.validation import _open_output_datatree


def _compare_datatrees(
    candidate,
    snapshot,
    *,
    scenarios: list[str] | None = None,
    variables: list[str] | None = None,
) -> DiffReport:
    """Diff ``candidate`` against ``snapshot`` across all leaves (no alignment).

    Runs :func:`~srm.snapshot.compare.compare` (which flags candidate leaves absent
    from the snapshot), then appends leaves present only in the snapshot so nothing is
    silently skipped. ``scenarios`` / ``variables`` restrict the reported leaves.
    """
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
    scenarios: list[str] | None = None,
    variables: list[str] | None = None,
) -> DiffReport:
    """Open two output stores on ``branch`` and compare them as-is.

    The stores must already be on the same grid (see the module docstring). Use the
    comparison notebook to subset the global snapshot to a regional candidate first.
    """
    candidate = _open_output_datatree(candidate_uri, branch=branch)
    snapshot = _open_output_datatree(snapshot_uri, branch=branch)
    return _compare_datatrees(candidate, snapshot, scenarios=scenarios, variables=variables)
