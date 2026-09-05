"""Comparison of BCSD output datasets and datatrees, exact by default.

The single :func:`compare` engine backs the comparison notebook (via
:func:`saidownscale.snapshot.runs.compare_runs`), so the notebook's verdict comes straight
from this code. All reductions are chunk-friendly so a few-GB South Africa diff runs
without loading whole arrays into memory.

A regional candidate and its regional snapshot are produced by the same code on the
same grid, so they are expected to match bit-for-bit: :func:`compare` defaults to
exact equality, with no per-variable tolerance band. Pass
:data:`saidownscale.snapshot.tolerances.TOLERANCES` (or a custom mapping) as ``tolerances`` to
restore an ``atol``/``rtol`` band instead — e.g. for the notebook's ``global`` mode,
which diffs against a differently-provenanced baseline where some regrid/dask
nondeterminism is expected rather than a finding.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import xarray as xr

from saidownscale.bcsd_config import METHOD_SEGMENTS
from saidownscale.snapshot.tolerances import Tolerance

#: Verdict for a leaf with no caller-supplied tolerance. Exact equality is the strictest
#: setting, so falling back to it can only tighten a comparison, never loosen one.
EXACT = Tolerance(rtol=0.0, atol=0.0)


@dataclass(frozen=True)
class LeafDiff:
    """Comparison result for one (group, variable) leaf.

    Parameters
    ----------
    path : str
        Group path of the leaf within the datatree (e.g. ``"g6_1p5k/tas"``).
    variable : str
        Canonical variable name, for reporting.
    max_abs_diff : float
        Maximum absolute difference over all cells (NaN if shapes mismatch).
    rmse : float
        Root-mean-square difference over all cells.
    frac_over_tol : float
        Fraction of cells exceeding tolerance (a shared NaN in both arrays is
        never a mismatch; with the default zero tolerance this is the fraction
        not exactly equal to the snapshot). Descriptive only, for ranking and
        plotting: ``within_tol`` is the verdict, and the two can disagree where
        the snapshot holds ``inf``.
    nan_mismatch_count : int
        Count of cells where candidate and snapshot disagree on NaN-ness.
    shape_mismatch : bool
        True if candidate and snapshot have different dimension names or shapes,
        or their coordinates do not align exactly.
    within_tol : bool
        True only if no cell is over tolerance, no NaN mismatch, no shape
        mismatch.
    """

    path: str
    variable: str
    max_abs_diff: float
    rmse: float
    frac_over_tol: float
    nan_mismatch_count: int
    shape_mismatch: bool
    within_tol: bool


def _compare_dataarray(
    candidate: xr.DataArray,
    snapshot: xr.DataArray,
    *,
    path: str,
    variable: str,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> LeafDiff:
    """Compare two DataArrays under an absolute/relative tolerance, exact by default.

    A cell passes when its absolute difference is at most ``atol + rtol * abs(snapshot)``
    — the same rule as :func:`xarray.testing.assert_allclose`. With the defaults
    (``rtol=atol=0``) that band collapses to plain equality: a cell passes only when it is
    numerically identical to the snapshot, or both are NaN. The function reduces the full
    element-wise comparison down to the scalar summary statistics carried by
    :class:`LeafDiff`, so callers get a compact verdict instead of a whole diff array.

    If the two arrays have different dimension names or shapes, or their coordinates
    do not align exactly, they cannot be compared cell-by-cell, so the function
    short-circuits: it returns a :class:`LeafDiff` with ``shape_mismatch=True``,
    ``within_tol=False``, and NaN/sentinel metrics rather than raising. Otherwise
    every metric is computed and ``within_tol`` comes from
    :func:`xarray.testing.assert_equal` (exact mode) or
    :func:`xarray.testing.assert_allclose` (band mode), not from ``frac_over_tol``.

    Parameters
    ----------
    candidate, snapshot : xr.DataArray
        New and baseline arrays. ``snapshot`` is the reference in the relative term.
    path : str
        Leaf group path, for reporting.
    variable : str
        Canonical variable name, for reporting.
    rtol, atol : float
        Relative and absolute tolerance, both zero by default (exact equality).
        ``atol`` sets a fixed floor that dominates near zero, while ``rtol`` scales
        the allowance with the magnitude of ``snapshot`` and dominates for large
        values.

    Returns
    -------
    LeafDiff
        The populated comparison result. ``within_tol`` is authoritative and is
        taken from the xarray assertion; the other fields (``max_abs_diff``,
        ``rmse``, ``frac_over_tol``, ``nan_mismatch_count``) quantify *how far*
        off the arrays are when they disagree. ``frac_over_tol`` can read 0.0 on a
        leaf the assertion rejects (an ``inf`` cell has no finite difference to
        exceed a tolerance), so rank on it but never gate on it.

    Notes
    -----
    All reductions call ``.compute()`` on lazy aggregates only, never on the
    full arrays, so a multi-GB diff is summarized without materializing it in
    memory. NaN cells are skipped in ``max_abs_diff`` and ``rmse`` via
    ``skipna=True``, and disagreement on NaN-ness is tracked separately through
    ``nan_mismatch_count``.
    """
    # Dimension names have to match before any arithmetic happens. Subtracting a
    # (y, x) array from a (lat, lon) array of the same shape does not raise, it
    # broadcasts to the outer product: O(N^2) on a real leaf, which exhausts memory
    # long before any verdict is reached.
    if candidate.dims != snapshot.dims or candidate.shape != snapshot.shape:
        return LeafDiff(
            path=path,
            variable=variable,
            max_abs_diff=float("nan"),
            rmse=float("nan"),
            frac_over_tol=1.0,
            nan_mismatch_count=-1,
            shape_mismatch=True,
            within_tol=False,
        )
    try:
        # join="exact" requires identical indexes, so this also catches the case
        # where shapes happen to match but the two arrays sit on different grids.
        candidate, snapshot = xr.align(candidate, snapshot, join="exact")
    except ValueError:
        return LeafDiff(
            path=path,
            variable=variable,
            max_abs_diff=float("nan"),
            rmse=float("nan"),
            frac_over_tol=1.0,
            nan_mismatch_count=-1,
            shape_mismatch=True,
            within_tol=False,
        )

    diff = candidate - snapshot  # signed error, per element
    abs_diff = abs(diff)  # magnitude of the error, ignoring sign
    nan_mismatch = candidate.isnull() != snapshot.isnull()  # NaN in one array but not the other
    # Per-element allowed difference, using numpy's isclose convention: a fixed floor
    # (atol) plus a term that scales with the reference value (rtol * |snapshot|).
    # NaN > tol is always False, so a shared NaN never counts as "over"; a one-sided
    # NaN is instead caught by the explicit nan_mismatch term below.
    tol = atol + rtol * abs(snapshot)
    over = (abs_diff > tol) | nan_mismatch

    max_abs_diff = float(abs_diff.max(skipna=True).compute())
    rmse = float(((diff**2).mean(skipna=True) ** 0.5).compute())
    frac_over_tol = float(over.mean().compute())
    nan_mismatch_count = int(nan_mismatch.sum().compute())

    # The verdict comes from xarray's own assertion, not from frac_over_tol. The
    # hand-rolled `abs_diff > tol` comparison above only ever inspects values, so it
    # inherits IEEE's rule that every comparison against NaN is False: a snapshot cell
    # holding +/-inf collapses `tol` to NaN (rtol=0) or inf (rtol>0), `over` is False
    # there, and a leaf differing by infinity reports as an exact match. The assertions
    # handle inf correctly (inf == inf passes, inf vs finite fails) and additionally
    # check index identity. Both are lazy reductions over dask, so this does not
    # materialize either array.
    try:
        if rtol == 0.0 and atol == 0.0:
            xr.testing.assert_equal(candidate, snapshot)
        else:
            xr.testing.assert_allclose(candidate, snapshot, rtol=rtol, atol=atol)
        within_tol = True
    except AssertionError:
        within_tol = False

    return LeafDiff(
        path=path,
        variable=variable,
        max_abs_diff=max_abs_diff,
        rmse=rmse,
        frac_over_tol=frac_over_tol,
        nan_mismatch_count=nan_mismatch_count,
        shape_mismatch=False,
        within_tol=within_tol,
    )


def _iter_leaf_datasets(obj: xr.Dataset | xr.DataTree) -> Iterator[tuple[str, xr.Dataset]]:
    """Yield ``(group_path, dataset)`` for every group that holds data variables.

    A plain Dataset yields a single ``("", dataset)`` pair; a DataTree yields one
    pair per node that has data variables.
    """
    if isinstance(obj, xr.Dataset):
        yield "", obj
        return
    if isinstance(obj, xr.DataTree):
        for path, node in obj.subtree_with_keys:
            ds = node.to_dataset(inherit=False)
            if ds.data_vars:
                yield path, ds
        return
    raise TypeError(f"Expected xr.Dataset or xr.DataTree, got {type(obj)}")


def _select_matching_branch(
    candidate: xr.Dataset | xr.DataTree, snapshot: xr.Dataset | xr.DataTree
) -> xr.Dataset | xr.DataTree:
    """Narrow ``candidate`` to the branch whose leaf paths line up with ``snapshot``.

    Method-dependent groups are namespaced under a leading downscaling-method segment
    (``bcsd``, ``qdmsd``; see :data:`saidownscale.bcsd_config.METHOD_SEGMENTS`), so a
    method-namespaced candidate's leaves sit one level deeper than a pre-namespacing
    snapshot's (``bcsd/g6_1p5k/tas/003/tas`` vs ``g6_1p5k/tas/003/tas``).
    :func:`compare` matches leaves by literal path, so diffing the two trees as-is would
    report every leaf on both sides as missing, even though the same data exists one
    level down. When the candidate's top level holds only method segments and the
    snapshot's does not, descend into the candidate's ``bcsd`` branch -- the layout every
    pre-namespacing snapshot corresponds to -- so leaf paths line up again. Any other
    combination (both namespaced, neither namespaced, a bare Dataset, or a candidate
    with no ``bcsd`` branch) is returned unchanged.
    """
    if not (isinstance(candidate, xr.DataTree) and isinstance(snapshot, xr.DataTree)):
        return candidate
    cand_children = set(candidate.children)
    if not cand_children or not cand_children <= METHOD_SEGMENTS:
        return candidate
    snap_children = set(snapshot.children)
    if snap_children and snap_children <= METHOD_SEGMENTS:
        return candidate  # snapshot is namespaced too; paths already line up
    if "bcsd" not in candidate.children:
        return candidate
    return candidate["bcsd"]


def compare(
    candidate: xr.Dataset | xr.DataTree,
    snapshot: xr.Dataset | xr.DataTree,
    *,
    tolerances: dict[str, Tolerance] | None = None,
) -> DiffReport:
    """Compare a candidate dataset/datatree against a snapshot, per variable.

    Parameters
    ----------
    candidate, snapshot : xr.Dataset or xr.DataTree
        New and baseline data. Must be the same type.
    tolerances : dict, optional
        Mapping of variable name to :class:`~saidownscale.snapshot.tolerances.Tolerance`.
        Defaults to ``None``, which compares every leaf for exact equality — the
        right default for a regional candidate against its regional snapshot,
        which should match bit-for-bit. Pass
        :data:`~saidownscale.snapshot.tolerances.TOLERANCES` (or a custom mapping) to
        restore an ``atol``/``rtol`` band instead. A variable absent from the
        mapping is compared exactly, so a partial mapping bands only the
        variables it names and leaves the rest strict.

    Returns
    -------
    DiffReport
        One :class:`LeafDiff` per (group, variable) leaf present in both inputs.
    """
    if type(candidate) is not type(snapshot):
        raise TypeError(f"candidate ({type(candidate)}) and snapshot ({type(snapshot)}) must match")

    # A method-namespaced candidate diffed against a pre-namespacing snapshot has its
    # leaves one level deeper than the snapshot's; see _select_matching_branch. Narrowed
    # first so every leaf below is looked up under matching paths.
    candidate = _select_matching_branch(candidate, snapshot)

    snapshot_groups = dict(_iter_leaf_datasets(snapshot))
    leaves: list[LeafDiff] = []
    for path, cand_ds in _iter_leaf_datasets(candidate):
        snap_ds = snapshot_groups.get(path)
        for variable, cand_da in cand_ds.data_vars.items():
            leaf_path = f"{path}/{variable}" if path else str(variable)
            if snap_ds is None or variable not in snap_ds.data_vars:
                leaves.append(
                    LeafDiff(
                        path=leaf_path,
                        variable=str(variable),
                        max_abs_diff=float("nan"),
                        rmse=float("nan"),
                        frac_over_tol=1.0,
                        nan_mismatch_count=-1,
                        shape_mismatch=True,
                        within_tol=False,
                    )
                )
                continue
            # A caller-supplied mapping is authoritative. A variable it omits is
            # compared exactly rather than falling back to the module-level band, so a
            # partial mapping can only tighten the check, never loosen it. The reverse
            # made `tolerances={}` the loosest setting available instead of the
            # strictest, which is the opposite of what the spelling suggests.
            tol = EXACT if tolerances is None else tolerances.get(str(variable), EXACT)
            rtol, atol = tol.rtol, tol.atol
            leaves.append(
                _compare_dataarray(
                    cand_da,
                    snap_ds[variable],
                    path=leaf_path,
                    variable=str(variable),
                    rtol=rtol,
                    atol=atol,
                )
            )
    return DiffReport(leaves=leaves)


@dataclass(frozen=True)
class InvariantCheck:
    """Result of one cross-variable physical-invariant check on the candidate.

    These are checked on the candidate alone (not diffed against the snapshot): a
    per-variable leaf diff compares tasmax and tasmin independently, so a broken
    reconcile that leaves each field matching the snapshot yet violates ``tasmax >= tasmin``
    would pass every leaf. The invariant closes that gap. The check itself lives in
    :mod:`saidownscale.snapshot.runs` (it needs the qaqc checker); this dataclass is just the
    reported shape.

    Parameters
    ----------
    path : str
        Identifier of the checked pairing, e.g. ``"ssp245/tasmax_ge_tasmin/008"``.
    holds : bool
        True when the invariant is satisfied at every cell.
    detail : str
        Empty when the invariant holds; otherwise a human-readable violation message
        (e.g. ``"tasmax < tasmin at 3 grid point(s)"``).
    """

    path: str
    holds: bool
    detail: str


@dataclass(frozen=True)
class DiffReport:
    """Aggregate comparison result across all leaves.

    Parameters
    ----------
    leaves : list of LeafDiff
        One entry per compared (group, variable) leaf.
    invariant_checks : list of InvariantCheck
        Cross-variable physical-invariant results on the candidate (e.g.
        ``tasmax >= tasmin``). Empty when no invariant applies to the compared leaves.
    """

    leaves: list[LeafDiff]
    invariant_checks: list[InvariantCheck] = field(default_factory=list)

    @property
    def within_tolerance(self) -> bool:
        """True only if every leaf exactly matches the snapshot (empty report is False).

        This reflects numeric drift vs the snapshot only. It does *not* fold in the
        physical-invariant checks — use :attr:`passed` for the overall merge verdict.
        """
        return bool(self.leaves) and all(leaf.within_tol for leaf in self.leaves)

    @property
    def invariants_hold(self) -> bool:
        """True if every cross-variable invariant on the candidate holds.

        Vacuously true when no invariant checks ran (e.g. a tas-only comparison).
        """
        return all(check.holds for check in self.invariant_checks)

    @property
    def passed(self) -> bool:
        """Overall merge verdict: exact match vs the snapshot *and* invariants hold."""
        return self.within_tolerance and self.invariants_hold

    def to_lines(self) -> list[str]:
        """Render a plain-text summary table."""
        header = f"{'leaf':40s} {'status':6s} {'max_abs':>12s} {'rmse':>12s} {'frac>tol':>10s}"
        rows = [header, "-" * len(header)]
        for leaf in self.leaves:
            status = "PASS" if leaf.within_tol else "FAIL"
            rows.append(
                f"{leaf.path:40s} {status:6s} {leaf.max_abs_diff:12.3e} "
                f"{leaf.rmse:12.3e} {leaf.frac_over_tol:10.3e}"
            )
        for check in self.invariant_checks:
            status = "PASS" if check.holds else "FAIL"
            rows.append(f"{check.path:40s} {status:6s} {check.detail}")
        return rows

    def to_table(self):
        """Render a ``rich`` table for console/CLI display."""
        from rich import box
        from rich.table import Table

        table = Table(show_header=True, header_style="bold", box=box.SIMPLE_HEAD)
        table.add_column("leaf", no_wrap=True)
        table.add_column("status", justify="center")
        table.add_column("max_abs_diff", justify="right")
        table.add_column("rmse", justify="right")
        table.add_column("frac>tol", justify="right")
        for leaf in self.leaves:
            status = "[green]✓[/green]" if leaf.within_tol else "[red]✗[/red]"
            table.add_row(
                leaf.path,
                status,
                f"{leaf.max_abs_diff:.3e}",
                f"{leaf.rmse:.3e}",
                f"{leaf.frac_over_tol:.3e}",
            )
        for check in self.invariant_checks:
            status = "[green]✓[/green]" if check.holds else "[red]✗[/red]"
            table.add_row(check.path, status, check.detail, "", "")
        return table
