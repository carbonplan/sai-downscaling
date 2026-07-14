"""Tolerance-aware comparison of BCSD output datasets and datatrees.

The single :func:`compare` engine backs the comparison notebook (via
:func:`srm.snapshot.runs.compare_runs`), so the notebook's verdict comes straight
from this code. All reductions are chunk-friendly so a few-GB South Africa diff runs
without loading whole arrays into memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import xarray as xr

from srm.snapshot.tolerances import TOLERANCES, Tolerance, tolerance_for


@dataclass(frozen=True)
class LeafDiff:
    """Comparison result for one (group, variable) leaf.

    Parameters
    ----------
    path : str
        Group path of the leaf within the datatree (e.g. ``"g6_1p5k/tas"``).
    variable : str
        Canonical variable name driving tolerance selection.
    max_abs_diff : float
        Maximum absolute difference over all cells (NaN if shapes mismatch).
    rmse : float
        Root-mean-square difference over all cells.
    frac_over_tol : float
        Fraction of cells exceeding ``atol + rtol * abs(snapshot)``.
    nan_mismatch_count : int
        Count of cells where candidate and snapshot disagree on NaN-ness.
    shape_mismatch : bool
        True if candidate and snapshot have different shapes.
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
    rtol: float,
    atol: float,
) -> LeafDiff:
    """Compare two DataArrays under absolute/relative tolerance.

    Both arrays are compared element-wise using the same tolerance rule as
    :func:`xarray.testing.assert_allclose`: a cell passes when its absolute
    difference is at most ``atol + rtol * abs(snapshot)``. The function reduces
    the full element-wise comparison down to the scalar summary statistics
    carried by :class:`LeafDiff`, so callers get a compact verdict instead of a
    whole diff array.

    If the two arrays have different shapes they cannot be compared cell-by-cell,
    so the function short-circuits: it returns a :class:`LeafDiff` with
    ``shape_mismatch=True``, ``within_tol=False``, and NaN/sentinel metrics
    rather than raising. Otherwise every metric is computed and ``within_tol``
    reflects the real element-wise verdict.

    Parameters
    ----------
    candidate, snapshot : xr.DataArray
        New and baseline arrays. ``snapshot`` is the reference in the relative
        term, matching :func:`xarray.testing.assert_allclose`.
    path : str
        Leaf group path, for reporting.
    variable : str
        Canonical variable name, for reporting.
    rtol, atol : float
        Relative and absolute tolerance. ``atol`` sets a fixed floor that
        dominates near zero, while ``rtol`` scales the allowance with the
        magnitude of ``snapshot`` and dominates for large values.

    Returns
    -------
    LeafDiff
        The populated comparison result. ``within_tol`` is authoritative and is
        taken from :func:`xarray.testing.assert_allclose`; the other fields
        (``max_abs_diff``, ``rmse``, ``frac_over_tol``, ``nan_mismatch_count``)
        quantify *how far* off the arrays are when they disagree.

    Notes
    -----
    All reductions call ``.compute()`` on lazy aggregates only, never on the
    full arrays, so a multi-GB diff is summarized without materializing it in
    memory. NaN cells are skipped in ``max_abs_diff`` and ``rmse`` via
    ``skipna=True``, and disagreement on NaN-ness is tracked separately through
    ``nan_mismatch_count``.
    """
    if candidate.shape != snapshot.shape:
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
    # Per-element allowed difference, using numpy's isclose convention:
    # a fixed floor (atol) plus a term that scales with the reference value
    # (rtol * |snapshot|). atol dominates near zero; rtol dominates for large values.
    tol = atol + rtol * abs(snapshot)
    over = abs_diff > tol  # elements whose error exceeds their tolerance
    nan_mismatch = candidate.isnull() != snapshot.isnull()  # NaN in one array but not the other

    max_abs_diff = float(abs_diff.max(skipna=True).compute())
    rmse = float(((diff**2).mean(skipna=True) ** 0.5).compute())
    frac_over_tol = float(over.mean().compute())
    nan_mismatch_count = int(nan_mismatch.sum().compute())

    try:
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
        Mapping of variable name to :class:`~srm.snapshot.tolerances.Tolerance`.
        Defaults to :data:`~srm.snapshot.tolerances.TOLERANCES`; unknown
        variables fall back to the default tolerance.

    Returns
    -------
    DiffReport
        One :class:`LeafDiff` per (group, variable) leaf present in both inputs.
    """
    if type(candidate) is not type(snapshot):
        raise TypeError(f"candidate ({type(candidate)}) and snapshot ({type(snapshot)}) must match")

    table = TOLERANCES if tolerances is None else tolerances
    snapshot_groups = dict(_iter_leaf_datasets(snapshot))
    leaves: list[LeafDiff] = []
    for path, cand_ds in _iter_leaf_datasets(candidate):
        snap_ds = snapshot_groups.get(path)
        for variable, cand_da in cand_ds.data_vars.items():
            leaf_path = f"{path}/{variable}" if path else str(variable)
            # Pick this variable's rtol/atol: exact match in the table, else a
            # sensible per-variable default. Its .rtol/.atol flow into the
            # atol + rtol * abs(snapshot) tolerance inside _compare_dataarray.
            tol = table.get(str(variable)) or tolerance_for(str(variable))
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
            leaves.append(
                _compare_dataarray(
                    cand_da,
                    snap_ds[variable],
                    path=leaf_path,
                    variable=str(variable),
                    rtol=tol.rtol,
                    atol=tol.atol,
                )
            )
    return DiffReport(leaves=leaves)


@dataclass(frozen=True)
class DiffReport:
    """Aggregate comparison result across all leaves.

    Parameters
    ----------
    leaves : list of LeafDiff
        One entry per compared (group, variable) leaf.
    """

    leaves: list[LeafDiff]

    @property
    def within_tolerance(self) -> bool:
        """True only if every leaf is within tolerance (empty report is False)."""
        return bool(self.leaves) and all(leaf.within_tol for leaf in self.leaves)

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
        return table
