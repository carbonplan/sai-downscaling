"""Snapshot comparison core for BCSD pipeline outputs.

This package provides the per-variable tolerances (:mod:`srm.snapshot.tolerances`)
and the :func:`srm.snapshot.compare.compare` engine. The
:func:`srm.snapshot.runs.compare_runs` wrapper opens two output stores and diffs a
candidate run against the global snapshot, which is what the comparison notebook
calls.
"""

from srm.snapshot.compare import DiffReport, LeafDiff, compare
from srm.snapshot.runs import compare_runs
from srm.snapshot.tolerances import DEFAULT_TOLERANCE, TOLERANCES, Tolerance, tolerance_for

__all__ = [
    "DEFAULT_TOLERANCE",
    "TOLERANCES",
    "DiffReport",
    "LeafDiff",
    "Tolerance",
    "compare",
    "compare_runs",
    "tolerance_for",
]
