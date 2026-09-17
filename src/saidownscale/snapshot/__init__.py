"""Snapshot comparison core for downscaling pipeline outputs.

This package provides the per-variable tolerances (:mod:`saidownscale.snapshot.tolerances`)
and the :func:`saidownscale.snapshot.compare.compare` engine. The
:func:`saidownscale.snapshot.runs.compare_runs` wrapper opens two output stores and diffs a
candidate run against the global snapshot, which is what the comparison notebook
calls.
"""

from saidownscale.snapshot.compare import DiffReport, LeafDiff, compare
from saidownscale.snapshot.runs import compare_runs
from saidownscale.snapshot.tolerances import DEFAULT_TOLERANCE, TOLERANCES, Tolerance, tolerance_for

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
