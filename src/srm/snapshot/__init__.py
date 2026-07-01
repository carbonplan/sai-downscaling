"""Snapshot regression testing for BCSD pipeline outputs.

Pairs syrupy-geo's snapshot storage and lifecycle with srm's climate-domain
comparison: per-variable tolerances (:mod:`srm.snapshot.tolerances`) and a
:func:`srm.snapshot.compare.compare` engine shared by the pytest gate, the
``bcsd compare`` CLI, and the comparison notebook.
"""

from srm.snapshot.compare import DiffReport, LeafDiff, compare
from srm.snapshot.tolerances import DEFAULT_TOLERANCE, TOLERANCES, Tolerance, tolerance_for

__all__ = [
    "DEFAULT_TOLERANCE",
    "TOLERANCES",
    "DiffReport",
    "LeafDiff",
    "Tolerance",
    "compare",
    "tolerance_for",
]
