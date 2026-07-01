"""syrupy-geo extension that judges snapshots with srm's per-variable tolerance.

syrupy-geo keeps the storage, lifecycle, and ``--snapshot-update`` flow; this
subclass replaces its hardcoded ``assert_allclose`` verdict with the srm
:func:`~srm.snapshot.compare.compare` engine so the gate uses the same
per-variable tolerances and rich diff as the CLI and notebook.
"""

from __future__ import annotations

import typing

from syrupy_geo.xarray import XarraySnapshotExtension

from srm.snapshot.compare import compare


class SrmXarraySnapshotExtension(XarraySnapshotExtension):
    """Xarray snapshot extension using srm's tolerance policy and diff."""

    def matches(self, *, serialized_data: typing.Any, snapshot_data: typing.Any) -> bool:
        """Return True when the candidate is within per-variable tolerance."""
        if snapshot_data is None:
            return False
        if type(serialized_data) is not type(snapshot_data):
            return False
        try:
            return compare(serialized_data, snapshot_data).within_tolerance
        except (TypeError, ValueError):
            return False

    def diff_lines(
        self, serialized_data: typing.Any, snapshot_data: typing.Any
    ) -> typing.Iterator[str]:
        """Yield the srm DiffReport table as plain text for failure output."""
        if snapshot_data is None:
            yield "No stored snapshot found."
            return
        report = compare(serialized_data, snapshot_data)
        yield "Snapshot comparison failed (srm per-variable tolerance)"
        yield from report.to_lines()
