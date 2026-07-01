"""Tracked pointers to the canonical global snapshot baselines.

"Update the snapshot to point at the new global run" (issue #410) is realized by
editing this file in a PR, so the baseline a comparison runs against is always a
reviewed, version-controlled change. The South Africa gate does not use this — it
compares against the syrupy-geo snapshot store; this is for the global path
(``bcsd compare`` and the snapshot workflow's ``scope: global``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalBaseline:
    """A global snapshot store and the icechunk branch to read it on.

    Parameters
    ----------
    uri : str
        ``s3://`` URI of the icechunk store.
    branch : str
        icechunk branch holding the blessed global run.
    """

    uri: str
    branch: str


# Repoint by editing this value in a PR once a new global run is approved.
# NOTE: data lives on a version branch (e.g. "v1"); "main" is an empty anchor and
# must not be used here. Confirm the real branch with the discovery command in
# Task 6 Step 3b before relying on `bcsd compare` against the baseline.
CESM2_WACCM_GLOBAL = GlobalBaseline(
    uri="s3://carbonplan-srm/output/production/CESM2-WACCM-ERA5-global.icechunk",
    branch="v0.7.0",
)
