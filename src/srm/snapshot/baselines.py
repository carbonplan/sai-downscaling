"""Tracked pointer to the canonical snapshot baseline.

``CESM2_WACCM_GLOBAL.branch`` is the single source of truth for the snapshot
branch. It is both the icechunk branch holding the blessed global run (read by
``bcsd compare`` and the snapshot workflow's ``scope: global``) and the branch the
South Africa gate runs on: the ``snapshot`` workflow derives ``BCSD_BRANCH`` from
this value, and the syrupy baseline is stored under ``snapshots/<branch>``.

Blessing a new baseline version is realized by editing this file in a PR (issue
#410's "update the snapshot to point at the new global run"), so the baseline is
always a reviewed, version-controlled change and the global and South Africa paths
move together.
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


# Bump ``branch`` here to start a new baseline version; the snapshot workflow and
# the South Africa gate both follow this value. Data must live on the named branch
# ("main" is an empty anchor and must not be used). Verified: "v0.7.0" holds the
# blessed global run.
CESM2_WACCM_GLOBAL = GlobalBaseline(
    uri="s3://carbonplan-srm/output/production/CESM2-WACCM-ERA5-global.icechunk",
    branch="v0.7.0",
)
