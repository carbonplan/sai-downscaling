"""Tracked pointer to the canonical global snapshot baseline.

``CESM2_WACCM_GLOBAL`` records the icechunk store and branch holding the approved
global run. The comparison notebook and :func:`srm.snapshot.runs.compare_runs` read
this pointer. Approving a new baseline version is a reviewed edit to this file (issue
#410's "update the snapshot to point at the new global run").
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
        icechunk branch holding the approved global run.
    """

    uri: str
    branch: str


# Bump ``branch`` here to start a new baseline version; the comparison notebook and
# ``compare_runs`` both read this value. Data must live on the named branch ("main"
# is an empty anchor and must not be used). Verified: "v0.10.0" holds the approved
# global run.
CESM2_WACCM_GLOBAL = GlobalBaseline(
    uri="s3://carbonplan-srm/output/production/CESM2-WACCM-ERA5-global.icechunk",
    branch="v0.10.0",
)
