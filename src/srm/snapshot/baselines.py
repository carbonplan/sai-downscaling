"""Tracked pointer to the canonical global and regional snapshot baseline.

``CESM2_WACCM_GLOBAL`` records the icechunk store and branch holding the approved
global run. The comparison notebook and :func:`srm.snapshot.runs.compare_runs` read
this pointer. Blessing a new baseline version is a reviewed edit to this file (issue
#410's "update the snapshot to point at the new global run").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Baseline:
    """A snapshot store and the icechunk branch to read it on.

    Parameters
    ----------
    uri : str
        ``s3://`` URI of the icechunk store.
    branch : str
        icechunk branch holding the approved global run.
    """

    uri: str
    branch: str


# Bump ``uri``/``branch`` here to start a new baseline version; the comparison notebook
# and ``compare_runs`` both read this value. Data must live on the named branch ("main"
# is an empty anchor and must not be used). Verified 2026-07-28: this store's branches
# are {"main", "v0.12.0"}, and "v0.12.0" holds the approved global run.
#
# Published production output moved off the private ``carbonplan-srm`` bucket to
# CarbonPlan's Source Cooperative repository. The bucket is public, but
# ``srm.validation._open_output_datatree`` opens it with ``from_env=True``; signed
# requests from another account are accepted, so no anonymous-access special case is
# needed here.
CESM2_WACCM_GLOBAL = Baseline(
    uri=(
        "s3://us-west-2.opendata.source.coop/carbonplan/srm-downscaling"
        "/output/production/CESM2-WACCM-ERA5-global.icechunk"
    ),
    branch="v0.12.0",
)

# Deliberately still on ``carbonplan-scratch`` while other non-production paths moved to
# ``s3://carbonplan-srm/scratch/``: copying 302 GiB to reach one of 20 branches buys
# nothing, and a baseline does not belong in a prefix meant to be deleted wholesale. The
# next release rebuilds it under ``scratch/snapshot/output/qa/``.
#
# ``branch`` is writable, not a frozen tag: the ``bcsd release`` freeze step has never run
# here, so a run with a matching ``BCSD_BRANCH`` could overwrite what this cites.
CESM2_WACCM_SOUTH_AFRICA = Baseline(
    uri=(
        "s3://carbonplan-srm/scratch/snapshot"
        "/output/qa/CESM2-WACCM-ERA5-lat-38.0to-19.0_lon13.0to36.0.icechunk"
    ),
    branch="pr-655-run-1",
)
