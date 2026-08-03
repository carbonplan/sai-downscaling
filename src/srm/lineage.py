"""
Ensemble member lineage resolution for GCM scenarios.

Maps ``(gcm, scenario, ensemble_member, variable)`` tuples to the corresponding
historical, SSP245 bridge, and parent SAI members required by the BCSD detrending
workflow. All lineage logic belongs here; no other module should encode member
relationships.
"""

from __future__ import annotations

# Lineage lookup: (gcm, scenario, ensemble_member, variable) ->
#   (historical_member, ssp245_member, ssp245_esgf_member, sai_parent)
# ssp245_member is None for non-SAI scenarios.
# ssp245_esgf_member is set when an ESGF SSP245 dataset is needed to fill a gap before the primary
# SSP245 bridge starts (MIROC G6-1.5K only: GeoMIP SSP245 starts 2020, leaving 2015–2019 gap).
# sai_parent is (scenario, member) and is set when a scenario continues an earlier SAI run rather
# than branching off SSP245 (CESM G6-1.5K-END only: it resumes G6-1.5K 002 in 2085, so the bridge
# needs the parent's 2035–2084 years on top of the SSP245 years).
# Source: docs/srm-provenance.csv parent_experiment_ensemble_ids column,
# confirmed by emails from Walker Lee (G6→SSP245) and Simone Tilmes (historical "001").
# That column carries a third entry for G6-1.5K-END, which is what sai_parent records.

# (historical_member, ssp245_member, ssp245_esgf_member, sai_parent)
LineageEntry = tuple[str, str | None, str | None, tuple[str, str] | None]

# The one SAI run that another scenario continues.
_G6_002 = ("G6-1.5K", "002")


def _build_lineage() -> dict[tuple[str, str, str, str], LineageEntry]:
    table: dict[tuple[str, str, str, str], LineageEntry] = {}

    def add(
        gcm: str,
        scenario: str,
        member: str,
        variables: tuple[str, ...],
        hist: str,
        ssp245: str | None = None,
        ssp245_esgf: str | None = None,
        sai_parent: tuple[str, str] | None = None,
    ) -> None:
        for var in variables:
            table[(gcm, scenario, member, var)] = (hist, ssp245, ssp245_esgf, sai_parent)

    _std = (
        "tas",
        "pr",
        "rsds",
        "hurs",
    )
    _tmx = ("tasmax", "tasmin", "dtr")

    # CESM2-WACCM G6-1.5K
    # Standard variables branch from r1/r2/r3i1p1f1 historical and SSP245 001/002/003 bridge.
    # tasmax/tasmin use the corrected historical run ("001") due to the CMIP6 tasmax/tasmin bug;
    # their SSP245 bridges are 009/007/008 (provisional for 001: some open questions).
    add("CESM2-WACCM", "G6-1.5K", "001", _std, "r1i1p1f1", "001")
    add("CESM2-WACCM", "G6-1.5K", "001", _tmx, "001", "009")
    add("CESM2-WACCM", "G6-1.5K", "002", _std, "r2i1p1f1", "002")
    add("CESM2-WACCM", "G6-1.5K", "002", _tmx, "001", "007")
    add("CESM2-WACCM", "G6-1.5K", "003", _std, "r3i1p1f1", "003")
    add("CESM2-WACCM", "G6-1.5K", "003", _tmx, "001", "008")

    # CESM2-WACCM G6-1.5K-END: SAI stops after 2084 and member 002 runs on to 2100.
    # The store holds only 2085-2100, so the bridge spans two runs: SSP245 for 2015-2034,
    # then the parent G6-1.5K 002 for the 2035-2084 SAI years the termination continues.
    # All three parents come from the provenance sheet, which records this run's chain as
    # [('historical','r2i1p1f1'),('ssp245','002'),('g6_1p5k','002')] for the standard
    # variables and [('historical','001'),('ssp245','007'),('g6_1p5k','002')] for
    # tasmax/tasmin. The first two match G6-1.5K 002 exactly, as expected for a run that
    # continues that realization rather than branching afresh.
    add("CESM2-WACCM", "G6-1.5K-END", "002", _std, "r2i1p1f1", "002", sai_parent=_G6_002)
    add("CESM2-WACCM", "G6-1.5K-END", "002", _tmx, "001", "007", sai_parent=_G6_002)

    # CESM2-WACCM SSP245 (no SAI bridge, ssp245_member is always None)
    # Members 001-005: standard variables only (tasmax/tasmin have the CMIP6 bug: not usable).
    # Members 006-010: tasmax/tasmin available via corrected run ("001"); all end in 2069
    # (007-010 on 2069-12-31, 006 a day earlier - see _MEMBER_TIME_BOUNDS in srm.validation).
    # Scenario label matches catalog key case: "SSP245".
    add("CESM2-WACCM", "SSP245", "001", _std, "r1i1p1f1")
    add("CESM2-WACCM", "SSP245", "002", _std, "r2i1p1f1")
    add("CESM2-WACCM", "SSP245", "003", _std, "r3i1p1f1")
    add("CESM2-WACCM", "SSP245", "004", _std, "r2i1p1f1")
    add("CESM2-WACCM", "SSP245", "005", _std, "r3i1p1f1")
    add("CESM2-WACCM", "SSP245", "006", _std, "r1i1p1f1")
    add("CESM2-WACCM", "SSP245", "006", _tmx, "001")
    add("CESM2-WACCM", "SSP245", "007", _std, "r2i1p1f1")
    add("CESM2-WACCM", "SSP245", "007", _tmx, "001")
    add("CESM2-WACCM", "SSP245", "008", _std, "r3i1p1f1")
    add("CESM2-WACCM", "SSP245", "008", _tmx, "001")
    add("CESM2-WACCM", "SSP245", "009", _std, "r2i1p1f1")
    add("CESM2-WACCM", "SSP245", "009", _tmx, "001")
    add("CESM2-WACCM", "SSP245", "010", _std, "r3i1p1f1")
    add("CESM2-WACCM", "SSP245", "010", _tmx, "001")

    _all = _std + _tmx

    # UKESM1-0-LL (code gcm name: "UKESM")
    # As of #355, SSP245 and G6-1.5K are each consolidated into a single icechunk
    # store keyed by ripf members (r2/r3/r12i1p1f2) covering all variables, matching
    # the historical store's member IDs. Lineage is therefore self-referential:
    # hist=self for both scenarios, ssp245_bridge=self for the G6-1.5K SAI bridge.
    for _m in ("r2i1p1f2", "r3i1p1f2", "r12i1p1f2"):
        add("UKESM", "SSP245", _m, _all, _m)
        add("UKESM", "G6-1.5K", _m, _all, _m, _m)

    # MIROC-ES2H GeoMIP runs (r01–r10, abbreviated IDs, not CMIP6 ripf format).
    # SSP245 = paired SSP245-continuation runs (formerly "baseline"); these serve as
    # both the G6-1.5K bridge and the standalone SSP245 output product.
    # Historical parent cycles: r01/r04/r07/r10→r1i1p4f2, r02/r05/r08→r2i1p4f2, r03/r06/r09→r3i1p4f2.
    # Source: JAMSTEC GeoMIP server (Shingo Watanabe).
    _miroc_g6_lineage = [
        ("r01", "r1i1p4f2"),
        ("r02", "r2i1p4f2"),
        ("r03", "r3i1p4f2"),
        ("r04", "r1i1p4f2"),
        ("r05", "r2i1p4f2"),
        ("r06", "r3i1p4f2"),
        ("r07", "r1i1p4f2"),
        ("r08", "r2i1p4f2"),
        ("r09", "r3i1p4f2"),
        ("r10", "r1i1p4f2"),
    ]
    for member, hist in _miroc_g6_lineage:
        # GeoMIP SSP245 ("baseline") starts 2020; ESGF SSP245 (hist-format member IDs) fills 2015–2019.
        add("MIROC-ES2H", "SSP245", member, _all, hist, ssp245_esgf=hist)
        add("MIROC-ES2H", "G6-1.5K", member, _all, hist, member, hist)

    return table


_LINEAGE = _build_lineage()


def resolve_member_lineage(
    gcm: str,
    scenario: str,
    ensemble_member: str,
    variable: str,
) -> LineageEntry:
    """Return (historical_member, ssp245_member, ssp245_esgf_member, sai_parent).

    ssp245_member is None for non-SAI scenarios.
    ssp245_esgf_member is set when an ESGF SSP245 dataset is needed to fill
    a gap before the primary SSP245 bridge starts (MIROC G6-1.5K only).
    sai_parent is ``(scenario, member)`` when this scenario continues an earlier
    SAI run whose years the bridge must also cover (CESM G6-1.5K-END only).
    Raises KeyError if the combination has no registered lineage.
    """
    key = (gcm, scenario, ensemble_member, variable)
    if key not in _LINEAGE:
        raise KeyError(
            f"No registered lineage for gcm={gcm!r}, scenario={scenario!r}, "
            f"ensemble_member={ensemble_member!r}, variable={variable!r}. "
            f"Add this combination to the lookup table in srm/lineage.py."
        )
    return _LINEAGE[key]


def get_lineage_entries(gcm: str, scenario: str) -> dict[tuple[str, str], LineageEntry]:
    """Return {(member, variable): (historical, ssp245, ssp245_esgf, sai_parent)}.

    Returns an empty dict if no lineage is registered for the given gcm/scenario.
    """
    return {
        (m, v): parents for (g, s, m, v), parents in _LINEAGE.items() if g == gcm and s == scenario
    }
