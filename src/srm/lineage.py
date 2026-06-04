from __future__ import annotations

# Lineage lookup: (gcm, scenario, ensemble_member, variable) -> (historical_member, ssp245_member, ssp245_esgf_member)
# ssp245_member is None for non-SAI scenarios.
# ssp245_esgf_member is set when an ESGF SSP245 dataset is needed to fill a gap before the primary
# SSP245 bridge starts (MIROC G6-1.5K only: GeoMIP SSP245 starts 2020, leaving 2015–2019 gap).
# Source: docs/srm-provenance.csv parent_experiment_ensemble_ids column,
# confirmed by emails from Walker Lee (G6→SSP245) and Simone Tilmes (historical "001").


def _build_lineage() -> dict[tuple[str, str, str, str], tuple[str, str | None, str | None]]:
    table: dict[tuple[str, str, str, str], tuple[str, str | None, str | None]] = {}

    def add(
        gcm: str,
        scenario: str,
        member: str,
        variables: tuple[str, ...],
        hist: str,
        ssp245: str | None = None,
        ssp245_esgf: str | None = None,
    ) -> None:
        for var in variables:
            table[(gcm, scenario, member, var)] = (hist, ssp245, ssp245_esgf)

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

    # CESM2-WACCM SSP245 (no SAI bridge, ssp245_member is always None)
    # Members 001-005: standard variables only (tasmax/tasmin have the CMIP6 bug: not usable).
    # Members 006-010: tasmax/tasmin available via corrected run ("001"); all end 2069-12-31.
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
    # Data arrives in two parallel sets with different member ID formats:
    #   ripf format (r2/r3/r12i1p1f2): hurs and rsds only — sourced from Matthew Henry (CEDA/Exeter)
    #   numeric format (001/002/003): tas, tasmax, tasmin, pr, dtr — sourced from NCAR Derecho
    # Numeric → historical ripf mapping (inferred from G6-1.5K parent chains in docs/srm-provenance.csv;
    # lineage chain not formally confirmed): 001→r12i1p1f2, 002→r2i1p1f2, 003→r3i1p1f2
    # G6-1.5K numeric members use numeric SSP245 members as SAI bridge (same ID series).
    _ukesm_hurs_rsds = ("hurs", "rsds")
    _ukesm_t_pr = ("tas", "tasmax", "tasmin", "pr", "dtr")

    # SSP245: ripf members carry hurs/rsds (self-consistent, hist=self)
    for _m in ("r2i1p1f2", "r3i1p1f2", "r12i1p1f2"):
        add("UKESM", "SSP245", _m, _ukesm_hurs_rsds, _m)

    # SSP245: numeric members carry tas/tasmax/tasmin/pr/dtr (not yet in buckets as of 2026-06)
    add("UKESM", "SSP245", "001", _ukesm_t_pr, "r12i1p1f2")
    add("UKESM", "SSP245", "002", _ukesm_t_pr, "r2i1p1f2")
    add("UKESM", "SSP245", "003", _ukesm_t_pr, "r3i1p1f2")

    # G6-1.5K: ripf members carry hurs/rsds; SSP245 bridge uses same ripf member
    add("UKESM", "G6-1.5K", "r2i1p1f2", _ukesm_hurs_rsds, "r2i1p1f2", "r2i1p1f2")
    add("UKESM", "G6-1.5K", "r3i1p1f2", _ukesm_hurs_rsds, "r3i1p1f2", "r3i1p1f2")
    add("UKESM", "G6-1.5K", "r12i1p1f2", _ukesm_hurs_rsds, "r12i1p1f2", "r12i1p1f2")

    # G6-1.5K: numeric members carry tas/tasmax/tasmin/pr/dtr; SSP245 bridge uses same numeric ID
    add("UKESM", "G6-1.5K", "001", _ukesm_t_pr, "r12i1p1f2", "001")
    add("UKESM", "G6-1.5K", "002", _ukesm_t_pr, "r2i1p1f2", "002")
    add("UKESM", "G6-1.5K", "003", _ukesm_t_pr, "r3i1p1f2", "003")

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
        add("MIROC-ES2H", "SSP245", member, _all, hist)  # not SAI; no bridge
        add("MIROC-ES2H", "G6-1.5K", member, _all, hist, member, hist)

    return table


_LINEAGE = _build_lineage()


def resolve_member_lineage(
    gcm: str,
    scenario: str,
    ensemble_member: str,
    variable: str,
) -> tuple[str, str | None, str | None]:
    """Return (historical_member, ssp245_member, ssp245_esgf_member).

    ssp245_member is None for non-SAI scenarios.
    ssp245_esgf_member is set when an ESGF SSP245 dataset is needed to fill
    a gap before the primary SSP245 bridge starts (MIROC G6-1.5K only).
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


def get_lineage_entries(
    gcm: str, scenario: str
) -> dict[tuple[str, str], tuple[str, str | None, str | None]]:
    """Return {(member, variable): (historical_member, ssp245_member, ssp245_esgf_member)}.

    Returns an empty dict if no lineage is registered for the given gcm/scenario.
    """
    return {
        (m, v): parents for (g, s, m, v), parents in _LINEAGE.items() if g == gcm and s == scenario
    }
