from __future__ import annotations

# Lineage lookup: (gcm, scenario, ensemble_member, variable) -> (historical_member, ssp245_member)
# ssp245_member is None for non-SAI scenarios.
# Source: srm-provenance.csv parent_experiment_ensemble_ids column,
# confirmed by emails from Walker Lee (G6→SSP245) and Simone Tilmes (historical "001").


def _build_lineage() -> dict[tuple[str, str, str, str], tuple[str, str | None]]:
    table: dict[tuple[str, str, str, str], tuple[str, str | None]] = {}

    def add(
        gcm: str,
        scenario: str,
        member: str,
        variables: tuple[str, ...],
        hist: str,
        ssp245: str | None = None,
    ) -> None:
        for var in variables:
            table[(gcm, scenario, member, var)] = (hist, ssp245)

    _std = ("tas", "pr", "rsds", "hurs")
    _tmx = ("tasmax", "tasmin")

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

    return table


_LINEAGE = _build_lineage()


def resolve_member_lineage(
    gcm: str,
    scenario: str,
    ensemble_member: str,
    variable: str,
) -> tuple[str, str | None]:
    """Return (historical_member, ssp245_member).

    ssp245_member is None for non-SAI scenarios (SSP245 runs do not need a
    bridge). Raises KeyError if the combination has no registered lineage.
    """
    key = (gcm, scenario, ensemble_member, variable)
    if key not in _LINEAGE:
        raise KeyError(
            f"No registered lineage for gcm={gcm!r}, scenario={scenario!r}, "
            f"ensemble_member={ensemble_member!r}, variable={variable!r}. "
            f"Add this combination to the lookup table in srm/lineage.py."
        )
    return _LINEAGE[key]
