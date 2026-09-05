"""
Ensemble member lineage resolution for GCM scenarios.

Maps ``(gcm, scenario, ensemble_member, variable)`` tuples to the corresponding
historical, SSP245 bridge, and parent SAI members required by the BCSD detrending
workflow. All lineage logic belongs here; no other module should encode member
relationships.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

# Lineage lookup: (gcm, scenario, ensemble_member, variable) -> LineageEntry.
# Source: docs/srm-provenance.csv parent_experiment_ensemble_ids column,
# confirmed by emails from Walker Lee (G6→SSP245) and Simone Tilmes (historical "001").
# That column carries a third entry for G6-1.5K-END, which is what sai_parent records.


@dataclass(frozen=True, slots=True, kw_only=True)
class ScenarioMember:
    """A ``(scenario, member)`` pair naming one run in the lineage table.

    Keyword-only because both fields are plain strings, so a positional swap would
    type-check cleanly and silently resolve the wrong run. ``__str__`` renders the
    ``scenario/member`` token that the CLI table, the pipeline log line, and the
    lineage notebook all display.
    """

    scenario: str
    member: str

    def __str__(self) -> str:
        return f"{self.scenario}/{self.member}"


@dataclass(frozen=True, slots=True, kw_only=True)
class LineageEntry:
    """Resolved parent members for one lineage key.

    Keyword-only and frozen on purpose. This was a positional 4-tuple that grew from two
    fields to four, and ``ssp245_bridge`` and ``ssp245_esgf_bridge`` are both ``str | None``
    and adjacent, so a positional mix-up type-checked cleanly and produced wrong output
    rather than an error.

    Attributes
    ----------
    historical : str
        Historical ensemble member this run branches from.
    ssp245_bridge : str or None
        SSP245 member bridging the pre-SAI years. ``None`` for non-SAI scenarios.
    ssp245_esgf_bridge : str or None
        ESGF SSP245 member filling a gap before the primary bridge starts, for a GCM whose
        scenario data begins after the historical period ends. No registered GCM sets this
        today; it is left in place for the next dataset that needs the gap filled.
    sai_parent : ScenarioMember or None
        Earlier SAI run this scenario continues rather than branching off SSP245. Set only
        for CESM G6-1.5K-END, which resumes G6-1.5K 002 in 2085, so its bridge needs the
        parent's 2035-2084 years on top of the SSP245 years.
    """

    historical: str
    ssp245_bridge: str | None = None
    ssp245_esgf_bridge: str | None = None
    sai_parent: ScenarioMember | None = None


# The one SAI run that another scenario continues.
_G6_002 = ScenarioMember(scenario="G6-1.5K", member="002")

# UKESM1-1-LL historical is a single UM suite, not a CMIP6 realization, so its ID is the suite name
# rather than a ripf label. Every UKESM1-1-LL scenario member and variable branches from this one run.
_UKESM_HIST = "u-by791"


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
        sai_parent: ScenarioMember | None = None,
    ) -> None:
        for var in variables:
            table[(gcm, scenario, member, var)] = LineageEntry(
                historical=hist,
                ssp245_bridge=ssp245,
                ssp245_esgf_bridge=ssp245_esgf,
                sai_parent=sai_parent,
            )

    _std = (
        "tas",
        "pr",
        "rsds",
        "hurs",
    )
    _tmx = ("tasmax", "tasmin", "dtr")

    # CESM2-WACCM6 G6-1.5K
    # Standard variables branch from r1/r2/r3i1p1f1 historical and SSP245 001/002/003 bridge.
    # tasmax/tasmin use the corrected historical run ("001") due to the CMIP6 tasmax/tasmin bug;
    # their SSP245 bridges are 009/007/008 (provisional for 001: some open questions).
    add("CESM2-WACCM6", "G6-1.5K", "001", _std, "r1i1p1f1", "001")
    add("CESM2-WACCM6", "G6-1.5K", "001", _tmx, "001", "009")
    add("CESM2-WACCM6", "G6-1.5K", "002", _std, "r2i1p1f1", "002")
    add("CESM2-WACCM6", "G6-1.5K", "002", _tmx, "001", "007")
    add("CESM2-WACCM6", "G6-1.5K", "003", _std, "r3i1p1f1", "003")
    add("CESM2-WACCM6", "G6-1.5K", "003", _tmx, "001", "008")

    # CESM2-WACCM6 G6-1.5K-END: SAI stops after 2084 and member 002 runs on to 2100.
    # The store holds only 2085-2100, so the bridge spans two runs: SSP245 for 2015-2034,
    # then the parent G6-1.5K 002 for the 2035-2084 SAI years the termination continues.
    # All three parents come from the provenance sheet, which records this run's chain as
    # [('historical','r2i1p1f1'),('ssp245','002'),('g6_1p5k','002')] for the standard
    # variables and [('historical','001'),('ssp245','007'),('g6_1p5k','002')] for
    # tasmax/tasmin. The first two match G6-1.5K 002 exactly, as expected for a run that
    # continues that realization rather than branching afresh.
    add("CESM2-WACCM6", "G6-1.5K-END", "002", _std, "r2i1p1f1", "002", sai_parent=_G6_002)
    add("CESM2-WACCM6", "G6-1.5K-END", "002", _tmx, "001", "007", sai_parent=_G6_002)

    # CESM2-WACCM6 SSP245 (no SAI bridge, ssp245_bridge is always None)
    # Members 001-005: standard variables only (tasmax/tasmin have the CMIP6 bug: not usable).
    # Members 006-010: tasmax/tasmin available via corrected run ("001"); all end in 2069
    # (007-010 on 2069-12-31, 006 a day earlier - see _MEMBER_TIME_BOUNDS in saidownscale.validation).
    # Scenario label matches catalog key case: "SSP245".
    add("CESM2-WACCM6", "SSP245", "001", _std, "r1i1p1f1")
    add("CESM2-WACCM6", "SSP245", "002", _std, "r2i1p1f1")
    add("CESM2-WACCM6", "SSP245", "003", _std, "r3i1p1f1")
    add("CESM2-WACCM6", "SSP245", "004", _std, "r2i1p1f1")
    add("CESM2-WACCM6", "SSP245", "005", _std, "r3i1p1f1")
    add("CESM2-WACCM6", "SSP245", "006", _std, "r1i1p1f1")
    add("CESM2-WACCM6", "SSP245", "006", _tmx, "001")
    add("CESM2-WACCM6", "SSP245", "007", _std, "r2i1p1f1")
    add("CESM2-WACCM6", "SSP245", "007", _tmx, "001")
    add("CESM2-WACCM6", "SSP245", "008", _std, "r3i1p1f1")
    add("CESM2-WACCM6", "SSP245", "008", _tmx, "001")
    add("CESM2-WACCM6", "SSP245", "009", _std, "r2i1p1f1")
    add("CESM2-WACCM6", "SSP245", "009", _tmx, "001")
    add("CESM2-WACCM6", "SSP245", "010", _std, "r3i1p1f1")
    add("CESM2-WACCM6", "SSP245", "010", _tmx, "001")

    _all = _std + _tmx

    # UKESM1-1-LL
    # Every file from this delivery is UKESM1-1-LL: source metadata and filenames that say
    # UKESM1-0-LL / UKESM1-1 are supplier labelling typos (confirmed by email), corrected on
    # ingest in saidownscale.input_data.ukesm.
    # SSP245 and G6-1.5K are each a single icechunk (r2/r3/r12i1p1f2) covering all variables.
    # The Historical scenario has a single ensemble_member: u-by791, so scenario members share that single historical parent.
    # SSP245 has no hurs.
    _ukesm_ssp245 = tuple(v for v in _all if v != "hurs")
    for _m in ("r2i1p1f2", "r3i1p1f2", "r12i1p1f2"):
        add("UKESM1-1-LL", "SSP245", _m, _ukesm_ssp245, _UKESM_HIST)
        add("UKESM1-1-LL", "G6-1.5K", _m, _all, _UKESM_HIST, _m)

    return table


_LINEAGE = _build_lineage()


def resolve_member_lineage(
    gcm: str,
    scenario: str,
    ensemble_member: str,
    variable: str,
) -> LineageEntry:
    """Return the :class:`LineageEntry` for one lineage key.

    ``ssp245_bridge`` is None for non-SAI scenarios. ``ssp245_esgf_bridge`` is set when an
    ESGF SSP245 dataset is needed to fill a gap before the primary SSP245 bridge starts;
    no registered GCM needs that today. ``sai_parent`` is a :class:`ScenarioMember` when this scenario
    continues an earlier SAI run whose years the bridge must also cover (CESM G6-1.5K-END
    only). Raises KeyError if the combination has no registered lineage.
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
    """Return ``{(member, variable): LineageEntry}`` for one GCM and scenario.

    Returns an empty dict if no lineage is registered for the given gcm/scenario.
    """
    return {
        (m, v): parents for (g, s, m, v), parents in _LINEAGE.items() if g == gcm and s == scenario
    }


def all_lineage_keys() -> list[tuple[str, str, str, str]]:
    """Return every registered ``(gcm, scenario, ensemble_member, variable)`` key, sorted."""
    return sorted(_LINEAGE)


# The provenance sheet labels GCMs and one scenario differently from the code. These maps
# are only for reconciling the two; nothing in the pipeline reads the sheet at runtime.
# A GCM absent from this map has its sheet rows skipped entirely, which is how the
# sheet's MIROC-ES2H rows stay out of the reconciliation without being deleted from a
# verbatim re-export.
PROVENANCE_GCM_ALIASES: dict[str, str] = {
    "CESM2(WACCM)": "CESM2-WACCM6",
    # Identity, kept so the sheet's UKESM rows are reconciled instead of skipped.
    "UKESM1-1-LL": "UKESM1-1-LL",
}
PROVENANCE_SCENARIO_ALIASES: dict[str, str] = {
    "G6-1.5K-end": "G6-1.5K-END",
}


def diff_against_provenance(
    csv_path: str | Path,
) -> dict[str, list[tuple[str, str, str, str]] | list[str]]:
    """Reconcile the lineage table against the provenance sheet.

    The pipeline resolves members from :data:`_LINEAGE`, while ``docs/srm-provenance.csv``
    is a re-export of the upstream tracking sheet. The two can drift, and nothing else
    notices when they do, so this returns the differences rather than assuming they agree.

    The sheet is the source of truth, so anything this reports is a defect to fix upstream
    rather than something to work around locally. ``docs/srm-provenance.csv`` is a verbatim
    re-export and must not be hand-patched: an edit there is clobbered by the next export
    and hides the very disagreement this function exists to surface.

    Returns a dict with four keys. ``only_in_code`` and ``only_in_sheet`` hold
    ``(gcm, scenario, member, variable)`` keys present on one side alone,
    ``parent_mismatch`` holds human-readable descriptions of keys both sides carry but
    disagree about, and ``malformed`` lists cells the sheet writes in a form that cannot
    be parsed. ``historical`` rows in the sheet are ignored, since the lineage table
    registers scenarios only.
    """
    import csv as _csv

    sheet: dict[tuple[str, str, str, str], dict[str, str]] = {}
    malformed: list[str] = []
    with open(csv_path, newline="") as handle:
        for row in _csv.DictReader(handle):
            gcm = PROVENANCE_GCM_ALIASES.get((row["gcm"] or "").strip())
            scenario_raw = (row["experiment_id"] or "").strip()
            parents_raw = (row["parent_experiment_ensemble_ids"] or "").strip()
            if gcm is None or scenario_raw in ("", "historical") or not parents_raw:
                continue
            scenario = PROVENANCE_SCENARIO_ALIASES.get(scenario_raw, scenario_raw)
            member = (row["ensemble_id"] or "").strip()
            variable = (row["variable"] or "").strip()
            key = (gcm, scenario, member, variable)
            try:
                parents = dict(ast.literal_eval(parents_raw))
            except (SyntaxError, ValueError):
                # Reported rather than skipped. The sheet is the source of truth, so a cell
                # it cannot express is a defect to fix upstream, and swallowing it here
                # would drop the row from every comparison below without a trace.
                malformed.append(f"{'/'.join(key)}: cannot parse {parents_raw!r}")
                continue
            sheet[key] = parents

    code = {k: v for k, v in _LINEAGE.items()}
    only_in_code = sorted(set(code) - set(sheet))
    only_in_sheet = sorted(set(sheet) - set(code))

    parent_mismatch: list[str] = []
    for key in sorted(set(code) & set(sheet)):
        entry = code[key]
        parents = sheet[key]
        expected = {"historical": entry.historical}
        if entry.ssp245_bridge is not None:
            expected["ssp245"] = entry.ssp245_bridge
        if entry.sai_parent is not None:
            expected["g6_1p5k"] = entry.sai_parent.member
        for role, want in expected.items():
            got = parents.get(role)
            if got != want:
                parent_mismatch.append(
                    f"{'/'.join(key)}: {role} is {want!r} in code, {got!r} in the sheet"
                )

    return {
        "only_in_code": only_in_code,
        "only_in_sheet": only_in_sheet,
        "parent_mismatch": parent_mismatch,
        "malformed": sorted(malformed),
    }
