"""Guard the Terms of Data Access copies against drift.

``TERMS_OF_DATA_ACCESS`` at the repository root is the canonical text. Every other copy, in the
docs site and anywhere else we publish it, is generated from that file and must match it byte for
byte. Edit the canonical file and copy it outward, never the other way round.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "TERMS_OF_DATA_ACCESS"
COPIES = [REPO_ROOT / "docs" / "access-data" / "terms-of-data-access.md"]

# The clauses the CarbonPlan Terms of Data Access guidance requires verbatim. They are listed
# separately from the whole-file comparison so a reworded canonical file fails loudly too.
REQUIRED_CLAUSES = [
    "## Our Terms of Use apply to SAI Downscaling",
    "you agree to CarbonPlan’s Terms Of Use",
    "https://carbonplan.org/terms",
    "data provided by CarbonPlan may originate from third parties",
    "projects — including SAI Downscaling — available strictly on an "
    "“as-is” and “as-available” basis",
    "without warranty of any kind",
    "accuracy, completeness, merchantability, fitness for a particular purpose, or noninfringement",
]


def test_canonical_file_exists() -> None:
    assert CANONICAL.is_file(), f"{CANONICAL.name} is the single source of truth and must exist"


@pytest.mark.parametrize("clause", REQUIRED_CLAUSES)
def test_canonical_keeps_required_clauses(clause: str) -> None:
    """The legal wording is quoted text"""
    assert clause in CANONICAL.read_text(), (
        f"{CANONICAL.name} no longer contains the required clause: {clause!r}"
    )


@pytest.mark.parametrize("copy", COPIES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_copies_match_canonical(copy: Path) -> None:
    assert copy.is_file(), f"{copy.relative_to(REPO_ROOT)} is missing"
    assert copy.read_text() == CANONICAL.read_text(), (
        f"{copy.relative_to(REPO_ROOT)} has drifted from {CANONICAL.name}. "
        f"Re-sync with: cp {CANONICAL.name} {copy.relative_to(REPO_ROOT)}"
    )


LICENSES_PAGE = REPO_ROOT / "docs" / "access-data" / "licenses.md"


def _documented_license_rows() -> set[tuple[str, str]]:
    """Return the (gcm, scenario_group) pairs the licenses page attributes."""
    rows: set[tuple[str, str]] = set()
    for line in LICENSES_PAGE.read_text().split("\n"):
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if len(cells) < 4 or cells[0] in {"GCM", "---"}:
            continue
        rows.add((cells[0], cells[1]))
    return rows


def _published_pairs() -> set[tuple[str, str]]:
    """Return every (gcm, scenario_group) the pipeline publishes, historical included.

    ``all_lineage_keys`` covers the scenario legs only, because historical has no SAI parent to
    resolve, so we add a historical leg per GCM to match what the output stores actually hold.
    """
    from saidownscale.config import SCENARIO_TO_GROUP
    from saidownscale.lineage import all_lineage_keys

    pairs = {(gcm, SCENARIO_TO_GROUP.get(scen, scen)) for gcm, scen, _, _ in all_lineage_keys()}
    return pairs | {(gcm, "historical") for gcm, _ in pairs}


def test_every_published_scenario_is_attributed() -> None:
    """A new GCM or scenario must not ship without a license and citation."""
    missing = _published_pairs() - _documented_license_rows()
    assert not missing, (
        "These published (GCM, scenario) pairs have no row in "
        f"{LICENSES_PAGE.relative_to(REPO_ROOT)}: {sorted(missing)}"
    )
