"""Guard the Terms of Data Access and licenses docs against drift from their sources."""

import re
from pathlib import Path

from saidownscale.licenses import INPUT_ATTRIBUTION, LICENSE_URLS, TERMS_OF_DATA_ACCESS

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "TERMS_OF_DATA_ACCESS"
COPIES = [REPO_ROOT / "docs" / "access-data" / "terms-of-data-access.md"]
LICENSES_PAGE = REPO_ROOT / "docs" / "access-data" / "licenses.md"
REQUIRED_CLAUSES = [
    "## Our Terms of Use Apply to SAI Downscaling",
    "you agree to CarbonPlan’s Terms Of Use",
    "https://carbonplan.org/terms",
    "data provided by CarbonPlan may originate from third parties",
    "projects — including SAI Downscaling — available strictly on an "
    "“as-is” and “as-available” basis",
    "without warranty of any kind",
    "accuracy, completeness, merchantability, fitness for a particular purpose, or noninfringement",
]
MIN_ATTRIBUTION_CHARS = 40


def _license_table_rows() -> list[tuple[str, str, str, str]]:
    rows = []
    for line in LICENSES_PAGE.read_text().split("\n"):
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if len(cells) != 4 or cells[0] == "GCM" or set(cells[0]) <= {"-", ":"}:
            continue
        rows.append(tuple(cells))
    return rows


def _published_pairs() -> set[tuple[str, str]]:
    """Every published (gcm, scenario_group); lineage keys omit historical, so add it per GCM."""
    from saidownscale.config import SCENARIO_TO_GROUP
    from saidownscale.lineage import all_lineage_keys

    pairs = {(gcm, SCENARIO_TO_GROUP.get(scen, scen)) for gcm, scen, _, _ in all_lineage_keys()}
    return pairs | {(gcm, "historical") for gcm, _ in pairs}


def test_canonical_terms_keep_required_clauses_and_copies_match(subtests) -> None:
    assert CANONICAL.is_file(), f"{CANONICAL.name} is the single source of truth and must exist"
    canonical = CANONICAL.read_text()
    for clause in REQUIRED_CLAUSES:
        with subtests.test(clause=clause):
            assert clause in canonical, (
                f"{CANONICAL.name} no longer contains the required clause: {clause!r}"
            )
    for copy in COPIES:
        rel = copy.relative_to(REPO_ROOT)
        with subtests.test(copy=str(rel)):
            assert copy.is_file(), f"{rel} is missing"
            assert copy.read_text() == canonical, (
                f"{rel} has drifted from {CANONICAL.name}. Re-sync with: cp {CANONICAL.name} {rel}"
            )


def test_terms_url_matches_where_the_page_is_published() -> None:
    """Read the Docs is multi-version, so the ``/en/latest/`` segment is required."""
    page = COPIES[0].relative_to(REPO_ROOT / "docs").with_suffix(".html").as_posix()
    assert TERMS_OF_DATA_ACCESS == f"https://sai-downscaling.readthedocs.io/en/latest/{page}"


def test_every_published_scenario_is_attributed_under_a_known_license(subtests) -> None:
    rows = _license_table_rows()
    with subtests.test("every published scenario is attributed"):
        missing = _published_pairs() - {(gcm, scenario) for gcm, scenario, _, _ in rows}
        assert not missing, (
            "These published (GCM, scenario) pairs have no row in "
            f"{LICENSES_PAGE.relative_to(REPO_ROOT)}: {sorted(missing)}"
        )
    for gcm, scenario, license_name, attribution in rows:
        with subtests.test(gcm=gcm, scenario=scenario):
            assert len(attribution) > MIN_ATTRIBUTION_CHARS, (
                f"{gcm}/{scenario} has no usable attribution text: {attribution!r}"
            )
            if license_name:
                assert license_name in LICENSE_URLS, (
                    f"{gcm}/{scenario} names an unrecognized license: {license_name!r}"
                )


def test_module_and_docs_table_agree() -> None:
    documented = {
        (gcm, scenario): (license_name or None, attribution)
        for gcm, scenario, license_name, attribution in _license_table_rows()
    }
    assert set(documented) == set(INPUT_ATTRIBUTION), (
        "the licenses table and saidownscale.licenses cover different simulations"
    )
    for key, (license_id, attribution) in documented.items():
        entry = INPUT_ATTRIBUTION[key]
        assert entry.license == license_id, f"{key}: license differs from the docs table"
        assert entry.references == re.sub(r"<(https?://[^>]+)>", r"\1", attribution), (
            f"{key}: citation text differs from the docs table (autolinks stripped)"
        )
