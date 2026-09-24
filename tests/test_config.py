"""Tests for saidownscale.config module."""

from saidownscale.config import SCENARIO_TO_GROUP


def test_scenario_to_group_values():
    assert SCENARIO_TO_GROUP["historical"] == "historical"
    assert SCENARIO_TO_GROUP["SSP245"] == "ssp245"
    assert SCENARIO_TO_GROUP["G6-1.5K"] == "g6_1p5k"
    assert SCENARIO_TO_GROUP["G6-1.5K-END"] == "g6_1p5k_end"
    assert SCENARIO_TO_GROUP["esgf-SSP245"] == "esgf_ssp245"


def test_scenario_to_group_pangeo_alias():
    assert SCENARIO_TO_GROUP["pangeo-historical"] == "historical"


def test_scenario_to_group_completeness():
    assert set(SCENARIO_TO_GROUP.keys()) == {
        "historical",
        "pangeo-historical",
        "SSP245",
        "ssp245",
        "G6-1.5K",
        "g6_1p5k",
        "esgf-SSP245",
        "esgf-ssp245",
        "esgf_ssp245",
        "G6-1.5K-END",
        "g6_1p5k_end",
        "SSP2-4.5",
        "G6-1.5K-SAI",
        "G6-1.5K-SAI-END",
    }


def test_published_names_resolve_back_to_their_group():
    """Stores publish these spellings, so a re-run against a patched store has to resolve them.

    Without the aliases the metadata script would read its own output as a path-versus-attrs
    disagreement and refuse the group. Every re-run has to be a no-op, so the names we write have
    to be names we also accept.
    """
    assert SCENARIO_TO_GROUP["SSP2-4.5"] == "ssp245"
    assert SCENARIO_TO_GROUP["G6-1.5K-SAI"] == "g6_1p5k"
    assert SCENARIO_TO_GROUP["G6-1.5K-SAI-END"] == "g6_1p5k_end"


def test_published_names_cover_every_group():
    """A group with no published name would raise mid-patch rather than at import."""
    from saidownscale.config import PUBLISHED_SCENARIO_NAMES

    assert set(PUBLISHED_SCENARIO_NAMES) == set(SCENARIO_TO_GROUP.values())


def test_config_spellings_are_untouched():
    """Config scenarios appear in every cache path, so renaming them would move every artifact.

    The published names are a separate mapping for that reason, and this pins the 2 apart. A well
    meant find-and-replace across both would orphan every artifact already written.
    """
    assert SCENARIO_TO_GROUP["SSP245"] == "ssp245"
    assert SCENARIO_TO_GROUP["G6-1.5K"] == "g6_1p5k"
    assert SCENARIO_TO_GROUP["G6-1.5K-END"] == "g6_1p5k_end"
