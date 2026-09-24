"""Tests for saidownscale.config."""

from saidownscale.config import PUBLISHED_SCENARIO_NAMES, SCENARIO_TO_GROUP


def test_scenario_to_group():
    """Config spellings key every cache path; published names are aliases so re-runs are no-ops."""
    assert SCENARIO_TO_GROUP == {
        "historical": "historical",
        "pangeo-historical": "historical",
        "SSP245": "ssp245",
        "ssp245": "ssp245",
        "G6-1.5K": "g6_1p5k",
        "g6_1p5k": "g6_1p5k",
        "esgf-SSP245": "esgf_ssp245",
        "esgf-ssp245": "esgf_ssp245",
        "esgf_ssp245": "esgf_ssp245",
        "G6-1.5K-END": "g6_1p5k_end",
        "g6_1p5k_end": "g6_1p5k_end",
        "SSP2-4.5": "ssp245",
        "G6-1.5K-SAI": "g6_1p5k",
        "G6-1.5K-SAI-END": "g6_1p5k_end",
    }


def test_published_names_cover_every_group():
    """A group with no published name would raise mid-patch rather than at import."""
    assert set(PUBLISHED_SCENARIO_NAMES) == set(SCENARIO_TO_GROUP.values())
