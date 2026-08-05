"""Tests for srm.config module."""

from srm.config import SCENARIO_TO_GROUP


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
    }
