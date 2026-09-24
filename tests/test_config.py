"""Tests for saidownscale.config."""

from saidownscale.config import SCENARIO_TO_GROUP


def test_scenario_to_group():
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
    }
