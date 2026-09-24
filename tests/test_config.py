"""Tests for saidownscale.config."""

from saidownscale.config import PUBLISHED_SCENARIO_NAMES, SCENARIO_TO_GROUP
from saidownscale.store_metadata import names_this_group


def test_scenario_to_group(subtests):
    """Published names stay out: this map keys every cache path."""
    with subtests.test("config spellings only"):
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
    with subtests.test("a published name resolves to no cache path"):
        for name in PUBLISHED_SCENARIO_NAMES.values():
            if name != "historical":
                assert name not in SCENARIO_TO_GROUP
    with subtests.test("but a patched store is still recognized"):
        for group, name in PUBLISHED_SCENARIO_NAMES.items():
            assert names_this_group(name, group)
    with subtests.test("an unrecognized name matches nothing"):
        assert not names_this_group("UTTER-NONSENSE", "ssp245")


def test_published_names_cover_every_group():
    """A group with no published name would raise mid-patch rather than at import."""
    assert set(PUBLISHED_SCENARIO_NAMES) == set(SCENARIO_TO_GROUP.values())
