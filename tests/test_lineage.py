"""Tests for saidownscale.lineage."""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import pytest

from saidownscale.lineage import (
    LineageEntry,
    ScenarioMember,
    all_lineage_keys,
    diff_against_provenance,
    resolve_member_lineage,
)

_STD = ("tas", "pr", "rsds", "hurs")
_TMX = ("tasmax", "tasmin", "dtr")
_UKESM_MEMBERS = ("r2i1p1f2", "r3i1p1f2", "r12i1p1f2")
_G6_002 = ScenarioMember(scenario="G6-1.5K", member="002")


def _expected_keys():
    cesm = "CESM2-WACCM6"
    keys = set(itertools.product([cesm], ["G6-1.5K"], ["001", "002", "003"], _STD + _TMX))
    keys |= set(itertools.product([cesm], ["G6-1.5K-END"], ["002"], _STD + _TMX))
    keys |= set(itertools.product([cesm], ["SSP245"], [f"{i:03d}" for i in range(1, 11)], _STD))
    keys |= set(itertools.product([cesm], ["SSP245"], ["006", "007", "008", "009", "010"], _TMX))
    keys |= set(itertools.product(["UKESM1-1-LL"], ["G6-1.5K"], _UKESM_MEMBERS, _STD + _TMX))
    ukesm_ssp245_vars = tuple(v for v in _STD + _TMX if v != "hurs")
    keys |= set(itertools.product(["UKESM1-1-LL"], ["SSP245"], _UKESM_MEMBERS, ukesm_ssp245_vars))
    return keys


def test_registered_keys_match_the_configured_members():
    assert set(all_lineage_keys()) == _expected_keys()


def test_every_entry_satisfies_the_lineage_invariants(subtests):
    for key in all_lineage_keys():
        gcm, scenario, member, variable = key
        with subtests.test(key="/".join(key)):
            entry = resolve_member_lineage(*key)
            assert entry.ssp245_esgf_bridge is None
            if gcm == "UKESM1-1-LL":
                assert entry.historical == "u-by791"
                assert entry.ssp245_bridge == (member if scenario == "G6-1.5K" else None)
            elif variable in _TMX:
                assert entry.historical == "001"
            else:
                assert entry.historical in ("r1i1p1f1", "r2i1p1f1", "r3i1p1f1")
            if scenario == "SSP245":
                assert entry.ssp245_bridge is None
            else:
                assert entry.ssp245_bridge is not None
            if scenario == "G6-1.5K-END":
                assert entry.sai_parent == _G6_002
                parent = resolve_member_lineage(gcm, "G6-1.5K", "002", variable)
                assert (entry.historical, entry.ssp245_bridge) == (
                    parent.historical,
                    parent.ssp245_bridge,
                )
            else:
                assert entry.sai_parent is None


def test_spot_checks(subtests):
    cases = [
        (("CESM2-WACCM6", "G6-1.5K", "001", "tas"), "r1i1p1f1", "001"),
        (("CESM2-WACCM6", "G6-1.5K", "002", "pr"), "r2i1p1f1", "002"),
        (("CESM2-WACCM6", "G6-1.5K", "003", "rsds"), "r3i1p1f1", "003"),
        (("CESM2-WACCM6", "G6-1.5K", "001", "tasmax"), "001", "009"),
        (("CESM2-WACCM6", "G6-1.5K", "002", "tasmin"), "001", "007"),
        (("CESM2-WACCM6", "G6-1.5K", "003", "dtr"), "001", "008"),
        (("CESM2-WACCM6", "G6-1.5K-END", "002", "tasmax"), "001", "007"),
        (("CESM2-WACCM6", "SSP245", "004", "tas"), "r2i1p1f1", None),
        (("CESM2-WACCM6", "SSP245", "006", "hurs"), "r1i1p1f1", None),
        (("CESM2-WACCM6", "SSP245", "010", "pr"), "r3i1p1f1", None),
    ]
    for key, historical, bridge in cases:
        with subtests.test(key="/".join(key)):
            entry = resolve_member_lineage(*key)
            assert (entry.historical, entry.ssp245_bridge) == (historical, bridge)


def test_unregistered_keys_raise_naming_every_key_field(subtests):
    cases = [
        ("UKESM1-0-LL", "G6-1.5K", "001", "tas"),
        ("CESM2-WACCM6", "ssp585", "001", "tas"),
        ("CESM2-WACCM6", "SSP245", "001", "tasmax"),
        ("CESM2-WACCM6", "SSP245", "005", "tasmin"),
        ("CESM2-WACCM6", "G6-1.5K", "004", "tas"),
        ("CESM2-WACCM6", "G6-1.5K-END", "001", "tas"),
        ("CESM2-WACCM6", "G6-1.5K-END", "003", "tas"),
        ("UKESM1-1-LL", "SSP245", "r2i1p1f2", "hurs"),
        ("UKESM1-1-LL", "SSP245", "001", "tas"),
        ("UKESM1-1-LL", "G6-1.5K", "001", "tas"),
        ("BADGCM", "badscenar", "999", "sfcWind"),
    ]
    for key in cases:
        with subtests.test(key="/".join(key)):
            with pytest.raises(KeyError) as exc_info:
                resolve_member_lineage(*key)
            msg = str(exc_info.value)
            assert all(field in msg for field in key)


def test_entry_types_are_frozen_keyword_only_and_not_tuples():
    entry = resolve_member_lineage("CESM2-WACCM6", "G6-1.5K-END", "002", "tas")
    with pytest.raises(TypeError):
        entry[0]
    with pytest.raises(TypeError):
        _hist, _ssp245 = entry
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.historical = "nope"
    with pytest.raises(TypeError):
        LineageEntry("r1i1p1f1", "001")
    with pytest.raises(TypeError):
        ScenarioMember("G6-1.5K", "002")
    assert entry.sai_parent != ("G6-1.5K", "002")
    with pytest.raises(TypeError):
        _scenario, _member = entry.sai_parent
    assert str(entry.sai_parent) == "G6-1.5K/002"


def test_lineage_table_reconciles_with_provenance_sheet(subtests):
    csv_path = Path(__file__).resolve().parent.parent / "docs" / "srm-provenance.csv"
    diff = diff_against_provenance(csv_path)
    assert set(diff) == {"only_in_code", "only_in_sheet", "parent_mismatch", "malformed"}
    assert diff["malformed"] == []
    cesm_sai = [m for m in diff["parent_mismatch"] if m.startswith("CESM2-WACCM6/G6-1.5K")]
    assert cesm_sai == []
    assert [k for k in diff["only_in_code"] if k[3] == "dtr"]
    for variable in _STD + ("tasmax", "tasmin"):
        with subtests.test(variable=variable):
            key = ("CESM2-WACCM6", "G6-1.5K-END", "002", variable)
            assert key not in diff["only_in_code"]
            assert key not in diff["only_in_sheet"]
