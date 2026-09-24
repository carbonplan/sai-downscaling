"""Tests for the CLI's _validate_lineage_members catalog cross-check."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from saidownscale.cli import _validate_lineage_members
from saidownscale.downscaling_config import DownscalingConfig

_G6_BASE = dict(
    gcm="CESM2-WACCM6",
    downscaling_method="BCSD",
    ensemble_member="001",
    scenario="G6-1.5K",
    predict_period_start=2015,
    predict_period_end=2084,
)


def _mock_dt_entry(hist_members=None, scenario_members=None, raise_on_open=False) -> MagicMock:
    entry = MagicMock()
    if raise_on_open:
        entry.to_xarray.side_effect = Exception("no S3 access in tests")
        return entry

    def _node(members):
        n = MagicMock()
        n.children = {m: MagicMock() for m in members}
        return n

    def getitem(path):
        if path.startswith("historical/"):
            if hist_members is None:
                raise KeyError(path)
            return _node(hist_members)
        for grp, members in (scenario_members or {}).items():
            if path.startswith(f"{grp}/"):
                return _node(members)
        raise KeyError(path)

    dt = MagicMock()
    dt.__getitem__.side_effect = getitem
    entry.to_xarray.return_value = dt
    return entry


def test_missing_resolved_member_raises(subtests):
    cases = [
        ("tas", ["r2i1p1f1", "r3i1p1f1"], ["001", "002", "003"], "r1i1p1f1"),
        ("tas", ["r1i1p1f1"], ["002", "003"], "ssp245"),
        ("tasmax", ["r1i1p1f1"], ["009"], "CESM2-WACCM6"),
    ]
    for variable, hist, scen, match in cases:
        with subtests.test(variable=variable, match=match):
            cfg = DownscalingConfig(variable=variable, **_G6_BASE)
            entry = _mock_dt_entry(hist_members=hist, scenario_members={"g6_1p5k": scen})
            with patch("saidownscale.datasets.catalog") as cat:
                cat.get.return_value = entry
                with pytest.raises(ValueError, match=match):
                    _validate_lineage_members([cfg])


def test_passes_or_skips_without_raising(subtests):
    g6_tas = DownscalingConfig(variable="tas", **_G6_BASE)
    no_scenario = DownscalingConfig(
        downscaling_method="BCSD", gcm="CESM2-WACCM6", variable="tas", ensemble_member="r1i1p1f1"
    )
    unknown_lineage = DownscalingConfig(variable="tas", **(_G6_BASE | {"gcm": "UKESM1-0-LL"}))
    present = _mock_dt_entry(
        hist_members=["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
        scenario_members={"g6_1p5k": ["001", "002", "003"]},
    )
    cases = [
        ("members_present", g6_tas, {"return_value": present}, True),
        ("store_unreachable", g6_tas, {"return_value": _mock_dt_entry(raise_on_open=True)}, True),
        ("store_not_in_catalog", g6_tas, {"side_effect": Exception("store not found")}, True),
        ("no_scenario", no_scenario, {}, False),
        ("unknown_lineage", unknown_lineage, {}, False),
    ]
    for name, cfg, cat_kwargs, opens_catalog in cases:
        with subtests.test(name):
            with patch("saidownscale.datasets.catalog") as cat:
                cat.get.configure_mock(**cat_kwargs)
                _validate_lineage_members([cfg])
            assert cat.get.called is opens_catalog


def test_deduplicates_store_lookups():
    cfgs = [DownscalingConfig(variable=v, **_G6_BASE) for v in ("tas", "pr", "rsds", "tasmax")]
    entry = _mock_dt_entry(
        hist_members=["r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "001", "002", "003"],
        scenario_members={"g6_1p5k": ["001", "002", "003", "007", "008", "009"]},
    )
    with patch("saidownscale.datasets.catalog") as cat:
        cat.get.return_value = entry
        _validate_lineage_members(cfgs)
    assert cat.get.call_count == 1
    assert cat.get.call_args.args[0] == "CESM2-WACCM6"
