"""Tests for lineage resolution and member validation.

_resolve_lineage has been removed from cli.py — lineage is now resolved inside
BCSDPipeline.__init__ using resolve_member_lineage from saidownscale.lineage. These tests
cover the lineage table directly and the CLI's _validate_lineage_members helper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from saidownscale.bcsd_config import BCSDConfig
from saidownscale.cli import _validate_lineage_members
from saidownscale.lineage import resolve_member_lineage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_G6_BASE = dict(
    gcm="CESM2-WACCM",
    ensemble_member="001",
    scenario="G6-1.5K",
    predict_period_start=2015,
    predict_period_end=2084,
)


def _mock_dt_entry(
    hist_members: list[str] | None = None,
    scenario_members: dict[str, list[str]] | None = None,
    raise_on_open: bool = False,
) -> MagicMock:
    """Mock a unified Datatree catalog entry with group-based member access.

    hist_members: ensemble members available under historical/{variable}
    scenario_members: mapping of scenario_group → member list (e.g. {"g6_1p5k": ["001", ...]})
    raise_on_open: if True, to_xarray() raises (unreachable store)
    """
    entry = MagicMock()
    if raise_on_open:
        entry.to_xarray.side_effect = Exception("no S3 access in tests")
        return entry

    dt = MagicMock()

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

    dt.__getitem__.side_effect = getitem
    entry.to_xarray.return_value = dt
    return entry


# ---------------------------------------------------------------------------
# resolve_member_lineage: lineage table correctness
# ---------------------------------------------------------------------------


class TestResolveLineage:
    def test_g6_member_001_standard_vars(self):
        entry = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "001", "tas")
        assert entry.historical == "r1i1p1f1"
        assert entry.ssp245_bridge == "001"

    def test_g6_member_002_standard_vars(self):
        entry = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "002", "pr")
        assert entry.historical == "r2i1p1f1"
        assert entry.ssp245_bridge == "002"

    def test_g6_member_003_standard_vars(self):
        entry = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "003", "tas")
        assert entry.historical == "r3i1p1f1"
        assert entry.ssp245_bridge == "003"

    def test_g6_member_001_tasmax_uses_corrected_historical(self):
        """tasmax member 001 → corrected historical run '001', SSP245 bridge '009'."""
        entry = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "001", "tasmax")
        assert entry.historical == "001"
        assert entry.ssp245_bridge == "009"

    def test_g6_member_002_tasmax_uses_corrected_historical(self):
        entry = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "002", "tasmax")
        assert entry.historical == "001"
        assert entry.ssp245_bridge == "007"

    def test_g6_member_003_tasmax_uses_corrected_historical(self):
        entry = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "003", "tasmax")
        assert entry.historical == "001"
        assert entry.ssp245_bridge == "008"

    def test_all_supported_g6_variables_resolved(self, subtests):
        for var in ("tas", "pr", "rsds", "tasmax"):
            with subtests.test(variable=var):
                entry = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "001", var)
                assert entry.historical is not None

    def test_unknown_gcm_raises_key_error(self):
        with pytest.raises(KeyError):
            resolve_member_lineage("UKESM1-0-LL", "G6-1.5K", "001", "tas")

    def test_unknown_scenario_raises_key_error(self):
        with pytest.raises(KeyError):
            resolve_member_lineage("CESM2-WACCM", "ssp585", "001", "tas")


# ---------------------------------------------------------------------------
# _validate_lineage_members: catalog cross-check
# ---------------------------------------------------------------------------


class TestValidateLineageMembers:
    def test_passes_when_hist_member_present(self):
        """G6/001/tas resolves to hist=r1i1p1f1 and ssp245=001; unified store has both — no error."""
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        # G6-1.5K → g6_1p5k group in the unified store
        entry = _mock_dt_entry(
            hist_members=["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
            scenario_members={"g6_1p5k": ["001", "002", "003"]},
        )
        with patch("saidownscale.datasets.catalog") as cat:
            cat.get.return_value = entry
            _validate_lineage_members([cfg])  # must not raise

    def test_raises_when_hist_member_absent(self):
        """ValueError raised when resolved historical member is absent from store."""
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        entry = _mock_dt_entry(
            hist_members=["r2i1p1f1", "r3i1p1f1"],  # missing r1i1p1f1
            scenario_members={"g6_1p5k": ["001", "002", "003"]},
        )
        with patch("saidownscale.datasets.catalog") as cat:
            cat.get.return_value = entry
            with pytest.raises(ValueError, match="r1i1p1f1"):
                _validate_lineage_members([cfg])

    def test_raises_when_ssp245_member_absent(self):
        """ValueError raised when resolved scenario member is absent from the scenario group."""
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        entry = _mock_dt_entry(
            hist_members=["r1i1p1f1"],  # hist ok
            scenario_members={"g6_1p5k": ["002", "003"]},  # missing 001
        )
        with patch("saidownscale.datasets.catalog") as cat:
            cat.get.return_value = entry
            with pytest.raises(ValueError, match="ssp245"):
                _validate_lineage_members([cfg])

    def test_error_message_includes_gcm_and_variable(self):
        cfg = BCSDConfig(variable="tasmax", **_G6_BASE)
        # tasmax/001 resolves to hist="001"; store is missing it
        entry = _mock_dt_entry(
            hist_members=["r1i1p1f1"],  # has r* but not "001"
            scenario_members={"g6_1p5k": ["009"]},
        )
        with patch("saidownscale.datasets.catalog") as cat:
            cat.get.return_value = entry
            with pytest.raises(ValueError) as exc_info:
                _validate_lineage_members([cfg])
        msg = str(exc_info.value)
        assert "CESM2-WACCM" in msg

    def test_skips_configs_without_scenario(self):
        """Configs with scenario=None are skipped — no catalog lookup."""
        cfg = BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member="r1i1p1f1")
        with patch("saidownscale.datasets.catalog") as cat:
            _validate_lineage_members([cfg])
        cat.get.assert_not_called()

    def test_skips_unknown_lineage_combos(self):
        """Combos not in the lineage table raise KeyError internally — silently skipped."""
        cfg = BCSDConfig(
            gcm="UKESM1-0-LL",
            variable="tas",
            ensemble_member="001",
            scenario="G6-1.5K",
            predict_period_start=2015,
            predict_period_end=2084,
        )
        with patch("saidownscale.datasets.catalog") as cat:
            _validate_lineage_members([cfg])  # must not raise
        cat.get.assert_not_called()

    def test_skips_when_catalog_members_none_and_store_unreachable(self):
        """S3 failure on to_xarray → silently skipped."""
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        entry = _mock_dt_entry(raise_on_open=True)  # to_xarray raises
        with patch("saidownscale.datasets.catalog") as cat:
            cat.get.return_value = entry
            _validate_lineage_members([cfg])  # must not raise

    def test_skips_when_store_not_in_catalog(self):
        """Exception from catalog.get → store unknown → silently skipped."""
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        with patch("saidownscale.datasets.catalog") as cat:
            cat.get.side_effect = Exception("store not found")
            _validate_lineage_members([cfg])  # must not raise

    def test_deduplicates_store_lookups(self):
        """Each GCM's unified store is opened at most once regardless of config count."""
        cfgs = [BCSDConfig(variable=var, **_G6_BASE) for var in ("tas", "pr", "rsds", "tasmax")]
        # All members across historical and G6-1.5K scenario groups
        entry = _mock_dt_entry(
            hist_members=["r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "001", "002", "003"],
            scenario_members={"g6_1p5k": ["001", "002", "003", "007", "008", "009"]},
        )
        with patch("saidownscale.datasets.catalog") as cat:
            cat.get.return_value = entry
            _validate_lineage_members(cfgs)
        # All 4 variables share the same GCM → catalog opened once
        assert cat.get.call_count == 1
        assert cat.get.call_args.args[0] == "CESM2-WACCM"
