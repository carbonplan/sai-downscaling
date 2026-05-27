"""Tests for lineage resolution and member validation.

_resolve_lineage has been removed from cli.py — lineage is now resolved inside
BCSDPipeline.__init__ using resolve_member_lineage from srm.lineage. These tests
cover the lineage table directly and the CLI's _validate_lineage_members helper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from srm.bcsd_config import BCSDConfig
from srm.cli import _validate_lineage_members
from srm.lineage import resolve_member_lineage

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


def _mock_catalog_entry(members: list[str] | None) -> MagicMock:
    entry = MagicMock()
    entry.ensemble_members = members
    if members is None:
        entry.to_xarray.side_effect = Exception("no S3 access in tests")
    return entry


# ---------------------------------------------------------------------------
# resolve_member_lineage: lineage table correctness
# ---------------------------------------------------------------------------


class TestResolveLineage:
    def test_g6_member_001_standard_vars(self):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "001", "tas")
        assert hist == "r1i1p1f1"
        assert ssp245 == "001"

    def test_g6_member_002_standard_vars(self):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "002", "pr")
        assert hist == "r2i1p1f1"
        assert ssp245 == "002"

    def test_g6_member_003_standard_vars(self):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "003", "tas")
        assert hist == "r3i1p1f1"
        assert ssp245 == "003"

    def test_g6_member_001_tasmax_uses_corrected_historical(self):
        """tasmax member 001 → corrected historical run '001', SSP245 bridge '009'."""
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "001", "tasmax")
        assert hist == "001"
        assert ssp245 == "009"

    def test_g6_member_002_tasmax_uses_corrected_historical(self):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "002", "tasmax")
        assert hist == "001"
        assert ssp245 == "007"

    def test_g6_member_003_tasmax_uses_corrected_historical(self):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "003", "tasmax")
        assert hist == "001"
        assert ssp245 == "008"

    def test_all_supported_g6_variables_resolved(self, subtests):
        for var in ("tas", "pr", "rsds", "tasmax"):
            with subtests.test(variable=var):
                hist, _ = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "001", var)
                assert hist is not None

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
        """G6/001/tas resolves to hist=r1i1p1f1 and ssp245=001; catalog has both — no error."""
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        hist_entry = _mock_catalog_entry(["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"])
        ssp_entry = _mock_catalog_entry(["001", "002", "003"])

        def catalog_get(store_name):
            return ssp_entry if "SSP245" in store_name else hist_entry

        with patch("srm.datasets.catalog") as cat:
            cat.get.side_effect = catalog_get
            _validate_lineage_members([cfg])  # must not raise

    def test_raises_when_hist_member_absent(self):
        """ValueError raised when resolved historical member is absent from store."""
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        # Catalog has other members but not r1i1p1f1
        entry = _mock_catalog_entry(["r2i1p1f1", "r3i1p1f1"])
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            with pytest.raises(ValueError, match="r1i1p1f1"):
                _validate_lineage_members([cfg])

    def test_raises_when_ssp245_member_absent(self):
        """ValueError raised when resolved SSP245 bridge member is absent from SSP245 store."""
        # G6/001/tas → ssp245="001"; mock catalog missing it for SSP245 store
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        entry_hist = _mock_catalog_entry(["r1i1p1f1"])  # hist store ok
        entry_ssp = _mock_catalog_entry(["002", "003"])  # SSP245 store missing 001

        def catalog_get(store_name):
            if "SSP245" in store_name:
                return entry_ssp
            return entry_hist

        with patch("srm.datasets.catalog") as cat:
            cat.get.side_effect = catalog_get
            with pytest.raises(ValueError, match="ssp245"):
                _validate_lineage_members([cfg])

    def test_error_message_includes_gcm_and_variable(self):
        cfg = BCSDConfig(variable="tasmax", **_G6_BASE)
        # tasmax/001 resolves to hist="001"; catalog doesn't have it
        entry = _mock_catalog_entry(["r1i1p1f1"])
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            with pytest.raises(ValueError) as exc_info:
                _validate_lineage_members([cfg])
        msg = str(exc_info.value)
        assert "CESM2-WACCM" in msg

    def test_skips_configs_without_scenario(self):
        """Configs with scenario=None are skipped — no catalog lookup."""
        cfg = BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member="r1i1p1f1")
        with patch("srm.datasets.catalog") as cat:
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
        with patch("srm.datasets.catalog") as cat:
            _validate_lineage_members([cfg])  # must not raise
        cat.get.assert_not_called()

    def test_skips_when_catalog_members_none_and_store_unreachable(self):
        """ensemble_members=None + S3 failure → silently skipped."""
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        entry = _mock_catalog_entry(None)  # ensemble_members is None, to_xarray raises
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            _validate_lineage_members([cfg])  # must not raise

    def test_skips_when_store_not_in_catalog(self):
        """KeyError from catalog.get → store unknown → silently skipped."""
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        with patch("srm.datasets.catalog") as cat:
            cat.get.side_effect = Exception("store not found")
            _validate_lineage_members([cfg])  # must not raise

    def test_deduplicates_store_lookups(self):
        """Each unique store is queried at most once regardless of config count."""
        cfgs = [BCSDConfig(variable=var, **_G6_BASE) for var in ("tas", "pr", "rsds", "tasmax")]
        entry = _mock_catalog_entry(
            ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "001", "002", "003", "009"]
        )
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            _validate_lineage_members(cfgs)
        # tas/pr/rsds → pangeo hist store; tasmax → standard hist store; all share SSP245
        called_stores = {call.args[0] for call in cat.get.call_args_list}
        assert "pangeo-CESM2-WACCM-historical-icechunk" in called_stores
        assert "CESM2-WACCM-historical-icechunk" in called_stores
        assert "CESM2-WACCM-SSP245-icechunk" in called_stores
        assert cat.get.call_count == 3
