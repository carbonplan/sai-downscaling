"""Tests for lineage resolution wiring and member validation in cli.py (PR 3)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from srm.bcsd_config import BCSDConfig
from srm.cli import _resolve_lineage, _validate_lineage_members

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
# _resolve_lineage: fields populated for CESM2-WACCM G6 configs
# ---------------------------------------------------------------------------


class TestResolveLineage:
    def test_g6_member_001_standard_vars(self):
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        [resolved] = _resolve_lineage([cfg])
        assert resolved.historical_ensemble_member == "r1i1p1f1"
        assert resolved.ssp245_ensemble_member == "001"

    def test_g6_member_002_standard_vars(self):
        cfg = BCSDConfig(variable="pr", **{**_G6_BASE, "ensemble_member": "002"})
        [resolved] = _resolve_lineage([cfg])
        assert resolved.historical_ensemble_member == "r2i1p1f1"
        assert resolved.ssp245_ensemble_member == "002"

    def test_g6_member_003_standard_vars(self):
        cfg = BCSDConfig(variable="tas", **{**_G6_BASE, "ensemble_member": "003"})
        [resolved] = _resolve_lineage([cfg])
        assert resolved.historical_ensemble_member == "r3i1p1f1"
        assert resolved.ssp245_ensemble_member == "003"

    def test_g6_member_001_tasmax_uses_corrected_historical(self):
        """tasmax member 001 → corrected historical run '001', SSP245 bridge '009'."""
        cfg = BCSDConfig(variable="tasmax", **_G6_BASE)
        [resolved] = _resolve_lineage([cfg])
        assert resolved.historical_ensemble_member == "001"
        assert resolved.ssp245_ensemble_member == "009"

    def test_g6_member_002_tasmax_uses_corrected_historical(self):
        cfg = BCSDConfig(variable="tasmax", **{**_G6_BASE, "ensemble_member": "002"})
        [resolved] = _resolve_lineage([cfg])
        assert resolved.historical_ensemble_member == "001"
        assert resolved.ssp245_ensemble_member == "007"

    def test_g6_member_003_tasmax_uses_corrected_historical(self):
        cfg = BCSDConfig(variable="tasmax", **{**_G6_BASE, "ensemble_member": "003"})
        [resolved] = _resolve_lineage([cfg])
        assert resolved.historical_ensemble_member == "001"
        assert resolved.ssp245_ensemble_member == "008"

    def test_all_supported_g6_variables_resolved(self, subtests):
        # hurs and tasmin are not yet in BCSDConfig.VariableName
        for var in ("tas", "pr", "rsds", "tasmax"):
            with subtests.test(variable=var):
                cfg = BCSDConfig(variable=var, **_G6_BASE)
                [resolved] = _resolve_lineage([cfg])
                assert resolved.historical_ensemble_member is not None

    def test_unknown_gcm_fields_stay_none(self):
        """KeyError for unknown GCMs is silently skipped — fields stay None."""
        cfg = BCSDConfig(
            gcm="UKESM1-0-LL",
            variable="tas",
            ensemble_member="001",
            scenario="G6-1.5K",
            predict_period_start=2015,
            predict_period_end=2084,
        )
        [resolved] = _resolve_lineage([cfg])
        assert resolved.historical_ensemble_member is None
        assert resolved.ssp245_ensemble_member is None

    def test_unknown_scenario_fields_stay_none(self):
        """Unknown scenario → KeyError silently skipped."""
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="001",
            scenario="ssp585",
            predict_period_start=2015,
            predict_period_end=2100,
        )
        [resolved] = _resolve_lineage([cfg])
        assert resolved.historical_ensemble_member is None

    def test_historical_only_run_not_resolved(self):
        """scenario=None configs are skipped entirely."""
        cfg = BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member="r1i1p1f1")
        [resolved] = _resolve_lineage([cfg])
        assert resolved.historical_ensemble_member is None
        assert resolved.ssp245_ensemble_member is None

    def test_original_config_object_not_mutated(self):
        """Resolution returns new BCSDConfig objects; originals are not changed."""
        cfg = BCSDConfig(variable="tas", **_G6_BASE)
        [resolved] = _resolve_lineage([cfg])
        assert cfg.historical_ensemble_member is None
        assert resolved.historical_ensemble_member == "r1i1p1f1"
        assert resolved is not cfg

    def test_multiple_configs_all_resolved(self):
        cfgs = [BCSDConfig(variable=v, **_G6_BASE) for v in ("tas", "pr", "rsds", "tasmax")]
        resolved = _resolve_lineage(cfgs)
        assert all(c.historical_ensemble_member is not None for c in resolved)
        assert len(resolved) == 4


# ---------------------------------------------------------------------------
# _validate_lineage_members
# ---------------------------------------------------------------------------


class TestValidateLineageMembers:
    def test_passes_when_hist_member_present(self):
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="001",
            historical_ensemble_member="r1i1p1f1",
        )
        entry = _mock_catalog_entry(["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"])
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            _validate_lineage_members([cfg])  # must not raise

    def test_raises_when_hist_member_absent(self):
        """ValueError raised when historical member is absent from store."""
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="001",
            historical_ensemble_member="NOT-IN-STORE",
        )
        entry = _mock_catalog_entry(["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"])
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            with pytest.raises(ValueError, match="NOT-IN-STORE"):
                _validate_lineage_members([cfg])

    def test_raises_when_ssp245_member_absent(self):
        """ValueError raised when SSP245 bridge member is absent from SSP245 store."""
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="001",
            scenario="G6-1.5K",
            predict_period_start=2015,
            predict_period_end=2084,
            ssp245_ensemble_member="MISSING-BRIDGE",
        )
        entry = _mock_catalog_entry(["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"])
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            with pytest.raises(ValueError, match="MISSING-BRIDGE"):
                _validate_lineage_members([cfg])

    def test_error_message_includes_gcm_and_variable(self):
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tasmax",
            ensemble_member="001",
            historical_ensemble_member="ABSENT",
        )
        entry = _mock_catalog_entry(["r1i1p1f1"])
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            with pytest.raises(ValueError) as exc_info:
                _validate_lineage_members([cfg])
        msg = str(exc_info.value)
        assert "CESM2-WACCM" in msg
        assert "tasmax" in msg
        assert "ABSENT" in msg

    def test_multiple_missing_members_listed_in_one_error(self):
        cfgs = [
            BCSDConfig(
                gcm="CESM2-WACCM",
                variable=var,
                ensemble_member="001",
                historical_ensemble_member="MISSING",
            )
            for var in ("tas", "pr")
        ]
        entry = _mock_catalog_entry(["r1i1p1f1"])
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            with pytest.raises(ValueError) as exc_info:
                _validate_lineage_members(cfgs)
        msg = str(exc_info.value)
        assert "tas" in msg
        assert "pr" in msg

    def test_skips_when_catalog_members_none_and_store_unreachable(self):
        """ensemble_members=None + S3 failure → silently skipped."""
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="001",
            historical_ensemble_member="001",
        )
        entry = _mock_catalog_entry(None)  # ensemble_members is None, to_xarray raises
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            _validate_lineage_members([cfg])  # must not raise

    def test_skips_when_store_not_in_catalog(self):
        """KeyError from catalog.get → store unknown → silently skipped."""
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="001",
            historical_ensemble_member="001",
        )
        with patch("srm.datasets.catalog") as cat:
            cat.get.side_effect = KeyError("store not found")
            _validate_lineage_members([cfg])  # must not raise

    def test_deduplicates_store_lookups(self):
        """Each unique store is queried at most once regardless of config count."""
        cfgs = [
            BCSDConfig(
                gcm="CESM2-WACCM",
                variable=var,
                ensemble_member="001",
                historical_ensemble_member="r1i1p1f1",
            )
            for var in ("tas", "pr", "rsds", "tasmax")
        ]
        entry = _mock_catalog_entry(["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"])
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            _validate_lineage_members(cfgs)
        cat.get.assert_called_once_with("CESM2-WACCM-historical-icechunk")

    def test_no_store_lookup_when_no_lineage_fields_set(self):
        """Configs without lineage fields never touch the catalog."""
        cfg = BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member="r1i1p1f1")
        with patch("srm.datasets.catalog") as cat:
            _validate_lineage_members([cfg])
        cat.get.assert_not_called()

    def test_hist_and_ssp245_stores_checked_separately(self):
        """historical and SSP245 stores use distinct catalog keys."""
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="001",
            scenario="G6-1.5K",
            predict_period_start=2015,
            predict_period_end=2084,
            historical_ensemble_member="r1i1p1f1",
            ssp245_ensemble_member="r1i1p1f1",
        )
        entry = _mock_catalog_entry(["r1i1p1f1"])
        with patch("srm.datasets.catalog") as cat:
            cat.get.return_value = entry
            _validate_lineage_members([cfg])

        called_stores = {call.args[0] for call in cat.get.call_args_list}
        assert "CESM2-WACCM-historical-icechunk" in called_stores
        assert "CESM2-WACCM-SSP245-icechunk" in called_stores
