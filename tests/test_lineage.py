"""Tests for srm.lineage.resolve_member_lineage and BCSDConfig lineage fields."""

from __future__ import annotations

import pytest

from srm.bcsd_config import BCSDConfig
from srm.lineage import resolve_member_lineage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STANDARD_VARS = ("tas", "pr", "rsds", "hurs")
_TMAX_MIN_VARS = ("tasmax", "tasmin")


# ---------------------------------------------------------------------------
# resolve_member_lineage: CESM2-WACCM G6-1.5K
# ---------------------------------------------------------------------------


class TestG6Lineage:
    """G6-1.5K lineage for all three CESM2-WACCM members and both variable groups."""

    @pytest.mark.parametrize("variable", _STANDARD_VARS)
    def test_g6_member_001_standard_vars(self, variable):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "001", variable)
        assert hist == "r1i1p1f1"
        assert ssp245 == "001"

    @pytest.mark.parametrize("variable", _TMAX_MIN_VARS)
    def test_g6_member_001_tmax_tmin(self, variable):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "001", variable)
        assert hist == "001"
        assert ssp245 == "009"

    @pytest.mark.parametrize("variable", _STANDARD_VARS)
    def test_g6_member_002_standard_vars(self, variable):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "002", variable)
        assert hist == "r2i1p1f1"
        assert ssp245 == "002"

    @pytest.mark.parametrize("variable", _TMAX_MIN_VARS)
    def test_g6_member_002_tmax_tmin(self, variable):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "002", variable)
        assert hist == "001"
        assert ssp245 == "007"

    @pytest.mark.parametrize("variable", _STANDARD_VARS)
    def test_g6_member_003_standard_vars(self, variable):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "003", variable)
        assert hist == "r3i1p1f1"
        assert ssp245 == "003"

    @pytest.mark.parametrize("variable", _TMAX_MIN_VARS)
    def test_g6_member_003_tmax_tmin(self, variable):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "003", variable)
        assert hist == "001"
        assert ssp245 == "008"

    def test_g6_ssp245_member_is_not_none(self, subtests):
        for member in ("001", "002", "003"):
            with subtests.test(member=member):
                _, ssp245 = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", member, "tas")
                assert ssp245 is not None


# ---------------------------------------------------------------------------
# resolve_member_lineage: CESM2-WACCM SSP245
# ---------------------------------------------------------------------------


class TestSSP245Lineage:
    """SSP245 lineage: no SAI bridge (ssp245_member always None)."""

    @pytest.mark.parametrize(
        ("member", "expected_hist"),
        [
            ("001", "r1i1p1f1"),
            ("002", "r2i1p1f1"),
            ("003", "r3i1p1f1"),
            ("004", "r2i1p1f1"),
            ("005", "r3i1p1f1"),
            ("007", "r2i1p1f1"),
            ("008", "r3i1p1f1"),
            ("009", "r2i1p1f1"),
            ("010", "r3i1p1f1"),
        ],
    )
    @pytest.mark.parametrize("variable", _STANDARD_VARS)
    def test_ssp245_standard_vars(self, member, expected_hist, variable):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "SSP245", member, variable)
        assert hist == expected_hist
        assert ssp245 is None

    @pytest.mark.parametrize("member", ("007", "008", "009", "010"))
    @pytest.mark.parametrize("variable", _TMAX_MIN_VARS)
    def test_ssp245_tmax_tmin_members(self, member, variable):
        hist, ssp245 = resolve_member_lineage("CESM2-WACCM", "SSP245", member, variable)
        assert hist == "001"
        assert ssp245 is None

    def test_ssp245_ssp245_member_always_none(self, subtests):
        members = ("001", "002", "003", "004", "005", "007", "008", "009", "010")
        for member in members:
            with subtests.test(member=member):
                _, ssp245 = resolve_member_lineage("CESM2-WACCM", "SSP245", member, "tas")
                assert ssp245 is None


# ---------------------------------------------------------------------------
# resolve_member_lineage: KeyError on unregistered combinations
# ---------------------------------------------------------------------------


class TestLineageKeyError:
    """Unregistered combinations raise KeyError with an informative message."""

    def test_unknown_gcm_raises(self):
        with pytest.raises(KeyError, match="gcm="):
            resolve_member_lineage("UKESM1-0-LL", "G6-1.5K", "001", "tas")

    def test_unknown_scenario_raises(self):
        with pytest.raises(KeyError, match="scenario="):
            resolve_member_lineage("CESM2-WACCM", "ssp585", "001", "tas")

    def test_excluded_member_006_raises(self):
        # SSP245 member 006 excluded: ends 2069-12-31
        with pytest.raises(KeyError, match="ensemble_member="):
            resolve_member_lineage("CESM2-WACCM", "SSP245", "006", "tas")

    def test_ssp245_001_tasmax_raises(self):
        # tasmax/tasmin not available for SSP245 001-005 (CMIP6 bug)
        with pytest.raises(KeyError, match="variable="):
            resolve_member_lineage("CESM2-WACCM", "SSP245", "001", "tasmax")

    def test_ssp245_005_tasmin_raises(self):
        with pytest.raises(KeyError):
            resolve_member_lineage("CESM2-WACCM", "SSP245", "005", "tasmin")

    def test_g6_unknown_member_raises(self):
        with pytest.raises(KeyError):
            resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "004", "tas")

    def test_error_message_contains_all_key_fields(self):
        with pytest.raises(KeyError) as exc_info:
            resolve_member_lineage("BADGCM", "badscenar", "999", "sfcWind")
        msg = str(exc_info.value)
        assert "BADGCM" in msg
        assert "badscenar" in msg
        assert "999" in msg
        assert "sfcWind" in msg


# ---------------------------------------------------------------------------
# BCSDConfig: new lineage fields
# ---------------------------------------------------------------------------


class TestBCSDConfigLineageFields:
    """historical_ensemble_member and ssp245_ensemble_member fields."""

    def test_defaults_to_none(self):
        cfg = BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member="001")
        assert cfg.historical_ensemble_member is None
        assert cfg.ssp245_ensemble_member is None

    def test_can_set_historical_member(self):
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="001",
            historical_ensemble_member="r1i1p1f1",
        )
        assert cfg.historical_ensemble_member == "r1i1p1f1"

    def test_can_set_ssp245_member(self):
        cfg = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tasmax",
            ensemble_member="001",
            scenario="G6-1.5K",
            predict_period_start=2035,
            predict_period_end=2084,
            historical_ensemble_member="001",
            ssp245_ensemble_member="009",
        )
        assert cfg.ssp245_ensemble_member == "009"

    def test_historical_member_changes_config_hash(self):
        base = BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member="001")
        with_lineage = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="001",
            historical_ensemble_member="r1i1p1f1",
        )
        assert base.config_hash != with_lineage.config_hash

    def test_ssp245_member_changes_config_hash(self):
        without = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tasmax",
            ensemble_member="001",
            scenario="G6-1.5K",
            predict_period_start=2035,
            predict_period_end=2084,
            historical_ensemble_member="001",
        )
        with_bridge = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tasmax",
            ensemble_member="001",
            scenario="G6-1.5K",
            predict_period_start=2035,
            predict_period_end=2084,
            historical_ensemble_member="001",
            ssp245_ensemble_member="009",
        )
        assert without.config_hash != with_bridge.config_hash

    def test_different_historical_members_produce_different_hashes(self, subtests):
        members = ("r1i1p1f1", "r2i1p1f1", "001")
        hashes = []
        for m in members:
            with subtests.test(member=m):
                cfg = BCSDConfig(
                    gcm="CESM2-WACCM",
                    variable="tas",
                    ensemble_member="001",
                    historical_ensemble_member=m,
                )
                hashes.append(cfg.config_hash)
        assert len(set(hashes)) == len(hashes)
