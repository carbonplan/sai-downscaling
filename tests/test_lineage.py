"""Tests for srm.lineage.resolve_member_lineage and BCSDConfig lineage fields."""

from __future__ import annotations

import pytest

from srm.lineage import resolve_member_lineage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STANDARD_VARS = ("tas", "pr", "rsds", "hurs")
_TMAX_MIN_VARS = ("tasmax", "tasmin", "dtr")
_UKESM_VARS = ("tas", "pr", "rsds", "hurs", "tasmax", "tasmin", "dtr")


# ---------------------------------------------------------------------------
# resolve_member_lineage: CESM2-WACCM G6-1.5K
# ---------------------------------------------------------------------------


class TestG6Lineage:
    """G6-1.5K lineage for all three CESM2-WACCM members and both variable groups."""

    @pytest.mark.parametrize("variable", _STANDARD_VARS)
    def test_g6_member_001_standard_vars(self, variable):
        hist, ssp245, *_ = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "001", variable)
        assert hist == "r1i1p1f1"
        assert ssp245 == "001"

    @pytest.mark.parametrize("variable", _TMAX_MIN_VARS)
    def test_g6_member_001_tmax_tmin(self, variable):
        hist, ssp245, *_ = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "001", variable)
        assert hist == "001"
        assert ssp245 == "009"

    @pytest.mark.parametrize("variable", _STANDARD_VARS)
    def test_g6_member_002_standard_vars(self, variable):
        hist, ssp245, *_ = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "002", variable)
        assert hist == "r2i1p1f1"
        assert ssp245 == "002"

    @pytest.mark.parametrize("variable", _TMAX_MIN_VARS)
    def test_g6_member_002_tmax_tmin(self, variable):
        hist, ssp245, *_ = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "002", variable)
        assert hist == "001"
        assert ssp245 == "007"

    @pytest.mark.parametrize("variable", _STANDARD_VARS)
    def test_g6_member_003_standard_vars(self, variable):
        hist, ssp245, *_ = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "003", variable)
        assert hist == "r3i1p1f1"
        assert ssp245 == "003"

    @pytest.mark.parametrize("variable", _TMAX_MIN_VARS)
    def test_g6_member_003_tmax_tmin(self, variable):
        hist, ssp245, *_ = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "003", variable)
        assert hist == "001"
        assert ssp245 == "008"

    def test_g6_ssp245_member_is_not_none(self, subtests):
        for member in ("001", "002", "003"):
            with subtests.test(member=member):
                _, ssp245, *_ = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", member, "tas")
                assert ssp245 is not None


# ---------------------------------------------------------------------------
# resolve_member_lineage: CESM2-WACCM G6-1.5K-END
# ---------------------------------------------------------------------------


class TestG6EndLineage:
    """G6-1.5K-END is the termination-shock continuation of G6-1.5K 002 from 2085.

    Its store holds only 2085-2100, so the detrend stitch needs both an SSP245
    segment (2015-2034) and the parent SAI segment (2035-2084) to reach it. The
    lineage table is what supplies all three parents to the bridge.
    """

    @pytest.mark.parametrize("variable", _STANDARD_VARS)
    def test_g6_end_standard_vars(self, variable):
        hist, ssp245, ssp245_esgf, sai_parent = resolve_member_lineage(
            "CESM2-WACCM", "G6-1.5K-END", "002", variable
        )
        assert hist == "r2i1p1f1"
        assert ssp245 == "002"
        assert ssp245_esgf is None
        assert sai_parent == ("G6-1.5K", "002")

    @pytest.mark.parametrize("variable", _TMAX_MIN_VARS)
    def test_g6_end_tmax_tmin(self, variable):
        hist, ssp245, ssp245_esgf, sai_parent = resolve_member_lineage(
            "CESM2-WACCM", "G6-1.5K-END", "002", variable
        )
        assert hist == "001"
        assert ssp245 == "007"
        assert ssp245_esgf is None
        assert sai_parent == ("G6-1.5K", "002")

    def test_g6_end_sai_parent_matches_g6_lineage(self, subtests):
        """The parent segment must carry the same lineage the END run continues.

        G6-1.5K 002 and G6-1.5K-END 002 branch from the same historical and SSP245
        members, so a mismatch here means the two segments come from different
        realizations. The provenance sheet records both chains, and they agree on
        the first two parents for every variable.
        """
        for variable in _STANDARD_VARS + _TMAX_MIN_VARS:
            with subtests.test(variable=variable):
                end = resolve_member_lineage("CESM2-WACCM", "G6-1.5K-END", "002", variable)
                parent = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", "002", variable)
                assert end[:3] == parent[:3]

    @pytest.mark.parametrize("member", ("001", "003"))
    def test_g6_end_unregistered_members_raise(self, member):
        """Only member 002 was run to termination."""
        with pytest.raises(KeyError):
            resolve_member_lineage("CESM2-WACCM", "G6-1.5K-END", member, "tas")

    def test_g6_has_no_sai_parent(self, subtests):
        """Plain G6-1.5K starts the SAI chain, so nothing precedes it."""
        for member in ("001", "002", "003"):
            with subtests.test(member=member):
                *_, sai_parent = resolve_member_lineage("CESM2-WACCM", "G6-1.5K", member, "tas")
                assert sai_parent is None


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
            ("006", "r1i1p1f1"),
            ("007", "r2i1p1f1"),
            ("008", "r3i1p1f1"),
            ("009", "r2i1p1f1"),
            ("010", "r3i1p1f1"),
        ],
    )
    @pytest.mark.parametrize("variable", _STANDARD_VARS)
    def test_ssp245_standard_vars(self, member, expected_hist, variable):
        hist, ssp245, *_ = resolve_member_lineage("CESM2-WACCM", "SSP245", member, variable)
        assert hist == expected_hist
        assert ssp245 is None

    @pytest.mark.parametrize("member", ("006", "007", "008", "009", "010"))
    @pytest.mark.parametrize("variable", _TMAX_MIN_VARS)
    def test_ssp245_tmax_tmin_members(self, member, variable):
        hist, ssp245, *_ = resolve_member_lineage("CESM2-WACCM", "SSP245", member, variable)
        assert hist == "001"
        assert ssp245 is None

    def test_ssp245_ssp245_member_always_none(self, subtests):
        members = ("001", "002", "003", "004", "005", "006", "007", "008", "009", "010")
        for member in members:
            with subtests.test(member=member):
                _, ssp245, *_ = resolve_member_lineage("CESM2-WACCM", "SSP245", member, "tas")
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
# resolve_member_lineage: UKESM SSP245
# ---------------------------------------------------------------------------


class TestUKESMSSP245Lineage:
    """SSP245 lineage: single ripf-keyed store covers all variables, hist=self."""

    @pytest.mark.parametrize("member", ("r2i1p1f2", "r3i1p1f2", "r12i1p1f2"))
    @pytest.mark.parametrize("variable", _UKESM_VARS)
    def test_members_hist_equals_self(self, member, variable):
        hist, ssp245, *_ = resolve_member_lineage("UKESM", "SSP245", member, variable)
        assert hist == member
        assert ssp245 is None

    @pytest.mark.parametrize("member", ("001", "002", "003"))
    def test_legacy_numeric_members_unregistered(self, member):
        with pytest.raises(KeyError):
            resolve_member_lineage("UKESM", "SSP245", member, "tas")


# ---------------------------------------------------------------------------
# resolve_member_lineage: UKESM G6-1.5K
# ---------------------------------------------------------------------------


class TestUKESMG6Lineage:
    """G6-1.5K lineage: single ripf-keyed store covers all variables, hist=self, ssp245_bridge=self."""

    @pytest.mark.parametrize("member", ("r2i1p1f2", "r3i1p1f2", "r12i1p1f2"))
    @pytest.mark.parametrize("variable", _UKESM_VARS)
    def test_members_self_consistent(self, member, variable):
        hist, ssp245, ssp245_esgf, sai_parent = resolve_member_lineage(
            "UKESM", "G6-1.5K", member, variable
        )
        assert hist == member
        assert ssp245 == member
        assert ssp245_esgf is None
        assert sai_parent is None

    @pytest.mark.parametrize("member", ("001", "002", "003"))
    def test_legacy_numeric_members_unregistered(self, member):
        with pytest.raises(KeyError):
            resolve_member_lineage("UKESM", "G6-1.5K", member, "tas")


# ---------------------------------------------------------------------------
# resolve_member_lineage: MIROC-ES2H G6-1.5K
# ---------------------------------------------------------------------------


class TestMirocLineage:
    """MIROC-ES2H lineage: ssp245_esgf_member set for both G6-1.5K and SSP245."""

    _MIROC_G6_LINEAGE = [
        ("r01", "r1i1p4f2"),
        ("r02", "r2i1p4f2"),
        ("r03", "r3i1p4f2"),
        ("r04", "r1i1p4f2"),
        ("r05", "r2i1p4f2"),
        ("r06", "r3i1p4f2"),
        ("r07", "r1i1p4f2"),
        ("r08", "r2i1p4f2"),
        ("r09", "r3i1p4f2"),
        ("r10", "r1i1p4f2"),
    ]

    @pytest.mark.parametrize(("member", "expected_hist"), _MIROC_G6_LINEAGE)
    def test_g6_hist_member(self, member, expected_hist):
        hist, *_ = resolve_member_lineage("MIROC-ES2H", "G6-1.5K", member, "tas")
        assert hist == expected_hist

    @pytest.mark.parametrize("member", [m for m, _ in _MIROC_G6_LINEAGE])
    def test_g6_ssp245_member_equals_geomip_member(self, member):
        _, ssp245, *_ = resolve_member_lineage("MIROC-ES2H", "G6-1.5K", member, "tas")
        assert ssp245 == member

    @pytest.mark.parametrize(("member", "expected_hist"), _MIROC_G6_LINEAGE)
    def test_g6_esgf_bridge_equals_hist_member(self, member, expected_hist):
        _, _, ssp245_esgf, _ = resolve_member_lineage("MIROC-ES2H", "G6-1.5K", member, "tas")
        assert ssp245_esgf == expected_hist

    @pytest.mark.parametrize(("member", "expected_hist"), _MIROC_G6_LINEAGE)
    def test_ssp245_no_sai_bridge_but_has_esgf_bridge(self, member, expected_hist):
        # SSP245 is not an SAI scenario so ssp245 (SAI bridge) is None.
        # ssp245_esgf is set because the GeoMIP SSP245 dataset starts in 2020;
        # ESGF SSP245 fills the 2015–2019 gap.
        _, ssp245, ssp245_esgf, _ = resolve_member_lineage("MIROC-ES2H", "SSP245", member, "tas")
        assert ssp245 is None
        assert ssp245_esgf == expected_hist
