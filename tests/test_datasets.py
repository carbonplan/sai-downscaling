import pytest

from srm.bcsd_config import BCSDConfig
from srm.datasets import Catalog, validate_configs_against_catalog


def _catalog_with_defective() -> Catalog:
    """Return a minimal Catalog whose CESM SSP245 entry carries defective_members."""
    cat = Catalog()
    return cat


def _make_config(
    gcm="CESM2-WACCM",
    variable="tas",
    member="r7i1p1f1",
) -> BCSDConfig:
    return BCSDConfig(gcm=gcm, variable=variable, ensemble_member=member)


def test_catalog(ds_info):
    """
    Tests loading the datasets from the catalog
    """
    ds = ds_info.to_xarray()
    # is there a better way in XRT to check the data exists?

    assert len(ds) > 0, f"dataset {ds_info.name} appears empty"


# ---------------------------------------------------------------------------
# BaseDataset.defective_members field
# ---------------------------------------------------------------------------


class TestDefectiveMembersField:
    def test_defaults_to_none(self):
        cat = Catalog()
        # Historical CESM dataset has no defective_members annotation
        ds = cat.get("CESM2-WACCM-Historical-icechunk")
        assert ds.defective_members is None

    def test_populated_for_ssp245(self):
        cat = Catalog()
        ds = cat.get("CESM2-WACCM-SSP245-icechunk")
        assert ds.defective_members is not None
        assert "tasmax" in ds.defective_members
        assert "tasmin" in ds.defective_members

    def test_populated_for_g6(self):
        cat = Catalog()
        ds = cat.get("CESM2-WACCM-G6-1.5K-icechunk")
        assert ds.defective_members is not None
        assert "tasmax" in ds.defective_members
        assert "tasmin" in ds.defective_members

    def test_defective_members_are_lists_of_strings(self):
        cat = Catalog()
        ds = cat.get("CESM2-WACCM-SSP245-icechunk")
        for var, members in ds.defective_members.items():
            assert isinstance(members, list)
            assert all(isinstance(m, str) for m in members)

    def test_known_defective_members_present(self):
        cat = Catalog()
        ds = cat.get("CESM2-WACCM-SSP245-icechunk")
        for var in ("tasmax", "tasmin"):
            for member in ("r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "r4i1p1f1", "r5i1p1f1"):
                assert member in ds.defective_members[var], (
                    f"{member} should be marked defective for {var}"
                )

    def test_other_variables_not_in_defective_members(self):
        cat = Catalog()
        ds = cat.get("CESM2-WACCM-SSP245-icechunk")
        for var in ("tas", "pr"):
            assert var not in ds.defective_members


# ---------------------------------------------------------------------------
# Catalog.get_defective_members
# ---------------------------------------------------------------------------


class TestGetDefectiveMembers:
    def test_returns_defective_members_for_tasmax(self):
        cat = Catalog()
        bad = cat.get_defective_members("CESM2-WACCM", "tasmax")
        assert isinstance(bad, set)
        for member in ("r1i1p1f1", "r2i1p1f1", "r3i1p1f1"):
            assert member in bad

    def test_returns_defective_members_for_tasmin(self):
        cat = Catalog()
        bad = cat.get_defective_members("CESM2-WACCM", "tasmin")
        assert "r1i1p1f1" in bad

    def test_returns_empty_set_for_tas(self):
        cat = Catalog()
        bad = cat.get_defective_members("CESM2-WACCM", "tas")
        assert bad == set()

    def test_returns_empty_set_for_different_gcm(self):
        cat = Catalog()
        # MIROC has no defective_members annotation
        bad = cat.get_defective_members("MIROC-ES2H", "tasmax")
        assert bad == set()

    def test_aggregates_across_multiple_datasets(self):
        # Both SSP245 and G6-1.5K CESM datasets annotate defective members;
        # the result should be the union.
        cat = Catalog()
        bad = cat.get_defective_members("CESM2-WACCM", "tasmax")
        # SSP245 marks r1–r5, G6-1.5K marks r1–r3; union covers at least r1–r5
        for member in ("r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "r4i1p1f1", "r5i1p1f1"):
            assert member in bad

    def test_case_insensitive_gcm_match(self):
        cat = Catalog()
        bad_lower = cat.get_defective_members("cesm2-waccm", "tasmax")
        bad_upper = cat.get_defective_members("CESM2-WACCM", "tasmax")
        assert bad_lower == bad_upper


# ---------------------------------------------------------------------------
# Catalog.is_defective
# ---------------------------------------------------------------------------


class TestIsDefective:
    def test_known_bad_combo_returns_true(self):
        cat = Catalog()
        assert cat.is_defective("CESM2-WACCM", "tasmax", "r1i1p1f1") is True

    def test_good_combo_returns_false(self):
        cat = Catalog()
        assert cat.is_defective("CESM2-WACCM", "tasmax", "r7i1p1f1") is False

    def test_tas_with_defective_member_returns_false(self):
        # r1 is only defective for tasmax/tasmin, not for tas
        cat = Catalog()
        assert cat.is_defective("CESM2-WACCM", "tas", "r1i1p1f1") is False

    def test_different_gcm_returns_false(self):
        cat = Catalog()
        assert cat.is_defective("MIROC-ES2H", "tasmax", "r1i1p1f1") is False


# ---------------------------------------------------------------------------
# validate_configs_against_catalog
# ---------------------------------------------------------------------------


class TestValidateConfigsAgainstCatalog:
    def test_clean_configs_do_not_raise(self):
        cat = Catalog()
        configs = [
            _make_config(gcm="CESM2-WACCM", variable="tasmax", member="r7i1p1f1"),
            _make_config(gcm="CESM2-WACCM", variable="tas", member="r1i1p1f1"),
        ]
        # Should not raise
        validate_configs_against_catalog(configs, cat)

    def test_defective_config_raises_value_error(self):
        cat = Catalog()
        configs = [_make_config(gcm="CESM2-WACCM", variable="tasmax", member="r1i1p1f1")]
        with pytest.raises(ValueError, match="defective"):
            validate_configs_against_catalog(configs, cat)

    def test_error_message_contains_issue_link(self):
        cat = Catalog()
        configs = [_make_config(gcm="CESM2-WACCM", variable="tasmax", member="r2i1p1f1")]
        with pytest.raises(ValueError, match="issues/156"):
            validate_configs_against_catalog(configs, cat)

    def test_error_message_lists_all_bad_combos(self):
        cat = Catalog()
        configs = [
            _make_config(gcm="CESM2-WACCM", variable="tasmax", member="r1i1p1f1"),
            _make_config(gcm="CESM2-WACCM", variable="tasmax", member="r2i1p1f1"),
            _make_config(gcm="CESM2-WACCM", variable="tas", member="r7i1p1f1"),  # clean
        ]
        with pytest.raises(ValueError) as exc_info:
            validate_configs_against_catalog(configs, cat)
        msg = str(exc_info.value)
        assert "tasmax" in msg
        assert "r1i1p1f1" in msg
        assert "r2i1p1f1" in msg

    def test_empty_config_list_does_not_raise(self):
        cat = Catalog()
        validate_configs_against_catalog([], cat)

    def test_only_bad_config_in_list_raises(self):
        cat = Catalog()
        configs = [_make_config(gcm="CESM2-WACCM", variable="tasmax", member="r3i1p1f1")]
        with pytest.raises(ValueError):
            validate_configs_against_catalog(configs, cat)

    def test_non_cesm_defective_member_patterns_not_flagged(self):
        """r1i1p1f1 for MIROC or UKESM is not marked defective."""
        cat = Catalog()
        configs = [
            _make_config(gcm="MIROC-ES2H", variable="tasmax", member="r1i1p1f1"),
        ]
        # Should not raise
        validate_configs_against_catalog(configs, cat)
