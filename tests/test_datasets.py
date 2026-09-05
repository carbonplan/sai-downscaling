import pytest

from saidownscale.datasets import catalog


def test_catalog(ds_info):
    """
    Tests loading the datasets from the catalog
    """
    if "-dev-" in ds_info.name:
        pytest.skip("dev store not yet written to S3")
    ds = ds_info.to_xarray()
    # is there a better way in XRT to check the data exists?

    assert len(ds) > 0, f"dataset {ds_info.name} appears empty"


class TestGcmCatalogNames:
    """Issue #598: catalog keys are the model names that output stores are named after."""

    def test_cesm_key_and_description(self):
        entry = catalog.get("CESM2-WACCM6")
        assert entry.name == "CESM2-WACCM6"
        assert entry.description == "CESM2.1.5-WACCM6(TSMLT)"
        assert str(entry.path) == "s3://carbonplan-srm/input/processed/cesm2-waccm.icechunk"

    def test_ukesm_key_and_description(self):
        entry = catalog.get("UKESM1-1-LL")
        assert entry.name == "UKESM1-1-LL"
        assert entry.description == "UKESM1.1-LL"
        assert str(entry.path) == "s3://carbonplan-srm/input/processed/ukesm.icechunk"

    @pytest.mark.parametrize("legacy", ["CESM2-WACCM", "UKESM"])
    def test_legacy_keys_are_gone(self, legacy):
        with pytest.raises(KeyError):
            catalog.get(legacy)

    def test_description_defaults_to_none(self):
        assert catalog.get("ERA5").description is None
