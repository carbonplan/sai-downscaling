import pytest

from saidownscale.datasets import catalog


def test_every_catalog_dataset_opens_non_empty(ds_info):
    if "-dev-" in ds_info.name:
        pytest.skip("dev store not yet written to S3")
    assert len(ds_info.to_xarray()) > 0, f"dataset {ds_info.name} appears empty"


def test_gcm_catalog_names(subtests):
    """#598: catalog keys are the model names output stores are named after."""
    prefix = "s3://us-west-2.opendata.source.coop/carbonplan/srm-downscaling/input/processed"
    for key, description in [
        ("CESM2-WACCM6", "CESM2.1.5-WACCM6(TSMLT)"),
        ("UKESM1-1-LL", "UKESM1.1-LL"),
    ]:
        with subtests.test(key=key):
            entry = catalog.get(key)
            assert (entry.name, entry.description) == (key, description)
            assert str(entry.path) == f"{prefix}/{key}.icechunk"

    with subtests.test(key="ERA5"):
        assert catalog.get("ERA5").description is None

    for legacy in ["CESM2-WACCM", "UKESM"]:
        with subtests.test(legacy=legacy), pytest.raises(KeyError):
            catalog.get(legacy)
