import pytest


def test_catalog(ds_info):
    """
    Tests loading the datasets from the catalog
    """
    if "-dev-" in ds_info.name:
        pytest.skip("dev store not yet written to S3")
    group = "historical" if "-unified-" in ds_info.name else None
    ds = ds_info.to_xarray(group=group)
    # is there a better way in XRT to check the data exists?

    assert len(ds) > 0, f"dataset {ds_info.name} appears empty"
