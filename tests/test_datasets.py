import pytest

from srm.datasets import Datatree


def test_catalog(ds_info):
    """
    Tests loading the datasets from the catalog
    """
    if "-dev-" in ds_info.name:
        pytest.skip("dev store not yet written to S3")
    result = ds_info.to_xarray()
    if isinstance(ds_info, Datatree):
        ds = result["historical"].ds
    else:
        ds = result
    assert len(ds) > 0, f"dataset {ds_info.name} appears empty"
