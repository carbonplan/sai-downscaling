def test_catalog(ds_info):
    """
    Tests loading the datasets from the catalog
    """
    ds = ds_info.to_xarray()
    # is there a better way in XRT to check the data exists?

    assert len(ds) > 0, f"dataset {ds_info.name} appears empty"
