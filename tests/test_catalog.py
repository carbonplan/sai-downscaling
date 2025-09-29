from srm import catalog
import icechunk
import xarray as xr
from typing import Literal

dataset_type = Literal["zarr", "icechunk"]


def test_catalog():
    datasets = catalog.datasets
    for _, ds_info in datasets.items():
        if ds_info.format == "icechunk":
            storage = icechunk.s3_storage(
                bucket=ds_info.bucket, prefix=ds_info.prefix, from_env=True
            )
            repo = icechunk.Repository.open(storage)
            session = repo.readonly_session("main")
            ds = xr.open_zarr(session.store, consolidated=False)
            # check dataset isn't empty
            assert len(ds.data_vars) > 0
        elif ds_info.format == "zarr":
            ds = xr.open_zarr(str(ds_info.path))
            assert len(ds.data_vars) > 0
        else:
            raise ValueError(f"dataset type: {ds_info.format}, not in {dataset_type}")
