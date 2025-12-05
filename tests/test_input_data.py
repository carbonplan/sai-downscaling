import icechunk
import pytest
import xarray as xr

from srm import catalog


@pytest.fixture(params=catalog.list())
def dataset_name(request):
    return request.param


@pytest.fixture
def dataset(dataset_name):
    dataset_obj = catalog.get(dataset_name)

    if dataset_obj.format == "icechunk":
        storage = icechunk.s3_storage(
            bucket=dataset_obj.bucket, prefix=dataset_obj.prefix, from_env=True
        )
        repo = icechunk.Repository.open(storage)
        session = repo.writable_session("main")
        ds = xr.open_zarr(session.store, consolidated=False)
    else:
        raise ValueError(f"Format: {dataset_obj.format} is not Icechunk. Update tests")

    return ds, dataset_name


def test_longitude_range(dataset):
    ds, name = dataset

    lon_names = ["lon", "longitude"]
    lon_coord = None
    for lon_name in lon_names:
        if lon_name in ds.coords:
            lon_coord = ds[lon_name]
            break

    assert lon_coord is not None, f"{name}: No longitude coordinate found"

    lon_values = lon_coord.values
    assert lon_values.min() >= -180, f"{name}: Longitude minimum {lon_values.min()} < -180"
    assert lon_values.max() <= 180, f"{name}: Longitude maximum {lon_values.max()} > 180"
