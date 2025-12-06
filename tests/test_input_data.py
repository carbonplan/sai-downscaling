import cf_xarray  # noqa
import icechunk
import pytest
import xarray as xr
from validators import DatasetValidator

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


class TestCatalogDatasets:
    """Validate input datasets in the catalog"""

    def test_longitude_valid(self, dataset):
        ds, name = dataset
        validator = DatasetValidator(ds)
        is_valid = validator.validate_lon(check_monotonic=True)
        assert is_valid, f"{name} lon validation failed: {validator.get_issues()}"

    def test_latitude_valid(self, dataset):
        ds, name = dataset
        validator = DatasetValidator(ds)
        is_valid = validator.validate_lat(check_monotonic=True)
        assert is_valid, f"{name} lat validation failed: {validator.get_issues()}"
