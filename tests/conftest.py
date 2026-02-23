import numpy as np
import pytest
import xarray as xr

from srm import catalog


@pytest.fixture(scope="session")
def dataset_catalog():
    """fixture for the catalog"""
    return catalog


@pytest.fixture(params=list(catalog.datasets.values()), ids=lambda ds: ds.name)
def ds_info(request):
    """Parametrize by dataset objects"""
    return request.param


@pytest.fixture
def ds_monotonic():
    return xr.Dataset(
        {"temp": (["lat", "lon"], np.random.rand(10, 20))},
        coords={
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(0, 360, 20, endpoint=False),
        },
    )


@pytest.fixture
def ds_non_monotonic():
    return xr.Dataset(
        {"temp": (["lat", "lon"], np.random.rand(10, 8))},
        coords={
            "lat": np.linspace(-90, 90, 10),
            "lon": [350, 355, 0, 5, 10, 180, 270, 300],
        },
    )


@pytest.fixture
def ds_0_360():
    return xr.Dataset(
        {"temp": (["lat", "lon"], np.random.rand(10, 20))},
        coords={
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(0, 360, 20, endpoint=False),
        },
    )
