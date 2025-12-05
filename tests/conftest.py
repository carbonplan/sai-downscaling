import numpy as np
import pytest
import xarray as xr


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
