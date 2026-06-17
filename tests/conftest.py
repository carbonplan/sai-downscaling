from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pytest
import xarray as xr

from srm import catalog
from srm.datasets import BaseDataset, Datatree
from srm.validation import _SCENARIO_TO_GROUP


@dataclass
class DatatreeGroupEntry:
    """Wraps one scenario group in a unified datatree store for test parametrization.

    Provides the same duck-type interface as BaseDataset (name, expected_vars,
    to_xarray) so DatasetChecker works without modification. The datatree is opened
    lazily at test runtime — never at collection time — and cached per GCM per worker
    process so each store is opened at most once.
    """

    name: str
    _entry: Datatree = field(repr=False)
    _group: str = field(repr=False)
    expected_vars: None = None

    _dt_cache: ClassVar[dict[str, xr.DataTree]] = {}

    def to_xarray(self) -> xr.Dataset:
        gcm_name = self._entry.name
        if gcm_name not in DatatreeGroupEntry._dt_cache:
            DatatreeGroupEntry._dt_cache[gcm_name] = self._entry.to_xarray()
        dt = DatatreeGroupEntry._dt_cache[gcm_name]
        if self._group not in dt.children:
            pytest.skip(f"Group '{self._group}' not present in datatree for {gcm_name}")
        return dt[self._group].to_dataset()


_DATATREE_PARAMS = [
    DatatreeGroupEntry(
        name=f"{entry.name}/{group}",
        _entry=entry,
        _group=group,
    )
    for entry in catalog.datasets.values()
    if isinstance(entry, Datatree)
    for group in _SCENARIO_TO_GROUP.values()
]


@pytest.fixture(scope="session")
def dataset_catalog():
    """fixture for the catalog"""
    return catalog


@pytest.fixture(
    params=[
        ds
        for ds in catalog.datasets.values()
        if isinstance(ds, BaseDataset) and not isinstance(ds, Datatree)
    ]
    + _DATATREE_PARAMS,
    ids=lambda ds: ds.name,
)
def ds_info(request):
    """Parametrize over all xarray-compatible catalog entries.

    Yields BaseDataset instances (Dataset, VirtualDataset) for the legacy per-scenario
    icechunk stores, and DatatreeGroupEntry instances for each scenario group in the
    unified per-GCM datatree stores. Both expose the same duck-type interface so
    DatasetChecker handles them identically.
    """
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
