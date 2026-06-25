from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pytest
import xarray as xr

from srm import catalog
from srm.cache import StoreLocation
from srm.config import SCENARIO_TO_GROUP, _ensure_root_group
from srm.datasets import BaseDataset, Datatree


def make_icechunk_group(loc: StoreLocation, branch: str = "main") -> None:
    """Create an icechunk store with a commit whose message equals ``loc.group``.

    ArtifactCache.exists() finds artifacts by commit message, so the message
    must match the group path. Pass ``branch=cache.branch`` — all artifacts use
    the same branch.
    """
    import icechunk
    import numpy as np
    import xarray as xr
    from icechunk.xarray import to_icechunk

    storage = icechunk.local_filesystem_storage(path=loc.store_path)
    repo = icechunk.Repository.open_or_create(storage)
    root_snapshot_id = _ensure_root_group(repo)
    if branch not in repo.list_branches():
        repo.create_branch(branch, root_snapshot_id)
    session = repo.writable_session(branch)
    ds = xr.Dataset({"dummy": xr.DataArray(np.array([1.0]), dims=["x"])})
    to_icechunk(ds, session, mode="w")
    session.commit(loc.group)


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
    for group in SCENARIO_TO_GROUP.values()
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

    Yields non-GCM BaseDataset instances (ERA5, NASA-NEX, GDEX-GMF) plus
    DatatreeGroupEntry instances for each scenario group in the unified per-GCM
    datatree stores. Both expose the same duck-type interface so DatasetChecker
    handles them identically.
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
