"""Diagnostic script for hurs units and value ranges across CMIP datasets."""

from srm.datasets import VirtualDataset, catalog
from srm.validation import GCM_OPTIONS

DAY_INDEX = 0

for name, entry in catalog.datasets.items():
    if not any(gcm in name for gcm in GCM_OPTIONS):
        continue

    ds = entry.to_xarray()
    if "hurs" not in ds:
        print(f"{name}: hurs not present")
        continue

    da = ds["hurs"].isel(time=DAY_INDEX)
    if "ensemble_member" in da.dims:
        da = da.isel(ensemble_member=0)

    units = ds["hurs"].attrs.get("units", "MISSING")
    spatial_min = float(da.min().compute())
    spatial_max = float(da.max().compute())
    sample = float(da.isel(lat=0, lon=0).compute())

    virtual_tag = " [virtual]" if isinstance(entry, VirtualDataset) else ""
    print(f"{name}{virtual_tag}:")
    print(f"  units attr : {units!r}")
    print(f"  spatial min: {spatial_min:.4g}")
    print(f"  spatial max: {spatial_max:.4g}")
    print(f"  sample[0,0]: {sample:.4g}")
