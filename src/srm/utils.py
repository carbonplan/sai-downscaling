from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr


def lon_to_180(ds: xr.Dataset, lon_name: str = "lon") -> xr.Dataset:
    """
    Convert longitude values from 0-360 to -180-180.

    Note: `longitude` is required dim/coord.

    Parameters
    ----------
    ds : xr.Dataset
        Input Xarray dataset

    Returns
    -------
    xr.Dataset
        Dataset with longitude coordinates converted to -180-180 range
    """

    ds.coords[lon_name] = (ds.coords[lon_name] + 180) % 360 - 180
    ds = ds.sortby(ds[lon_name])
    return ds
