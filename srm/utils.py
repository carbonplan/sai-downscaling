from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr


def lon_to_180(ds: xr.Dataset) -> xr.Dataset:
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
    lon = ds["longitude"].where(ds["longitude"] < 180, ds["longitude"] - 360)
    ds = ds.assign_coords(longitude=lon)
    return ds
