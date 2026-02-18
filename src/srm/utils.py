from __future__ import annotations
#brendanc comment
import time
from typing import TYPE_CHECKING

import pint_xarray
import xarray as xr

if TYPE_CHECKING:
    import xarray as xr


class Timer:
    """
    Context manager for timing code blocks with optional verbose output.

    Parameters
    ----------
    name : str
        Descriptive name for the timed operation
    verbose : bool, default=True
        If True, prints timing information when exiting the context

    Examples
    --------
    >>> with Timer("data loading", verbose=True):
    ...     data = load_data()
    data loading: 2.34 seconds

    >>> with Timer("processing", verbose=False):
    ...     process_data()  # No output
    """

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time: float | None = None
        self.elapsed: float | None = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"{self.name}: {self.elapsed:.2f} seconds")
        return False  # Don't suppress exceptions


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
    return ds.sortby(ds[lon_name])


def rename_variables(ds, model):
    if model == "CESM2-WACCM":
        ds = ds.rename({"TREFHT": "tas"})
        ds = ds.rename({"TREFHTMX": "tasmax"})
        ds = ds.rename({"TREFHTMN": "tasmin"})
        ds = ds.rename({"PRECT": "pr"})
        ds = ds.rename({"FSDS": "rsds"})

    elif model == "ERA5":
        ds = ds.rename({"2m_temperature": "tas"})
        ds = ds.rename({"maximum_2m_temperature_since_previous_post_processing": "tasmax"})
        ds = ds.rename({"minimum_2m_temperature_since_previous_post_processing": "tasmin"})
        ds = ds.rename({"mean_total_precipitation_rate": "pr"})
        ds = ds.rename({"mean_surface_downward_short_wave_radiation_flux": "rsds"})
    return ds


def rename_coords(ds):
    if "lon" in ds.coords:
        ds = ds.rename({"lon": "longitude"})
    if "lat" in ds.coords:
        ds = ds.rename({"lat": "latitude"})
    return ds

import xarray as xr

def convert_precip_units(da: xr.DataArray) -> xr.DataArray:
    """Convert precipitation units to mm/day."""

    pint_xarray.unit_registry.enable_contexts("hydro")
    result = da.pint.quantify().pint.to("mm/day").pint.dequantify().astype(da.dtype)
    return result
