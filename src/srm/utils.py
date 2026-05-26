from __future__ import annotations

import pint_xarray
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


def convert_precip_units(da: xr.DataArray) -> xr.DataArray:
    """Convert precipitation units to mm/day."""

    pint_xarray.unit_registry.enable_contexts("hydro")
    result = da.pint.quantify().pint.to("mm/day").pint.dequantify().astype(da.dtype)
    return result


def to_proleptic_gregorian(ds: xr.Dataset) -> xr.Dataset:
    """Convert any GCM Dataset to proleptic_gregorian via linear interpolation.

    - noleap    : inserts NaN on Feb 29 of each leap year, then linearly interpolates
    - 360_day   : maps dates by position within the year (align_on='year'), inserts NaN
                  on ~6 missing days per year, then linearly interpolates
    - gregorian / standard : type-cast only, no data change
    """
    calendar = ds.time.dt.calendar
    if calendar in ("proleptic_gregorian", "gregorian", "standard"):
        return ds.convert_calendar("proleptic_gregorian", use_cftime=False)
    align = "year" if calendar == "360_day" else None
    return ds.convert_calendar(
        "proleptic_gregorian", align_on=align, missing=float("nan"), use_cftime=False
    ).interpolate_na(dim="time")
