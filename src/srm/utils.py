from __future__ import annotations

from typing import TYPE_CHECKING

import icechunk
import xarray as xr

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

def icechunk_store_to_dataset(dataset_name, catalog):
    catalog_entry = catalog.get(dataset_name)
    icechunk_store = icechunk.s3_storage(
        bucket=catalog_entry.bucket,
        prefix=catalog_entry.prefix,
        from_env=True,
    )
    icechunk_session = icechunk.Repository.open(icechunk_store).readonly_session("main")
    ds = xr.open_zarr(icechunk_session.store, consolidated=False)
    return ds

def rename_variables(ds, model):
    if model == 'CESM2-WACCM':
        ds = ds.rename({"TREFHT": "TASMEAN"})
        ds = ds.rename({"TREFHTMX": "TASMAX"})
        ds = ds.rename({"TREFHTMN": "TASMIN"})
        ds = ds.rename({"PRECT": "PREC"})
        ds = ds.rename({"FSDS": "RSDS"})

    elif model == 'ERA5':
        ds = ds.rename({"2m_temperature": "TASMEAN"})
        ds = ds.rename({"maximum_2m_temperature_since_previous_post_processing": "TASMAX"})
        ds = ds.rename({"minimum_2m_temperature_since_previous_post_processing": "TASMIN"})
        ds = ds.rename({"mean_total_precipitation_rate": "PREC"})
        ds = ds.rename({"mean_surface_downward_short_wave_radiation_flux": "RSDS"})
    return ds

def rename_coords(ds):
    if 'lon' in ds.coords:
        ds = ds.rename({'lon': 'longitude'})
    if 'lat' in ds.coords:
        ds  = ds.rename({'lat': 'latitude'})
    return ds

def convert_precip_units(da):
    # convert m/s to mm/day for intuition
    # TODO: change this to using metpy for safer unit conversions
    return da * 1000 * 86400


def clean_up_dataset(ds, model):
    ds = rename_variables(ds, model)
    ds = rename_coords(ds)
    # make sure coords go from -180 to 180 and not to 360
    # this won't do anything if it's already on the -180-180 scale
    ds = lon_to_180(ds)
    # add a geographic coordinate system
    ds = ds.proj.assign_crs(spatial_ref="epsg:4326")
    ds['PREC'] = convert_precip_units(ds['PREC'])
    return ds


