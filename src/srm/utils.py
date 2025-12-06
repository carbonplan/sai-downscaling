from __future__ import annotations

from typing import TYPE_CHECKING

import icechunk
import xarray as xr

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
    return ds.sortby(ds[lon_name])

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
        ds = ds.rename({"TREFHT": "tas"})
        ds = ds.rename({"TREFHTMX": "tasmax"})
        ds = ds.rename({"TREFHTMN": "tasmin"})
        ds = ds.rename({"PRECT": "pr"})
        ds = ds.rename({"FSDS": "rsds"})

    elif model == 'ERA5':
        ds = ds.rename({"2m_temperature": "tas"})
        ds = ds.rename({"maximum_2m_temperature_since_previous_post_processing": "tasmax"})
        ds = ds.rename({"minimum_2m_temperature_since_previous_post_processing": "tasmin"})
        ds = ds.rename({"mean_total_precipitation_rate": "pr"})
        ds = ds.rename({"mean_surface_downward_short_wave_radiation_flux": "rsds"})
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
    return da * 86400


def clean_up_dataset(ds, model):
    ds = rename_variables(ds, model)
    ds = rename_coords(ds)
    # make sure coords go from -180 to 180 and not to 360
    # this won't do anything if it's already on the -180-180 scale
    ds = lon_to_180(ds, lon_name='longitude')
    # add a geographic coordinate system
    ds = ds.proj.assign_crs(spatial_ref="epsg:4326")
    ds['pr'] = convert_precip_units(ds['pr'])
    return ds
