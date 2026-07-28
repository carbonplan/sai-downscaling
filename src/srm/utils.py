"""
General-purpose utilities shared across pipeline stages and analysis code.

Provides coordinate normalization (:func:`lon_to_180`), variable extraction
(:func:`get_variable`), unit conversion helpers, and icechunk store accessors. These
helpers carry no pipeline-specific logic and no stage dependencies.
"""

from __future__ import annotations

from typing import Literal

import boto3
import icechunk
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


def decode_time_from_bounds(ds: xr.Dataset) -> xr.Dataset:
    """Rebuild the time axis from CF time bounds, dropping zero-width records.

    CAM stamps interval statistics at the *end* of the averaging interval, and writes one
    extra zero-width record per history stream holding the instantaneous initial state.
    Both are documented conventions rather than defects (CAM User Guide "Model Output";
    ESCOMP/CAM#159 calls the end-stamping a "longstanding quirk"). Reading ``time``
    verbatim therefore labels every daily mean one day late and admits a snapshot as
    though it were a mean, which is issue #521. That same snapshot is why ``tasmax ==
    tasmin == tas`` on the first day, so it is also the root cause of issue #424.

    ``time_bnds`` is the authority. Each record is restamped to the start of the interval
    it covers, and records whose interval has zero width are dropped. The bounds variable
    name is read from ``time.attrs["bounds"]`` rather than hardcoded, because CAM writes
    ``time_bnds`` while CLM writes ``time_bounds`` (ESCOMP/CESM#194).

    Apply this only where the bounds come from the source. Bounds we synthesise ourselves
    via :func:`srm.input_data.etl_utils.add_cf_bounds` are derived from stamp midpoints and
    do not describe the true aggregation window, so decoding from those would shift the
    data rather than correct it.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose ``time`` coordinate declares CF bounds. A dataset that declares none
        is returned unchanged, so the call is safe on sources that never carried them.

    Returns
    -------
    xr.Dataset
        Stamped at interval starts, with any zero-width records removed.
    """
    bounds_name = ds.time.attrs.get("bounds")
    if bounds_name is None or bounds_name not in ds:
        return ds

    bounds = ds[bounds_name]
    extra_dims = [d for d in bounds.dims if d != "time"]
    if len(extra_dims) != 1:
        raise ValueError(
            f"ambiguous bounds layout for {bounds_name!r}: expected exactly one non-time "
            f"dimension, found {extra_dims}. Decode each member before broadcasting the "
            "bounds over other dimensions."
        )

    bounds_dim = extra_dims[0]
    lower = bounds.isel({bounds_dim: 0})
    upper = bounds.isel({bounds_dim: 1})
    keep = (lower != upper).values
    return ds.isel(time=keep).assign_coords(time=lower.isel(time=keep).values)


def to_proleptic_gregorian(ds: xr.Dataset) -> xr.Dataset:
    """Convert any GCM Dataset to proleptic_gregorian via linear interpolation.

    - noleap    : inserts NaN on Feb 29 of each leap year, then linearly interpolates
    - 360_day   : maps dates by position within the year (align_on='year'), inserts NaN
                  on ~6 missing days per year, then linearly interpolates
    - gregorian / standard : type-cast only, no data change
    """
    # Drop time/lat/lon bounds vars: their cftime dtype combined with chunking only the
    # "time" dim triggers an xarray bug (zip() length mismatch in `_get_chunk`), and
    # they aren't needed downstream.
    bnds_vars = [
        v for v in list(ds.data_vars) + list(ds.coords) if str(v).endswith(("_bnds", "_bounds"))
    ]
    ds = ds.drop_vars(bnds_vars, errors="ignore")

    calendar = ds.time.dt.calendar
    if calendar in ("proleptic_gregorian", "gregorian", "standard"):
        return ds.convert_calendar("proleptic_gregorian", use_cftime=False)
    align: Literal["year"] | None = "year" if calendar == "360_day" else None
    ds = (
        ds.convert_calendar(
            "proleptic_gregorian", align_on=align, missing=float("nan"), use_cftime=False
        )
        .chunk({"time": -1})
        .interpolate_na(dim="time")
    )
    # enforce numpy datetime64, not some mixed float/cftime
    return ds.assign_coords(time=ds.time.values)


def get_variable(ds: xr.Dataset, variable: str) -> xr.DataArray:
    """Return a variable from ds, deriving dtr = tasmax - tasmin when not stored."""
    if variable == "dtr" and "dtr" not in ds:
        dtr = (ds["tasmax"] - ds["tasmin"]).rename("dtr")
        dtr.attrs.update({"units": "K", "long_name": "Diurnal Temperature Range"})
        return dtr
    return ds[variable]


def resolve_s3_glob(path):
    """Resolve a single * wildcard in an S3 path to a real path."""
    bucket, prefix = path.replace("s3://", "").split("/", 1)

    before, after = prefix.split("*/", 1)

    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=before, Delimiter="/")

    matches = [f"s3://{bucket}/{cp['Prefix']}{after}" for cp in response.get("CommonPrefixes", [])]

    if not matches:
        raise ValueError(f"No S3 paths matched: {path}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches: {matches}")

    return matches[0]


def open_icechunk(path, group=None, branch="main"):
    bucket, prefix = path.replace("s3://", "").rstrip("/").split("/", 1)
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch)

    if group is not None:
        ds = xr.open_dataset(session.store, engine="zarr", group=group, chunks={})
    else:
        ds = xr.open_dataset(session.store, engine="zarr", chunks={})
    return ds
