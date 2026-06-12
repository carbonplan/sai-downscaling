import json
import logging

import click
import dask
import icechunk
import xarray as xr
from obspec_utils.readers import EagerStoreReader
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser

from srm.config import VarSpec

logger = logging.getLogger(__name__)


def compute_wind_speed(
    ds_u: xr.Dataset, ds_v: xr.Dataset, u_var_name: str, v_var_name: str
) -> xr.Dataset:
    import xclim

    winds = xclim.indicators.convert.wind_speed_from_vector(
        uas=ds_u[u_var_name], vas=ds_v[v_var_name]
    )
    return xr.merge(winds)[["sfcWind"]]


def get_var_specs(catalog_entry) -> dict[str, VarSpec]:
    return {var.name: var for var in catalog_entry.expected_vars}


def CMORIZE_pr(ds: xr.Dataset, var_name: str) -> xr.Dataset:
    # CMORization CESM script says multiply by 1000)
    ds[var_name] = ds[var_name] * 1000.0
    return ds


def CMORIZE_hurs(ds: xr.Dataset, var_name: str) -> xr.Dataset:
    # convert fraction (0-1) to percent (0-100)
    ds[var_name] = ds[var_name] * 100
    return ds


def trim_negative_precipitation(ds: xr.Dataset) -> xr.Dataset:
    if "pr" in ds.data_vars:
        ds["pr"] = ds["pr"].clip(min=0)
    return ds


def determine_write_mode(repo: icechunk.Repository, branch: str = "main") -> str:
    history = list(repo.ancestry(branch=branch))
    if len(history) <= 1:
        return "w"
    else:
        return "a"


def build_encoding_dict(ds: xr.Dataset, chunks: dict, shards: dict) -> dict:
    encoding = {}
    for var_name in ds.data_vars:
        if var_name.endswith("_bounds") or var_name.endswith("_bnds"):
            continue
        var = ds[var_name]
        var_chunks = tuple(chunks[d] for d in var.dims)
        var_shards = tuple(shards[d] for d in var.dims)
        encoding[var_name] = {
            "chunks": var_chunks,
            "shards": var_shards,
        }
    return encoding


def add_cf_bounds(ds: xr.Dataset, coord_names: list[str] = None) -> xr.Dataset:
    import cf_xarray  # noqa ignore

    if coord_names is None:
        coord_names = ["time", "lat", "lon"]

    bnds_to_drop = [
        v
        for v in list(ds.data_vars) + list(ds.coords)
        if v.endswith("_bnds") or v.endswith("_bounds")
    ]
    ds = ds.drop_vars(bnds_to_drop, errors="ignore")

    for coord in coord_names:
        if coord in ds.coords or coord in ds.dims:
            ds = ds.cf.add_bounds(coord)

    bounds_vars = [v for v in ds.data_vars if v.endswith("_bounds")]
    if bounds_vars:
        ds = ds.set_coords(bounds_vars)

    return ds


def apply_ensemble_provenance(
    ds: xr.Dataset,
    derivation_logic: str,
    member_provenance: dict | None = None,
    ensemble_coord: str = "ensemble_member",
) -> xr.Dataset:
    """Set ensemble_member provenance on dataset global attrs and ensemble coord attrs."""

    ds.attrs["ensemble_derivation_logic"] = derivation_logic
    if ensemble_coord in ds.coords:
        coord_attrs: dict = {
            "long_name": "Ensemble Member Identifier",
            "derivation_method": derivation_logic,
        }
        if member_provenance is not None:
            coord_attrs["member_specific_provenance"] = json.dumps(member_provenance)
        ds[ensemble_coord].attrs.update(coord_attrs)
    return ds


def update_variable_attrs(ds: xr.Dataset, var_specs: dict[str, VarSpec]) -> xr.Dataset:
    for var_name, spec in var_specs.items():
        if var_name in ds.data_vars:
            ds[var_name].attrs["units"] = spec.units
            if spec.long_name:
                ds[var_name].attrs["long_name"] = spec.long_name
            if spec.cell_methods:
                ds[var_name].attrs["cell_methods"] = spec.cell_methods
    return ds


def write_dataset_to_icechunk(
    ds: xr.Dataset,
    session,
    encoding: dict = None,
    shards: dict = None,
    commit_message: str = None,
    write_mode: str = "a",
):
    """
    Write dataset to icechunk with optional rechunking.

    For virtual datasets: set encoding=None and shards=None
    For materialized datasets: provide encoding and shards for rechunking
    """
    from icechunk.xarray import to_icechunk

    if shards is not None:
        ds = ds.chunk(shards)
    to_icechunk(ds, session, encoding=encoding, mode=write_mode)

    if commit_message:
        session.commit(commit_message)


def open_netcdf_from_s3(store, path: str, drop_variables: list[str] | None = None) -> xr.Dataset:
    """Open a NetCDF file on S3 by reading it fully into memory."""
    reader = EagerStoreReader(store, path)
    return xr.open_dataset(reader, engine="h5netcdf", chunks="auto", drop_variables=drop_variables)


def variable_in_store(repo: icechunk.Repository, variable: str) -> bool:
    try:
        session = repo.readonly_session("main")
        existing = xr.open_dataset(session.store, engine="zarr", decode_times=False)
        return variable in existing.data_vars
    except Exception:
        return False


def group_paths_by_member(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Group (member_id, path) pairs into {member_id: [paths]}."""
    member_paths: dict[str, list[str]] = {}
    for member, path in pairs:
        member_paths.setdefault(member, []).append(path)
    return member_paths


def resolve_variables(variable: tuple[str, ...], all_variables: bool, catalog_entry) -> list[str]:
    """Resolve --variable/--all-variables CLI options against a catalog entry."""
    if all_variables:
        return [var.name for var in catalog_entry.expected_vars]
    if variable:
        return list(variable)
    raise click.UsageError("Must specify either --variable or --all-variables")


def write_variable_to_icechunk(
    ds: xr.Dataset,
    repo: icechunk.Repository,
    *,
    variable: str,
    scenario: str,
    chunks: dict,
    shards: dict,
    overwrite: bool,
    var_in_store: bool,
) -> None:
    """Write one variable to icechunk with shared overwrite semantics.

    overwrite + existing variable -> in-place r+ update (no encoding change);
    otherwise append/write with fresh chunk/shard encoding.
    """
    session = repo.writable_session("main")
    if overwrite and var_in_store:
        write_mode = "r+"
    elif overwrite:
        write_mode = "a"
    else:
        write_mode = determine_write_mode(repo)
    encoding = build_encoding_dict(ds, chunks, shards)
    logger.info("variable=%s writing to icechunk write_mode=%s", variable, write_mode)
    write_dataset_to_icechunk(
        ds,
        session,
        encoding=None if overwrite else encoding,
        shards=None if overwrite else shards,
        commit_message=f"{scenario}: {variable}" + (" (overwrite)" if overwrite else ""),
        write_mode=write_mode,
    )


def virtualize_netcdf(
    url: str,
    registry: ObjectStoreRegistry,
    parser: HDFParser,
    loadable_variables: list[str] | None = None,
    drop_variables: list[str] | None = None,
    preprocess_fn: callable = None,
) -> xr.Dataset:
    """
    Opens a single file virtually and applies the preprocessor with URL context.
    """
    from virtualizarr import open_virtual_dataset

    ds = open_virtual_dataset(
        url,
        registry=registry,
        parser=parser,
        loadable_variables=loadable_variables,
        drop_variables=drop_variables,
    )
    if preprocess_fn:
        ds = preprocess_fn(ds, url=url)

    return ds


def virtualize_and_combine(
    urls: list[str],
    registry: ObjectStoreRegistry,
    parser: HDFParser,
    preprocess_fn: callable = None,
    loadable_variables: list[str] | None = None,
    drop_variables: list[str] | None = None,
) -> xr.Dataset:
    """
    Parallellizes the virtualization of multiple files and combines them.
    """

    delayed_datasets = [
        dask.delayed(virtualize_netcdf)(
            url, registry, parser, loadable_variables, drop_variables, preprocess_fn
        )
        for url in urls
    ]

    ds_list = list(dask.compute(*delayed_datasets))
    return xr.combine_by_coords(
        ds_list,
        coords="minimal",
        data_vars="minimal",
        compat="override",
        combine_attrs="override",
    )


def load_dtr_from_store(bucket: str, prefix: str, region: str = "us-west-2") -> xr.Dataset:
    """Load DTR (diurnal temperature range) from an existing icechunk store.

    Computes dtr = tasmax - tasmin. tasmax and tasmin must already be present
    in the store before calling this.
    """
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region=region)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    ds = xr.open_dataset(session.store, engine="zarr", chunks="auto")
    if "tasmax" not in ds or "tasmin" not in ds:
        raise ValueError("tasmax and tasmin must be processed before dtr")
    dtr = (ds["tasmax"] - ds["tasmin"]).rename("dtr")
    dtr.attrs["units"] = "K"
    return dtr.to_dataset()
