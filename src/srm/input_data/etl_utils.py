# etl_utils.py
import dask
import icechunk
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr.parsers import HDFParser

from srm.config import VarSpec, init_repo

ENSEMBLE_MEMBER_MAPPING = {f"{i:03d}": f"r{i}i1p1f1" for i in [1, 2, 3, 4, 5, 7, 8, 9, 10]}


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
    # convert fraction to %, multiply by 1000
    ds[var_name] = ds[var_name] * 1000
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


def load_dtr_from_store(
    bucket: str, prefix: str, shards: dict, region: str = "us-west-2"
) -> xr.Dataset:
    """Load DTR (diurnal temperature range) from an existing icechunk store.

    Computes dtr = tasmax - tasmin. tasmax and tasmin must already be present
    in the store before calling this.
    """
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region=region)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    ds = xr.open_dataset(session.store, engine="zarr", chunks=shards)
    if "tasmax" not in ds or "tasmin" not in ds:
        raise ValueError("tasmax and tasmin must be processed before dtr")
    dtr = (ds["tasmax"] - ds["tasmin"]).rename("dtr")
    dtr.attrs["units"] = "K"
    return dtr.to_dataset()


def remap_ensemble_members(bucket: str, prefix: str, mapping: dict) -> None:
    repo, session = init_repo(bucket, prefix, readonly=False)
    root = zarr.open_group(session.store, mode="r+")
    em_array = root["ensemble_member"]
    em_array[:] = [mapping.get(v, v) for v in em_array[:]]
    session.commit("remap ensemble_member coords to CMIP6 ripf labels")
