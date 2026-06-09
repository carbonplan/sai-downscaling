import json
import logging

import boto3
import dask
import icechunk
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from virtualizarr.parsers import HDFParser

from srm.config import VarSpec

logger = logging.getLogger(__name__)
console = Console()


def get_aws_creds() -> dict:
    """Return AWS credentials and region from the active boto3 session."""
    sesh = boto3.Session()
    creds = sesh.get_credentials()
    return {
        "region": sesh.region_name,
        "aws_access_key_id": creds.access_key,
        "aws_secret_access_key": creds.secret_key,
    }


def make_fixed_ensemble_preprocess(member: str):
    """Return a preprocess function that stamps a known ensemble member onto every file."""

    def fn(ds: xr.Dataset, url: str | None = None) -> xr.Dataset:  # noqa: ARG001
        return ds.expand_dims({"ensemble_member": [member]})

    return fn


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
    encoding: dict | None = None,
    shards: dict | None = None,
    commit_message: str | None = None,
    write_mode: str = "a",
    repo: icechunk.Repository | None = None,
    group: str | None = None,
):
    """
    Write dataset to icechunk with optional rechunking.

    For virtual datasets: set encoding=None and shards=None
    For materialized datasets: provide encoding and shards for rechunking

    If repo is provided, old snapshots are expired and garbage collected after
    each commit, keeping storage bounded when variables are rewritten.

    group, if given, writes into a zarr sub-group (e.g. ``"ssp245"``).
    """
    import zarr
    from icechunk.xarray import to_icechunk

    if shards is not None:
        ds = ds.chunk(shards)

    is_overwrite = False
    if write_mode == "a" and encoding:
        # zarr rejects encoding specs for arrays that already exist in append mode;
        # existing arrays keep the encoding they were written with.
        target = zarr.open_group(session.store, path=group)
        existing = set(target.array_keys())
        is_overwrite = bool(set(encoding.keys()) & existing)
        encoding = {k: v for k, v in encoding.items() if k not in existing}

    to_icechunk(ds, session, encoding=encoding, mode=write_mode, group=group)

    if commit_message:
        session.commit(commit_message)

    if repo is not None and commit_message and is_overwrite:
        console.print(Text.from_ansi(str(repo.ancestry_graph(branch="main"))))
        history = list(repo.ancestry(branch="main"))
        if len(history) > 2:
            # keep one rollback point: expire everything older than the second-to-last commit
            keep_from = history[1].written_at
            n_expired = len(repo.expire_snapshots(older_than=keep_from))
            gc_result = repo.garbage_collect(keep_from)
            logger.info("GC: expired %d snapshots, collected %s", n_expired, gc_result)
    if repo is not None:
        console.print(Text.from_ansi(str(repo.ancestry_graph(branch="main"))))


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


def _init_repo_from_uri(uri: str):
    """Open or create an icechunk repository at a local path or ``s3://`` URI.

    Parameters
    ----------
    uri : str
        Either a local filesystem path (e.g. ``/tmp/test``) or an S3 URI
        of the form ``s3://bucket/prefix``.

    Returns
    -------
    tuple[icechunk.Repository, icechunk.Session]
        The repository and a writable session on ``main``.
    """
    if uri.startswith("s3://"):
        parts = uri[len("s3://") :].split("/", 1)
        bucket, prefix = parts[0], parts[1] if len(parts) > 1 else ""
        storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region="us-west-2")
    else:
        storage = icechunk.local_filesystem_storage(uri)
    repo = icechunk.Repository.open_or_create(storage)
    return repo, repo.writable_session("main")


def _display_dry_run_result(ds: xr.Dataset, variable: str, store: str | None = None) -> None:
    """Compute a dry-run dataset and render a summary panel to the console.

    Parameters
    ----------
    ds : xr.Dataset
        The (possibly lazy) dataset to compute and summarise.
    variable : str
        Label used in the panel title.
    store : str or None, optional
        Destination URI shown in the panel subtitle. When ``None`` the subtitle
        reads ``(display only, no write)``.
    """
    logger.info("Computing sample result for %s...", variable)
    ds = ds.compute()

    table = Table(show_header=True, header_style="bold green")
    table.add_column("Variable", style="cyan")
    table.add_column("Shape")
    table.add_column("Dims")
    table.add_column("Units", style="yellow")
    table.add_column("Min", justify="right", style="blue")
    table.add_column("Max", justify="right", style="blue")

    for var_name in ds.data_vars:
        da = ds[var_name]
        table.add_row(
            var_name,
            str(da.shape),
            " × ".join(da.dims),
            da.attrs.get("units", "—"),
            f"{float(da.min()):.4g}",
            f"{float(da.max()):.4g}",
        )

    time_start = str(ds.time.values[0])[:10]
    time_end = str(ds.time.values[-1])[:10]
    subtitle = f"[dim]→ {store}[/dim]" if store else "[dim](display only, no write)[/dim]"
    console.print(
        Panel(
            table,
            title=f"[bold green]✓ {variable}[/] | {time_start} → {time_end}",
            subtitle=subtitle,
            border_style="green",
        )
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
