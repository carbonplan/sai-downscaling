import json
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError
from typing import Any

import boto3
import click
import dask
import icechunk
import xarray as xr
from obspec_utils.readers import BlockStoreReader
from obspec_utils.registry import ObjectStoreRegistry
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from virtualizarr.parsers import HDFParser

from srm.config import VarSpec

logger = logging.getLogger(__name__)
console = Console()


def setup_logging() -> None:
    """Configure root logger with a Rich handler backed by the shared console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )


def get_aws_creds() -> dict:
    """Return AWS credentials and region from the active boto3 session."""
    sesh = boto3.Session()
    creds = sesh.get_credentials().get_frozen_credentials()
    result = {
        "region": sesh.region_name,
        "aws_access_key_id": creds.access_key,
        "aws_secret_access_key": creds.secret_key,
    }
    if creds.token:
        result["aws_session_token"] = creds.token
    return result


def run_with_cluster_retry[T](
    make_client: Callable[[], Any],
    process_one: Callable[[T], None],
    items: Iterable[T],
    *,
    max_retries: int = 3,
    log: logging.Logger | None = None,
) -> None:
    """Call ``process_one`` for each item, recreating the cluster on connection loss.

    If the dask scheduler connection is lost mid-computation (e.g. a spot
    instance reclaim), ``CancelledError`` propagates from ``process_one``.
    On that error the cluster is recreated via ``make_client`` and the same
    item is retried, up to ``max_retries`` attempts.
    """
    log = log or logger
    client = make_client()
    try:
        for item in items:
            for attempt in range(1, max_retries + 1):
                try:
                    process_one(item)
                    break
                except CancelledError:
                    if attempt == max_retries:
                        raise
                    log.warning(
                        "%s attempt=%d/%d: cluster connection lost, "
                        "recreating cluster and retrying",
                        item,
                        attempt,
                        max_retries,
                    )
                    try:
                        client.shutdown()
                    except OSError:
                        pass
                    client = make_client()
    finally:
        try:
            client.shutdown()
        except OSError:
            pass


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


def determine_write_mode(
    repo: icechunk.Repository, branch: str = "main", group: str | None = None
) -> str:
    """Pick "w" for a fresh target, "a" otherwise.

    With group: "w" only when the group does not exist yet (never "w" on an
    existing group — that would clobber it while sibling groups share the repo).
    """
    if group is not None:
        import zarr

        session = repo.readonly_session(branch)
        try:
            zarr.open_group(session.store, path=group, mode="r")
            return "a"
        except (FileNotFoundError, KeyError):
            return "w"
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
    branch: str = "main",
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
        # BasicConflictSolver is the icechunk 2 way: auto-resolves non-overlapping concurrent writes.
        session.commit(commit_message, rebase_with=icechunk.BasicConflictSolver())

    if repo is not None and commit_message and is_overwrite:
        console.print(Text.from_ansi(str(repo.ancestry_graph(branch=branch))))
        # Expiry and garbage collection are repository-wide, not per-branch, so only prune when
        # writing main. Pruning during a regeneration would be computed from the wrong timeline
        # and could drop snapshots that the other branch, or a rollback of main to its
        # pre-regeneration tip, still depends on.
        if branch == "main":
            history = list(repo.ancestry(branch=branch))
            if len(history) > 2:
                # keep one rollback point: expire everything older than the second-to-last commit
                keep_from = history[1].written_at
                n_expired = len(repo.expire_snapshots(older_than=keep_from))
                gc_result = repo.garbage_collect(keep_from)
                logger.info("GC: expired %d snapshots, collected %s", n_expired, gc_result)
        else:
            logger.info("branch=%s: skipping snapshot expiry, only main is pruned", branch)
    if repo is not None:
        console.print(Text.from_ansi(str(repo.ancestry_graph(branch=branch))))


def open_netcdf_from_s3(store, path: str, drop_variables: list[str] | None = None) -> xr.Dataset:
    """Open a NetCDF file on S3 via a block-cached reader (64 × 4 MB LRU, no full-file load)."""
    reader = BlockStoreReader(store, path, block_size=4_194_304)
    return xr.open_dataset(reader, engine="h5netcdf", chunks="auto", drop_variables=drop_variables)


def variable_in_store(
    repo: icechunk.Repository, variable: str, group: str | None = None, branch: str = "main"
) -> bool:
    try:
        session = repo.readonly_session(branch)
        existing = xr.open_dataset(session.store, engine="zarr", group=group, decode_times=False)
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
    group: str | None = None,
    commit_message: str | None = None,
    branch: str = "main",
) -> None:
    """Write one variable to icechunk with shared overwrite semantics.

    overwrite + existing variable -> in-place r+ update (no encoding change);
    otherwise append/write with fresh chunk/shard encoding.
    group, if given, targets a zarr sub-group within the repo (e.g. ``"ssp245"``).
    commit_message, if given, overrides the default ``"{scenario}: {variable}"`` message.
    branch selects the icechunk branch to write to; a branch cut from the root snapshot is how a
    store gets regenerated when its time axis changed, since ``r+`` needs matching shapes and
    ``determine_write_mode`` will not choose ``"w"`` for a group that already exists.
    """
    session = repo.writable_session(branch)
    if overwrite and var_in_store:
        write_mode = "r+"
    elif overwrite:
        write_mode = "a"
    else:
        write_mode = determine_write_mode(repo, branch=branch, group=group)
    encoding = build_encoding_dict(ds, chunks, shards)
    logger.info(
        "variable=%s group=%s writing to icechunk write_mode=%s", variable, group, write_mode
    )
    write_dataset_to_icechunk(
        ds,
        session,
        encoding=None if overwrite else encoding,
        shards=shards,
        commit_message=commit_message
        or (f"{scenario}: {variable}" + (" (overwrite)" if overwrite else "")),
        write_mode=write_mode,
        repo=repo,
        group=group,
        branch=branch,
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
