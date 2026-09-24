# COILED vm-type r8g.24xlarge
# COILED region us-west-2
# COILED tag project=SRM

import dataclasses
import json
import logging
import re

import dask
import icechunk
import obstore as obs
import typer
import xarray as xr
import zarr
from obstore.store import from_url

from saidownscale.config import SCENARIO_TO_GROUP, VarSpec, VarStandards, init_repo
from saidownscale.input_data.etl_utils import (
    CMORIZE_pr,
    _display_dry_run_result,
    _init_repo_from_uri,
    apply_published_metadata,
    build_encoding_dict,
    console,
    determine_write_mode,
    get_aws_creds,
    group_paths_by_member,
    label_ensemble_coord,
    open_netcdf_from_s3,
    raw_netcdf_prefix,
    setup_logging,
    trim_negative_precipitation,
    update_variable_attrs,
    variable_in_store,
    write_dataset_to_icechunk,
    write_variable_to_icechunk,
)
from saidownscale.utils import decode_time_from_bounds, lon_to_180, to_proleptic_gregorian

zarr.config.set({"async.concurrency": 128})
dask.config.set(scheduler="threads")

setup_logging()
log = logging.getLogger(__name__)


SHARED_VARIABLES = ["tas", "rsds", "hurs", "pr", "tasmax", "tasmin"]
SHARED_ENSEMBLE_MEMBERS = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]

BUCKET = "carbonplan-srm"
UNIFIED_PREFIX = "input/processed/cesm2-waccm.icechunk"


NETCDF_PREFIX: dict[str, str] = {
    "historical": raw_netcdf_prefix("CESM2-WACCM", "historical"),
    "SSP245": raw_netcdf_prefix("CESM2-WACCM", "ssp245"),
    "G6-1.5K": raw_netcdf_prefix("CESM2-WACCM", "g6-1p5k"),
    "G6-1.5K-END": raw_netcdf_prefix("CESM2-WACCM", "g6-1p5k-end"),
}

ENSEMBLE_MEMBERS: dict[str, list[str]] = {
    # NOTE: email confirmation that this ensemble member maps to 001
    "historical": ["001"],
    "SSP245": ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"],
    "G6-1.5K": ["001", "002", "003"],
    "G6-1.5K-END": ["002"],  # single run, case G6-1p5K-termination_002
}

TIME_RANGE: dict[str, str] = {
    "historical": "1850-2014",
    "SSP245": "2015-2099",
    "G6-1.5K": "2035-2084",
    # Decoded from time_bnds: the two source files stamp 2085-01-02..2095-01-01 and
    # 2095-01-02..2101-01-01, which decode to a contiguous 2085-01-01..2100-12-31.
    "G6-1.5K-END": "2085-2100",
}

OUTPUT_CHUNKS: dict[str, int] = {"ensemble_member": 1, "time": 30, "lat": 192, "lon": 288}
OUTPUT_SHARDS: dict[str, int] = {"ensemble_member": 1, "time": 480, "lat": 192, "lon": 288}

ALL_SCENARIOS = ["pangeo-historical", "historical", "SSP245", "G6-1.5K", "G6-1.5K-END"]

# Issue #424 (TREFHTMX == TREFHTMN == TREFHT on each run's first day) was a symptom of the
# zero-width initial-state record described in issue #521: an instantaneous field has no
# within-day spread, so its max, min and mean coincide. decode_time_from_bounds drops that
# record outright, so the per-group backfill table this module used to carry is gone.

# SSP245 members 001-005 have tasmax == tasmin across the whole series (upstream
# CMIP6 bug, issue #156). Values exist in the store but must never be used, so
# mask them to NaN.
INVALID_TASMAX_TASMIN_MEMBERS: dict[str, list[str]] = {
    "SSP245": ["001", "002", "003", "004", "005"],
}

VAR_SPECS: dict[str, VarSpec] = {
    f.default.name: f.default for f in dataclasses.fields(VarStandards)
}

_DRY_RUN_STEPS = 365

# CESM case names embed the member as a 3-digit segment before ".cam.", separated by a
# dot in the CMIP6-style cases and by an underscore in the termination run:
#   b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001.cam.h1....nc
#   G6-1p5K-termination_002.cam.h1....nc
MEMBER_PATTERN = re.compile(r"[._](\d{3})\.cam\.")

CESM_WACCM_VARIABLE_MAPPING: dict[str, str] = {
    "FSDS": "rsds",
    "TREFHT": "tas",
    "TREFHTMX": "tasmax",
    "TREFHTMN": "tasmin",
    "RHREFHT": "hurs",
    "PRECT": "pr",
}

CESM_UNIT_MAPPING: dict[str, str] = {
    "pr": "kg m-2 s-1",
    "tas": "K",
    "tasmax": "K",
    "tasmin": "K",
    "hurs": "%",
    "rsds": "W m-2",
}

CMORIZATION_FUNCTIONS: dict = {"pr": CMORIZE_pr}

CESM_DROP_VARIABLES: list[str] = [
    "gw",
    "hyam",
    "hybm",
    "P0",
    "hyai",
    "hybi",
    "ndbase",
    "nsbase",
    "nbdate",
    "nbsec",
    "mdt",
    "date",
    "datesec",
    # NOTE: time_bnds is deliberately NOT dropped here. It is the CF authority for the time
    # axis (issue #521) and is consumed by decode_time_from_bounds in _preprocess_cesm;
    # to_proleptic_gregorian drops it afterwards.
    "date_written",
    "time_written",
    "ndcur",
    "nscur",
    "co2vmr",
    "ch4vmr",
    "n2ovmr",
    "f11vmr",
    "f12vmr",
    "sol_tsi",
    "nsteph",
    "ilev",
    "lev",
]


def _capture_provenance(ds: xr.Dataset, url: str) -> dict:
    """Extracts identification markers and records exactly which keys were found."""
    found_in = []
    info = {"source_url": url}

    # Check attributes for ensemble/variant info
    for key in ["variant_label", "case", "parent_variant_label", "parent_id", "tracking_id"]:
        val = ds.attrs.get(key)
        if val:
            info[key] = val
            found_in.append(f"attr:{key}")

    info["provenance_sources"] = list(set(found_in))
    return info


def _attach_source_manifest(ds: xr.Dataset, url: str) -> xr.Dataset:
    prov = _capture_provenance(ds, url)
    ds.attrs["_source_manifest"] = json.dumps(prov)
    return ds


#: Key into the licenses table, which is not the ``model`` attr: that still says CESM2-WACCM.
GCM_KEY = "CESM2-WACCM6"


def _finalize_metadata(ds: xr.Dataset, scenario: str) -> xr.Dataset:
    """Records the parsing and the lineage chain"""

    # No experiment lineage is recorded. It was derived from ``parent_experiment_id``, which only
    # CMORized CMIP6 output carries, so on the raw CAM deliveries it read "unknown_parent -> " plus
    # the scenario already recorded beside it. Where the parent is known it survives verbatim in
    # ``parent_experiment_id``, which is the attribute a reader should use.
    etl_attrs = {
        "scenario": scenario,
        "model": "CESM2-WACCM",
        "processing_steps": (
            "time_drop_duplicates, lon_to_180, lat_lon_sort, trim_negative_precip, convert_calendar_to_proleptic_gregorian"
        ),
    }

    if any(v in CMORIZATION_FUNCTIONS for v in ds.data_vars):
        etl_attrs["processing_steps"] += ", cmorization_unit_conversion"

    ds.attrs.update(etl_attrs)

    if "_source_manifest" in ds.attrs:
        del ds.attrs["_source_manifest"]

    return label_ensemble_coord(apply_published_metadata(ds, GCM_KEY, scenario))


def get_CESM_WACCM_ds(scenario: str) -> xr.Dataset:
    """Fetches Pangeo Zarr stores and builds a comprehensive provenance manifest."""
    import intake
    from zarr.storage import ObjectStore

    experiment_id = scenario.removeprefix("pangeo-")
    cat = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    subset = cat.search(
        source_id=["CESM2-WACCM"],
        experiment_id=experiment_id,
        variable_id=SHARED_VARIABLES,
        member_id=SHARED_ENSEMBLE_MEMBERS,
        table_id="day",
    )

    datasets = []
    #  store the full audit trail for every member
    full_manifest = {}

    for zstore_url in subset.df.zstore:
        gcs_store = from_url(zstore_url, skip_signature=True)
        zarr_store = ObjectStore(gcs_store)

        ds = xr.open_dataset(
            zarr_store, engine="zarr", consolidated=True, chunks="auto"
        ).drop_encoding()

        # The CMORized CMIP6 copies carry CAM's convention unchanged: end-of-interval stamps
        # and a zero-width initial-state record (issue #521). Decode per member, before the
        # expand_dims/combine below gives time_bnds an ensemble_member dimension and makes
        # the bounds axis ambiguous.
        ds = decode_time_from_bounds(ds)

        member_id = ds.attrs.get("variant_label", "unknown")
        # Store all raw attributes and the URL for this specific member
        full_manifest[member_id] = _capture_provenance(ds, zstore_url)

        ds = _attach_source_manifest(ds, zstore_url)
        ds = ds.expand_dims({"ensemble_member": [member_id]})
        datasets.append(ds)

    # Combine all members into a single dataset
    combined = xr.combine_by_coords(
        datasets,
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )

    # Store the full audit trail as  JSON
    combined.ensemble_member.attrs["member_specific_provenance"] = json.dumps(full_manifest)

    return combined


def _get_cesm_var_from_cmip6(cmip6_var: str) -> str:
    reverse_mapping = {v: k for k, v in CESM_WACCM_VARIABLE_MAPPING.items()}
    return reverse_mapping.get(cmip6_var, cmip6_var)


def _member_from_path(path: str, ensemble_members: list[str]) -> str:
    match = MEMBER_PATTERN.search(path.split("/")[-1])
    if match:
        return match.group(1)
    # NCAR historical files carry no member segment; a single configured member is implied
    if len(ensemble_members) == 1:
        return ensemble_members[0]
    return "unknown"


def _get_netcdf_urls(scenario: str, variable: str) -> list[tuple[str, str]]:
    """Return (member_id, s3_path) pairs for one variable."""
    aws = get_aws_creds()
    region = aws.pop("region")
    store = from_url(f"s3://{BUCKET}", region=region, **aws)
    stream = obs.list_with_delimiter(store, prefix=NETCDF_PREFIX[scenario], return_arrow=True)
    paths = list(stream["objects"]["path"].to_numpy())
    cesm_var = _get_cesm_var_from_cmip6(variable)
    ensemble_members = ENSEMBLE_MEMBERS[scenario]

    result = []
    for path in sorted(paths):
        if not (path.endswith(".nc") and f".{cesm_var}." in path):
            continue
        member = _member_from_path(path, ensemble_members)
        if ensemble_members and member not in ensemble_members:
            continue
        result.append((member, path))
    return result


def _standardize_vars(ds: xr.Dataset) -> xr.Dataset:
    for var, cmip6_var in CESM_WACCM_VARIABLE_MAPPING.items():
        if var in ds.data_vars:
            ds = ds.rename({var: cmip6_var})
    return ds


def _is_pangeo_scenario(scenario: str) -> bool:
    return scenario.startswith("pangeo-")


def _trim_time_range(ds: xr.Dataset, scenario: str) -> xr.Dataset:
    """Clamp to the scenario's canonical extent; CESM source files overrun it."""
    start_year, end_year = TIME_RANGE[scenario.removeprefix("pangeo-")].split("-")
    return ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))


def _preprocess_cesm(ds: xr.Dataset, scenario: str, var: str, subset: bool = False) -> xr.Dataset:
    # First, before anything else touches the time axis. CAM stamps interval statistics at the
    # END of their averaging window and prefixes each history stream with a zero-width
    # initial-state record (issue #521), so the raw `time` values label every daily mean one
    # day late. This also has to precede the coord prune below, which would discard time_bnds,
    # and to_proleptic_gregorian, which drops bounds variables outright.
    ds = decode_time_from_bounds(ds)

    keep_coords = set(ds.dims) | {"lat", "lon", "time"}
    ds = ds.drop_vars([c for c in ds.coords if c not in keep_coords], errors="ignore")
    ds = ds.drop_duplicates(dim="time", keep="first")
    ds = to_proleptic_gregorian(ds)
    ds = ds.drop_encoding()
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])
    ds = _standardize_vars(ds)
    ds = trim_negative_precipitation(ds)

    if var in CMORIZATION_FUNCTIONS:
        ds = CMORIZATION_FUNCTIONS[var](ds, var)

    ds = _trim_time_range(ds, scenario)

    if subset:
        ds = ds.isel(time=slice(0, 365))
    return ds


def _mask_invalid_tasmax_tasmin(ds: xr.Dataset, scenario: str, variable: str) -> xr.Dataset:
    """Issue #156: NaN out tasmax/tasmin for members with a corrupt full series."""
    if variable not in ("tasmax", "tasmin"):
        return ds
    members = INVALID_TASMAX_TASMIN_MEMBERS.get(scenario)
    if not members or "ensemble_member" not in ds.dims:
        return ds
    keep = ~ds.ensemble_member.isin(members)
    ds[variable] = ds[variable].where(keep)
    log.info("variable=%s masked invalid members %s to NaN (issue #156)", variable, members)
    return ds


def _update_attrs(ds: xr.Dataset, var_specs: dict, scenario: str) -> xr.Dataset:
    for var_name in ds.data_vars:
        if var_name in CESM_UNIT_MAPPING:
            ds[var_name].attrs["units"] = CESM_UNIT_MAPPING[var_name]

    ds = update_variable_attrs(ds, var_specs)
    ds = _finalize_metadata(ds, scenario)
    return ds


def _run_dry_run(
    ds: xr.Dataset,
    label: str,
    group: str | None,
    dry_run_output: str | None,
    commit_message: str | None,
) -> None:
    """Display a truncated preview of ``ds`` and, optionally, write it to ``dry_run_output``."""
    ds = ds.isel(time=slice(0, _DRY_RUN_STEPS))
    with zarr.config.set({"async.concurrency": 8}):
        _display_dry_run_result(ds, label, store=dry_run_output)
        if dry_run_output is None:
            return

        repo, session = _init_repo_from_uri(dry_run_output)
        write_mode = determine_write_mode(repo, group=group)
        encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
        write_dataset_to_icechunk(
            ds,
            session,
            encoding=encoding,
            shards=OUTPUT_SHARDS,
            commit_message=f"dry-run: {commit_message or label}",
            write_mode=write_mode,
            repo=repo,
            group=group,
        )
        log.info("dry-run write done: %s -> %s", label, dry_run_output)
        read_session = repo.readonly_session("main")
        written = xr.open_dataset(read_session.store, engine="zarr", chunks="auto", group=group)
        console.print(written)


def _process_single_variable(
    scenario: str,
    variable: str,
    repo: icechunk.Repository,
    var_specs: dict,
    overwrite: bool,
    subset: bool,
    dry_run: bool = False,
    dry_run_output: str | None = None,
    commit_message: str | None = None,
    branch: str = "main",
) -> None:
    group = SCENARIO_TO_GROUP[scenario]
    ensemble_members = ENSEMBLE_MEMBERS[scenario]
    log.info("variable=%s group=%s start", variable, group)

    # dry-run only previews _DRY_RUN_STEPS timesteps; subsetting per-member up front
    # avoids reading/concatenating the full time series just to truncate it later.
    subset = subset or dry_run

    var_in_store = variable_in_store(repo, variable, group=group, branch=branch)
    if not dry_run and not overwrite and var_in_store:
        log.info("variable=%s skip: already in store (use --overwrite to replace)", variable)
        return

    aws = get_aws_creds()
    obstore_inst = from_url(f"s3://{BUCKET}", region=aws.pop("region"), **aws)
    url_pairs = _get_netcdf_urls(scenario, variable)
    if not url_pairs:
        log.warning("variable=%s no files found, skipping", variable)
        return
    log.info("variable=%s found %d files", variable, len(url_pairs))

    member_paths = group_paths_by_member(url_pairs)

    member_datasets = []
    member_manifest = {}
    for member, paths in sorted(member_paths.items()):
        log.info("variable=%s member=%s opening %d file(s)", variable, member, len(paths))
        time_slices = [
            open_netcdf_from_s3(obstore_inst, p, CESM_DROP_VARIABLES) for p in sorted(paths)
        ]
        member_manifest[member] = _capture_provenance(time_slices[0], paths[0])
        member_ds = (
            xr.concat(time_slices, dim="time", data_vars="minimal")
            if len(time_slices) > 1
            else time_slices[0]
        )
        member_ds = _preprocess_cesm(member_ds, scenario, variable, subset=subset)
        member_ds = member_ds.expand_dims({"ensemble_member": [member]})
        member_datasets.append(member_ds)
        log.info("variable=%s member=%s shape=%s", variable, member, dict(member_ds.dims))

    # join="outer" unions time axes: SSP245 member 006 ends one day short
    # of 007-010 and gets NaN for the missing final day
    ds = (
        xr.concat(member_datasets, dim="ensemble_member", join="outer")
        if len(member_datasets) > 1
        else member_datasets[0]
    )
    ds = ds[[variable]]
    # reindex to full member list; fills any missing members with NaN
    if ensemble_members and "ensemble_member" in ds.dims:
        ds = ds.reindex(ensemble_member=ensemble_members)
    ds = _mask_invalid_tasmax_tasmin(ds, scenario, variable)
    ds.ensemble_member.attrs["member_specific_provenance"] = json.dumps(member_manifest)
    log.info("variable=%s concat done shape=%s", variable, dict(ds.dims))

    ds = _update_attrs(ds, var_specs, scenario)

    if dry_run:
        _run_dry_run(ds, variable, group, dry_run_output, commit_message)
        return

    write_variable_to_icechunk(
        ds,
        repo,
        variable=variable,
        scenario=scenario,
        chunks=OUTPUT_CHUNKS,
        shards=OUTPUT_SHARDS,
        overwrite=overwrite,
        var_in_store=var_in_store,
        group=group,
        commit_message=commit_message,
        branch=branch,
    )
    log.info("variable=%s done", variable)


def _run_process(
    scenario: str,
    variables: list[str],
    overwrite: bool,
    subset: bool,
    store_prefix: str | None = None,
    dry_run: bool = False,
    dry_run_output: str | None = None,
    commit_message: str | None = None,
    branch: str = "main",
) -> None:
    log.info(
        "scenario=%s group=%s variables=%s overwrite=%s subset=%s dry_run=%s",
        scenario,
        SCENARIO_TO_GROUP[scenario],
        variables,
        overwrite,
        subset,
        dry_run,
    )
    repo, _ = init_repo(BUCKET, store_prefix or UNIFIED_PREFIX, readonly=False, branch=branch)
    for var in variables:
        _process_single_variable(
            scenario,
            var,
            repo,
            VAR_SPECS,
            overwrite,
            subset,
            dry_run=dry_run,
            dry_run_output=dry_run_output,
            commit_message=commit_message,
            branch=branch,
        )
    log.info("scenario=%s all variables complete", scenario)


def _run_pangeo_process(
    scenario: str,
    variables: list[str],
    store_prefix: str | None = None,
    dry_run: bool = False,
    dry_run_output: str | None = None,
    commit_message: str | None = None,
    branch: str = "main",
) -> None:
    """Merge Pangeo historical members (r1/r2/r3i1p1f1) into the ``historical`` group.

    The group must already hold NCAR/ESGF member "001" (written by
    ``--scenario historical``). Both datasets are reindexed onto the union of
    ensemble members and time steps (NaN-filling variables a member lacks,
    e.g. tasmax/tasmin for the Pangeo members) and the group is rewritten.
    """
    group = SCENARIO_TO_GROUP[scenario]

    ds = get_CESM_WACCM_ds(scenario)
    available = [v for v in variables if v in ds]
    ds = ds[available]
    ds = to_proleptic_gregorian(ds)
    ds = trim_negative_precipitation(ds)
    ds = _trim_time_range(ds, scenario)
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])
    ds = _update_attrs(ds, VAR_SPECS, scenario)

    if dry_run:
        _run_dry_run(ds, "pangeo-historical", group, dry_run_output, commit_message)
        return

    repo, session = init_repo(BUCKET, store_prefix or UNIFIED_PREFIX, readonly=False, branch=branch)
    try:
        existing = xr.open_dataset(
            repo.readonly_session(branch).store, engine="zarr", group=group, chunks="auto"
        )
    except (FileNotFoundError, KeyError) as err:
        raise RuntimeError(
            f"group '{group}' not found in unified store; run --scenario historical "
            "(member 001) before --scenario pangeo-historical"
        ) from err

    members = list(dict.fromkeys([*existing.ensemble_member.values, *ds.ensemble_member.values]))
    time_index = existing.indexes["time"].union(ds.indexes["time"])
    merged = existing.reindex(ensemble_member=members, time=time_index).combine_first(
        ds.reindex(ensemble_member=members, time=time_index)
    )
    merged.attrs = {**existing.attrs, **ds.attrs, "scenario": "historical"}
    merged["ensemble_member"].attrs = {
        "long_name": "Ensemble Member Identifier",
        "derivation_method": (
            "'001': corrected NCAR/ESGF historical run (all variables); "
            "r1/r2/r3i1p1f1: Pangeo CMIP6 historical (NaN where a variable is missing)"
        ),
    }

    encoding = build_encoding_dict(merged, OUTPUT_CHUNKS, OUTPUT_SHARDS)
    write_dataset_to_icechunk(
        merged,
        session,
        encoding=encoding,
        shards=OUTPUT_SHARDS,
        commit_message=commit_message
        or f"historical: merge pangeo members {SHARED_ENSEMBLE_MEMBERS}",
        write_mode="w",
        repo=repo,
        group=group,
    )
    log.info("scenario=pangeo-historical group=%s done", group)


app = typer.Typer()


@app.command()
def process(
    variable: list[str] = typer.Option([], "--variable", help="Specific variable(s) to process"),
    scenario: list[str] = typer.Option(
        ...,
        "--scenario",
        help=(
            "Scenario(s) to process; repeat to run several sequentially in order given. "
            f"Choices: {ALL_SCENARIOS}"
        ),
    ),
    all_variables: bool = typer.Option(
        False, "--all-variables", help="Process all expected variables (SHARED_VARIABLES)"
    ),
    subset: bool = typer.Option(False, "--subset/--no-subset"),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing variable arrays in-place (r+ mode)"
    ),
    store_prefix: str | None = typer.Option(
        None,
        "--store-prefix",
        help="Override the unified store prefix (e.g. a dev path for test runs)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=f"Run transforms on a {_DRY_RUN_STEPS}-step sample and display results, without writing.",
    ),
    dry_run_output: str | None = typer.Option(
        None,
        "--dry-run-output",
        help=(
            "Write the dry-run sample to this location instead of discarding it. "
            "Accepts a local path or an s3:// URI. Only used with --dry-run."
        ),
    ),
    commit_message: str | None = typer.Option(
        None, "--commit-message", help="Override the default icechunk commit message."
    ),
    branch: str = typer.Option(
        "main",
        "--branch",
        help=(
            "icechunk branch to write to. Use a branch cut from the repository's root snapshot "
            "to regenerate a store whose time axis changed: the groups must be absent for a "
            "fresh write, and promotion is then repo.reset_branch('main', tip) rather than a "
            "copy of the whole store."
        ),
    ),
) -> None:
    """Open NetCDF files from S3, concat, rechunk, and write to the unified per-GCM
    icechunk store under each scenario's zarr group. One variable at a time.

    Ordering: historical must complete before pangeo-historical (which merges
    Pangeo members into the historical group); list them in that order.
    """
    if all_variables:
        variables = SHARED_VARIABLES
    elif variable:
        variables = list(variable)
    else:
        raise typer.BadParameter("Must specify either --variable or --all-variables")

    for scen in scenario:
        if scen not in ALL_SCENARIOS:
            raise typer.BadParameter(f"scenario must be one of {ALL_SCENARIOS}, got {scen!r}")
        if _is_pangeo_scenario(scen):
            _run_pangeo_process(
                scen,
                variables,
                store_prefix=store_prefix,
                dry_run=dry_run,
                dry_run_output=dry_run_output,
                commit_message=commit_message,
                branch=branch,
            )
        else:
            _run_process(
                scen,
                variables,
                overwrite,
                subset,
                store_prefix=store_prefix,
                dry_run=dry_run,
                dry_run_output=dry_run_output,
                commit_message=commit_message,
                branch=branch,
            )


@app.command(name="mask-invalid-tasmax-tasmin")
def mask_invalid_tasmax_tasmin(
    store_prefix: str | None = typer.Option(None, "--store-prefix"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """One-off (issue #156): NaN SSP245 members 001-005 tasmax/tasmin in the
    existing unified store.

    """
    scenario = "SSP245"
    group = SCENARIO_TO_GROUP[scenario]  # "ssp245"
    members = INVALID_TASMAX_TASMIN_MEMBERS[scenario]
    repo, _ = init_repo(BUCKET, store_prefix or UNIFIED_PREFIX, readonly=False)
    existing = xr.open_dataset(
        repo.readonly_session("main").store, engine="zarr", group=group, chunks="auto"
    )

    keep = ~existing.ensemble_member.isin(members)
    kept_members = [m for m in existing.ensemble_member.values.tolist() if m not in members]
    for var in ("tasmax", "tasmin"):
        if var not in existing.data_vars:
            log.warning("variable=%s not in group=%s, skipping", var, group)
            continue
        ds = existing[[var]]
        ds[var] = ds[var].where(keep)

        if dry_run:
            # Run the actual mask with no write/commit.
            sample = ds[var].isel(time=slice(0, 30))
            masked_nan = bool(sample.sel(ensemble_member=members).isnull().all().compute())
            kept_finite = bool(sample.sel(ensemble_member=kept_members).notnull().all().compute())
            log.info(
                "dry-run: %s -> members %s all-NaN=%s, kept members finite=%s (no write)",
                var,
                members,
                masked_nan,
                kept_finite,
            )
            continue

        write_variable_to_icechunk(
            ds,
            repo,
            variable=var,
            scenario=scenario,
            chunks=OUTPUT_CHUNKS,
            shards=OUTPUT_SHARDS,
            overwrite=True,
            var_in_store=True,
            group=group,
            commit_message=f"issue #156: NaN {var} for ssp245 members {members}",
        )
        log.info("variable=%s masked invalid members %s to NaN in store", var, members)


if __name__ == "__main__":
    app()
