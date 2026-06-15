# COILED vm-type r8g.4xlarge
# COILED region us-west-2
# COILED tag project=SRM

import dataclasses
import logging
import subprocess

import dask
import icechunk
import obstore as obs
import typer
import xarray as xr
import zarr
from obstore.store import from_url

from srm.config import VarSpec, VarStandards, init_repo
from srm.input_data.etl_utils import (
    _display_dry_run_result,
    _init_repo_from_uri,
    apply_ensemble_provenance,
    build_encoding_dict,
    console,
    determine_write_mode,
    get_aws_creds,
    group_paths_by_member,
    open_netcdf_from_s3,
    setup_logging,
    trim_negative_precipitation,
    update_variable_attrs,
    variable_in_store,
    write_dataset_to_icechunk,
    write_variable_to_icechunk,
)
from srm.utils import lon_to_180, to_proleptic_gregorian

zarr.config.set({"async.concurrency": 128})
dask.config.set(scheduler="threads")

setup_logging()
log = logging.getLogger(__name__)


# Variables that may come from the private T/PR NetCDF archive instead of S3_INPUT_PREFIX
T_PR_VARS: list[str] = ["pr", "tas", "tasmin", "tasmax"]

UKESM_VARIABLES = ["tas", "rsds", "hurs", "pr", "tasmax", "tasmin"]

BUCKET = "carbonplan-srm"
UNIFIED_PREFIX = "input/processed/ukesm.icechunk"

SCENARIO_TO_GROUP: dict[str, str] = {
    "historical": "historical",
    "SSP245": "ssp245",
    "G6-1.5K": "g6_1p5k",
}

# Raw NetCDF prefixes on S3 — primary source for each scenario
S3_INPUT_PREFIX: dict[str, str] = {
    "historical": "input/tensor/UKESM/netcdf/historical",
    "SSP245": "input/tensor/UKESM/transfer/SSP2-4.5/",
    "G6-1.5K": "input/tensor/UKESM/transfer/G6-1.5K",
}

# Private T/PR NetCDF prefixes — secondary source for pr/tas/tasmin/tasmax
T_PR_INPUT_PREFIX: dict[str, str] = {
    "SSP245": "input/tensor/UKESM/UKESM_SSP245_T_PR_NETCDF/UKESM_SSP245_T_PR",
    "G6-1.5K": "input/tensor/UKESM/UKESM_G6-1.5K_T_PR_NETCDF/UKESM_G6-1.5K_T_PR",
}

# Filename prefixes to look for in T_PR_INPUT_PREFIX, per variable
T_PR_VAR_LOOKUP: dict[str, dict[str, list[str]]] = {
    "SSP245": {
        "pr": ["pr_day_"],
        # r2 member has tas/tasmin/tasmax combined in one file
        "tas": ["tas_day_", "tas_mean_min_max_day_"],
        "tasmin": ["tasmin_day_", "tas_mean_min_max_day_"],
        "tasmax": ["tasmax_day_", "tas_mean_min_max_day_"],
    },
    "G6-1.5K": {
        # tas_day files contain tas, tasmin, and tasmax
        "pr": ["pr_day_"],
        "tas": ["tas_day_"],
        "tasmin": ["tas_day_"],
        "tasmax": ["tas_day_"],
    },
}

# T_PR files use legacy UM variable names; map to CMIP6 names.
# air_temperature cell_methods: max=tasmax, min=tasmin, mean=tas
T_PR_VAR_RENAME: dict[str, dict[str, str]] = {
    "SSP245": {
        "precipitation_flux": "pr",
        "air_temperature": "tasmax",
        "air_temperature_0": "tasmin",
        "air_temperature_1": "tas",
    },
    "G6-1.5K": {
        "precipitation_flux": "pr",
        "air_temperature": "tasmax",
        "air_temperature_0": "tasmin",
        "air_temperature_1": "tas",
    },
}

# UM suite ID / legacy member ID -> CMIP6 ripf mapping for T_PR files
T_PR_MEMBER_RENAME: dict[str, dict[str, str]] = {
    "SSP245": {"r2i1p1f1": "r2i1p1f2"},
    "G6-1.5K": {
        "u-dp583": "r2i1p1f2",
        "u-dp690": "r3i1p1f2",
        "u-dp691": "r12i1p1f2",
    },
}

ENSEMBLE_MEMBERS: dict[str, list[str]] = {
    "historical": ["r2i1p1f2", "r3i1p1f2", "r12i1p1f2"],
    "SSP245": ["r12i1p1f2", "r2i1p1f2", "r3i1p1f2"],
    "G6-1.5K": ["r12i1p1f2", "r2i1p1f2", "r3i1p1f2"],
}

# T_PR files span 2015-2100; clip G6-1.5K to match hurs/rsds time range
TIME_RANGE: dict[str, str] = {
    "SSP245": "2015-2099",
    "G6-1.5K": "2035-2084",
}

ALL_SCENARIOS = list(SCENARIO_TO_GROUP.keys())

# --- Historical fetch (CEDA -> S3) ---

HISTORICAL_SOURCE_BASE_URL = (
    "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical"
)
HISTORICAL_TIME_SLICES: list[str] = ["18500101-19491230", "19500101-20141230"]
HISTORICAL_ENSEMBLE_DATES: dict[str, str] = {
    "r2i1p1f2": "d20190708",
    "r3i1p1f2": "d20190708",
    "r12i1p1f2": "d20191210",
}

OUTPUT_CHUNKS: dict[str, int] = {"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192}
OUTPUT_SHARDS: dict[str, int] = {"ensemble_member": 1, "time": 960, "lat": 144, "lon": 192}

VAR_SPECS: dict[str, VarSpec] = {
    f.default.name: f.default for f in dataclasses.fields(VarStandards)
}

# 360 daily steps ~= one calendar year; used for --subset and dry runs
_DRY_RUN_STEPS = 360


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _get_netcdf_urls(scenario: str, variable: str) -> list[tuple[str, str]]:
    """Return (member_id, s3_path) pairs for one variable."""
    aws = get_aws_creds()
    region = aws.pop("region")
    store = from_url(f"s3://{BUCKET}", region=region, **aws)

    if variable in T_PR_VARS and scenario in T_PR_INPUT_PREFIX:
        fname_prefixes = T_PR_VAR_LOOKUP[scenario].get(variable, [f"{variable}_day_"])
        stream = obs.list_with_delimiter(
            store, prefix=T_PR_INPUT_PREFIX[scenario], return_arrow=True
        )
        paths = list(stream["objects"]["path"].to_numpy())
        result = []
        for path in paths:
            fname = path.split("/")[-1]
            if not (fname.endswith(".nc") and any(fname.startswith(p) for p in fname_prefixes)):
                continue
            member = (
                fname.split("_gn_")[0].split("_")[-1]
                if "_gn_" in fname
                else fname.split(".nc")[0].split("_")[-2]
            )
            result.append((member, path))
        return result

    members = ENSEMBLE_MEMBERS[scenario]
    stream = obs.list_with_delimiter(store, prefix=S3_INPUT_PREFIX[scenario], return_arrow=True)
    paths = list(stream["objects"]["path"].to_numpy())
    result = []
    for path in paths:
        if not path.endswith(".nc"):
            continue
        if f"{variable}_".lower() not in path.lower():
            continue
        if members and not any(f"_{m}_" in path for m in members):
            continue
        try:
            member = path.split(".nc")[0].split("_gn")[0].split("_")[-1]
        except IndexError:
            member = "unknown"
        result.append((member, path))
    return result


def _preprocess_ukesm(ds: xr.Dataset, scenario: str, subset: bool = False) -> xr.Dataset:
    if subset:
        ds = ds.isel(time=slice(0, _DRY_RUN_STEPS))

    # Keep only standard spatial/temporal coords; drop everything else before calendar
    # conversion to avoid auxiliary object-dtype coords (e.g. forecast_reference_time)
    # getting float NaN mixed in during convert_calendar, which breaks CF encoding at write
    keep_coords = set(ds.dims) | {"lat", "lon", "latitude", "longitude", "time"}
    ds = ds.drop_vars([c for c in ds.coords if c not in keep_coords], errors="ignore")

    # Drop data vars that use a bnds dimension (time_bnds, forecast_period_bnds, etc.)
    # chunk({"time": -1}) in to_proleptic_gregorian fails on multi-dim bnds variables
    bnds_data_vars = [v for v in ds.data_vars if any("bnds" in d for d in ds[v].dims)]
    if bnds_data_vars:
        ds = ds.drop_vars(bnds_data_vars)

    if scenario != "historical":
        ds = ds.drop_duplicates(dim="time", keep="first")

    ds = to_proleptic_gregorian(ds)
    ds = ds.drop_encoding()

    # CMORize: rename latitude/longitude -> lat/lon if needed
    rename_map = {
        k: v
        for k, v in [("latitude", "lat"), ("longitude", "lon")]
        if k in ds.dims and v not in ds.dims
    }
    if rename_map:
        ds = ds.rename(rename_map)

    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])

    if scenario in TIME_RANGE:
        start_year, end_year = TIME_RANGE[scenario].split("-")
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    return trim_negative_precipitation(ds)


def _derivation_logic(scenario: str, variable: str | None = None) -> str:
    if variable in T_PR_VARS and scenario in T_PR_INPUT_PREFIX:
        member_rename = T_PR_MEMBER_RENAME.get(scenario, {})
        if member_rename:
            mapping_str = ", ".join(f"{k}->{v}" for k, v in member_rename.items())
            return (
                f"UM suite IDs extracted from filename and remapped to CMIP6 ripf: {mapping_str}. "
                "Source files are private T/PR NetCDFs."
            )
        return (
            "Extracted from filename position index: filename.split('_')[1] "
            "(e.g. '001'). Positional ID stored under 'ensemble_member' dim. "
            "Source files are private T/PR NetCDFs."
        )
    return "Extracted from CMIP6 filename: path.split('.nc')[0].split('_gn')[0].split('_')[-1]."


def _update_attrs(
    ds: xr.Dataset, var_specs: dict, scenario: str, variable: str | None = None
) -> xr.Dataset:
    ds = update_variable_attrs(ds, var_specs)
    ds.attrs.update(
        {
            "scenario": scenario,
            "model": "UKESM1-0-LL",
            "Conventions": "CF-1.8",
        }
    )
    return apply_ensemble_provenance(ds, _derivation_logic(scenario, variable))


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
) -> None:
    group = SCENARIO_TO_GROUP[scenario]
    log.info("variable=%s group=%s start", variable, group)

    # dry-run only previews _DRY_RUN_STEPS timesteps; subsetting per-member up front
    # avoids reading/concatenating the full time series just to truncate it later.
    subset = subset or dry_run

    var_in_store = variable_in_store(repo, variable, group=group)
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
    for member, path in url_pairs:
        log.info("  member=%s path=%s", member, path)

    member_paths = group_paths_by_member(url_pairs)

    member_rename = T_PR_MEMBER_RENAME.get(scenario, {})
    if member_rename:
        member_paths = {member_rename.get(m, m): paths for m, paths in member_paths.items()}

    var_rename = T_PR_VAR_RENAME.get(scenario, {})

    member_datasets = []
    for member, paths in sorted(member_paths.items()):
        log.info("variable=%s member=%s opening %d file(s)", variable, member, len(paths))
        time_slices = [open_netcdf_from_s3(obstore_inst, p) for p in sorted(paths)]
        member_ds = (
            xr.concat(time_slices, dim="time", data_vars="minimal")
            if len(time_slices) > 1
            else time_slices[0]
        )
        member_ds = _preprocess_ukesm(member_ds, scenario, subset=subset)
        # Rename per-member before concat so members with mixed naming conventions
        # (e.g. UM legacy names vs CMIP6 standard) align on the same variable names.
        if var_rename:
            member_ds = member_ds.rename({k: v for k, v in var_rename.items() if k in member_ds})
        member_ds = member_ds.expand_dims({"ensemble_member": [member]})
        member_datasets.append(member_ds)
        log.info("variable=%s member=%s shape=%s", variable, member, dict(member_ds.dims))

    ds = xr.concat(member_datasets, dim="ensemble_member")
    if variable in ds:
        ds = ds[[variable]]
    log.info("variable=%s concat done shape=%s", variable, dict(ds.dims))

    ds = _update_attrs(ds, var_specs, scenario, variable)

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
    repo, _ = init_repo(BUCKET, store_prefix or UNIFIED_PREFIX, readonly=False)
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
        )
    log.info("scenario=%s all variables complete", scenario)


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


def _fetch_ukesm_historical(variables: list[str]) -> None:
    import warnings

    warnings.warn(
        "This fetches raw netcdf files from CEDA and moves them to s3. "
        "You must have rclone installed and configured."
    )

    urls = []
    for ens in ENSEMBLE_MEMBERS["historical"]:
        date_str = HISTORICAL_ENSEMBLE_DATES[ens]
        for var in variables:
            for t_range in HISTORICAL_TIME_SLICES:
                folder_path = f"{HISTORICAL_SOURCE_BASE_URL}/{ens}/day/{var}/gn/files/{date_str}"
                file_name = f"{var}_day_UKESM1-0-LL_historical_{ens}_gn_{t_range}.nc"
                urls.append(f"{folder_path}/{file_name}")

    urls_file = "UKESM-historical-urls.txt"
    with open(urls_file, "w") as f:
        f.write("\n".join(urls))

    target_remote = f"aws:{BUCKET}/{S3_INPUT_PREFIX['historical']}/"
    command = [
        "rclone",
        "copyurl",
        "--urls",
        urls_file,
        target_remote,
        "--progress",
        "--no-clobber",
        "--transfers",
        "4",
        "--s3-upload-concurrency",
        "8",
        "--s3-chunk-size",
        "64M",
        "--buffer-size",
        "32M",
        "--s3-no-check-bucket",
        "--disable-http2",
        "--retries",
        "3",
        "--low-level-retries",
        "10",
    ]
    log.info("running rclone command:\n%s", " \\\n    ".join(command))
    subprocess.run(command)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.command()
def fetch(
    variable: list[str] = typer.Option(..., "--variable", help="UKESM variables to fetch"),
    scenario: str = typer.Option(..., "--scenario", help=f"Choices: {ALL_SCENARIOS}"),
) -> None:
    """Fetch raw UKESM NetCDF files and copy to S3 (historical only)."""
    if scenario != "historical":
        raise typer.BadParameter(f"fetch not implemented for scenario: {scenario!r}")
    _fetch_ukesm_historical(list(variable))


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
        False, "--all-variables", help="Process all expected variables (UKESM_VARIABLES)"
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
        help=f"Run transforms on a {_DRY_RUN_STEPS}-step sample and display results, "
        "without writing.",
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
) -> None:
    """Open NetCDF files from S3, concat, rechunk, and write to the unified per-GCM
    icechunk store under each scenario's zarr group. One variable at a time.
    """
    if all_variables:
        variables = UKESM_VARIABLES
    elif variable:
        variables = list(variable)
    else:
        raise typer.BadParameter("Must specify either --variable or --all-variables")

    for scen in scenario:
        if scen not in ALL_SCENARIOS:
            raise typer.BadParameter(f"scenario must be one of {ALL_SCENARIOS}, got {scen!r}")
        _run_process(
            scen,
            variables,
            overwrite,
            subset,
            store_prefix=store_prefix,
            dry_run=dry_run,
            dry_run_output=dry_run_output,
            commit_message=commit_message,
        )


if __name__ == "__main__":
    app()
