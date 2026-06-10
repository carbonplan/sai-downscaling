# COILED vm-type r8g.4xlarge
# COILED region us-west-2

import dataclasses
import logging
import subprocess

import obstore as obs
import typer
import xarray as xr
import zarr
from obspec_utils.readers import EagerStoreReader
from obstore.store import from_url

from srm.config import (
    ClusterConfig,
    VarSpec,
    VarStandards,
    init_repo,
    setup_cluster,
    setup_local_client,
)
from srm.input_data.etl_utils import (
    _display_dry_run_result,
    _init_repo_from_uri,
    apply_ensemble_provenance,
    build_encoding_dict,
    console,
    determine_write_mode,
    get_aws_creds,
    run_with_cluster_retry,
    setup_logging,
    trim_negative_precipitation,
    update_variable_attrs,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180, to_proleptic_gregorian

zarr.config.set({"async.concurrency": 128})
setup_logging()
logger = logging.getLogger(__name__)

# --- Variable mappings ---

VAR_SPECS: dict[str, VarSpec] = {
    f.default.name: f.default for f in dataclasses.fields(VarStandards)
}

# Variables that may come from the private T/PR NetCDF archive instead of s3_input_prefix
T_PR_VARS: list[str] = ["pr", "tas", "tasmin", "tasmax"]

# --- S3 paths ---

BUCKET = "carbonplan-srm"

# Unified per-GCM store — all scenarios as zarr groups, written by `process`
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
# G6-1.5K air_temperature cell_methods: max=tasmax, min=tasmin, mean=tas
T_PR_VAR_RENAME: dict[str, dict[str, str]] = {
    "SSP245": {
        "precipitation_flux": "pr",
        "air_temperature": "tas",
    },
    "G6-1.5K": {
        "precipitation_flux": "pr",
        "air_temperature": "tasmax",
        "air_temperature_0": "tasmin",
        "air_temperature_1": "tas",
    },
}

# UM suite ID -> CMIP6 ripf mapping for T_PR files
T_PR_MEMBER_RENAME: dict[str, dict[str, str]] = {
    "G6-1.5K": {
        "u-dp583": "r2i1p1f2",
        "u-dp690": "r3i1p1f2",
        "u-dp691": "r12i1p1f2",
    },
}

# Members used to filter S3_INPUT_PREFIX listings (non-T_PR variables).
# G6-1.5K hurs/rsds are stored on S3 under positional IDs ("001"/"002"/"003"),
# distinct from the T_PR ripf member IDs.
ENSEMBLE_MEMBERS: dict[str, list[str]] = {
    "historical": ["r2i1p1f2", "r3i1p1f2", "r12i1p1f2"],
    "SSP245": ["r12i1p1f2", "r2i1p1f2", "r3i1p1f2"],
    "G6-1.5K": ["001", "002", "003"],
}

# T_PR files span 2015-2100; clip G6-1.5K to match hurs/rsds time range
TIME_RANGE: dict[str, str] = {
    "SSP245": "2015-2099",
    "G6-1.5K": "2035-2084",
}

EXPECTED_VARS: dict[str, list[str]] = {
    "historical": ["pr", "tas", "tasmin", "tasmax", "hurs", "rsds"],
    "SSP245": ["hurs", "pr", "rsds", "tas", "tasmax", "tasmin"],
    "G6-1.5K": ["hurs", "rsds", "pr", "tas", "tasmin", "tasmax"],
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

PROCESS_CLUSTER: dict = {
    "n_workers": [2, 8],
    "worker_vm_types": ["r8g.4xlarge"],
    "scheduler_vm_types": "c8g.xlarge",
}

# Retries for a single variable if the cluster connection is lost mid-computation
# (e.g. spot reclamation), recreating the cluster between attempts.
MAX_VAR_RETRIES = 3

# 360 daily steps ~= one calendar year; used for --subset and dry runs
_SUBSET_TIME_STEPS = 360


# --- Helpers ---


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


def _open_netcdf_from_s3(store, path: str) -> xr.Dataset:
    reader = EagerStoreReader(store, path)
    return xr.open_dataset(reader, engine="h5netcdf", chunks="auto")


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


# --- Pipeline functions ---


def preprocess_ukesm(ds: xr.Dataset, scenario: str, subset: bool = False) -> xr.Dataset:
    """Standardize UKESM coordinates and trim to scenario time range."""
    if subset:
        ds = ds.isel(time=slice(0, _SUBSET_TIME_STEPS))

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

    ds = ds.pipe(to_proleptic_gregorian).drop_encoding()

    # CMORize: rename latitude/longitude -> lat/lon if needed
    rename_map = {
        k: v
        for k, v in [("latitude", "lat"), ("longitude", "lon")]
        if k in ds.dims and v not in ds.dims
    }
    if rename_map:
        ds = ds.rename(rename_map)

    ds = ds.pipe(lon_to_180, lon_name="lon").sortby(["lat", "lon"])

    if scenario in TIME_RANGE:
        start_year, end_year = TIME_RANGE[scenario].split("-")
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    return ds.pipe(trim_negative_precipitation)


def finalize_ukesm_metadata(
    ds: xr.Dataset, scenario: str, variable: str | None = None
) -> xr.Dataset:
    """Stamp scenario/model global attrs and record ensemble provenance."""
    ds = update_variable_attrs(ds, VAR_SPECS)
    ds.attrs.update(
        {
            "scenario": scenario,
            "model": "UKESM1-0-LL",
            "Conventions": "CF-1.8",
        }
    )
    return apply_ensemble_provenance(ds, _derivation_logic(scenario, variable))


def load_ukesm_var(variable: str, scenario: str, subset: bool = False) -> xr.Dataset:
    """Load and concatenate a single UKESM variable across ensemble members from S3 NetCDF."""
    aws = get_aws_creds()
    region = aws.pop("region")
    store = from_url(f"s3://{BUCKET}", region=region, **aws)

    url_pairs = _get_netcdf_urls(scenario, variable)
    if not url_pairs:
        raise ValueError(f"no files found for variable={variable} scenario={scenario}")
    logger.info("variable=%s found %d files", variable, len(url_pairs))
    for member, path in url_pairs:
        logger.info("  member=%s path=%s", member, path)

    member_paths: dict[str, list[str]] = {}
    for member, path in url_pairs:
        member_paths.setdefault(member, []).append(path)

    member_rename = T_PR_MEMBER_RENAME.get(scenario, {})
    if member_rename:
        member_paths = {member_rename.get(m, m): paths for m, paths in member_paths.items()}

    member_datasets = []
    for member, paths in sorted(member_paths.items()):
        logger.info("variable=%s member=%s opening %d file(s)", variable, member, len(paths))
        time_slices = [_open_netcdf_from_s3(store, p) for p in sorted(paths)]
        member_ds = (
            xr.concat(time_slices, dim="time", data_vars="minimal")
            if len(time_slices) > 1
            else time_slices[0]
        )
        member_ds = preprocess_ukesm(member_ds, scenario, subset=subset)
        member_datasets.append(member_ds.expand_dims({"ensemble_member": [member]}))
        logger.info("variable=%s member=%s shape=%s", variable, member, dict(member_ds.dims))

    ds = xr.concat(member_datasets, dim="ensemble_member")
    var_rename = T_PR_VAR_RENAME.get(scenario, {})
    if var_rename:
        ds = ds.rename({k: v for k, v in var_rename.items() if k in ds and v not in ds})
    if variable in ds:
        ds = ds[[variable]]
    logger.info("variable=%s concat done shape=%s", variable, dict(ds.dims))
    return ds


def process_ukesm_var(
    variable: str,
    scenario: str,
    output_prefix: str | None = None,
    subset: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    dry_run_output: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Process a single UKESM variable and write it to the unified per-GCM icechunk store."""
    out_prefix = output_prefix or UNIFIED_PREFIX
    group = SCENARIO_TO_GROUP[scenario]

    logger.info("Loading %s (%s) from S3 NetCDF", variable, scenario)
    raw = load_ukesm_var(variable, scenario, subset=subset or dry_run)

    ds = raw.pipe(finalize_ukesm_metadata, scenario, variable)

    if dry_run:
        with zarr.config.set({"async.concurrency": 8}):
            _display_dry_run_result(ds, variable, store=dry_run_output)
            if dry_run_output is not None:
                logger.info("Writing dry-run sample for %s to %s", variable, dry_run_output)
                repo, session = _init_repo_from_uri(dry_run_output)
                write_mode = determine_write_mode(repo)
                encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
                write_dataset_to_icechunk(
                    ds,
                    session,
                    encoding=encoding,
                    shards=OUTPUT_SHARDS,
                    commit_message=f"dry-run: {commit_message or variable}",
                    write_mode=write_mode,
                    repo=repo,
                    group=group,
                )
                logger.info("✓ Dry-run write done: %s -> %s", variable, dry_run_output)
                read_session = repo.readonly_session("main")  # re-open after commit
                written = xr.open_dataset(
                    read_session.store, engine="zarr", chunks="auto", group=group
                )
                console.print(written)
        return

    repo, session = init_repo(BUCKET, out_prefix, readonly=False)

    var_in_store = False
    try:
        existing = xr.open_dataset(
            repo.readonly_session("main").store, engine="zarr", group=group, decode_times=False
        )
        var_in_store = variable in existing.data_vars
    except Exception:
        pass

    if not overwrite and var_in_store:
        logger.info("variable=%s skip: already in store (use --overwrite to replace)", variable)
        return

    if overwrite and var_in_store:
        write_mode = "r+"
    elif overwrite:
        write_mode = "a"
    else:
        write_mode = determine_write_mode(repo)

    encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
    logger.info(
        "Writing %s/%s -> s3://%s/%s/%s (mode=%s)",
        scenario,
        variable,
        BUCKET,
        out_prefix,
        group,
        write_mode,
    )
    write_dataset_to_icechunk(
        ds,
        session,
        encoding=None if overwrite else encoding,
        shards=None if overwrite else OUTPUT_SHARDS,
        commit_message=commit_message
        or f"{scenario}: {variable}" + (" (overwrite)" if overwrite else ""),
        write_mode=write_mode,
        repo=repo,
        group=group,
    )
    logger.info("✓ Done: %s/%s -> %s", scenario, variable, group)


def process_ukesm_pipeline(
    variables: list[str],
    scenario: str,
    output_prefix: str | None = None,
    subset: bool = False,
    overwrite: bool = False,
    use_coiled: bool = False,
    dry_run: bool = False,
    dry_run_output: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Run the UKESM process pipeline for one or more variables."""
    if dry_run:
        dest = dry_run_output or "(display only, no write)"
        logger.info("Dry run: %d daily steps per variable -> %s", _SUBSET_TIME_STEPS, dest)
        for var in variables:
            process_ukesm_var(
                var,
                scenario,
                output_prefix=output_prefix,
                subset=subset,
                dry_run=True,
                dry_run_output=dry_run_output,
                commit_message=commit_message,
            )
        return

    def _make_client():
        return (
            setup_cluster(ClusterConfig(**PROCESS_CLUSTER)) if use_coiled else setup_local_client()
        )

    def _process_one(var: str) -> None:
        process_ukesm_var(
            var,
            scenario,
            output_prefix=output_prefix,
            subset=subset,
            overwrite=overwrite,
            commit_message=commit_message,
        )

    run_with_cluster_retry(_make_client, _process_one, variables, max_retries=MAX_VAR_RETRIES)


# --- Fetch helpers ---


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
    logger.info("Running rclone command:\n%s", " \\\n    ".join(command))
    subprocess.run(command)


# --- CLI ---

app = typer.Typer()


@app.command()
def fetch(
    variable: list[str] = typer.Option(..., help="UKESM variables to fetch"),
    scenario: str = typer.Option(..., help=f"Choices: {ALL_SCENARIOS}"),
) -> None:
    """Fetch raw UKESM NetCDF files and copy to S3 (historical only)."""
    if scenario != "historical":
        raise typer.BadParameter(f"fetch not implemented for scenario: {scenario}")
    _fetch_ukesm_historical(list(variable))


@app.command()
def process(
    variable: list[str] = typer.Option([], help="CMIP6 variable names"),
    scenario: str = typer.Option(..., help=f"Choices: {ALL_SCENARIOS}"),
    output: str | None = typer.Option(None, "--output", help="Override destination prefix."),
    coiled: bool = typer.Option(False, "--coiled/--local", help="Use Coiled cluster."),
    all_variables: bool = typer.Option(False, "--all-variables"),
    subset: bool = typer.Option(False, "--subset/--no-subset"),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing variable arrays in-place (r+ mode)."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=f"Run transforms on a {_SUBSET_TIME_STEPS}-step sample and display results.",
    ),
    dry_run_output: str | None = typer.Option(
        None,
        "--dry-run-output",
        help=(
            "Write the dry-run sample to this location instead of discarding it. "
            "Accepts a local path or an S3 URI. Only used with --dry-run."
        ),
    ),
    commit_message: str | None = typer.Option(
        None,
        "--commit-message",
        help="Icechunk commit message. Defaults to '{scenario}: {variable}' when not set.",
    ),
) -> None:
    """Process UKESM variables and write them into the unified per-GCM icechunk store."""
    variables = EXPECTED_VARS[scenario] if (all_variables or not variable) else list(variable)
    process_ukesm_pipeline(
        variables=variables,
        scenario=scenario,
        output_prefix=output,
        subset=subset,
        overwrite=overwrite,
        use_coiled=coiled,
        dry_run=dry_run,
        dry_run_output=dry_run_output,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    app()
