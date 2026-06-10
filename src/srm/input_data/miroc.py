import dataclasses
import logging
import subprocess

import icechunk
import obstore as obs
import typer
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import from_url
from virtualizarr.parsers import HDFParser

from srm.config import (
    ClusterConfig,
    VarSpec,
    VarStandards,
    init_repo,
    setup_cluster,
    setup_local_client,
)
from srm.input_data.etl_utils import (
    CMORIZE_hurs,
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
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180, to_proleptic_gregorian

zarr.config.set({"async.concurrency": 128})
setup_logging()
logger = logging.getLogger(__name__)

# --- Source URLs ---

CMIP6_SOURCE_BASE_URL = "https://www.jamstec.go.jp/swpub/public/CMIP6/MIROC-ES2H"
GEOMIP_SOURCE_BASE_URL = "https://www.jamstec.go.jp/swpub/public/GeoMIP"

# --- Variable constants ---

MIROC_VARIABLES: list[str] = ["hurs", "pr", "rsds", "tas", "tasmax", "tasmin"]

VAR_SPECS: dict[str, VarSpec] = {
    f.default.name: f.default for f in dataclasses.fields(VarStandards)
}

# --- Per-scenario lookup tables ---

BUCKET = "carbonplan-srm"

# Virtual icechunk stores — written by `virtualize`, read by `process`
VIRTUAL_PREFIX: dict[str, str] = {
    "historical": "input/tensor/MIROC-ES2H/historical/icechunk/MIROC-ES2H-historical-virtual.icechunk",
    "esgf-ssp245": "input/tensor/MIROC-ES2H/ssp245/icechunk/MIROC-ES2H-SSP245-virtual.icechunk",
    "ssp245": "input/tensor/MIROC-ES2H/baseline/icechunk/MIROC-ES2H-baseline-virtual.icechunk",
    "G6-1.5K": "input/tensor/MIROC-ES2H/G6-1.5K/icechunk/MIROC-ES2H-G6-1.5K-virtual.icechunk",
}

# Raw NetCDF prefixes on S3 — used only by `fetch` and `virtualize`
NETCDF_PREFIX: dict[str, str] = {
    "historical": "input/tensor/MIROC-ES2H/historical/netcdf",
    "esgf-ssp245": "input/tensor/MIROC-ES2H/ssp245/netcdf",
    "ssp245": "input/tensor/MIROC-ES2H/baseline/netcdf",
    "G6-1.5K": "input/tensor/MIROC-ES2H/G6-1.5K/netcdf",
}

# Unified per-GCM store — all scenarios as zarr groups, written by `process`
UNIFIED_PREFIX = "input/processed/miroc-es2h.icechunk"

SCENARIO_TO_GROUP: dict[str, str] = {
    "historical": "historical",
    "esgf-ssp245": "ssp245",  # CMIP6 members r1–r3, time 2015–2100
    "ssp245": "ssp245",  # GeoMIP baseline members r01–r10 — writes to same group
    "G6-1.5K": "g6_1p5k",
}

ENSEMBLE_MEMBERS: dict[str, list[str]] = {
    "historical": ["r1i1p4f2", "r2i1p4f2", "r3i1p4f2"],
    "esgf-ssp245": ["r1i1p4f2", "r2i1p4f2", "r3i1p4f2"],
    "ssp245": [f"r{i:02d}" for i in range(1, 11)],
    "G6-1.5K": [f"r{i:02d}" for i in range(1, 11)],
}

# SSP245/G6-1.5K/GeoMIP-baseline raw hurs is stored as fraction (0–1) despite units='%'
# due to a CMOR labeling bug. Historical is correctly in %.
CMORIZE_HURS: dict[str, bool] = {
    "historical": False,
    "esgf-ssp245": True,
    "ssp245": True,
    "G6-1.5K": True,
}

TIME_RANGE: dict[str, str] = {
    "historical": "1850-2014",
    "esgf-ssp245": "2015-2100",
    "ssp245": "2015-2100",  # stitched: 2015-2019 from esgf-ssp245, 2020+ from GeoMIP baseline
}

# CMIP6 DRS experiment ID used in JAMSTEC URL paths (differs from our scenario key)
CMIP6_EXPERIMENT_ID: dict[str, str] = {
    "historical": "historical",
    "esgf-ssp245": "ssp245",
}

CMIP6_ENSEMBLE_VERSIONS: dict[str, dict[str, str]] = {
    "historical": {"r1i1p4f2": "v20220214", "r2i1p4f2": "v20220214", "r3i1p4f2": "v20220214"},
    "esgf-ssp245": {"r1i1p4f2": "v20220214", "r2i1p4f2": "v20220214", "r3i1p4f2": "v20220214"},
}

CMIP6_YEAR_RANGES: dict[str, range] = {
    "historical": range(1850, 2015),
    "esgf-ssp245": range(2015, 2101),
}

# GeoMIP scenario name on the JAMSTEC server (baseline = SSP245 continuation)
GEOMIP_SCENARIO_NAME: dict[str, str] = {
    "ssp245": "baseline",
    "G6-1.5K": "G6-1.5K-SAI",
}

# Scenarios that use the CMIP6 DRS URL / preprocess pattern
_CMIP6_SCENARIOS: frozenset[str] = frozenset({"historical", "esgf-ssp245"})

# GeoMIP SSP245 (baseline) starts 2020; ESGF SSP245 fills the 2015–2019 gap.
# Maps each GeoMIP member to its ESGF counterpart. Source: lineage.py _miroc_g6_lineage.
_SSP245_ESGF_MEMBER: dict[str, str] = {
    "r01": "r1i1p4f2",
    "r02": "r2i1p4f2",
    "r03": "r3i1p4f2",
    "r04": "r1i1p4f2",
    "r05": "r2i1p4f2",
    "r06": "r3i1p4f2",
    "r07": "r1i1p4f2",
    "r08": "r2i1p4f2",
    "r09": "r3i1p4f2",
    "r10": "r1i1p4f2",
}

OUTPUT_CHUNKS: dict[str, int] = {"ensemble_member": 1, "time": 30, "lat": 192, "lon": 288}
OUTPUT_SHARDS: dict[str, int] = {"ensemble_member": 1, "time": 480, "lat": 192, "lon": 288}

VIRTUALIZE_CLUSTER: dict = {
    "n_workers": [4, 24],
    "worker_vm_types": ["r8g.8xlarge"],
    "scheduler_vm_types": "c8g.2xlarge",
}
PROCESS_CLUSTER: dict = {
    "n_workers": [4, 16],
    "worker_vm_types": ["r8g.4xlarge"],
    "scheduler_vm_types": "c8g.xlarge",
}

# Retries for a single variable if the cluster connection is lost mid-computation
# (e.g. spot reclamation), recreating the cluster between attempts.
MAX_VAR_RETRIES = 3

# 365 daily steps = one calendar year; enough to verify the full transform chain
_DRY_RUN_STEPS = 365

ALL_SCENARIOS = list(SCENARIO_TO_GROUP.keys())


# --- Virtual store loader ---


def load_miroc_virtual(var: str, scenario: str) -> xr.Dataset:
    """Load a single MIROC-ES2H variable from a virtual icechunk store.

    Virtual chunks point back to S3, so the repository must be opened with a
    VirtualChunkContainer that authorizes those reads.
    """
    storage = icechunk.s3_storage(
        bucket=BUCKET, prefix=VIRTUAL_PREFIX[scenario], region="us-west-2"
    )
    repo_config = icechunk.RepositoryConfig.default()
    repo_config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            f"s3://{BUCKET}/", store=icechunk.s3_store(region="us-west-2")
        )
    )
    virtual_chunk_credentials = icechunk.containers_credentials(
        {f"s3://{BUCKET}/": icechunk.s3_credentials(from_env=True)}
    )
    repo = icechunk.Repository.open_or_create(
        storage, repo_config, authorize_virtual_chunk_access=virtual_chunk_credentials
    )
    session = repo.readonly_session("main")
    return xr.open_dataset(
        session.store, engine="zarr", chunks="auto", consolidated=False, zarr_format=3
    )[[var]]


def _load_ssp245_with_gap_fill(var: str) -> xr.Dataset:
    """Load GeoMIP SSP245 (baseline, 2020+) stitched with ESGF SSP245 (2015–2019).

    GeoMIP baseline runs start 2020, leaving a 5-year gap. ESGF SSP245 members
    r1/r2/r3i1p4f2 provide the bridge; each GeoMIP member is relabelled to its
    ESGF parent before concatenation via _SSP245_ESGF_MEMBER.
    """
    geomip = load_miroc_virtual(var, "ssp245")
    esgf_bridge = load_miroc_virtual(var, "esgf-ssp245").sel(time=slice("2015", "2019"))

    gap_pieces = [
        esgf_bridge.sel(ensemble_member=[_SSP245_ESGF_MEMBER[m]]).assign_coords(ensemble_member=[m])
        for m in ENSEMBLE_MEMBERS["ssp245"]
    ]
    gap_ds = xr.concat(gap_pieces, dim="ensemble_member")
    combined = xr.concat([gap_ds, geomip], dim="time")
    combined.attrs.update(
        {
            "gap_fill_source": "MIROC-ES2H-esgf-SSP245-virtual",
            "gap_fill_period": "2015-2019",
            "gap_fill_member_map": ", ".join(
                f"{m}→{e}" for m, e in sorted(_SSP245_ESGF_MEMBER.items())
            ),
            "processing_steps": (
                "2015–2019: ESGF SSP245 (r1/r2/r3i1p4f2) relabelled to GeoMIP member IDs; "
                "2020–2100: GeoMIP baseline run"
            ),
        }
    )
    return combined


# --- Pipeline functions ---


def preprocess_miroc(ds: xr.Dataset) -> xr.Dataset:
    """Standardize MIROC-ES2H coordinates to CMIP6 conventions."""
    return (
        ds.drop_duplicates(dim="time", keep="first")
        .pipe(to_proleptic_gregorian)
        .drop_encoding()
        .pipe(lon_to_180, lon_name="lon")
        .sortby(["lat", "lon"])
        .pipe(trim_negative_precipitation)
    )


def cmorize_miroc(ds: xr.Dataset, scenario: str) -> xr.Dataset:
    """Convert hurs from fraction to percent for scenarios affected by the CMOR labeling bug."""
    if CMORIZE_HURS[scenario] and "hurs" in ds.data_vars:
        return CMORIZE_hurs(ds, "hurs")
    return ds


def finalize_miroc_metadata(ds: xr.Dataset, scenario: str) -> xr.Dataset:
    """Stamp scenario/model global attrs and record ensemble provenance."""
    if scenario in _CMIP6_SCENARIOS:
        derivation = (
            "Extracted from CMIP6 DRS filename: url.split('.nc')[0].split('_gn')[0].split('_')[-1]."
        )
    else:
        derivation = "Raw filename suffix: url.split('.nc')[0].split('_')[-1] (e.g. 'r01')."

    attrs: dict = {"scenario": scenario, "model": "MIROC-ES2H", "Conventions": "CF-1.8"}
    if scenario in TIME_RANGE:
        attrs["time_range"] = TIME_RANGE[scenario]
    ds.attrs.update(attrs)
    return apply_ensemble_provenance(ds, derivation)


# --- Pipeline ---


def process_miroc_var(
    var: str,
    scenario: str,
    output_prefix: str | None = None,
    subset: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    dry_run_output: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Process a single MIROC-ES2H variable and write it to the unified per-GCM icechunk store."""
    group = SCENARIO_TO_GROUP[scenario]
    out_prefix = output_prefix or UNIFIED_PREFIX

    logger.info("Loading %s (%s) from virtual store", var, scenario)
    if scenario == "ssp245":
        logger.info("Stitching 2015–2019 gap from esgf-ssp245 into GeoMIP baseline")
        raw = _load_ssp245_with_gap_fill(var)
    else:
        raw = load_miroc_virtual(var, scenario)
    if subset:
        raw = raw.isel(time=slice(0, 365))
    ds = (
        (raw.isel(time=slice(0, _DRY_RUN_STEPS)) if dry_run else raw)
        .pipe(preprocess_miroc)
        .pipe(cmorize_miroc, scenario)
        .pipe(update_variable_attrs, VAR_SPECS)
        .pipe(finalize_miroc_metadata, scenario)
    )

    if dry_run:
        with zarr.config.set({"async.concurrency": 8}):
            _display_dry_run_result(ds, var, store=dry_run_output)
            if dry_run_output is not None:
                logger.info("Writing dry-run sample for %s to %s", var, dry_run_output)
                repo, session = _init_repo_from_uri(dry_run_output)
                write_mode = determine_write_mode(repo)
                encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
                write_dataset_to_icechunk(
                    ds,
                    session,
                    encoding=encoding,
                    shards=OUTPUT_SHARDS,
                    commit_message=f"dry-run: {commit_message or var}",
                    write_mode=write_mode,
                    repo=repo,
                    group=group,
                )
                logger.info("✓ Dry-run write done: %s → %s", var, dry_run_output)
                read_session = repo.readonly_session("main")  # re-open after commit
                written = xr.open_dataset(
                    read_session.store, engine="zarr", chunks="auto", group=group
                )
                console.print(written)
        return

    repo, session = init_repo(BUCKET, out_prefix, readonly=False)
    write_mode = "r+" if overwrite else determine_write_mode(repo)
    logger.info(
        "Writing %s/%s → s3://%s/%s/%s (mode=%s)",
        scenario,
        var,
        BUCKET,
        out_prefix,
        group,
        write_mode,
    )
    encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
    write_dataset_to_icechunk(
        ds,
        session,
        encoding=encoding,
        shards=OUTPUT_SHARDS,
        commit_message=commit_message or f"{group}: {var}",
        write_mode=write_mode,
        repo=repo,
        group=group,
    )
    logger.info("✓ Done: %s/%s → %s", scenario, var, group)


def process_miroc_pipeline(
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
    """Run the MIROC-ES2H process pipeline for one or more variables."""
    if dry_run:
        dest = dry_run_output or "(display only, no write)"
        logger.info("Dry run: %d daily steps per variable → %s", _DRY_RUN_STEPS, dest)
        for var in variables:
            process_miroc_var(
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
        process_miroc_var(
            var,
            scenario,
            output_prefix=output_prefix,
            subset=subset,
            overwrite=overwrite,
            commit_message=commit_message,
        )

    run_with_cluster_retry(_make_client, _process_one, variables, max_retries=MAX_VAR_RETRIES)


# --- Fetch helpers ---


def _build_cmip6_urls(variables: list[str], scenario: str) -> list[str]:
    """Build all per-year NetCDF URLs for a CMIP6 scenario without directory crawling."""
    experiment = CMIP6_EXPERIMENT_ID[scenario]
    ensemble_versions = CMIP6_ENSEMBLE_VERSIONS[scenario]
    urls = []
    for ens in ENSEMBLE_MEMBERS[scenario]:
        version = ensemble_versions[ens]
        for var in variables:
            for year in CMIP6_YEAR_RANGES[scenario]:
                t_range = f"{year}0101-{year}1231"
                folder = f"{CMIP6_SOURCE_BASE_URL}/{experiment}/{ens}/day/{var}/gn/{version}"
                fname = f"{var}_day_MIROC-ES2H_{experiment}_{ens}_gn_{t_range}.nc"
                urls.append(f"{folder}/{fname}")
    return urls


def _build_geomip_urls(variables: list[str], scenario: str) -> list[str]:
    """Build GeoMIP NetCDF URLs from HTML directory listing."""
    import pandas as pd

    geomip_name = GEOMIP_SCENARIO_NAME[scenario]
    base = f"{GEOMIP_SOURCE_BASE_URL}/{geomip_name}/MIROC-ES2H/day/"
    df = pd.read_html(base)[0][["Name"]].iloc[2:].dropna().reset_index(drop=True)
    urls = []
    for name in df["Name"].tolist():
        if not name.endswith(".nc"):
            continue
        if any(name.startswith(f"{var}_") for var in variables):
            if any(f"_{ens}.nc" in name for ens in ENSEMBLE_MEMBERS[scenario]):
                urls.append(f"{base}{name}")
    return urls


def _filter_existing_urls(urls: list[str], scenario: str) -> list[str]:
    """Filter out URLs whose filenames already exist in S3 via obstore list.

    rclone --no-clobber does the same check but has to HEAD each file
    individually against the jamstec server, which is slow on a flaky remote.
    Listing the destination bucket first is waaay faster.
    """
    aws = get_aws_creds()
    region = aws.pop("region")
    store = from_url(
        f"s3://{BUCKET}/{NETCDF_PREFIX[scenario]}/",
        region=region,
        **aws,
    )
    existing_names: set[str] = set()
    for batch in obs.list(store):
        for item in batch:
            existing_names.add(item["path"].split("/")[-1])

    logger.info("  %d files already present in S3", len(existing_names))
    filtered = [u for u in urls if u.split("/")[-1] not in existing_names]
    logger.info(
        "  Skipping %d already-uploaded files, %d to fetch",
        len(urls) - len(filtered),
        len(filtered),
    )
    return filtered


def _rclone_copy_urls(urls: list[str], urls_file: str, scenario: str) -> None:
    """Write urls_file and invoke rclone copyurl to S3."""
    with open(urls_file, "w") as f:
        f.write("\n".join(urls))

    command = [
        "rclone",
        "copyurl",
        "--urls",
        urls_file,
        f"aws:{BUCKET}/{NETCDF_PREFIX[scenario]}/",
        "--progress",
        "--no-clobber",
        "--checkers",
        "8",
        "--transfers",
        "2",
        "--s3-chunk-size",
        "16M",
        "--s3-upload-concurrency",
        "2",
        "--buffer-size",
        "256M",
        "--fast-list",
        "--s3-no-check-bucket",
        "--disable-http2",
        "--retries",
        "20",
        "--low-level-retries",
        "40",
        "--retries-sleep",
        "30s",
        "--timeout",
        "30m",
        "--contimeout",
        "60s",
    ]
    logger.info("Running rclone command:\n%s", " \\\n    ".join(command))
    subprocess.run(command)


# --- Virtualize helpers ---


def _get_cmip6_netcdf_urls(variables: list[str], scenario: str) -> list[str]:
    """List CMIP6 NetCDF files from S3, filtered by variable."""
    aws = get_aws_creds()
    region = aws.pop("region")
    store = from_url(f"s3://{BUCKET}", region=region, **aws)
    stream = obs.list_with_delimiter(store, prefix=NETCDF_PREFIX[scenario], return_arrow=True)
    all_paths = list(stream["objects"]["path"].to_numpy())
    return [
        f"s3://{BUCKET}/{path}"
        for path in sorted(all_paths)
        if path.endswith(".nc")
        and any(path.split("/")[-1].startswith(f"{var}_") for var in variables)
    ]


def _get_geomip_netcdf_urls(variables: list[str], scenario: str) -> list[str]:
    """Construct GeoMIP NetCDF S3 URLs from known filename pattern."""
    geomip_name = GEOMIP_SCENARIO_NAME[scenario]
    return [
        f"s3://{BUCKET}/{NETCDF_PREFIX[scenario]}/{var}_{geomip_name}_{ens}.nc"
        for var in variables
        for ens in ENSEMBLE_MEMBERS[scenario]
    ]


def _preprocess_cmip6_ensemble(ds: xr.Dataset, url: str | None = None) -> xr.Dataset:
    """Extract CMIP6 ensemble member (e.g. r1i1p4f2) from URL and add as dimension."""
    if url is None:
        raise ValueError("url parameter is required to determine ensemble member")
    ensemble = url.split(".nc")[0].split("_gn")[0].split("_")[-1]
    return ds.expand_dims({"ensemble_member": [ensemble]})


def _preprocess_geomip_ensemble(ds: xr.Dataset, url: str | None = None) -> xr.Dataset:
    """Extract GeoMIP ensemble member suffix from URL filename and add as dimension."""
    if url is None:
        raise ValueError("url parameter is required to determine ensemble member")
    ensemble = url.split(".nc")[0].split("_")[-1]  # e.g. "r01"
    return ds.expand_dims({"ensemble_member": [ensemble]})


# --- CLI ---

app = typer.Typer()


@app.command()
def fetch(
    variable: list[str] = typer.Option([], help="Variables to fetch (defaults to all)"),
    scenario: str = typer.Option(..., help=f"Choices: {ALL_SCENARIOS}"),
) -> None:
    """Fetch MIROC-ES2H NetCDF files from JAMSTEC and copy to S3."""
    import warnings

    variables = list(variable) if variable else MIROC_VARIABLES
    if scenario in _CMIP6_SCENARIOS:
        warnings.warn(
            "Fetches raw NetCDF files from JAMSTEC and copies to S3. rclone must be installed."
        )
        urls = _build_cmip6_urls(variables, scenario)
    else:
        warnings.warn(
            "Fetches raw NetCDF files from a slow JAMSTEC server and copies to S3. rclone must be installed."
        )
        urls = _build_geomip_urls(variables, scenario)

    urls = _filter_existing_urls(urls, scenario)
    if not urls:
        logger.info("Nothing to fetch — all files already in S3.")
        return
    prefix = "CMIP6" if scenario in _CMIP6_SCENARIOS else "GeoMIP"
    urls_file = f"MIROC-ES2H-{prefix}-{scenario}-urls.txt"
    logger.info("Queued %d files → %s", len(urls), urls_file)
    _rclone_copy_urls(urls, urls_file, scenario)


@app.command()
def virtualize(
    scenario: str = typer.Option(..., help=f"Choices: {ALL_SCENARIOS}"),
    coiled: bool = typer.Option(False, "--coiled/--local"),
) -> None:
    """Virtualize NetCDF files on S3 into a virtual icechunk store."""
    from obspec_utils.readers import BufferedStoreReader
    from obspec_utils.wrappers import CachingReadableStore, SplittingReadableStore

    aws = get_aws_creds()
    region = aws.pop("region")
    base_store = from_url(f"s3://{BUCKET}", region=region, **aws)
    registry = ObjectStoreRegistry(
        {f"s3://{BUCKET}": CachingReadableStore(SplittingReadableStore(base_store))}
    )
    parser = HDFParser(reader_factory=BufferedStoreReader)

    if scenario in _CMIP6_SCENARIOS:
        netcdf_urls = _get_cmip6_netcdf_urls(MIROC_VARIABLES, scenario)
        preprocess_fn = _preprocess_cmip6_ensemble
    else:
        netcdf_urls = _get_geomip_netcdf_urls(MIROC_VARIABLES, scenario)
        preprocess_fn = _preprocess_geomip_ensemble

    client = setup_cluster(ClusterConfig(**VIRTUALIZE_CLUSTER)) if coiled else setup_local_client()
    try:
        combined_ds = virtualize_and_combine(
            urls=netcdf_urls,
            registry=registry,
            parser=parser,
            loadable_variables=["lat", "lon", "time", "time_bnds", "lat_bnds", "lon_bnds"],
            preprocess_fn=preprocess_fn,
        )
        repo_config = icechunk.RepositoryConfig.default()
        repo_config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                f"s3://{BUCKET}/", store=icechunk.s3_store(region="us-west-2")
            )
        )
        storage = icechunk.s3_storage(
            bucket=BUCKET, prefix=VIRTUAL_PREFIX[scenario], region="us-west-2"
        )
        repo = icechunk.Repository.open_or_create(storage, repo_config)
        session = repo.writable_session("main")
        combined_ds.vz.to_icechunk(session.store)
        session.commit(f"{scenario}: virtualized {MIROC_VARIABLES}")
        repo.save_config()
    finally:
        client.shutdown()


@app.command()
def process(
    variable: list[str] = typer.Option([], help="CMIP6 variable names"),
    scenario: str = typer.Option(..., help=f"Choices: {ALL_SCENARIOS}"),
    output: str | None = typer.Option(
        None, "--output", help="Override destination s3:// URI or local path."
    ),
    coiled: bool = typer.Option(False, "--coiled/--local", help="Use Coiled cluster."),
    all_variables: bool = typer.Option(False, "--all-variables"),
    subset: bool = typer.Option(False, "--subset/--no-subset"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=f"Run transforms on a {_DRY_RUN_STEPS}-step sample and display results.",
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
        help="Icechunk commit message. Defaults to '{group}: {variable}' when not set.",
    ),
) -> None:
    """Process MIROC-ES2H variables and write them into the unified per-GCM icechunk store."""
    variables = MIROC_VARIABLES if (all_variables or not variable) else list(variable)
    process_miroc_pipeline(
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
