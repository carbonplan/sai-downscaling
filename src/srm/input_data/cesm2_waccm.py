import dataclasses
import logging

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
    CMORIZE_pr,
    _display_dry_run_result,
    _init_repo_from_uri,
    apply_ensemble_provenance,
    build_encoding_dict,
    console,
    determine_write_mode,
    get_aws_creds,
    make_fixed_ensemble_preprocess,
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

# --- Variable mappings ---

CESM_TO_CMIP6: dict[str, str] = {
    "FSDS": "rsds",
    "TREFHT": "tas",
    "TREFHTMX": "tasmax",
    "TREFHTMN": "tasmin",
    "RHREFHT": "hurs",
    "PRECT": "pr",
}
CMIP6_TO_CESM: dict[str, str] = {v: k for k, v in CESM_TO_CMIP6.items()}

CESM_UNIT_MAPPING: dict[str, str] = {
    "pr": "kg m-2 s-1",
    "tas": "K",
    "tasmax": "K",
    "tasmin": "K",
    "hurs": "%",
    "rsds": "W m-2",
}

CMORIZATION_MAP = {"pr": CMORIZE_pr}

# Variables dropped during NetCDF virtualization (CESM-specific aux fields)
DROP_VARS: list[str] = [
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
    "time_bnds",
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
]

SHARED_VARIABLES: list[str] = ["tas", "rsds", "hurs", "pr", "tasmax", "tasmin"]
PANGEO_ENSEMBLE_MEMBERS: list[str] = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]
# r1/r2/r3i1p1f1 have no tasmax/tasmin in any public archive (not in pangeo, CEDA, or NASA NEX)
PANGEO_VARIABLES: list[str] = ["tas", "rsds", "hurs", "pr"]

VAR_SPECS: dict[str, VarSpec] = {
    f.default.name: f.default for f in dataclasses.fields(VarStandards)
}

# --- S3 paths ---

BUCKET = "carbonplan-srm"

# Virtual (reference) icechunk stores — written by `virtualize`, read by `process`
_CESM_HIST_IC = "input/tensor/CESM2/CESM2-WACCM-Historical/icechunk"
_CESM_SSP_IC = "input/tensor/CESM2/CESM2-WACCM-SSP245/icechunk"
_CESM_G6_IC = "input/tensor/CESM2/CESM2-WACCM-G6-1.5K/icechunk"
VIRTUAL_PREFIX: dict[str, str] = {
    "historical": f"{_CESM_HIST_IC}/CESM2-WACCM-historical-virtual.icechunk",
    "G6-1.5K": f"{_CESM_G6_IC}/CESM2-WACCM-G6-1.5K-virtual.icechunk",
    # SSP245 is split: 001-005 lack tasmax/tasmin; 006 ends one day short of 007-010
    "ssp245_5": f"{_CESM_SSP_IC}/CESM2-WACCM-SSP245-001-005-virtual.icechunk",
    "ssp245_6": f"{_CESM_SSP_IC}/CESM2-WACCM-SSP245-006-virtual.icechunk",
    "ssp245_7_10": f"{_CESM_SSP_IC}/CESM2-WACCM-SSP245-007-010-virtual.icechunk",
}

# Raw NetCDF prefixes on S3 — used only by `virtualize`
NETCDF_PREFIX: dict[str, str] = {
    "historical": "input/tensor/CESM2/CESM2-WACCM-Historical/netcdf",
    "ssp245": "input/tensor/CESM2/CESM2-WACCM-SSP245/netcdf",
    "G6-1.5K": "input/tensor/CESM2/CESM2-WACCM-G6-1.5K/netcdf",
}

# Unified per-GCM store — all scenarios as zarr groups, written by `process`
UNIFIED_PREFIX = "input/tensor/cesm2-waccm.icechunk"

SCENARIO_TO_GROUP: dict[str, str] = {
    "historical": "historical",
    "pangeo-historical": "historical",
    "ssp245": "ssp245",
    "G6-1.5K": "g6_1p5k",
}

ENSEMBLE_MEMBERS: dict[str, list[str]] = {
    "historical": ["001"],
    "pangeo-historical": PANGEO_ENSEMBLE_MEMBERS,
    "ssp245": ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"],
    "G6-1.5K": ["001", "002", "003"],
}

# SSP245 split: tasmax/tasmin only exist in members 006-010
SSP245_VARS_5: list[str] = ["rsds", "tas", "hurs", "pr"]
SSP245_VARS_6_10: list[str] = ["rsds", "tas", "tasmax", "tasmin", "hurs", "pr"]
SSP245_VARS_6_10_ONLY: list[str] = ["tasmax", "tasmin"]
SSP245_SUBSET_5: list[str] = ["001", "002", "003", "004", "005"]
SSP245_SUBSET_6: list[str] = ["006"]
SSP245_SUBSET_7_10: list[str] = ["007", "008", "009", "010"]

OUTPUT_CHUNKS: dict[str, int] = {"ensemble_member": 1, "time": 30, "lat": 192, "lon": 288}
OUTPUT_SHARDS: dict[str, int] = {"ensemble_member": 1, "time": 480, "lat": 192, "lon": 288}

VIRTUALIZE_CLUSTER: dict = {
    "n_workers": [6, 24],
    "worker_vm_types": ["r8g.4xlarge"],
    "scheduler_vm_types": "c8g.2xlarge",
}

PROCESS_CLUSTER: dict = {
    "n_workers": [6, 16],
    "worker_vm_types": ["r8g.12xlarge"],
    "scheduler_vm_types": "c8g.xlarge",
}

# 365 daily steps = one calendar year; enough to verify the full transform chain
_DRY_RUN_STEPS = 365

ALL_SCENARIOS = list(SCENARIO_TO_GROUP.keys())


# --- Helpers ---


def _get_netcdf_urls(variables: list[str], scenario: str) -> list[str]:
    aws = get_aws_creds()
    region = aws.pop("region")
    store = from_url(f"s3://{BUCKET}", region=region, **aws)
    stream = obs.list_with_delimiter(store, prefix=NETCDF_PREFIX[scenario], return_arrow=True)
    netcdf_list = list(stream["objects"]["path"].to_numpy())
    cesm_vars = [CMIP6_TO_CESM.get(v, v) for v in variables]
    return [
        f"s3://{BUCKET}/{path}"
        for path in netcdf_list
        if path.endswith(".nc") and any(f".{cv}." in path for cv in cesm_vars)
    ]


def _preprocess_ensemble(ds: xr.Dataset, url: str | None = None) -> xr.Dataset:
    if url is None:
        raise ValueError("url is required to determine ensemble member")
    case = ds.attrs.get("case", "")
    ensemble = case.rsplit(".")[-1] if case else "unknown"
    return ds.expand_dims({"ensemble_member": [ensemble]})


# --- Pipeline stages ---


def load_cesm_virtual(cesm_var: str, virtual_key: str) -> xr.Dataset:
    """Load a single CESM2-WACCM variable from a virtual icechunk store.

    Virtual chunks point back to S3, so the repository must be opened with a
    VirtualChunkContainer that authorizes those reads.
    """
    storage = icechunk.s3_storage(
        bucket=BUCKET, prefix=VIRTUAL_PREFIX[virtual_key], region="us-west-2"
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
    )[[cesm_var]]


def load_pangeo_cesm(variables: list[str]) -> xr.Dataset:
    """Load CESM2-WACCM from the Pangeo GCS Zarr catalog and combine ensemble members."""
    import intake
    from zarr.storage import ObjectStore

    cat = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    subset = cat.search(
        source_id=["CESM2-WACCM"],
        experiment_id="historical",
        variable_id=variables,
        member_id=PANGEO_ENSEMBLE_MEMBERS,
        table_id="day",
    )
    datasets = []
    for zstore_url in subset.df.zstore:
        gcs_store = from_url(zstore_url, skip_signature=True)
        ds = xr.open_dataset(
            ObjectStore(gcs_store), engine="zarr", consolidated=True, chunks="auto"
        ).drop_encoding()
        member_id = ds.attrs.get("variant_label", "unknown")
        datasets.append(ds.expand_dims({"ensemble_member": [member_id]}))
    return xr.combine_by_coords(
        datasets, coords="minimal", compat="override", combine_attrs="drop_conflicts"
    )


def preprocess_cesm(ds: xr.Dataset) -> xr.Dataset:
    """Standardize CESM coord/var names to CMIP6 conventions."""
    return (
        ds.drop_duplicates(dim="time", keep="first")
        .drop_encoding()
        .drop_vars(["ilev", "lev"], errors="ignore")
        .pipe(lon_to_180, lon_name="lon")
        .sortby(["lat", "lon"])
        .rename({k: v for k, v in CESM_TO_CMIP6.items() if k in ds.data_vars})
        .pipe(trim_negative_precipitation)
    )


def cmorize_cesm(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Set canonical units and apply cmorization for CESM variables."""
    for v in list(ds.data_vars):
        if v in CESM_UNIT_MAPPING:
            ds[v].attrs["units"] = CESM_UNIT_MAPPING[v]
    if var in CMORIZATION_MAP:
        ds = CMORIZATION_MAP[var](ds, var)
    return ds


def finalize_metadata(ds: xr.Dataset, scenario: str) -> xr.Dataset:
    """Stamp scenario/model attrs and record ensemble provenance."""
    parent_exp = ds.attrs.get("parent_experiment_id", "unknown_parent")
    ds.attrs.update(
        {
            "scenario": scenario,
            "model": "CESM2-WACCM",
            "experiment_lineage": f"{parent_exp} -> {scenario}",
            "processing_steps": (
                "time_drop_duplicates, lon_to_180, lat_lon_sort, "
                "trim_negative_precip, convert_calendar_to_proleptic_gregorian"
            ),
        }
    )
    return apply_ensemble_provenance(
        ds, derivation_logic="Ensemble member derived from url or variant_label"
    )


def _load_ssp245_var(var: str, cesm_var: str, subset: bool = False) -> xr.Dataset:
    """Load SSP245 var from the correct virtual stores and align ensemble/time dims."""
    # tasmax/tasmin only exist in members 006-010
    virtual_keys = (
        ["ssp245_6", "ssp245_7_10"]
        if var in SSP245_VARS_6_10_ONLY
        else ["ssp245_5", "ssp245_6", "ssp245_7_10"]
    )
    subsets = [load_cesm_virtual(cesm_var, key).pipe(preprocess_cesm) for key in virtual_keys]
    ds = xr.concat(subsets, dim="ensemble_member", join="outer") if len(subsets) > 1 else subsets[0]
    ds = ds.reindex(ensemble_member=ENSEMBLE_MEMBERS["ssp245"])

    # Member 006 ends one day short of 001-005/007-010; fill with NaN via reindex
    ref_time = load_cesm_virtual(cesm_var, virtual_keys[-1])["time"]
    if len(ds.time) < len(ref_time):
        ds = ds.reindex(time=ref_time)

    if subset:
        ds = ds.isel(time=slice(0, 365))
    return ds


def _load_pangeo_historical_var(var: str, dry_run: bool = False) -> xr.Dataset:
    """Load and standardize a single Pangeo historical (r1/r2/r3i1p1f1) variable."""
    logger.info("Loading %s from Pangeo GCS catalog", var)
    raw = (
        load_pangeo_cesm([var])
        .pipe(lon_to_180, lon_name="lon")
        .sortby(["lat", "lon"])
        .pipe(trim_negative_precipitation)
    )
    return (
        (raw.isel(time=slice(0, _DRY_RUN_STEPS)) if dry_run else raw)
        .pipe(to_proleptic_gregorian)
        .pipe(update_variable_attrs, VAR_SPECS)
        .pipe(finalize_metadata, "pangeo-historical")
    )


def _build_pangeo_historical_dataset(dry_run: bool = False) -> xr.Dataset:
    """Load and merge the Pangeo historical variables (tas, rsds, hurs, pr)."""
    pieces = [_load_pangeo_historical_var(var, dry_run=dry_run) for var in PANGEO_VARIABLES]
    return xr.merge(pieces, combine_attrs="override")


def merge_pangeo_into_historical(
    output_prefix: str | None = None,
    dry_run: bool = False,
    dry_run_output: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Merge Pangeo historical (r1/r2/r3i1p1f1) into the unified ``historical`` group.

    The ``historical`` group already holds ESGF member ``001`` (all 6 variables,
    1978-2015). r1/r2/r3i1p1f1 cover tas/rsds/hurs/pr only, over 1850-2015. Both
    datasets are reindexed onto the union of ``ensemble_member`` labels and ``time``
    steps (NaN-filling members/variables with no data for a given period), combined,
    and the group is rewritten in full.
    """
    group = SCENARIO_TO_GROUP["pangeo-historical"]
    pangeo_ds = _build_pangeo_historical_dataset(dry_run=dry_run)

    if dry_run:
        with zarr.config.set({"async.concurrency": 8}):
            _display_dry_run_result(pangeo_ds, "pangeo-historical", store=dry_run_output)
            if dry_run_output is not None:
                logger.info("Writing dry-run sample for pangeo-historical to %s", dry_run_output)
                repo, session = _init_repo_from_uri(dry_run_output)
                write_mode = determine_write_mode(repo)
                encoding = build_encoding_dict(pangeo_ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
                write_dataset_to_icechunk(
                    pangeo_ds,
                    session,
                    encoding=encoding,
                    shards=OUTPUT_SHARDS,
                    commit_message=f"dry-run: {commit_message or 'pangeo-historical'}",
                    write_mode=write_mode,
                    repo=repo,
                    group=group,
                )
                logger.info("✓ Dry-run write done: pangeo-historical → %s", dry_run_output)
                read_session = repo.readonly_session("main")  # re-open after commit
                written = xr.open_dataset(
                    read_session.store, engine="zarr", chunks="auto", group=group
                )
                console.print(written)
        return

    out_prefix = output_prefix or UNIFIED_PREFIX
    repo, session = init_repo(BUCKET, out_prefix, readonly=False)

    try:
        existing = xr.open_dataset(
            repo.readonly_session("main").store,
            engine="zarr",
            group=group,
            chunks="auto",
            consolidated=False,
            zarr_format=3,
        )
    except (FileNotFoundError, KeyError):
        existing = None

    if existing is None or existing.sizes.get("ensemble_member", 0) == 0:
        merged = pangeo_ds
    else:
        members = list(
            dict.fromkeys([*existing.ensemble_member.values, *pangeo_ds.ensemble_member.values])
        )
        time_index = existing.indexes["time"].union(pangeo_ds.indexes["time"])
        existing_r = existing.reindex(ensemble_member=members, time=time_index)
        pangeo_r = pangeo_ds.reindex(ensemble_member=members, time=time_index)
        merged = existing_r.combine_first(pangeo_r)
        merged.attrs = {**existing.attrs, **pangeo_ds.attrs, "scenario": "historical"}
        merged["ensemble_member"].attrs = {
            "long_name": "Ensemble Member Identifier",
            "derivation_method": (
                "'001': Simone Tilmes corrected ESGF historical run (all variables); "
                "r1/r2/r3i1p1f1: Pangeo CMIP6 historical (tas/rsds/hurs/pr only)"
            ),
        }

    logger.info("Writing merged historical group → s3://%s/%s/%s", BUCKET, out_prefix, group)
    encoding = build_encoding_dict(merged, OUTPUT_CHUNKS, OUTPUT_SHARDS)
    write_dataset_to_icechunk(
        merged,
        session,
        encoding=encoding,
        shards=OUTPUT_SHARDS,
        commit_message=commit_message or "merge pangeo-historical (r1/r2/r3i1p1f1) into historical",
        write_mode="w",
        repo=repo,
        group=group,
    )
    logger.info("✓ Done: pangeo-historical → %s", group)


def process_cesm_var(
    var: str,
    scenario: str,
    output_prefix: str | None = None,
    subset: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    dry_run_output: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Process a single CESM2-WACCM variable and write it to the unified per-GCM icechunk store."""
    out_prefix = output_prefix or UNIFIED_PREFIX
    group = SCENARIO_TO_GROUP[scenario]
    cesm_var = CMIP6_TO_CESM.get(var, var)

    if scenario == "ssp245":
        logger.info("Loading %s (%s) from SSP245 virtual stores", var, cesm_var)
        raw = _load_ssp245_var(var, cesm_var, subset).pipe(cmorize_cesm, var)
        ds = (
            (raw.isel(time=slice(0, _DRY_RUN_STEPS)) if dry_run else raw)
            .pipe(to_proleptic_gregorian)
            .pipe(update_variable_attrs, VAR_SPECS)
            .pipe(finalize_metadata, scenario)
        )
    else:
        logger.info("Loading %s (%s) from virtual store (%s)", var, cesm_var, scenario)
        raw = load_cesm_virtual(cesm_var, scenario).pipe(preprocess_cesm).pipe(cmorize_cesm, var)
        ds = (
            (raw.isel(time=slice(0, _DRY_RUN_STEPS)) if dry_run else raw)
            .pipe(to_proleptic_gregorian)
            .pipe(update_variable_attrs, VAR_SPECS)
            .pipe(finalize_metadata, scenario)
        )
        if "ensemble_member" in ds.dims:
            ds = ds.reindex(ensemble_member=ENSEMBLE_MEMBERS[scenario])

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
    encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
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


def process_cesm_pipeline(
    variables: list[str],
    scenario: str,
    output_prefix: str | None = None,
    use_coiled: bool = False,
    subset: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    dry_run_output: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Run the CESM2-WACCM process pipeline for one or more variables."""
    if dry_run:
        dest = dry_run_output or "(display only, no write)"
        logger.info("Dry run: %d daily steps per variable → %s", _DRY_RUN_STEPS, dest)
        if scenario == "pangeo-historical":
            merge_pangeo_into_historical(
                output_prefix=output_prefix,
                dry_run=True,
                dry_run_output=dry_run_output,
                commit_message=commit_message,
            )
            return
        for var in variables:
            process_cesm_var(
                var,
                scenario,
                output_prefix=output_prefix,
                subset=subset,
                dry_run=True,
                dry_run_output=dry_run_output,
                commit_message=commit_message,
            )
        return

    client = setup_cluster(ClusterConfig(**PROCESS_CLUSTER)) if use_coiled else setup_local_client()
    try:
        if scenario == "pangeo-historical":
            merge_pangeo_into_historical(
                output_prefix=output_prefix,
                commit_message=commit_message,
            )
            return
        for var in variables:
            process_cesm_var(
                var,
                scenario,
                output_prefix=output_prefix,
                subset=subset,
                overwrite=overwrite,
                commit_message=commit_message,
            )
    finally:
        client.shutdown()


# --- CLI ---

app = typer.Typer()


@app.command()
def virtualize(
    scenario: str = typer.Option(..., help=f"Choices: {ALL_SCENARIOS}"),
    coiled: bool = typer.Option(False, "--coiled/--local"),
) -> None:
    """Virtualize CESM2-WACCM NetCDF files into an icechunk virtual store."""
    from obspec_utils.readers import BufferedStoreReader
    from obspec_utils.wrappers import CachingReadableStore, SplittingReadableStore

    if scenario == "pangeo-historical":
        raise typer.BadParameter("pangeo-historical is already zarr; no virtualization needed.")

    aws = get_aws_creds()
    region = aws.pop("region")
    base_store = from_url(f"s3://{BUCKET}", region=region, **aws)
    registry = ObjectStoreRegistry(
        {f"s3://{BUCKET}": CachingReadableStore(SplittingReadableStore(base_store))}
    )
    parser = HDFParser(drop_variables=DROP_VARS, reader_factory=BufferedStoreReader)
    loadable_variables = ["lat", "lon", "time"]
    ensemble_members = ENSEMBLE_MEMBERS[scenario]
    preprocess_fn = (
        make_fixed_ensemble_preprocess(ensemble_members[0])
        if len(ensemble_members) == 1
        else _preprocess_ensemble
    )

    client = setup_cluster(ClusterConfig(**VIRTUALIZE_CLUSTER)) if coiled else setup_local_client()
    try:
        if scenario == "ssp245":
            repo_config = icechunk.RepositoryConfig.default()
            repo_config.set_virtual_chunk_container(
                icechunk.VirtualChunkContainer(
                    f"s3://{BUCKET}/", store=icechunk.s3_store(region=region)
                )
            )
            for vkey, members_subset, variables in [
                ("ssp245_5", SSP245_SUBSET_5, SSP245_VARS_5),
                ("ssp245_6", SSP245_SUBSET_6, SSP245_VARS_6_10),
                ("ssp245_7_10", SSP245_SUBSET_7_10, SSP245_VARS_6_10),
            ]:
                urls = [
                    u
                    for u in _get_netcdf_urls(variables, "ssp245")
                    if any(f"CMIP6-SSP2-4.5-WACCM.{m}." in u for m in members_subset)
                ]
                ds = virtualize_and_combine(
                    urls=urls,
                    registry=registry,
                    parser=parser,
                    loadable_variables=loadable_variables,
                    drop_variables=["ilev", "lev"],
                    preprocess_fn=preprocess_fn,
                )
                logger.info("virtualized ssp245 %s: %s", members_subset, ds)
                storage = icechunk.s3_storage(
                    bucket=BUCKET, prefix=VIRTUAL_PREFIX[vkey], region=region
                )
                repo = icechunk.Repository.open_or_create(storage, repo_config)
                session = repo.writable_session("main")
                ds.vz.to_icechunk(session.store)
                session.commit(f"ssp245: virtualized {members_subset} variables {variables}")
                repo.save_config()
        else:
            repo_config = icechunk.RepositoryConfig.default()
            repo_config.set_virtual_chunk_container(
                icechunk.VirtualChunkContainer(
                    f"s3://{BUCKET}/", store=icechunk.s3_store(region="us-west-2")
                )
            )
            urls = _get_netcdf_urls(SHARED_VARIABLES, scenario)
            ds = virtualize_and_combine(
                urls=urls,
                registry=registry,
                parser=parser,
                loadable_variables=loadable_variables,
                drop_variables=["ilev", "lev"],
                preprocess_fn=preprocess_fn,
            )
            storage = icechunk.s3_storage(
                bucket=BUCKET, prefix=VIRTUAL_PREFIX[scenario], region="us-west-2"
            )
            repo = icechunk.Repository.open_or_create(storage, repo_config)
            session = repo.writable_session("main")
            ds.vz.to_icechunk(session.store)
            session.commit(f"{scenario}: virtualized {SHARED_VARIABLES}")
            repo.save_config()
    finally:
        client.shutdown()


@app.command()
def process(
    variable: list[str] = typer.Option([], help="CMIP6 variable names"),
    scenario: str = typer.Option(..., help=f"Choices: {ALL_SCENARIOS}"),
    output: str | None = typer.Option(None, "--output", help="Override destination prefix."),
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
    """Process CESM2-WACCM variables and write them into the unified per-GCM icechunk store."""
    if scenario == "pangeo-historical":
        if variable and not all_variables and set(variable) - set(PANGEO_VARIABLES):
            raise typer.BadParameter(
                f"pangeo-historical only provides {PANGEO_VARIABLES} "
                "(r1/r2/r3i1p1f1 have no tasmax/tasmin in any public archive)."
            )
        variables = PANGEO_VARIABLES
    else:
        variables = SHARED_VARIABLES if (all_variables or not variable) else list(variable)
    process_cesm_pipeline(
        variables=variables,
        scenario=scenario,
        output_prefix=output,
        use_coiled=coiled,
        subset=subset,
        overwrite=overwrite,
        dry_run=dry_run,
        dry_run_output=dry_run_output,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    app()
