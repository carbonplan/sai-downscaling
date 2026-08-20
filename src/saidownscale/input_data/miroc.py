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

from saidownscale.config import SCENARIO_TO_GROUP, VarSpec, VarStandards, init_repo
from saidownscale.input_data.etl_utils import (
    CMORIZE_hurs,
    _display_dry_run_result,
    _init_repo_from_uri,
    apply_ensemble_provenance,
    build_encoding_dict,
    console,
    determine_write_mode,
    get_aws_creds,
    group_paths_by_member,
    open_netcdf_from_s3,
    raw_netcdf_prefix,
    setup_logging,
    trim_negative_precipitation,
    update_variable_attrs,
    variable_in_store,
    write_dataset_to_icechunk,
    write_variable_to_icechunk,
)
from saidownscale.utils import lon_to_180, to_proleptic_gregorian

zarr.config.set({"async.concurrency": 128})
dask.config.set(scheduler="threads")

setup_logging()
log = logging.getLogger(__name__)


CMIP6_SOURCE_BASE_URL = "https://www.jamstec.go.jp/swpub/public/CMIP6/MIROC-ES2H"
GEOMIP_SOURCE_BASE_URL = "https://www.jamstec.go.jp/swpub/public/GeoMIP"

MIROC_VARIABLES = ["hurs", "pr", "rsds", "tas", "tasmax", "tasmin"]

BUCKET = "carbonplan-srm"
UNIFIED_PREFIX = "input/processed/miroc-es2h.icechunk"


# "ssp245" reads the GeoMIP baseline drop; the ETL scenario key and the source
# directory name differ here by design.
NETCDF_PREFIX: dict[str, str] = {
    "historical": raw_netcdf_prefix("MIROC-ES2H", "historical"),
    "esgf-ssp245": raw_netcdf_prefix("MIROC-ES2H", "esgf-ssp245"),
    "ssp245": raw_netcdf_prefix("MIROC-ES2H", "baseline"),
    "G6-1.5K": raw_netcdf_prefix("MIROC-ES2H", "g6-1p5k"),
}

ENSEMBLE_MEMBERS: dict[str, list[str]] = {
    "historical": ["r1i1p4f2", "r2i1p4f2", "r3i1p4f2"],
    "esgf-ssp245": ["r1i1p4f2", "r2i1p4f2", "r3i1p4f2"],
    "ssp245": [f"r{i:02d}" for i in range(1, 11)],
    "G6-1.5K": [f"r{i:02d}" for i in range(1, 11)],
}

# SSP245/G6-1.5K/GeoMIP-baseline raw hurs is stored as fraction (0-1) despite units='%'
# due to a CMOR labeling bug. Historical and esgf-ssp245 are correctly in %.
CMORIZE_HURS: dict[str, bool] = {
    "historical": False,
    "esgf-ssp245": False,
    "ssp245": True,
    "G6-1.5K": True,
}

TIME_RANGE: dict[str, str] = {
    "historical": "1850-2014",
    "esgf-ssp245": "2015-2099",
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

# GeoMIP SSP245 (baseline) starts 2020; ESGF SSP245 fills the 2015-2019 gap.
# Maps each GeoMIP member to its ESGF counterpart.
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

OUTPUT_CHUNKS: dict[str, int] = {"ensemble_member": 1, "time": 60, "lat": 128, "lon": 256}
OUTPUT_SHARDS: dict[str, int] = {"ensemble_member": 1, "time": 960, "lat": 128, "lon": 256}

ALL_SCENARIOS = ["historical", "esgf-ssp245", "ssp245", "G6-1.5K"]

VAR_SPECS: dict[str, VarSpec] = {
    f.default.name: f.default for f in dataclasses.fields(VarStandards)
}

_DRY_RUN_STEPS = 365


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _get_netcdf_urls(scenario: str, variable: str) -> list[tuple[str, str]]:
    """Return (member_id, s3_path) pairs for one variable."""
    aws = get_aws_creds()
    region = aws.pop("region")
    store = from_url(f"s3://{BUCKET}", region=region, **aws)
    stream = obs.list_with_delimiter(store, prefix=NETCDF_PREFIX[scenario], return_arrow=True)
    paths = list(stream["objects"]["path"].to_numpy())

    is_cmip6 = scenario in _CMIP6_SCENARIOS
    ensemble_members = ENSEMBLE_MEMBERS[scenario]

    result = []
    for path in sorted(paths):
        fname = path.split("/")[-1]
        if not (fname.endswith(".nc") and fname.startswith(f"{variable}_")):
            continue
        if is_cmip6:
            # {var}_day_MIROC-ES2H_{experiment}_{ens}_gn_{trange}.nc
            member = fname.split(".nc")[0].split("_gn")[0].split("_")[-1]
        else:
            # {var}_{geomip_scenario}_{rXX}.nc
            member = fname.split(".nc")[0].split("_")[-1]
        if ensemble_members and member not in ensemble_members:
            continue
        result.append((member, path))
    return result


def _preprocess_miroc(ds: xr.Dataset, scenario: str, subset: bool = False) -> xr.Dataset:
    # Keep only standard spatial/temporal coords and drop bnds variables
    # (time_bnds etc.) before calendar conversion; chunk({"time": -1}) in
    # to_proleptic_gregorian fails on multi-dim bnds variables
    keep_coords = set(ds.dims) | {"lat", "lon", "time"}
    ds = ds.drop_vars([c for c in ds.coords if c not in keep_coords], errors="ignore")
    bnds_data_vars = [v for v in ds.data_vars if any("bnds" in d for d in ds[v].dims)]
    if bnds_data_vars:
        ds = ds.drop_vars(bnds_data_vars)
    ds = ds.drop_duplicates(dim="time", keep="first")
    ds = to_proleptic_gregorian(ds)
    ds = ds.drop_encoding()
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])
    ds = trim_negative_precipitation(ds)

    if CMORIZE_HURS[scenario] and "hurs" in ds.data_vars:
        ds = CMORIZE_hurs(ds, "hurs")

    if scenario in TIME_RANGE:
        start_year, end_year = TIME_RANGE[scenario].split("-")
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    if subset:
        ds = ds.isel(time=slice(0, 365))
    return ds


def _derivation_logic(scenario: str) -> str:
    """For documenting how we get the ensemble_member, ie from attrs or filepath."""
    if scenario in _CMIP6_SCENARIOS:
        return (
            "Extracted from CMIP6 DRS filename: url.split('.nc')[0].split('_gn')[0].split('_')[-1]."
        )
    return "Raw filename suffix: url.split('.nc')[0].split('_')[-1] (e.g. 'r01')."


def _update_attrs(ds: xr.Dataset, var_specs: dict, scenario: str) -> xr.Dataset:
    ds = update_variable_attrs(ds, var_specs)

    global_attrs: dict = {
        "scenario": scenario,
        "model": "MIROC-ES2H",
        "Conventions": "CF-1.8",
    }
    if scenario in TIME_RANGE:
        global_attrs["time_range"] = TIME_RANGE[scenario]
    ds.attrs.update(global_attrs)

    return apply_ensemble_provenance(ds, _derivation_logic(scenario))


def _stitch_esgf_gap(geomip: xr.Dataset, variable: str, repo: icechunk.Repository) -> xr.Dataset:
    """Prepend ESGF SSP245 2015-2019 to the GeoMIP baseline run (which starts 2020).

    Reads the bridge from the unified store's ``esgf_ssp245`` group; each GeoMIP
    member gets its ESGF counterpart via _SSP245_ESGF_MEMBER. Requires
    --scenario esgf-ssp245 to have been processed first.
    """
    session = repo.readonly_session("main")
    try:
        esgf = xr.open_dataset(session.store, engine="zarr", group="esgf_ssp245", chunks="auto")[
            [variable]
        ]
    except (FileNotFoundError, KeyError) as err:
        raise RuntimeError(
            f"esgf_ssp245 group missing variable '{variable}' in the unified store; "
            "run --scenario esgf-ssp245 before --scenario ssp245 (gap-fill dependency)"
        ) from err

    bridge = esgf.sel(time=slice("2015", "2019"))
    gap_pieces = [
        bridge.sel(ensemble_member=[_SSP245_ESGF_MEMBER[m]]).assign_coords(ensemble_member=[m])
        for m in geomip.ensemble_member.values
    ]
    gap_ds = xr.concat(gap_pieces, dim="ensemble_member")
    combined = xr.concat([gap_ds, geomip], dim="time")
    combined.attrs.update(
        {
            "gap_fill_source": "esgf_ssp245 group (MIROC-ES2H ESGF SSP245)",
            "gap_fill_period": "2015-2019",
            "gap_fill_member_map": ", ".join(
                f"{m}->{e}" for m, e in sorted(_SSP245_ESGF_MEMBER.items())
            ),
        }
    )
    return combined


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

    member_paths = group_paths_by_member(url_pairs)

    member_datasets = []
    for member, paths in sorted(member_paths.items()):
        log.info("variable=%s member=%s opening %d file(s)", variable, member, len(paths))
        time_slices = [open_netcdf_from_s3(obstore_inst, p) for p in sorted(paths)]
        member_ds = (
            xr.concat(time_slices, dim="time", data_vars="minimal")
            if len(time_slices) > 1
            else time_slices[0]
        )
        member_ds = _preprocess_miroc(member_ds, scenario, subset=subset)
        member_ds = member_ds.expand_dims({"ensemble_member": [member]})
        member_datasets.append(member_ds)
        log.info("variable=%s member=%s shape=%s", variable, member, dict(member_ds.dims))

    ds = xr.concat(member_datasets, dim="ensemble_member")
    ds = ds[[variable]]
    log.info("variable=%s concat done shape=%s", variable, dict(ds.dims))

    if scenario == "ssp245":
        ds = _stitch_esgf_gap(ds, variable, repo)
        log.info("variable=%s gap-filled 2015-2019 from esgf_ssp245", variable)

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
    store = from_url(f"s3://{BUCKET}/{NETCDF_PREFIX[scenario]}/", region=region, **aws)

    existing_names: set[str] = set()
    for batch in obs.list(store):
        for item in batch:
            existing_names.add(item["path"].split("/")[-1])

    log.info("%d files already present in S3", len(existing_names))
    filtered = [u for u in urls if u.split("/")[-1] not in existing_names]
    log.info(
        "skipping %d already-uploaded files, %d to fetch",
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
    log.info("running rclone command:\n%s", " \\\n    ".join(command))
    subprocess.run(command)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.command()
def fetch(
    variable: list[str] = typer.Option(
        [], "--variable", help="Variables to fetch (defaults to all)"
    ),
    scenario: str = typer.Option(..., "--scenario", help=f"Choices: {ALL_SCENARIOS}"),
) -> None:
    """Fetch MIROC-ES2H NetCDF files from JAMSTEC and copy to S3."""
    import warnings

    if scenario not in ALL_SCENARIOS:
        raise typer.BadParameter(f"scenario must be one of {ALL_SCENARIOS}, got {scenario!r}")

    variables = list(variable) if variable else MIROC_VARIABLES
    if scenario in _CMIP6_SCENARIOS:
        warnings.warn(
            "Fetches raw NetCDF files from JAMSTEC and copies to S3. rclone must be installed."
        )
        urls = _build_cmip6_urls(variables, scenario)
    else:
        warnings.warn(
            "Fetches raw NetCDF files from a slow JAMSTEC server and copies to S3. "
            "rclone must be installed."
        )
        urls = _build_geomip_urls(variables, scenario)

    urls = _filter_existing_urls(urls, scenario)
    if not urls:
        log.info("nothing to fetch, all files already in S3")
        return

    prefix = "CMIP6" if scenario in _CMIP6_SCENARIOS else "GeoMIP"
    urls_file = f"MIROC-ES2H-{prefix}-{scenario}-urls.txt"
    log.info("queued %d files -> %s", len(urls), urls_file)
    _rclone_copy_urls(urls, urls_file, scenario)


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
        False, "--all-variables", help="Process all expected variables (MIROC_VARIABLES)"
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

    Ordering: esgf-ssp245 must complete before ssp245 (the ssp245 group
    gap-fills 2015-2019 from the esgf_ssp245 group); list them in that order.
    """
    if all_variables:
        variables = MIROC_VARIABLES
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
