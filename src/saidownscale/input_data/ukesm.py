# COILED vm-type r8g.4xlarge
# COILED region us-west-2
# COILED tag project=SRM

import dataclasses
import logging

import dask
import icechunk
import obstore as obs
import typer
import xarray as xr
import zarr
from obstore.store import from_url

from saidownscale.config import SCENARIO_TO_GROUP, VarSpec, VarStandards, init_repo
from saidownscale.input_data.etl_utils import (
    _display_dry_run_result,
    _init_repo_from_uri,
    apply_ensemble_provenance,
    build_encoding_dict,
    console,
    determine_write_mode,
    get_aws_creds,
    group_paths_by_member,
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
logging.getLogger("fsspec").setLevel(logging.WARNING)


T_PR_VARS: list[str] = ["pr", "tas", "tasmin", "tasmax"]

UKESM_VARIABLES = ["tas", "rsds", "pr", "tasmax", "tasmin"]

BUCKET = "carbonplan-srm"
UNIFIED_PREFIX = "input/processed/ukesm.icechunk"


# Raw NetCDF prefixes on S3
S3_INPUT_PREFIX: dict[str, str] = {
    "historical": raw_netcdf_prefix("UKESM", "historical"),
    # SSP245 source updated 09-2026: https://gws-access.jasmin.ac.uk/public/macloud/
    "SSP245": raw_netcdf_prefix("UKESM1-1LL", "ssp245"),
    "G6-1.5K": raw_netcdf_prefix("UKESM", "g6-1p5k"),
}

# T/PR NetCDF prefixes for G6-1.5K
T_PR_INPUT_PREFIX: dict[str, str] = {
    "G6-1.5K": raw_netcdf_prefix("UKESM", "g6-1p5k-t-pr"),
}

# --- 2026 SSP245 ssp245 update (JASMIN) ---
# In 09-2026, we received updated the ssp245 source files from (https://gws-access.jasmin.ac.uk/public/macloud/).
# These files are all genuinely UKESM1-1-LL. Landed under a separate UKESM1-1LL/ raw dir so the old UKESM/
# drops stay intact for comparison and rollback if needed.
# Delivered at https://gws-access.jasmin.ac.uk/public/macloud/SSP245_ukesm1p1/{suite}/,
# one directory per UM suite, and copied verbatim into
# s3://carbonplan-srm/input/raw/UKESM1-1LL/netcdf/ssp245/ (9 files, ~51 GB).
SSP245_SUITE_TO_MEMBER: dict[str, str] = {
    "u-dp902": "r2i1p1f2",
    "u-dp903": "r3i1p1f2",
    "u-dp904": "r12i1p1f2",
}

SSP245_FILE_TEMPLATE = "{stem}_UKESM1-1-LL_SSP245_{suite}_201501-210012.nc"

# Which file each variable is read from. tas/tasmin/tasmax share one file and are
# separated by cell_methods (see _um_var_rename).
SSP245_VAR_TO_STEM: dict[str, str] = {
    "tas": "tas_mean_min_max_day",
    "tasmin": "tas_mean_min_max_day",
    "tasmax": "tas_mean_min_max_day",
    "pr": "pr_day",
    "rsds": "rsds_day",
}

# Filename prefixes to look for in G6-1.5K T_PR_INPUT_PREFIX, per variable
T_PR_VAR_LOOKUP: dict[str, dict[str, list[str]]] = {
    "G6-1.5K": {
        # tas_day files contain tas, tasmin, and tasmax
        "pr": ["pr_day_"],
        "tas": ["tas_day_"],
        "tasmin": ["tas_day_"],
        "tasmax": ["tas_day_"],
    },
}

# UM suite ID / legacy member ID -> CMIP6 ripf mapping for T_PR files
T_PR_MEMBER_RENAME: dict[str, dict[str, str]] = {
    "G6-1.5K": {
        "u-dp583": "r2i1p1f2",
        "u-dp690": "r3i1p1f2",
        "u-dp691": "r12i1p1f2",
    },
}

# Historical is a single UM suite; the suite ID is the member ID (no CMIP6 ripf exists).
HISTORICAL_MEMBER = "u-by791"

ENSEMBLE_MEMBERS: dict[str, list[str]] = {
    "historical": [HISTORICAL_MEMBER],
    "SSP245": ["r12i1p1f2", "r2i1p1f2", "r3i1p1f2"],
    "G6-1.5K": ["r12i1p1f2", "r2i1p1f2", "r3i1p1f2"],
}

# T_PR files span 2015-2100; clip G6-1.5K to match hurs/rsds time range.
# Historical is already 1850-2014; the entry is a guard, not a real trim.
# The 2026 SSP245 delivery runs to 2100-12-30, but ssp245 is clipped to 2099 project-wide
# (cesm2_waccm.py) so the scenario shares one time axis across GCMs. The extra
# year stays in the raw drop on S3 if that convention is ever revisited for all three.
TIME_RANGE: dict[str, str] = {
    "historical": "1850-2014",
    "SSP245": "2015-2099",
    "G6-1.5K": "2035-2084",
}

ALL_SCENARIOS = list(ENSEMBLE_MEMBERS.keys())

# --- Model identity ---
# Every UKESM file delivered by this supplier is UKESM1-1-LL, across all scenarios.
# Some source NetCDFs carry source_id/model_id/parent_source_id = "UKESM1-0-LL" and some
# filenames say "UKESM1-1" instead of "UKESM1-1-LL". The data provider confirmed these are
# typos in the file metadata, not a different model. We therefore overwrite the stale
# identity attrs rather than trust them, and record the overwrite in MODEL_ATTR_NOTE so
# the correction stays auditable downstream.
#
# The updated SSP245 NetCDFs (https://gws-access.jasmin.ac.uk/public/macloud/)
#  don't have any model identity attrs, so nothing there contradicts the filename.
MODEL = "UKESM1-1-LL"
_MODEL_IDENTITY_ATTRS = ("source_id", "model_id", "parent_source_id")
MODEL_ATTR_NOTE = (
    f"Model identity attrs from the source files were overwritten with {MODEL}; the data "
    "supplier confirmed the source_id/filename model labels are typos and that all files "
    f"from this delivery are {MODEL}."
)

# --- Historical (single UM suite, delivered directly to S3) ---

_HIST_STEM = f"daily_UKESM1-1-LL_historical_{HISTORICAL_MEMBER}_185001-201512.nc"
# tas/tasmin/tasmax share one combined file
_HIST_TEMP_FILE = f"tas-min-mn-max_{_HIST_STEM}"
HISTORICAL_FILES: dict[str, str] = {
    "hurs": f"hurs_{_HIST_STEM}",
    "pr": f"pr_{_HIST_STEM}",
    "rsds": f"rsds_{_HIST_STEM}",
    "tas": _HIST_TEMP_FILE,
    "tasmin": _HIST_TEMP_FILE,
    "tasmax": _HIST_TEMP_FILE,
}

# UM-native files use UM/CF standard names; map to CMIP6 names.
HISTORICAL_VAR_RENAME: dict[str, str] = {
    "relative_humidity": "hurs",
    "precipitation_flux": "pr",
    "surface_downwelling_shortwave_flux_in_air": "rsds",
}

# The combined temperature file yields air_temperature, air_temperature_0, air_temperature_1
_UM_CELL_METHOD_TO_VAR: dict[str, str] = {
    "maximum": "tasmax",
    "minimum": "tasmin",
    "mean": "tas",
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


def _list_netcdfs(prefix: str) -> list[str]:
    """List every .nc object directly under an S3 prefix."""
    aws = get_aws_creds()
    region = aws.pop("region")
    store = from_url(f"s3://{BUCKET}", region=region, **aws)
    stream = obs.list_with_delimiter(store, prefix=prefix, return_arrow=True)
    return [str(p) for p in stream["objects"]["path"].to_numpy() if str(p).endswith(".nc")]


def _get_netcdf_urls(scenario: str, variable: str) -> list[tuple[str, str]]:
    """Return (member_id, s3_path) pairs for one variable."""
    if scenario == "historical":
        # Single ensemble, known filenames.
        if variable not in HISTORICAL_FILES:
            return []
        return [
            (HISTORICAL_MEMBER, f"{S3_INPUT_PREFIX['historical']}/{HISTORICAL_FILES[variable]}")
        ]

    if scenario == "SSP245":
        # Filenames are fully determined by suite x stem, so build them rather than listing
        # and parsing: the member comes from SSP245_SUITE_TO_MEMBER by construction, and the
        # suite ID in the filename is never mistaken for the date range.
        if variable not in SSP245_VAR_TO_STEM:
            return []
        stem = SSP245_VAR_TO_STEM[variable]
        return [
            (
                member,
                f"{S3_INPUT_PREFIX['SSP245']}/"
                + SSP245_FILE_TEMPLATE.format(stem=stem, suite=suite),
            )
            for suite, member in SSP245_SUITE_TO_MEMBER.items()
        ]

    # G6-1.5K is the one scenario still fed by two drops: rsds comes from the ARISE-CMOR
    # set, while pr/tas/tasmin/tasmax come from the private T/PR archive, whose filenames
    # carry UM suite ids that T_PR_MEMBER_RENAME maps to ripf downstream of discovery.
    if variable in T_PR_VARS:
        prefix = T_PR_INPUT_PREFIX[scenario]
        fname_prefixes = T_PR_VAR_LOOKUP[scenario].get(variable, [f"{variable}_day_"])
        members: list[str] = []
    else:
        prefix = S3_INPUT_PREFIX[scenario]
        fname_prefixes = [f"{variable}_"]
        members = ENSEMBLE_MEMBERS[scenario]

    result = []
    for path in _list_netcdfs(prefix):
        fname = path.split("/")[-1]
        if not any(fname.startswith(p) for p in fname_prefixes):
            continue
        if members and not any(f"_{m}_" in fname for m in members):
            continue
        # CMOR names end ..._<member>_gn_<dates>.nc; raw UM ones end ..._<suite>_<dates>.nc.
        member = (
            fname.split("_gn_")[0].split("_")[-1]
            if "_gn_" in fname
            else fname.split(".nc")[0].split("_")[-2]
        )
        result.append((member, path))
    return result


def _um_var_rename(ds: xr.Dataset) -> dict[str, str]:
    """Map UM/CF standard names to CMIP6 names.

    Used by every drop written straight out of the Unified Model — the historical files and
    the 2026 SSP245 delivery, which share this layout.

    ``air_temperature*`` vars are resolved by their ``cell_methods`` statistic rather than by
    the order h5netcdf assigned the ``_0``/``_1`` suffixes, so a file that writes the three
    temperatures in a different order still maps correctly.
    """
    rename: dict[str, str] = {}
    for name in ds.data_vars:
        name = str(name)
        if name in HISTORICAL_VAR_RENAME:
            rename[name] = HISTORICAL_VAR_RENAME[name]
        elif name.startswith("air_temperature"):
            cell_methods = ds[name].attrs.get("cell_methods", "")
            matched = [v for m, v in _UM_CELL_METHOD_TO_VAR.items() if m in cell_methods]
            if len(matched) != 1:
                raise ValueError(
                    f"cannot resolve {name!r} to tas/tasmin/tasmax from "
                    f"cell_methods={cell_methods!r}"
                )
            rename[name] = matched[0]
    log.info("UM var rename: %s", rename)
    return rename


def _preprocess_ukesm(ds: xr.Dataset, scenario: str, subset: bool = False) -> xr.Dataset:
    if subset:
        # Dry-run only: seek to the window before head-sampling. The T/PR files open before
        # the G6-1.5K window, so sampling the head of the file leaves nothing behind once the
        # authoritative clip below runs. Sampling here rather than only after that clip keeps
        # to_proleptic_gregorian's chunk({"time": -1}) from pulling the whole series.
        # Start bound only: the raw axis is 360_day, where "12-31" does not exist.
        if scenario in TIME_RANGE:
            start_year = TIME_RANGE[scenario].split("-")[0]
            ds = ds.sel(time=slice(f"{start_year}-01-01", None))
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

    # Sample AFTER the clip: the T/PR source files open before the G6-1.5K window, so
    # taking the head of the file first leaves nothing behind once the clip is applied.
    if subset:
        ds = ds.isel(time=slice(0, _DRY_RUN_STEPS))

    return trim_negative_precipitation(ds)


def _derivation_logic(scenario: str, variable: str | None = None) -> str:
    if scenario == "historical":
        return (
            f"Single ensemble_member {HISTORICAL_MEMBER}; the ID is stored as the "
            "'ensemble_member' value because the source files carry no CMIP6 ripf ID."
        )
    if scenario == "SSP245":
        mapping_str = ", ".join(f"{k}->{v}" for k, v in SSP245_SUITE_TO_MEMBER.items())
        return (
            "Filenames are constructed from the UM suite ID, which is mapped to its CMIP6 "
            f"ripf: {mapping_str}. Single-source 2026 delivery; no member ID is parsed out "
            "of the filename."
        )
    if variable in T_PR_VARS and scenario in T_PR_INPUT_PREFIX:
        mapping_str = ", ".join(f"{k}->{v}" for k, v in T_PR_MEMBER_RENAME[scenario].items())
        return (
            f"UM suite IDs extracted from filename and remapped to CMIP6 ripf: {mapping_str}. "
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
            "model": MODEL,
            "Conventions": "CF-1.8",
        }
    )
    # See MODEL_ATTR_NOTE: the identity attrs carried in from the source NetCDFs are not
    # trusted, so any that are present are overwritten and the overwrite is recorded.
    overwritten = [key for key in _MODEL_IDENTITY_ATTRS if key in ds.attrs]
    if overwritten:
        for key in overwritten:
            ds.attrs[key] = MODEL
        ds.attrs["model_id_correction"] = MODEL_ATTR_NOTE
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
    branch: str = "main",
) -> None:
    group = SCENARIO_TO_GROUP[scenario]
    log.info("variable=%s group=%s start", variable, group)

    # dry-run only previews _DRY_RUN_STEPS timesteps; subsetting per-member up front
    # avoids reading/concatenating the full time series just to truncate it later.
    subset = subset or dry_run

    var_in_store = variable_in_store(repo, variable, group=group, branch=branch)
    if not dry_run and not overwrite and var_in_store:
        log.info("variable=%s skip: already in store (use --overwrite to replace)", variable)
        return

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

    member_datasets = []
    for member, paths in sorted(member_paths.items()):
        log.info("variable=%s member=%s opening %d file(s)", variable, member, len(paths))
        time_slices = [
            xr.open_dataset(f"s3://{BUCKET}/{path}", engine="h5netcdf", chunks={})
            for path in sorted(paths)
        ]
        member_ds = (
            xr.concat(time_slices, dim="time", data_vars="minimal")
            if len(time_slices) > 1
            else time_slices[0]
        )
        member_ds = member_ds.rename(_um_var_rename(member_ds))[[variable]]
        if scenario == "historical":
            member_ds = member_ds.chunk({"time": OUTPUT_SHARDS["time"]})
        member_ds = _preprocess_ukesm(member_ds, scenario, subset=subset)
        member_ds = member_ds.expand_dims({"ensemble_member": [member]})
        member_datasets.append(member_ds)
        log.info("variable=%s member=%s shape=%s", variable, member, dict(member_ds.sizes))

    ds = xr.concat(member_datasets, dim="ensemble_member")
    if variable in ds:
        ds = ds[[variable]]
    log.info("variable=%s concat done shape=%s", variable, dict(ds.sizes))

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.callback()
def cli() -> None:
    """UKESM input-data ETL. Keeps 'process' as an explicit subcommand."""


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
            branch=branch,
        )


if __name__ == "__main__":
    app()
