# COILED vm-type r8g.16xlarge
# COILED region us-west-2
# COILED tag project=SRM

import dataclasses
import logging
import time

import dask
import icechunk
import numpy as np
import typer
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk
from obspec_utils.readers import EagerStoreReader
from obstore.store import HTTPStore

from srm.bcsd_config import VariableConfig
from srm.config import VarSpec, VarStandards
from srm.input_data.etl_utils import (
    _display_dry_run_result,
    _init_repo_from_uri,
    add_cf_bounds,
    build_encoding_dict,
    console,
    determine_write_mode,
    setup_logging,
    update_variable_attrs,
    write_dataset_to_icechunk,
)
from srm.qaqc import periodic_rolling
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 128})
dask.config.set(scheduler="threads")

setup_logging()
log = logging.getLogger(__name__)

# --- Source ---

OSDF_BASE = "https://osdf-director.osg-htc.org"
DTN_BASE = "https://dtn-pas.bois.nrp.internet2.edu:8443"
OSDF_PATH_3HRLY = "/ncar/gdex/d314000/0.25deg/3hrly"
OSDF_PATH_DAILY = "/ncar/gdex/d314000/0.25deg/daily"
GDEX_0P25_YEARS = range(1948, 2011)

# tas is only available as 3-hourly and must be resampled to daily mean.
THRHRLY_TO_CMIP6: dict[str, str] = {"tas": "tas"}

# Vars available as pre-computed daily NetCDF3 files on the https server.
DAILY_TO_CMIP6: dict[str, str] = {
    "prcp": "pr",
    "dlwrf": "rlds",
    "dswrf": "rsds",
    "pres": "ps",
    "shum": "huss",
    "tmin": "tasmin",
    "tmax": "tasmax",
}

GDEX_TO_CMIP6: dict[str, str] = {**THRHRLY_TO_CMIP6, **DAILY_TO_CMIP6}
GDEX_VARS: list[str] = list(GDEX_TO_CMIP6.keys())

VAR_SPECS: dict[str, VarSpec] = {
    f.default.name: f.default for f in dataclasses.fields(VarStandards)
}

# --- S3 paths ---

BUCKET = "carbonplan-srm"

OUTPUT_URI = f"s3://{BUCKET}/input/processed/gdex-gmf.icechunk"
OUTPUT_CHUNKS: dict[str, int] = {"time": 1, "lat": 720, "lon": 1440}
OUTPUT_SHARDS: dict[str, int] = {"time": 30, "lat": 720, "lon": 1440}

_DRY_RUN_STEPS = 365
# FILL_MAX = 1e-9
# FILL_MIN = 0.9 * FILL_MAX
TRAIN_PERIOD = ("1960", "2008")  # matches the BCSD training window
VARIABLE_CONFIG_WINDOW = VariableConfig.model_fields["running_window_length"].default


# --- Source URL helpers ---


def make_osdf_urls(
    variable: str,
    years: range | list[int] = GDEX_0P25_YEARS,
) -> list[str]:
    if variable in THRHRLY_TO_CMIP6:
        return [
            f"{OSDF_BASE}{OSDF_PATH_3HRLY}/{year}/{variable}_0p25_3hourly_{year}-{year}.nc"
            for year in years
        ]
    return [f"{DTN_BASE}{OSDF_PATH_DAILY}/{variable}_0p25_daily_{year}-{year}.nc" for year in years]


def _open_year(year: int, variable: str) -> xr.Dataset:
    """Open a single year's GDEX NetCDF file over HTTP via EagerStoreReader.

    3-hourly files (tas) are NetCDF4/HDF5; daily files are NetCDF3 Classic.
    """
    url = make_osdf_urls(variable=variable, years=[year])[0]
    is_thrhrly = variable in THRHRLY_TO_CMIP6
    base = OSDF_BASE if is_thrhrly else DTN_BASE
    if is_thrhrly:
        # EagerStoreReader downloads the full file before h5py opens it,
        # avoiding a bunch of small HTTP round-trips that OSDF drops under load.
        store = HTTPStore.from_url(
            base, client_options={"timeout": "1800s", "connect_timeout": "60s"}
        )
        path = url.removeprefix(base)
        return xr.open_dataset(
            EagerStoreReader(store, path, request_size=64 * 1024 * 1024, max_concurrent_requests=4),
            engine="h5netcdf",
            chunks="auto",
        )
    else:
        # 4 × 64 MB concurrent requests; many small requests overwhelmed the slow server.
        store = HTTPStore.from_url(
            base, client_options={"timeout": "900s", "connect_timeout": "30s"}
        )
        path = url.removeprefix(base)
        return xr.open_dataset(
            EagerStoreReader(store, path, request_size=64 * 1024 * 1024, max_concurrent_requests=4),
            engine="scipy",
            chunks="auto",
        )


@dask.delayed
def _load_year(year: int, variable: str, needs_resample: bool) -> xr.Dataset:
    """Download one year via obspec-utils, preprocess, return xr.Dataset."""
    for attempt in range(4):
        try:
            ds = _open_year(year, variable)[[variable]]
            if needs_resample:
                ds = _resample_daily(ds)
            return ds.pipe(_rename_to_cmip6, variable).pipe(preprocess_gdex)
        except Exception as e:
            if attempt == 3:
                raise
            wait = 30 * (2**attempt)  # 30s, 60s, 120s
            log.warning(
                "attempt %d failed for %s %d: %s — retrying in %ds",
                attempt + 1,
                variable,
                year,
                e,
                wait,
            )
            time.sleep(wait)


# --- Pipeline functions ---


def preprocess_gdex(ds: xr.Dataset) -> xr.Dataset:
    """Standardize GDEX coordinates: drop singleton ``z``, normalize lat/lon names, sort."""
    if "z" in ds.dims:
        ds = ds.squeeze("z", drop=True)
    # Daily (NetCDF3) files use 'latitude'/'longitude'; rename to match 3-hourly convention.
    rename = {}
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    return ds.pipe(lon_to_180, lon_name="lon").sortby(["lat", "lon"]).drop_encoding()


def finalize_gdex_metadata(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    """Stamp source/provenance global attrs for the covered period."""
    ds.attrs.update(
        {
            "valid_time_start": f"{start_year}-01-01",
            "valid_time_stop": f"{end_year}-12-31",
            "source": "Princeton Global Forcing (Sheffield et al. 2006), 0.25deg daily",
            "url": "https://gdex.ucar.edu/datasets/d314000/",
        }
    )
    return ds


def _resample_daily(ds: xr.Dataset) -> xr.Dataset:
    return ds.resample(time="D").mean()


def _rename_to_cmip6(ds: xr.Dataset, gdex_var: str) -> xr.Dataset:
    cmip6_name = GDEX_TO_CMIP6[gdex_var]
    if cmip6_name != gdex_var:
        ds = ds.rename({gdex_var: cmip6_name})
    return ds


def _process_gdex_variable(
    variable: str,
    start_year: int,
    end_year: int,
    repo: icechunk.Repository,
    dry_run: bool,
    dry_run_output: str | None,
    commit_message: str | None,
) -> None:
    """Process and write one GDEX variable (given as its native GDEX name) to the icechunk store."""
    cmip6_name = GDEX_TO_CMIP6[variable]
    needs_resample = variable in THRHRLY_TO_CMIP6

    if not dry_run:
        return

    raw = _open_year(start_year, variable)[[variable]]
    if needs_resample:
        raw = _resample_daily(raw)
    raw = raw.isel(time=slice(0, _DRY_RUN_STEPS))
    ds = (
        raw.pipe(_rename_to_cmip6, variable)
        .pipe(preprocess_gdex)
        .pipe(add_cf_bounds)
        .pipe(update_variable_attrs, VAR_SPECS)
        .pipe(finalize_gdex_metadata, start_year, start_year)
    )
    _display_dry_run_result(ds, cmip6_name, store=dry_run_output)
    if dry_run_output is not None:
        log.info("Writing dry-run sample for %s to %s", cmip6_name, dry_run_output)
        dr_repo, dr_session = _init_repo_from_uri(dry_run_output)
        dr_write_mode = determine_write_mode(dr_repo)
        encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
        write_dataset_to_icechunk(
            ds,
            dr_session,
            encoding=encoding,
            shards=OUTPUT_SHARDS,
            commit_message=f"dry-run: {commit_message or cmip6_name}",
            write_mode=dr_write_mode,
            repo=dr_repo,
        )
        log.info("dry-run write done: %s -> %s", cmip6_name, dry_run_output)
        read_session = dr_repo.readonly_session("main")
        written = xr.open_dataset(read_session.store, engine="zarr", chunks="auto")
        console.print(written)


def process_gdex(
    variables: list[str],
    start_year: int = 1950,
    end_year: int = 2008,
    output_uri: str = OUTPUT_URI,
    dry_run: bool = False,
    dry_run_output: str | None = None,
    commit_message: str | None = None,
    overwrite: bool = False,
) -> None:
    """Materialize GDEX variables (PGF 0.25deg, resampled to daily) to the icechunk store."""
    repo, _ = _init_repo_from_uri(output_uri)

    if dry_run:
        for variable in variables:
            _process_gdex_variable(
                variable=variable,
                start_year=start_year,
                end_year=end_year,
                repo=repo,
                dry_run=True,
                dry_run_output=dry_run_output,
                commit_message=commit_message,
            )
        return

    cmip6_names = [GDEX_TO_CMIP6[v] for v in variables]
    msg = commit_message or f"GDEX: {', '.join(cmip6_names)}"

    if not overwrite and any(c.message == msg for c in repo.ancestry(branch="main")):
        log.info("already committed, skipping: %s", msg)
        return

    var_datasets = []
    for variable in variables:
        needs_resample = variable in THRHRLY_TO_CMIP6
        years = list(range(start_year, end_year + 1))
        delayed_list = [_load_year(yr, variable, needs_resample) for yr in years]
        log.info("fetching %d years for %s in parallel", len(years), variable)
        year_datasets: list[xr.Dataset] = list(dask.compute(*delayed_list))
        var_datasets.append(xr.concat(year_datasets, dim="time"))

    ds = (
        xr.merge(var_datasets)
        .pipe(add_cf_bounds)
        .pipe(update_variable_attrs, VAR_SPECS)
        .pipe(finalize_gdex_metadata, start_year, end_year)
    )

    session = repo.writable_session("main")
    write_mode = "w" if overwrite else determine_write_mode(repo)
    log.info("writing %s shape=%s mode=%s", cmip6_names, dict(ds.dims), write_mode)
    encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)

    write_dataset_to_icechunk(
        ds,
        session,
        encoding=encoding,
        shards=OUTPUT_SHARDS,
        commit_message=msg,
        write_mode=write_mode,
        repo=repo,
        group=None,
    )
    log.info("Done: %s", msg)


# --- Chronic-zero precipitation patch: scan and fill (issue #517) ---


def open_pr(uri: str, branch: str = "main") -> xr.DataArray:
    repo, _ = _init_repo_from_uri(uri)
    return xr.open_dataset(
        repo.readonly_session(branch).store, engine="zarr", chunks=OUTPUT_SHARDS
    )["pr"]


def _any_within_window(mask: xr.DataArray, window: int = VARIABLE_CONFIG_WINDOW) -> xr.DataArray:
    """Is the ``(dayofyear, ...)`` mask true anywhere in the centered window? Wraps the year."""
    return periodic_rolling(mask.astype("int8"), dim="dayofyear", window=window, agg="max") > 0


def always_dry_doy_windows(pr: xr.DataArray, window: int = VARIABLE_CONFIG_WINDOW) -> xr.DataArray:
    """``(dayofyear, lat, lon)`` mask of windows with no rain on any day, in any year."""
    doy_max = pr.groupby("time.dayofyear").max()
    observed = doy_max.notnull().any("dayofyear")
    return ~_any_within_window(doy_max > 0, window) & observed


def patch_dry_pixels(
    branch: str = "fix-dry-pixels",
) -> None:
    """Fill chronically dry precipitation windows onto a non-main icechunk branch.

    Scans the training window for day-of-year windows that never see rain, then replaces the
    exact zeros on those days with ``FILL_CONST``. The whole record is rewritten, not just the
    training years, so the fill stays consistent across the store.
    """
    if branch == "main":
        raise ValueError("refusing to patch main; use a dedicated branch, then merge")

    pr = open_pr(OUTPUT_URI)

    always_dry = always_dry_doy_windows(pr.sel(time=slice(*TRAIN_PERIOD))).compute()

    observed = int(pr.isel(time=0).notnull().sum().compute())
    flagged_doy_cells = int(always_dry.sum())
    flagged_cells = int(always_dry.any("dayofyear").sum())
    log.info(
        "flagged %d doy-cells across %d cells (%.2f%% of %d observed)",
        flagged_doy_cells,
        flagged_cells,
        100 * flagged_cells / observed,
        observed,
    )
    if flagged_cells == 0:
        raise ValueError("scan flagged nothing! No changes made")

    fill_doys = _any_within_window(always_dry, VARIABLE_CONFIG_WINDOW).reindex(
        dayofyear=np.arange(1, 367), fill_value=False
    )

    doy = pr["time"].dt.dayofyear
    fill_days = (
        fill_doys.chunk({"dayofyear": -1})
        .sel(dayofyear=doy)
        .drop_vars("dayofyear")
        .chunk({"time": OUTPUT_SHARDS["time"]})
    )

    FILL_CONST = 1e-6
    patched = xr.where(fill_days & (pr == 0), FILL_CONST, pr).astype(pr.dtype).rename("pr")

    # Jittered alternative, drawn from a narrow band so the filled window has non-zero variance.
    # def _draw(block, block_id=None):
    #     rng = np.random.default_rng((0, *block_id))
    #     return rng.uniform(FILL_MIN, FILL_MAX, size=block.shape).astype(block.dtype)

    # fill = pr.copy(data=pr.data.map_blocks(_draw, dtype=pr.dtype))
    # patched = xr.where(fill_days & (pr == 0), fill, pr).astype(pr.dtype).rename("pr")

    patched.attrs = pr.attrs

    log.info("Done: %s", patched)
    repo, _ = _init_repo_from_uri(OUTPUT_URI)
    if branch not in repo.list_branches():
        repo.create_branch(branch, repo.lookup_branch("main"))
        log.info("created branch %s from the main tip", branch)
    session = repo.writable_session(branch)

    to_icechunk(patched.to_dataset(name="pr"), session, region="auto", mode="r+")

    snapshot = session.commit("fixed empty rolling windows without rain")
    console.print(f"committed {snapshot} to branch {branch}")
    console.print(f"main tip unchanged: {repo.lookup_branch('main')}")

    # Promote, once the branch has been reviewed.
    # repo.reset_branch("main", repo.lookup_branch(branch))


# --- CLI ---

app = typer.Typer()


def _resolve_variables(
    variable: list[str], all_variables: bool, valid_vars: list[str]
) -> list[str]:
    if all_variables:
        return valid_vars
    if variable:
        invalid = set(variable) - set(valid_vars)
        if invalid:
            raise typer.BadParameter(f"Unknown variable(s): {sorted(invalid)}. Valid: {valid_vars}")
        return list(variable)
    raise typer.BadParameter("Specify at least one --variable or use --all-variables.")


@app.command()
def process(
    start_year: int = typer.Option(1950, help="First year to include."),
    end_year: int = typer.Option(2008, help="Last year to include."),
    variable: list[str] = typer.Option([], "--variable", help="Variable(s) to process."),
    all_variables: bool = typer.Option(False, "--all", help="Process all GDEX vars."),
    output: str = typer.Option(OUTPUT_URI, "--output", help="Destination s3:// URI or local path."),
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
        help="Icechunk commit message.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Force mode='w' to cleanly re-materialize a populated store (clobbers it).",
    ),
) -> None:
    """Materialize GDEX variables to the unified icechunk store.

    Accepts native GDEX daily names; renames to CMIP6 in output (prcp→pr, dlwrf→rlds, tmin→tasmin, tmax→tasmax).
    Valid variables: tas, prcp, dlwrf, tmin, tmax.
    """
    variables = _resolve_variables(variable, all_variables, GDEX_VARS)
    process_gdex(
        variables=variables,
        start_year=start_year,
        end_year=end_year,
        output_uri=output,
        dry_run=dry_run,
        dry_run_output=dry_run_output,
        commit_message=commit_message,
        overwrite=overwrite,
    )


@app.command()
def fix_dry_pixels(
    branch: str = typer.Option("fix-dry-pixels", "--branch", help="Target branch; never main."),
) -> None:
    """Fill chronically dry precipitation windows (issue #517).

    Cells that are exactly zero across a whole running window in every training year break the
    parametric tail fits in the BCSD debiaser. Writes to a dedicated branch; main is untouched.
    """
    patch_dry_pixels(branch=branch)


if __name__ == "__main__":
    app()
