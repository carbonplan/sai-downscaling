# COILED vm-type r8g.2xlarge
# COILED region us-west-2
# COILED tag project=SRM

import dataclasses
import logging
import time

import dask
import icechunk
import typer
import xarray as xr
import zarr
from obspec_utils.readers import EagerStoreReader
from obstore.store import HTTPStore

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


if __name__ == "__main__":
    app()
