# COILED vm-type r8g.4xlarge
# COILED region us-west-2
# COILED tag project=SRM

import dataclasses
import logging

import dask
import icechunk
import typer
import xarray as xr
import zarr
from obspec_utils.readers import BlockStoreReader
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import from_url
from virtualizarr.parsers import HDFParser

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
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 128})
dask.config.set(scheduler="threads")

setup_logging()
log = logging.getLogger(__name__)

# --- Source ---

OSDF_BASE = "https://osdf-director.osg-htc.org"
OSDF_PATH_PREFIX = "/ncar/gdex/d314000/0.25deg/3hrly"
GDEX_0P25_YEARS = range(1948, 2011)

VAR_SPECS: dict[str, VarSpec] = {
    f.default.name: f.default for f in dataclasses.fields(VarStandards)
}

# --- S3 paths ---

BUCKET = "carbonplan-srm"

# Virtual (reference) icechunk store — written by `virtualize`, read by `process`
VIRTUAL_PREFIX = "input/tensor/NCAR/GDEX-GMF-virtual.icechunk"

# Materialized output — written by `process`
OUTPUT_URI = f"s3://{BUCKET}/input/processed/gdex-gmf.icechunk"
OUTPUT_CHUNKS: dict[str, int] = {"time": 1, "lat": 720, "lon": 1440}
OUTPUT_SHARDS: dict[str, int] = {"time": 30, "lat": 720, "lon": 1440}

# 365 daily steps = one calendar year; enough to verify the full transform chain
_DRY_RUN_STEPS = 365


# --- Source URL helpers ---


def make_osdf_urls(
    years: range | list[int] = GDEX_0P25_YEARS,
    base_url: str = OSDF_BASE,
) -> list[str]:
    """Build OSDF URLs for the GDEX 0.25deg 3-hourly ``tas`` NetCDF files."""
    return [
        f"{base_url}{OSDF_PATH_PREFIX}/{year}/tas_0p25_3hourly_{year}-{year}.nc" for year in years
    ]


def _open_year(store, year: int) -> xr.Dataset:
    """Open a single year's GDEX NetCDF file directly from OSDF over HTTP."""
    url = make_osdf_urls(years=[year])[0]
    path = url.removeprefix(OSDF_BASE)
    return xr.open_dataset(BlockStoreReader(store, path), engine="h5netcdf", chunks="auto")


# --- Virtual store loader ---


def load_gdex_virtual() -> xr.Dataset:
    """Load ``tas`` from the GDEX virtual icechunk store.

    Virtual chunks point back to OSDF over HTTP, so the repository must be
    opened with a VirtualChunkContainer that authorizes those reads.
    """
    storage = icechunk.s3_storage(bucket=BUCKET, prefix=VIRTUAL_PREFIX, region="us-west-2")
    repo_config = icechunk.RepositoryConfig.default()
    repo_config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(OSDF_BASE + "/", store=icechunk.http_store())
    )
    repo = icechunk.Repository.open_or_create(storage, repo_config)
    session = repo.readonly_session("main")
    return xr.open_dataset(
        session.store, engine="zarr", chunks="auto", consolidated=False, zarr_format=3
    )[["tas"]]


# --- Pipeline functions ---


def preprocess_gdex(ds: xr.Dataset) -> xr.Dataset:
    """Standardize GDEX coordinates: drop singleton ``z``, shift lon to [-180, 180], sort."""
    if "z" in ds.dims:
        ds = ds.squeeze("z", drop=True)
    return ds.pipe(lon_to_180, lon_name="lon").sortby(["lat", "lon"]).drop_encoding()


def finalize_gdex_metadata(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    """Stamp source/provenance global attrs for the covered period."""
    ds.attrs.update(
        {
            "valid_time_start": f"{start_year}-01-01",
            "valid_time_stop": f"{end_year}-12-31",
            "source": "Princeton Global Forcing (Sheffield et al. 2006), 0.25deg 3-hourly resampled to daily",
            "url": "https://gdex.ucar.edu/datasets/d314000/",
        }
    )
    return ds


def process_gdex(
    start_year: int = 1950,
    end_year: int = 2008,
    use_virtual: bool = False,
    output_uri: str = OUTPUT_URI,
    dry_run: bool = False,
    dry_run_output: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Materialize GDEX ``tas`` (PGF 0.25deg, resampled to daily) to the icechunk store.

    Parameters
    ----------
    start_year : int, optional
        First year to include. Default is 1950.
    end_year : int, optional
        Last year to include. Default is 2008.
    use_virtual : bool, optional
        If True, read from the virtual icechunk store. Otherwise, stream
        per-year NetCDF files directly from OSDF and resume from the last
        committed year. Default is False.
    output_uri : str, optional
        Destination ``s3://`` URI or local path. Defaults to ``OUTPUT_URI``.
    dry_run : bool, optional
        If True, process a small sample and display the result without
        writing to ``output_uri``. If ``dry_run_output`` is also given, the
        sample is written there. Default is False.
    dry_run_output : str or None, optional
        Local path or ``s3://`` URI to write the dry-run sample to. Ignored
        when ``dry_run`` is False. Default is None.
    commit_message : str or None, optional
        Icechunk commit message. Defaults to ``"GDEX: tas"`` (or
        ``"GDEX: tas {year}"`` per year in direct mode) when not set.
    """
    if use_virtual:
        log.info("Loading tas from virtual icechunk (%d-%d)", start_year, end_year)
        raw = load_gdex_virtual().resample(time="D").mean()
        raw = raw.sel(time=slice(str(start_year), str(end_year)))
        if dry_run:
            raw = raw.isel(time=slice(0, _DRY_RUN_STEPS))
        ds = (
            raw.pipe(preprocess_gdex)
            .pipe(add_cf_bounds)
            .pipe(update_variable_attrs, VAR_SPECS)
            .pipe(finalize_gdex_metadata, start_year, end_year)
        )

        if dry_run:
            _display_dry_run_result(ds, "tas", store=dry_run_output)
            if dry_run_output is not None:
                log.info("Writing dry-run sample for tas to %s", dry_run_output)
                repo, session = _init_repo_from_uri(dry_run_output)
                write_mode = determine_write_mode(repo)
                encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
                write_dataset_to_icechunk(
                    ds,
                    session,
                    encoding=encoding,
                    shards=OUTPUT_SHARDS,
                    commit_message=f"dry-run: {commit_message or 'tas'}",
                    write_mode=write_mode,
                    repo=repo,
                )
                log.info("dry-run write done: tas -> %s", dry_run_output)
                read_session = repo.readonly_session("main")
                written = xr.open_dataset(read_session.store, engine="zarr", chunks="auto")
                console.print(written)
            return

        repo, session = _init_repo_from_uri(output_uri)
        write_mode = determine_write_mode(repo)
        log.info("Writing tas to %s (mode=%s)", output_uri, write_mode)
        encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
        write_dataset_to_icechunk(
            ds,
            session,
            encoding=encoding,
            shards=OUTPUT_SHARDS,
            commit_message=commit_message or "GDEX: tas",
            write_mode=write_mode,
            repo=repo,
        )
        log.info("Done: tas")
        return

    # --- Direct mode: stream per-year NetCDF files from OSDF, resuming from history ---
    from icechunk.xarray import to_icechunk

    repo, session = _init_repo_from_uri(output_uri)
    write_mode = determine_write_mode(repo)
    store = from_url(OSDF_BASE)

    if dry_run:
        raw = _open_year(store, start_year)[["tas"]].resample(time="D").mean()
        raw = raw.isel(time=slice(0, _DRY_RUN_STEPS))
        ds = (
            raw.pipe(preprocess_gdex)
            .pipe(add_cf_bounds)
            .pipe(update_variable_attrs, VAR_SPECS)
            .pipe(finalize_gdex_metadata, start_year, start_year)
        )
        _display_dry_run_result(ds, "tas", store=dry_run_output)
        if dry_run_output is not None:
            log.info("Writing dry-run sample for tas to %s", dry_run_output)
            dr_repo, dr_session = _init_repo_from_uri(dry_run_output)
            dr_write_mode = determine_write_mode(dr_repo)
            encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
            write_dataset_to_icechunk(
                ds,
                dr_session,
                encoding=encoding,
                shards=OUTPUT_SHARDS,
                commit_message=f"dry-run: {commit_message or 'tas'}",
                write_mode=dr_write_mode,
                repo=dr_repo,
            )
            log.info("dry-run write done: tas -> %s", dry_run_output)
            read_session = dr_repo.readonly_session("main")
            written = xr.open_dataset(read_session.store, engine="zarr", chunks="auto")
            console.print(written)
        return

    commits = [c.message for c in repo.ancestry(branch="main")]
    year_commits = [c for c in commits if c.startswith("GDEX: tas ")]
    resume_year = int(year_commits[0].split()[-1]) + 1 if year_commits else start_year
    if resume_year > start_year:
        log.info("resuming from %d (last committed: %d)", resume_year, resume_year - 1)

    encoding = None
    for year in range(resume_year, end_year + 1):
        log.info("processing tas %d", year)
        for attempt in range(3):
            try:
                raw = _open_year(store, year)[["tas"]].resample(time="D").mean()
                ds = raw.pipe(preprocess_gdex)

                if write_mode == "w" and year == resume_year:
                    ds = (
                        ds.pipe(add_cf_bounds)
                        .pipe(update_variable_attrs, VAR_SPECS)
                        .pipe(finalize_gdex_metadata, start_year, end_year)
                    )
                    encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
                    write_dataset_to_icechunk(
                        ds,
                        session,
                        encoding=encoding,
                        shards=OUTPUT_SHARDS,
                        commit_message=commit_message or f"GDEX: tas {year}",
                        write_mode=write_mode,
                    )
                else:
                    ds = (
                        ds.pipe(add_cf_bounds, coord_names=["time"])
                        .pipe(update_variable_attrs, VAR_SPECS)
                        .chunk(OUTPUT_SHARDS)
                    )
                    to_icechunk(ds, session, append_dim="time", align_chunks=True)
                    session.commit(commit_message or f"GDEX: tas {year}")
                break
            except Exception as e:
                if attempt == 2:
                    raise
                log.warning("attempt %d failed for %d: %s, retrying", attempt + 1, year, e)
                session = repo.writable_session("main")

        session = repo.writable_session("main")
    log.info("Done: tas")


# --- CLI ---

app = typer.Typer()


@app.command()
def virtualize(
    start_year: int = typer.Option(min(GDEX_0P25_YEARS), help="First year to virtualize."),
    end_year: int = typer.Option(max(GDEX_0P25_YEARS), help="Last year to virtualize."),
) -> None:
    """Virtualize GDEX 0.25deg 3-hourly NetCDF files (OSDF) into a virtual icechunk store."""
    osdf_store = from_url(OSDF_BASE)
    registry = ObjectStoreRegistry({OSDF_BASE: osdf_store})
    parser = HDFParser()

    years = range(start_year, end_year + 1)
    urls = make_osdf_urls(years=years)
    log.info("virtualizing %d files (tas x %d years)", len(urls), len(list(years)))

    virt_ds = virtualize_and_combine(
        urls=urls,
        registry=registry,
        parser=parser,
        loadable_variables=["lat", "lon", "time"],
    )
    repo_config = icechunk.RepositoryConfig.default()
    repo_config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(OSDF_BASE + "/", store=icechunk.http_store())
    )
    storage = icechunk.s3_storage(bucket=BUCKET, prefix=VIRTUAL_PREFIX, region="us-west-2")
    repo = icechunk.Repository.open_or_create(storage, repo_config)
    session = repo.writable_session("main")
    virt_ds.vz.to_icechunk(session.store)
    session.commit(f"GDEX: virtualized 0.25deg 3 hourly ({start_year}-{end_year})")
    repo.save_config()
    log.info("written to s3://%s/%s", BUCKET, VIRTUAL_PREFIX)


@app.command()
def process(
    start_year: int = typer.Option(1950, help="First year to include."),
    end_year: int = typer.Option(2008, help="Last year to include."),
    use_virtual: bool = typer.Option(
        False, "--use-virtual/--direct", help="Read from the virtual icechunk store."
    ),
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
        help="Icechunk commit message. Defaults to 'GDEX: tas' (or per-year in direct mode).",
    ),
) -> None:
    """Materialize GDEX tas to the unified icechunk store."""
    process_gdex(
        start_year=start_year,
        end_year=end_year,
        use_virtual=use_virtual,
        output_uri=output,
        dry_run=dry_run,
        dry_run_output=dry_run_output,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    app()
