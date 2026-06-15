# COILED vm-type r8g.4xlarge
# COILED region us-west-2

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

from srm.config import init_repo
from srm.input_data.etl_utils import (
    add_cf_bounds,
    build_encoding_dict,
    determine_write_mode,
    setup_logging,
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 128})
dask.config.set(scheduler="threads")

setup_logging()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OSDF_BASE = "https://osdf-director.osg-htc.org"
OSDF_PATH_PREFIX = "/ncar/gdex/d314000/0.25deg/3hrly"
GDEX_0P25_YEARS = range(1948, 2011)

VIRTUAL_S3_PATH = "s3://carbonplan-srm/input/tensor/NCAR/GDEX-GMF-virtual.icechunk"

OUTPUT_BUCKET = "carbonplan-srm"
OUTPUT_PREFIX = "input/processed/gdex-gmf.icechunk"
OUTPUT_CHUNKS: dict[str, int] = {"time": 1, "lat": 720, "lon": 1440}
OUTPUT_SHARDS: dict[str, int] = {"time": 30, "lat": 720, "lon": 1440}

DEFAULT_START_YEAR = 1950
DEFAULT_END_YEAR = 2008


def make_osdf_urls(
    years: range | list[int] = GDEX_0P25_YEARS,
    base_url: str = OSDF_BASE,
) -> list[str]:
    return [
        f"{base_url}{OSDF_PATH_PREFIX}/{year}/tas_0p25_3hourly_{year}-{year}.nc" for year in years
    ]


def _open_virtual_store(readonly: bool = True) -> tuple[icechunk.Repository, icechunk.Session]:
    from cloudpathlib import CloudPath

    p = CloudPath(VIRTUAL_S3_PATH)
    storage = icechunk.s3_storage(bucket=p.bucket, prefix=p.key, region="us-west-2")

    repo_config = icechunk.RepositoryConfig.default()
    repo_config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(OSDF_BASE + "/", store=icechunk.http_store())
    )

    if readonly:
        repo = icechunk.Repository.open(storage, config=repo_config)
        return repo, repo.readonly_session("main")
    else:
        repo = icechunk.Repository.open_or_create(storage, repo_config)
        return repo, repo.writable_session("main")


def _preprocess_gdex(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    ds = ds.sel(time=slice(str(start_year), str(end_year)))
    if "z" in ds.dims:
        ds = ds.squeeze("z", drop=True)
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])
    bounds_vars = [v for v in ds.data_vars if v.endswith("_bounds") or v.endswith("_bnds")]
    if bounds_vars:
        ds = ds.set_coords(bounds_vars)
    return ds.drop_encoding()


def _update_attrs(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    ds = add_cf_bounds(ds)
    ds.attrs.update(
        {
            "valid_time_start": f"{start_year}-01-01",
            "valid_time_stop": f"{end_year}-12-31",
            "source": "Princeton Global Forcing (Sheffield et al. 2006), 0.25deg 3-hourly resampled to daily",
            "url": "https://gdex.ucar.edu/datasets/d314000/",
        }
    )
    return ds


def virtualize_pipeline(
    years: range | list[int] = GDEX_0P25_YEARS,
) -> None:
    osdf_store = from_url(OSDF_BASE)
    registry = ObjectStoreRegistry({OSDF_BASE: osdf_store})
    parser = HDFParser()

    urls = make_osdf_urls(years=years)
    log.info("virtualizing %d files (tas × %d years)", len(urls), len(list(years)))

    virt_ds = virtualize_and_combine(
        urls=urls,
        registry=registry,
        parser=parser,
        loadable_variables=["lat", "lon", "time"],
    )

    _, session = _open_virtual_store(readonly=False)
    virt_ds.vz.to_icechunk(session.store)
    session.commit("GDEX: virtualized 0.25deg 3 hourly (1948-2010)")
    log.info("written to %s", VIRTUAL_S3_PATH)


def _open_year(store: object, year: int) -> xr.Dataset:
    url = make_osdf_urls(years=[year])[0]
    path = url.removeprefix(OSDF_BASE)
    return xr.open_dataset(BlockStoreReader(store, path), engine="h5netcdf", chunks="auto")


def process_pipeline(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    use_virtual: bool = False,
) -> None:
    from icechunk.xarray import to_icechunk

    repo, session = init_repo(OUTPUT_BUCKET, OUTPUT_PREFIX, readonly=False)
    write_mode = determine_write_mode(repo)

    if use_virtual:
        log.info("processing tas from virtual icechunk")
        _, virt_session = _open_virtual_store(readonly=True)
        ds = xr.open_dataset(
            virt_session.store,
            engine="zarr",
            consolidated=False,
            zarr_format=3,
            chunks="auto",
        )[["tas"]]
        ds = ds.resample(time="D").mean()
        ds = _preprocess_gdex(ds, start_year, end_year)
        ds = _update_attrs(ds, start_year, end_year)
        encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
        write_dataset_to_icechunk(
            ds,
            session,
            encoding=encoding,
            shards=OUTPUT_SHARDS,
            commit_message="GDEX: tas",
            write_mode=write_mode,
        )
    else:
        # resume: find last committed year from store history
        commits = [c.message for c in repo.ancestry(branch="main")]
        year_commits = [c for c in commits if c.startswith("GDEX: tas ")]
        resume_year = int(year_commits[0].split()[-1]) + 1 if year_commits else start_year
        if resume_year > start_year:
            log.info("resuming from %d (last committed: %d)", resume_year, resume_year - 1)

        store = from_url(OSDF_BASE)
        encoding = None
        for year in range(resume_year, end_year + 1):
            log.info("processing tas %d", year)

            for attempt in range(3):
                try:
                    ds = _open_year(store, year)[["tas"]]
                    ds = ds.resample(time="D").mean()
                    ds = _preprocess_gdex(ds, year, year)

                    if write_mode == "w" and year == resume_year:
                        ds = _update_attrs(ds, start_year, end_year)
                        encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
                        write_dataset_to_icechunk(
                            ds,
                            session,
                            encoding=encoding,
                            shards=OUTPUT_SHARDS,
                            commit_message=f"GDEX: tas {year}",
                            write_mode=write_mode,
                        )
                    else:
                        ds = add_cf_bounds(ds, coord_names=["time"])
                        ds = ds.chunk(OUTPUT_SHARDS)
                        to_icechunk(ds, session, append_dim="time", align_chunks=True)
                        session.commit(f"GDEX: tas {year}")
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    log.warning("attempt %d failed for %d: %s, retrying", attempt + 1, year, e)
                    session = repo.writable_session("main")

            session = repo.writable_session("main")


app = typer.Typer()


@app.command()
def virtualize(
    start_year: int = typer.Option(min(GDEX_0P25_YEARS), "--start-year"),
    end_year: int = typer.Option(max(GDEX_0P25_YEARS), "--end-year"),
) -> None:
    """Virtualize GDEX 0.25deg NetCDF3 files into icechunk via OSDF."""
    virtualize_pipeline(years=range(start_year, end_year + 1))


@app.command()
def process(
    start_year: int = typer.Option(DEFAULT_START_YEAR, "--start-year"),
    end_year: int = typer.Option(DEFAULT_END_YEAR, "--end-year"),
    use_virtual: bool = typer.Option(
        False, "--use-virtual/--direct", help="Read from virtual icechunk instead of direct OSDF"
    ),
) -> None:
    """Materialize GDEX tas from virtual icechunk (or direct OSDF HTTP) to icechunk."""
    process_pipeline(start_year=start_year, end_year=end_year, use_virtual=use_virtual)


if __name__ == "__main__":
    app()
