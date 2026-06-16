from dataclasses import dataclass, field
from typing import Any

import click
import icechunk
import xarray as xr
import zarr
from obspec_utils.readers import BlockStoreReader
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import from_url
from virtualizarr.parsers import HDFParser

from srm import catalog
from srm.config import ClusterConfig, init_repo, setup_cluster, setup_local_client
from srm.input_data.etl_utils import (
    add_cf_bounds,
    build_encoding_dict,
    determine_write_mode,
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 128})
OSDF_BASE = "https://osdf-director.osg-htc.org"
OSDF_PATH_PREFIX = "/ncar/gdex/d314000/0.25deg/3hrly"
GDEX_0P25_YEARS = range(1948, 2011)


VIRTUAL_S3_PATH = "s3://carbonplan-srm/input/tensor/NCAR/GDEX-GMF-virtual.icechunk"
MATERIALIZED_CATALOG_KEY = "GDEX-GMF"


@dataclass
class GDEXConfig:
    start_year: int = 1950
    end_year: int = 2008

    virtualize_cluster: dict = field(
        default_factory=lambda: {
            "n_workers": [2, 36],
            "worker_vm_types": ["c8g.2xlarge"],
            "scheduler_vm_types": "c8g.xlarge",
        }
    )
    process_cluster: dict = field(
        default_factory=lambda: {
            "n_workers": [2, 36],
            "worker_vm_types": ["c8g.2xlarge"],
            "scheduler_vm_types": "c8g.xlarge",
        }
    )

    def __post_init__(self):
        mat_cat = catalog.get(MATERIALIZED_CATALOG_KEY)
        self.bucket = mat_cat.bucket
        self.prefix = mat_cat.prefix
        self.encoding = {
            "chunks": mat_cat.expected_chunks,
            "shards": mat_cat.expected_shards,
        }


def make_osdf_urls(
    years: range | list[int] = GDEX_0P25_YEARS,
    base_url: str = OSDF_BASE,
) -> list[str]:
    return [
        f"{base_url}{OSDF_PATH_PREFIX}/{year}/tas_0p25_3hourly_{year}-{year}.nc" for year in years
    ]


def _open_virtual_store(readonly: bool = True) -> tuple[icechunk.Repository, Any]:
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


def _update_attrs(ds: xr.Dataset, config: GDEXConfig) -> xr.Dataset:
    ds = add_cf_bounds(ds)
    ds.attrs.update(
        {
            "valid_time_start": f"{config.start_year}-01-01",
            "valid_time_stop": f"{config.end_year}-12-31",
            "source": "Princeton Global Forcing (Sheffield et al. 2006), 0.25deg 3-hourly resampled to daily",
            "url": "https://gdex.ucar.edu/datasets/d314000/",
        }
    )
    return ds


def virtualize_pipeline(
    years: range | list[int] = GDEX_0P25_YEARS,
    use_coiled: bool = False,
    verbose: bool = True,
) -> None:
    client = (
        setup_cluster(ClusterConfig(**GDEXConfig().virtualize_cluster))
        if use_coiled
        else setup_local_client()
    )

    try:
        osdf_store = from_url(OSDF_BASE)
        registry = ObjectStoreRegistry({OSDF_BASE: osdf_store})
        parser = HDFParser()

        urls = make_osdf_urls(years=years)
        if verbose:
            print(f"virtualizing {len(urls)} files (tas × {len(list(years))} years)")

        virt_ds = virtualize_and_combine(
            urls=urls,
            registry=registry,
            parser=parser,
            loadable_variables=["lat", "lon", "time"],
        )

        _, session = _open_virtual_store(readonly=False)
        virt_ds.vz.to_icechunk(session.store)
        session.commit("GDEX: virtualized 0.25deg 3 hourly (1948-2010)")

        if verbose:
            print(f"written to {VIRTUAL_S3_PATH}")
    finally:
        client.shutdown()


def _open_year(store: Any, year: int) -> xr.Dataset:
    url = make_osdf_urls(years=[year])[0]
    path = url.removeprefix(OSDF_BASE)
    return xr.open_dataset(BlockStoreReader(store, path), engine="h5netcdf", chunks="auto")


def process_pipeline(
    start_year: int = 1950,
    end_year: int = 2008,
    use_virtual: bool = False,
    use_coiled: bool = False,
    verbose: bool = True,
) -> None:
    from icechunk.xarray import to_icechunk

    config = GDEXConfig(start_year=start_year, end_year=end_year)
    client = (
        setup_cluster(ClusterConfig(**config.process_cluster))
        if use_coiled
        else setup_local_client()
    )

    try:
        repo, session = init_repo(config.bucket, config.prefix, readonly=False)
        write_mode = determine_write_mode(repo)

        if use_virtual:
            if verbose:
                print("processing tas from virtual icechunk")
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
            ds = _update_attrs(ds, config)
            encoding = build_encoding_dict(ds, config.encoding["chunks"], config.encoding["shards"])
            write_dataset_to_icechunk(
                ds,
                session,
                encoding=encoding,
                shards=config.encoding["shards"],
                commit_message="GDEX: tas",
                write_mode=write_mode,
            )
        else:
            # resume: find last committed year from store history
            commits = [c.message for c in repo.ancestry(branch="main")]
            year_commits = [c for c in commits if c.startswith("GDEX: tas ")]
            resume_year = int(year_commits[0].split()[-1]) + 1 if year_commits else start_year
            if resume_year > start_year and verbose:
                print(f"resuming from {resume_year} (last committed: {resume_year - 1})")

            store = from_url(OSDF_BASE)
            encoding = None
            for year in range(resume_year, end_year + 1):
                if verbose:
                    print(f"processing tas {year}")

                for attempt in range(3):
                    try:
                        ds = _open_year(store, year)[["tas"]]
                        ds = ds.resample(time="D").mean()
                        ds = _preprocess_gdex(ds, year, year)

                        if write_mode == "w" and year == resume_year:
                            ds = _update_attrs(ds, config)
                            encoding = build_encoding_dict(
                                ds, config.encoding["chunks"], config.encoding["shards"]
                            )
                            write_dataset_to_icechunk(
                                ds,
                                session,
                                encoding=encoding,
                                shards=config.encoding["shards"],
                                commit_message=f"GDEX: tas {year}",
                                write_mode=write_mode,
                            )
                        else:
                            ds = add_cf_bounds(ds, coord_names=["time"])
                            ds = ds.chunk(config.encoding["shards"])
                            to_icechunk(ds, session, append_dim="time", align_chunks=True)
                            session.commit(f"GDEX: tas {year}")
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        if verbose:
                            print(f"attempt {attempt + 1} failed for {year}: {e}, retrying")
                        session = repo.writable_session("main")

                session = repo.writable_session("main")
    finally:
        client.shutdown()


@click.group()
def cli():
    pass


@cli.command()
@click.option("--start-year", type=int, default=min(GDEX_0P25_YEARS))
@click.option("--end-year", type=int, default=max(GDEX_0P25_YEARS))
@click.option("--coiled/--local", default=False)
def virtualize(start_year, end_year, coiled):
    """Virtualize GDEX 0.25deg NetCDF3 files into icechunk via OSDF."""
    virtualize_pipeline(years=range(start_year, end_year + 1), use_coiled=coiled, verbose=True)


@cli.command()
@click.option("--start-year", type=int, default=1950)
@click.option("--end-year", type=int, default=2008)
@click.option("--use-virtual/--direct", default=False)
@click.option("--coiled/--local", default=False)
def process(start_year, end_year, use_virtual, coiled):
    """Materialize GDEX tas from virtual icechunk (or direct OSDF HTTP) to icechunk."""
    process_pipeline(
        start_year=start_year,
        end_year=end_year,
        use_virtual=use_virtual,
        use_coiled=coiled,
        verbose=True,
    )


if __name__ == "__main__":
    cli()
