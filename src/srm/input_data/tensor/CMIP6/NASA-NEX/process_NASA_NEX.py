import logging

import click
import icechunk
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import from_url
from virtualizarr import open_virtual_mfdataset
from virtualizarr.parsers import HDFParser

from srm.config import (
    setup_cluster,
    setup_local_client,
)


def gen_urls(start_year, end_year, scenario):
    """Generate URLs for given scenario"""
    return [
        f"s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/CESM2-WACCM/{scenario}/r3i1p1f1/tas/tas_day_CESM2-WACCM_{scenario}_r3i1p1f1_gn_{year}_v2.0.nc"
        for year in range(start_year, end_year + 1)
    ]


def virtualize(urls: list[str]) -> xr.Dataset:
    """Virtualize netCDF files into a xarray Dataset."""
    bucket = "s3://nex-gddp-cmip6"
    store = from_url(bucket, region="us-west-2", skip_signature=True)
    registry = ObjectStoreRegistry({bucket: store})
    parser = HDFParser()

    vds = open_virtual_mfdataset(
        urls,
        registry=registry,
        parser=parser,
        parallel="dask",
        combine="by_coords",
        combine_attrs="drop_conflicts",
    )
    return vds.expand_dims({"ensemble_member": ["r3i1p1f1"]})


def write(vds: xr.Dataset, scenario: str):
    """Write virtualized dataset to Icechunk storage."""
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "s3://nex-gddp-cmip6/",
            store=icechunk.s3_store(region="us-west-2"),
        ),
    )
    storage = icechunk.s3_storage(
        bucket="carbonplan-srm",
        prefix=f"input/tensor/nasa-nex/{scenario}/virtual.icechunk",
        from_env=True,
    )
    repo = icechunk.Repository.open_or_create(storage, config)
    session = repo.writable_session("main")

    vds.vz.to_icechunk(session.store)
    snapshot_id = session.commit("nasa-nex-tas")
    print(snapshot_id)
    repo.save_config()


def process_urls(urls: list[str], scenario: str):
    try:
        vds = virtualize(urls)
        write(vds, scenario)
    except Exception as e:
        logging.error(f"An error occurred while processing URLs {e}")
        return


@click.group()
def cli():
    pass


@cli.command()
@click.option("--scenario", type=click.Choice(["SSP245", "historical"]), required=True)
@click.option("--coiled/--local", default=False, help="use coiled")
def process(scenario: str, coiled: bool):
    """Virtualize netcdf files into virtual Icechunk dataset"""
    logging.info(f"Processing scenario: {scenario}")

    if scenario == "SSP245":
        urls = gen_urls(start_year=2015, end_year=2100, scenario="ssp245")
    elif scenario == "historical":
        urls = gen_urls(start_year=1950, end_year=2014, scenario="historical")

    setup_client(coiled)
    process_urls(urls, scenario)


def setup_client(coiled: bool):
    """Set up computation client."""
    from srm.config import ClusterConfig

    return setup_cluster(ClusterConfig()) if coiled else setup_local_client()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cli()
