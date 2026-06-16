# COILED vm-type r8g.4xlarge
# COILED region us-west-2
# COILED tag project=SRM

"""ETL script for creating virtual zarr stores of a subset of the NASA-NEX dataset, specifically:
CESM2-WACCM, [historical, SSP245], r3i1p1f1, tas.
"""

import logging

import dask
import icechunk
import typer
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import from_url
from virtualizarr import open_virtual_mfdataset
from virtualizarr.parsers import HDFParser

from srm.input_data.etl_utils import setup_logging

dask.config.set(scheduler="threads")

setup_logging()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_BUCKET = "s3://nex-gddp-cmip6"
OUTPUT_BUCKET = "carbonplan-srm"
OUTPUT_PREFIX = "input/processed/nasa-nex"

MODEL = "CESM2-WACCM"
ENSEMBLE_MEMBER = "r3i1p1f1"
VARIABLE = "tas"

ALL_SCENARIOS = ["historical", "SSP245"]

# Year ranges per scenario (inclusive)
YEAR_RANGE: dict[str, tuple[int, int]] = {
    "historical": (1950, 2014),
    "SSP245": (2015, 2100),
}

# Scenario name as used in the NASA-NEX S3 path
_S3_SCENARIO_NAME: dict[str, str] = {
    "historical": "historical",
    "SSP245": "ssp245",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen_urls(scenario: str) -> list[str]:
    """Generate per-year NASA-NEX S3 URLs for the given scenario."""
    s3_scenario = _S3_SCENARIO_NAME[scenario]
    start, end = YEAR_RANGE[scenario]
    return [
        f"{SOURCE_BUCKET}/NEX-GDDP-CMIP6/{MODEL}/{s3_scenario}/{ENSEMBLE_MEMBER}"
        f"/{VARIABLE}/{VARIABLE}_day_{MODEL}_{s3_scenario}_{ENSEMBLE_MEMBER}_gn_{year}_v2.0.nc"
        for year in range(start, end + 1)
    ]


def _virtualize(urls: list[str]) -> xr.Dataset:
    """Virtualize netCDF files from NASA-NEX into a single xarray Dataset."""
    store = from_url(SOURCE_BUCKET, region="us-west-2", skip_signature=True)
    registry = ObjectStoreRegistry({SOURCE_BUCKET: store})
    parser = HDFParser()

    vds = open_virtual_mfdataset(
        urls,
        registry=registry,
        parser=parser,
        parallel="dask",
        combine="by_coords",
        combine_attrs="drop_conflicts",
    )
    return vds.expand_dims({"ensemble_member": [ENSEMBLE_MEMBER]})


def _write(vds: xr.Dataset, scenario: str) -> None:
    """Write the virtualized dataset to the icechunk store on S3."""
    s3_scenario = _S3_SCENARIO_NAME[scenario]
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            f"{SOURCE_BUCKET}/",
            store=icechunk.s3_store(region="us-west-2"),
        ),
    )
    storage = icechunk.s3_storage(
        bucket=OUTPUT_BUCKET,
        prefix=f"{OUTPUT_PREFIX}/{s3_scenario}/virtual.icechunk",
        from_env=True,
    )
    repo = icechunk.Repository.open_or_create(storage, config)
    session = repo.writable_session("main")

    vds.vz.to_icechunk(session.store)
    snapshot_id = session.commit(f"nasa-nex: {VARIABLE} {scenario}")
    log.info("committed snapshot: %s", snapshot_id)
    repo.save_config()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.command()
def process(
    scenario: str = typer.Option(
        ...,
        "--scenario",
        help=f"Scenario to virtualize. Choices: {ALL_SCENARIOS}",
    ),
) -> None:
    """Virtualize NASA-NEX NetCDF files and write a virtual icechunk store."""
    if scenario not in ALL_SCENARIOS:
        raise typer.BadParameter(f"scenario must be one of {ALL_SCENARIOS}, got {scenario!r}")

    log.info("scenario=%s variable=%s start", scenario, VARIABLE)
    urls = _gen_urls(scenario)
    log.info("scenario=%s found %d URLs", scenario, len(urls))

    vds = _virtualize(urls)
    _write(vds, scenario)
    log.info("scenario=%s done", scenario)


if __name__ == "__main__":
    app()
