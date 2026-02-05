import subprocess
from dataclasses import dataclass, field

import boto3
import click
import icechunk
import obstore as obs
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import from_url
from virtualizarr.parsers import HDFParser

from srm import catalog
from srm.config import (
    init_repo,
    setup_cluster,
    setup_local_client,
)
from srm.input_data.etl_config import BaseETLConfig
from srm.input_data.etl_utils import (
    add_cf_bounds,
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    trim_negative_precipitation,
    update_variable_attrs,
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 64})


@dataclass
class BaseUKESM_Config(BaseETLConfig):
    s3_input_prefix: str = ""

    ensemble_members: list = field(default_factory=lambda: ["001", "002", "003"])

    drop_variables: list = field(
        default_factory=lambda: [
            "time_bnds",
            "lat_bnds",
            "lon_bnds",
            "height",
        ]
    )

    aws_creds: dict = field(default_factory=dict)

    # Cluster config for virtualization (metadata-only, needs many small workers)
    virtualize_cluster: dict = field(
        default_factory=lambda: {
            "n_workers": [2, 50],
            "worker_vm_types": ["r8g.2xlarge"],
            "scheduler_vm_types": "c8g.2xlarge",
        }
    )

    # Cluster config for processing (data-heavy, needs fewer large workers)
    process_cluster: dict = field(
        default_factory=lambda: {
            "n_workers": [1, 50],
            "worker_vm_types": ["r8g.2xlarge"],
            "scheduler_vm_types": "c8g.2xlarge",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        sesh = boto3.Session()
        creds = sesh.get_credentials()
        self.aws_creds = {
            "aws_access_key_id": creds.access_key,
            "aws_secret_access_key": creds.secret_key,
        }


@dataclass
class UKESM_SSP245_Config(BaseUKESM_Config):
    scenario: str = "SSP245"
    catalog_key: str = "UKESM-SSP245-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/transfer/SSP2-4.5/"


@dataclass
class UKESM_G6_1p5K_Config(BaseUKESM_Config):
    scenario: str = "G6-1.5K"
    catalog_key: str = "UKESM-G6-1.5K-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/transfer/G6-1.5K"


@dataclass
class UKESM_SSP245_SFCWIND_Config(BaseUKESM_Config):
    scenario: str = "SSP245"
    catalog_key: str = "UKESM-SSP245-SFCWIND-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/transfer/SSP2-4.5/"


@dataclass
class UKESM_G6_1p5K_SFCWIND_Config(BaseUKESM_Config):
    scenario: str = "G6-1.5K"
    catalog_key: str = "UKESM-G6-1.5K-SFCWIND-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/transfer/G6-1.5K"


@dataclass
class UKESM_Historical_Config(BaseUKESM_Config):
    scenario: str = "historical"
    catalog_key: str = "UKESM-historical-icechunk"
    time_range: str = "1850-1949"
    source_base_url: str = (
        "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical"
    )
    s3_input_prefix: str = "input/tensor/UKESM/netcdf/historical"

    ensemble_members: list = field(
        default_factory=lambda: [
            "r1i1p1f2",
            "r2i1p1f2",
            "r3i1p1f2",
            "r4i1p1f2",
            "r5i1p1f3",
            "r6i1p1f3",
            "r7i1p1f3",
            "r8i1p1f2",
            "r9i1p1f2",
            "r10i1p1f2",
        ]
    )


SCENARIO_CONFIG_MAP = {
    "SSP245": UKESM_SSP245_Config,
    "G6-1.5K": UKESM_G6_1p5K_Config,
    "SSP245-SFCWIND": UKESM_SSP245_SFCWIND_Config,
    "G6-1.5K-SFCWIND": UKESM_G6_1p5K_SFCWIND_Config,
    "historical": UKESM_Historical_Config,
}


def _fetch_ukesm_historical(variables: list[str], config: UKESM_Historical_Config) -> None:
    import warnings

    warnings.warn(
        "This fetches raw netcdf files from CEDA and moves them to s3. You must have rclone installed and configured."
    )
    time_slices = ["18500101-19491230", "19500101-20141230"]
    ensemble_dates = {
        "r1i1p1f2": "d20190627",
        "r2i1p1f2": "d20190708",
        "r3i1p1f2": "d20190708",
        "r4i1p1f2": "d20190708",
        "r5i1p1f3": "d20191115",
        "r6i1p1f3": "d20191113",
        "r7i1p1f3": "d20191011",
        "r8i1p1f2": "d20190708",
        "r9i1p1f2": "d20191015",
        "r10i1p1f2": "d20191213",
    }

    urls = []
    for ens in config.ensemble_members:
        date_str = ensemble_dates.get(ens)

        for var in variables:
            for t_range in time_slices:
                folder_path = f"{config.source_base_url}/{ens}/day/{var}/gn/files/{date_str}"
                file_name = f"{var}_day_UKESM1-0-LL_historical_{ens}_gn_{t_range}.nc"
                urls.append(f"{folder_path}/{file_name}")

    urls_file = "UKESM-historical-urls.txt"
    with open(urls_file, "w") as f:
        f.write("\n".join(urls))

    target_remote = f"aws:{config.s3_bucket}/{config.s3_input_prefix}/"
    command = [
        "rclone",
        "copyurl",
        "--urls",
        urls_file,
        target_remote,
        "--progress",
        "--no-clobber",
        "--transfers",
        "4",
        "--s3-upload-concurrency",
        "8",
        "--s3-chunk-size",
        "64M",
        "--buffer-size",
        "32M",
        "--s3-no-check-bucket",
        "--disable-http2",
        "--retries",
        "3",
        "--low-level-retries",
        "10",
    ]
    subprocess.run(command)


def _get_netcdf_urls(config: BaseUKESM_Config, variables: list[str]) -> list[str]:
    store = from_url(f"s3://{config.s3_bucket}", region="us-west-2", **config.aws_creds)

    stream = obs.list_with_delimiter(store, prefix=config.s3_input_prefix, return_arrow=True)
    netcdf_list = list(stream["objects"]["path"].to_numpy())
    filtered_urls = [
        f"s3://{config.s3_bucket}/{path}"
        for path in netcdf_list
        if path.endswith(".nc") and any(f"{var}_".lower() in path.lower() for var in variables)
    ]
    return filtered_urls


def _preprocess_ensemble(ds: xr.Dataset, url: str = None) -> xr.Dataset:
    """
    Extract ensemble member from URL and expand dims.

    url parameter is required when using VirtualiZarr's open_virtual_mfdataset
    since the dataset encoding doesn't contain the source path.
    """
    if url is None:
        raise ValueError("url parameter is required to determine ensemble member")

    ensemble = url.split(".nc")[0].split("_gn")[0].split("_")[-1]
    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    return ds


def _preprocess_ukesm(
    ds: xr.Dataset,
    config: BaseUKESM_Config,
    subset: bool = False,
) -> xr.Dataset:
    # Drop any duplicate times (do this first while still lazy)
    ds = ds.drop_duplicates(dim="time", keep="first")

    ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False, align_on="date")
    ds = ds.drop_encoding()
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])

    if isinstance(config, UKESM_Historical_Config) and hasattr(config, "time_range"):
        start_year, end_year = config.time_range.split("-")
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    ds = trim_negative_precipitation(ds)
    if subset:
        ds = ds.isel(time=slice(0, 365))
    return ds


def _update_attrs(ds: xr.Dataset, var_specs: dict, config: BaseUKESM_Config) -> xr.Dataset:
    ds = update_variable_attrs(ds, var_specs)
    ds = add_cf_bounds(ds)

    attrs = {
        "scenario": config.scenario,
        "model": "UKESM1-0-LL",
        "Conventions": "CF-1.8",
    }

    if isinstance(config, UKESM_Historical_Config):
        attrs["time_range"] = config.time_range

    ds.attrs.update(attrs)

    return ds


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--variable",
    multiple=True,
    required=True,
    help="UKESM variables to fetch",
)
@click.option(
    "--scenario",
    type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())),
    required=True,
    help="which UKESM scenario to fetch",
)
def fetch(variable, scenario):
    config_class = SCENARIO_CONFIG_MAP.get(scenario)
    if config_class is None:
        raise ValueError(f"unknown scenario: {scenario}")

    config = config_class()

    if isinstance(config, UKESM_Historical_Config):
        _fetch_ukesm_historical(list(variable), config)
    else:
        raise ValueError(f"fetch not implemented for scenario: {scenario}")


@click.command()
@click.option(
    "--variable",
    multiple=True,
    required=True,
    help="UKESM variables to virtualize",
)
@click.option(
    "--scenario",
    type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())),
    required=True,
    help="which UKESM scenario to virtualize",
)
@click.option("--coiled/--local", default=False)
def virtualize(variable, scenario, coiled):
    """Stage 1: Virtualize netcdf files and write virtual references to icechunk"""
    from cloudpathlib import S3Path
    from obspec_utils.readers import BufferedStoreReader
    from obspec_utils.wrappers import CachingReadableStore, SplittingReadableStore

    config_class = SCENARIO_CONFIG_MAP.get(scenario)
    if config_class is None:
        raise ValueError(f"unknown scenario: {scenario}")

    config = config_class()

    if coiled:
        from srm.config import ClusterConfig

        virt_cluster_config = ClusterConfig(**config.virtualize_cluster)
        client = setup_cluster(virt_cluster_config)
    else:
        client = setup_local_client()

    ukesm_cat = catalog.get(config.catalog_key)

    virtual_path = ukesm_cat.virtual_path
    if not virtual_path:
        raise ValueError(f"No virtual_path defined for {config.catalog_key}")

    virtual_s3_path = S3Path(virtual_path)
    virtual_bucket = virtual_s3_path.bucket
    virtual_prefix = str(virtual_s3_path.key)

    try:
        # repo_config = icechunk.RepositoryConfig.default()
        # repo_config.set_virtual_chunk_container(
        #     icechunk.VirtualChunkContainer(
        #         f"s3://{config.s3_bucket}/",
        #         store=icechunk.s3_store(region="us-west-2"),
        #     ),
        # )

        # repo = icechunk.Repository.open_or_create(storage, repo_config)

        for var in list(variable):
            repo_config = icechunk.RepositoryConfig.default()
            repo_config.set_virtual_chunk_container(
                icechunk.VirtualChunkContainer(
                    "s3://carbonplan-srm/",
                    store=icechunk.s3_store(region="us-west-2"),
                ),
            )
            storage = icechunk.s3_storage(
                bucket=virtual_bucket,
                prefix=virtual_prefix,
                region="us-west-2",
            )
            repo = icechunk.Repository.open_or_create(storage, repo_config)
            session = repo.writable_session("main")
            # write_mode = determine_write_mode(repo)

            netcdf_urls = _get_netcdf_urls(config, [var])

            base_store = from_url(
                f"s3://{config.s3_bucket}", region="us-west-2", **config.aws_creds
            )

            splitting_store = SplittingReadableStore(base_store)

            max_size = 512 * 1024 * 1024
            caching_store = CachingReadableStore(splitting_store, max_size=max_size)

            registry = ObjectStoreRegistry({f"s3://{config.s3_bucket}": caching_store})
            parser = HDFParser(
                drop_variables=config.drop_variables, reader_factory=BufferedStoreReader
            )

            ds = virtualize_and_combine(
                urls=netcdf_urls,
                registry=registry,
                parser=parser,
                loadable_variables=["lat", "lon", "time"],
                preprocess_fn=_preprocess_ensemble,
            )

            ds.vz.to_icechunk(session.store, group=var)
            session.commit(f"{scenario}: {var}")
            print(f"var finished:{var}")

            repo.save_config()

    finally:
        client.shutdown()


@click.command()
@click.option(
    "--variable",
    multiple=True,
    required=True,
    help="UKESM variables to process",
)
@click.option(
    "--scenario",
    type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())),
    required=True,
    help="which UKESM scenario to process",
)
@click.option("--coiled/--local", default=False)
@click.option("--subset/--no-subset", default=False)
def process(variable, scenario, coiled, subset):
    """Stage 2: Read virtual icechunk, postprocess, rechunk, and write materialized data"""
    from cloudpathlib import S3Path

    config_class = SCENARIO_CONFIG_MAP.get(scenario)
    if config_class is None:
        raise ValueError(f"unknown scenario: {scenario}")

    config = config_class()

    if coiled:
        # Use process-specific cluster config (fewer large workers)
        from srm.config import ClusterConfig

        proc_cluster_config = ClusterConfig(**config.process_cluster)
        client = setup_cluster(proc_cluster_config)
    else:
        client = setup_local_client()

    ukesm_cat = catalog.get(config.catalog_key)
    var_specs = get_var_specs(ukesm_cat)

    # Extract bucket and prefix from virtual_path using cloudpathlib
    virtual_path = ukesm_cat.virtual_path
    if not virtual_path:
        raise ValueError(f"No virtual_path defined for {config.catalog_key}")

    virtual_s3_path = S3Path(virtual_path)
    virtual_bucket = virtual_s3_path.bucket
    virtual_prefix = str(virtual_s3_path.key)

    try:
        for var in list(variable):
            # Read from virtual icechunk store
            virtual_repo, virtual_session = init_repo(virtual_bucket, virtual_prefix, readonly=True)
            virtual_store = virtual_session.store()
            ds = xr.open_zarr(virtual_store, consolidated=False, zarr_format=3)
            ds = ds[[var]]

            # Postprocess
            ds = _preprocess_ukesm(ds, config, subset=subset)
            print(ds)
            ds = _update_attrs(ds, var_specs, config)

            # Write to final icechunk with proper chunking
            repo, session = init_repo(ukesm_cat.bucket, ukesm_cat.prefix, readonly=False)
            write_mode = determine_write_mode(repo)

            encoding = build_encoding_dict(ds, config.encoding["chunks"], config.encoding["shards"])

            write_dataset_to_icechunk(
                ds,
                session,
                encoding=encoding,
                shards=config.encoding["shards"],
                commit_message=f"{scenario}: {var}",
                write_mode=write_mode,
            )

    finally:
        client.shutdown()


cli.add_command(fetch)
cli.add_command(virtualize)
cli.add_command(process)


if __name__ == "__main__":
    cli()
