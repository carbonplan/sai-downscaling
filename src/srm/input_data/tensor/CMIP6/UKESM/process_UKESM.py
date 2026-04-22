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
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    load_dtr_from_store,
    trim_negative_precipitation,
    update_variable_attrs,
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 128})


T_PR_VARS = ["pr", "tas", "tasmin", "tasmax"]

VAR_PREFIX_MAP = {
    "pr": "PRECT",
    "tas": "T",
    "tasmin": "T",
    "tasmax": "T",
}


@dataclass
class BaseUKESM_Config(BaseETLConfig):
    s3_input_prefix: str = ""
    ensemble_members: list = field(default_factory=lambda: ["001", "002", "003"])
    drop_variables: list = field(
        default_factory=lambda: ["time_bnds", "lat_bnds", "lon_bnds", "height"]
    )
    aws_creds: dict = field(default_factory=dict)

    virtualize_cluster: dict = field(
        default_factory=lambda: {
            "n_workers": [4, 24],
            "worker_vm_types": ["r8g.8xlarge"],
            "scheduler_vm_types": "c8g.2xlarge",
        }
    )

    process_cluster: dict = field(
        default_factory=lambda: {
            "n_workers": [4, 16],
            "worker_vm_types": ["r8g.4xlarge"],
            "scheduler_vm_types": "c8g.xlarge",
        }
    )

    def __post_init__(self):
        sesh = boto3.Session()
        creds = sesh.get_credentials()
        self.aws_creds = {
            "aws_access_key_id": creds.access_key,
            "aws_secret_access_key": creds.secret_key,
        }


@dataclass
class UKESM_SSP245_Config(BaseUKESM_Config):
    scenario: str = "SSP245"
    catalog_key: str = "UKESM-SSP245-virtual"
    materialized_key: str = "UKESM-SSP245-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/transfer/SSP2-4.5/"


@dataclass
class UKESM_SSP245_T_PR_Config(BaseUKESM_Config):
    scenario: str = "SSP245"
    catalog_key: str = "UKESM-SSP245-t-pr-virtual"
    materialized_key: str = "UKESM-SSP245-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/netcdf/ssp245"


@dataclass
class UKESM_G6_1p5K_Config(BaseUKESM_Config):
    scenario: str = "G6-1.5K"
    catalog_key: str = "UKESM-G6-1.5K-virtual"
    materialized_key: str = "UKESM-G6-1.5K-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/transfer/G6-1.5K"


@dataclass
class UKESM_G6_1p5K_T_PR_Config(BaseUKESM_Config):
    scenario: str = "G6-1.5K"
    catalog_key: str = "UKESM-G6-1.5K-t-pr-virtual"
    materialized_key: str = "UKESM-G6-1.5K-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/G6-1.5K/netcdf"


@dataclass
class UKESM_Historical_Config(BaseUKESM_Config):
    scenario: str = "historical"
    materialized_key: str = "UKESM-historical-icechunk"
    catalog_key: str = "UKESM-historical-virtual"
    s3_input_prefix: str = "input/tensor/UKESM/netcdf/historical"
    source_base_url: str = (
        "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical"
    )

    ensemble_members: list = field(
        default_factory=lambda: [
            # "r1i1p1f2",
            # "r2i1p1f2",
            # "r3i1p1f2",
            # "r4i1p1f2",
            # "r5i1p1f3",
            # "r6i1p1f3",
            # "r7i1p1f3",
            # "r8i1p1f2",
            # "r9i1p1f2",
            # "r10i1p1f2",
            "r12i1p1f2"
        ]
    )


SCENARIO_CONFIG_MAP = {
    "SSP245": UKESM_SSP245_Config,
    "SSP245-t-pr": UKESM_SSP245_T_PR_Config,
    "G6-1.5K": UKESM_G6_1p5K_Config,
    "G6-1.5K-t-pr": UKESM_G6_1p5K_T_PR_Config,
    "historical": UKESM_Historical_Config,
}

T_PR_SCENARIOS = {UKESM_SSP245_T_PR_Config, UKESM_G6_1p5K_T_PR_Config}


def _fetch_ukesm_historical(variables: list[str], config: UKESM_Historical_Config) -> None:
    import warnings

    warnings.warn(
        "This fetches raw netcdf files from CEDA and moves them to s3. You must have rclone installed and configured."
    )
    time_slices = ["18500101-19491230", "19500101-20141230"]
    ensemble_dates = {
        # "r1i1p1f2": "d20190627",
        # "r2i1p1f2": "d20190708",
        # "r3i1p1f2": "d20190708",
        # "r4i1p1f2": "d20190708",
        # "r5i1p1f3": "d20191115",
        # "r6i1p1f3": "d20191113",
        # "r7i1p1f3": "d20191011",
        # "r8i1p1f2": "d20190708",
        # "r9i1p1f2": "d20191015",
        # "r10i1p1f2": "d20191213",
        "r12i1p1f2": "d20191210"
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


def _get_netcdf_urls_t_pr(config: BaseUKESM_Config, variables: list[str]) -> list[str]:
    store = from_url(f"s3://{config.s3_bucket}", region="us-west-2", **config.aws_creds)
    urls = []

    for var in variables:
        prefix_dir = VAR_PREFIX_MAP[var]
        prefix = f"{config.s3_input_prefix}/{prefix_dir}"
        stream = obs.list_with_delimiter(store, prefix=prefix, return_arrow=True)
        paths = list(stream["objects"]["path"].to_numpy())
        urls.extend(
            f"s3://{config.s3_bucket}/{path}"
            for path in paths
            if path.endswith(".nc") and f"_{var}_" in path and "_rechunked.nc" in path
        )
    return urls


def _preprocess_ensemble(ds: xr.Dataset, url: str = None) -> xr.Dataset:
    if url is None:
        raise ValueError("url parameter is required to determine ensemble member")
    ensemble = url.split(".nc")[0].split("_gn")[0].split("_")[-1]
    ds = ds.expand_dims({"ensemble_member_inferred": [ensemble]})
    return ds


def _preprocess_ensemble_t_pr(ds: xr.Dataset, url: str = None) -> xr.Dataset:
    """Extract ensemble member from filename pattern: PREFIX_001_var_corrected.nc
    and normalize latitude/longitude dim names to lat/lon."""
    if url is None:
        raise ValueError("url parameter is required to determine ensemble member")
    filename = url.split("/")[-1]
    member_idx = filename.split("_")[1]
    ds = ds.expand_dims({"ensemble_member_inferred": [member_idx]})
    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    return ds


def _preprocess_ukesm(
    ds: xr.Dataset,
    config: BaseUKESM_Config,
    subset: bool = False,
) -> xr.Dataset:
    # OUR TEMP/PR data has values from 2015-2099. it should be 2035-2085
    if isinstance(config, (UKESM_G6_1p5K_Config, UKESM_G6_1p5K_T_PR_Config)):
        ds = ds.sel(time=slice("2035-01-01", "2084-12-30"))

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
    # ds = add_cf_bounds(ds)

    if type(config) in T_PR_SCENARIOS:
        source = (
            "Parsed from the rechunked filename numeric suffix by taking the second "
            "underscore-separated segment "
            "(for example 'T_001_tas_corrected_rechunked.nc' → '001'). "
            "Source files are NetCDF3 produced by the CONVSH converter from UK Met Office "
            "Unified Model output and contain no CMIP6 metadata. "
            "Confirmed mapping from Matthew Henry: "
            "001 = r2i1p1f2, 002 = r3i1p1f2, 003 = r12i1p1f2. "
            "Note: the variant_label attribute in the original source files is incorrect "
            "due to a postprocessing oversight; the filenames and this mapping are authoritative."
        )
    elif isinstance(config, UKESM_Historical_Config):
        source = (
            "Parsed from the CEDA Data Reference Syntax filename by splitting on '_gn' "
            "and taking the preceding segment "
            "(for example 'tas_day_UKESM1-0-LL_historical_r1i1p1f2_gn_18500101-19491230.nc' "
            "→ 'r1i1p1f2'). "
            "Values are confirmed to match the variant_label global attribute in the source files."
        )
    else:
        source = (
            "Parsed from the ARISE Data Reference Syntax filename by splitting on '_gn' "
            "and taking the preceding segment "
            "(for example 'tas_day_UKESM1-1-LL_G6-1p5K-SAI_r12i1p1f2_gn_20350101-20491230.nc' "
            "→ 'r12i1p1f2'). "
            "Values are confirmed to match the variant_label global attribute in the source files."
        )
    ds.attrs.update(
        {
            "scenario": config.scenario,
            "model": "UKESM1-0-LL",
            "Conventions": "CF-1.8",
            "ensemble_member_source": source,
        }
    )

    return ds


@click.group()
def cli():
    pass


@click.command()
@click.option("--variable", multiple=True, required=True, help="UKESM variables to fetch")
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
@click.option("--scenario", type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())), required=True)
@click.option("--coiled/--local", default=False)
def virtualize(scenario, coiled):
    """Virtualize netcdf files into virtual icechunk dataset"""
    from obspec_utils.readers import BufferedStoreReader
    from obspec_utils.wrappers import CachingReadableStore, SplittingReadableStore

    config = SCENARIO_CONFIG_MAP[scenario]()
    virt_cat = catalog.get(config.catalog_key)
    variables = [var.name for var in virt_cat.expected_vars]

    if coiled:
        from srm.config import ClusterConfig

        client = setup_cluster(ClusterConfig(**config.virtualize_cluster))
    else:
        client = setup_local_client()

    is_t_pr = type(config) in T_PR_SCENARIOS

    try:
        base_store = from_url(f"s3://{config.s3_bucket}", region="us-west-2", **config.aws_creds)
        registry = ObjectStoreRegistry(
            {f"s3://{config.s3_bucket}": CachingReadableStore(SplittingReadableStore(base_store))}
        )

        if is_t_pr:
            parser = HDFParser(reader_factory=BufferedStoreReader)
            netcdf_urls = _get_netcdf_urls_t_pr(config, variables)
            preprocess_fn = _preprocess_ensemble_t_pr
            loadable_variables = ["latitude", "longitude", "time"]
        else:
            parser = HDFParser(
                drop_variables=config.drop_variables, reader_factory=BufferedStoreReader
            )
            netcdf_urls = _get_netcdf_urls(config, variables)
            preprocess_fn = _preprocess_ensemble
            loadable_variables = ["lat", "lon", "time"]

        combined_ds = virtualize_and_combine(
            urls=netcdf_urls,
            registry=registry,
            parser=parser,
            loadable_variables=loadable_variables,
            preprocess_fn=preprocess_fn,
        )

        repo_config = icechunk.RepositoryConfig.default()
        repo_config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                f"s3://{config.s3_bucket}/", store=icechunk.s3_store(region="us-west-2")
            )
        )

        storage = icechunk.s3_storage(
            bucket=virt_cat.bucket, prefix=virt_cat.prefix, region="us-west-2"
        )
        repo = icechunk.Repository.open_or_create(storage, repo_config)
        session = repo.writable_session("main")

        combined_ds.vz.to_icechunk(session.store)
        session.commit(f"{scenario}: virtualized variables {variables}")
        repo.save_config()

    finally:
        client.shutdown()


@click.command()
@click.option("--variable", multiple=True, help="Specific variables to process")
@click.option("--scenario", type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())), required=True)
@click.option("--coiled/--local", default=False)
@click.option("--all-variables", is_flag=True, help="process all expected variables from catalog")
@click.option("--subset/--no-subset", default=False)
def process(variable, scenario, coiled, all_variables, subset):
    """Read virtual icechunk stores, postprocess, rechunk, shard and write to icechunk"""
    config = SCENARIO_CONFIG_MAP[scenario]()

    if coiled:
        from srm.config import ClusterConfig

        client = setup_cluster(ClusterConfig(**config.process_cluster))
    else:
        client = setup_local_client()

    mat_key = getattr(config, "materialized_key", f"UKESM-{scenario}-icechunk")
    materialized_cat = catalog.get(mat_key)
    var_specs = get_var_specs(materialized_cat)

    if all_variables:
        variables = [var.name for var in materialized_cat.expected_vars]

    elif variable:
        variables = list(variable)
    else:
        raise click.UsageError("Must specify either --variable or --all-variables")

    try:
        for var in variables:
            if var.lower() == "dtr":
                ds = load_dtr_from_store(
                    materialized_cat.bucket, materialized_cat.prefix, config.encoding["shards"]
                )
            else:
                if var in T_PR_VARS and hasattr(config, "materialized_key"):
                    t_pr_key = f"UKESM-{config.scenario}-t-pr-virtual"
                    virt_ds = catalog.get(t_pr_key).to_xarray()
                else:
                    virt_ds = catalog.get(config.catalog_key).to_xarray()
                ds = virt_ds[[var]]
                ds = _preprocess_ukesm(ds, config, subset=subset)

            ds = _update_attrs(ds, var_specs, config)

            repo, session = init_repo(
                materialized_cat.bucket, materialized_cat.prefix, readonly=False
            )
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
