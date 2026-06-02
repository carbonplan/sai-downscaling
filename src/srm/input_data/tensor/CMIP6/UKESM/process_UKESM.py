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
    apply_ensemble_provenance,
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    load_dtr_from_store,
    trim_negative_precipitation,
    update_variable_attrs,
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180, to_proleptic_gregorian

zarr.config.set({"async.concurrency": 128})


T_PR_VARS = ["pr", "tas", "tasmin", "tasmax"]

VAR_PREFIX_MAP = {
    "pr": "PRECT",
    "tas": "T",
    "tasmin": "T",
    "tasmax": "T",
}

# Maps raw variable names in G6-1.5K-SAI T files to CF standard names
T_VAR_MAP = {
    "temp": "tasmax",
    "temp_1": "tasmin",
    "temp_2": "tas",
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
    t_pr_catalog_key: str = "UKESM-SSP245-t-pr-virtual"
    materialized_key: str = "UKESM-SSP245-icechunk"
    t_pr_materialized_key: str = "UKESM-SSP245-t-pr-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/transfer/SSP2-4.5/"


@dataclass
class UKESM_SSP245_T_PR_Config(BaseUKESM_Config):
    scenario: str = "SSP245"
    catalog_key: str = "UKESM-SSP245-t-pr-virtual"
    materialized_key: str = "UKESM-SSP245-t-pr-icechunk"
    # rechunked NetCDF4 output prefix
    s3_input_prefix: str = "input/tensor/UKESM/netcdf/ssp245"
    # raw source prefix for NetCDF3 files to be prepared
    s3_raw_prefix: str = "input/tensor/UKESM/SSP245_transfer_from_NCAR"


@dataclass
class UKESM_G6_1p5K_Config(BaseUKESM_Config):
    scenario: str = "G6-1.5K"
    catalog_key: str = "UKESM-G6-1.5K-virtual"
    t_pr_catalog_key: str = "UKESM-G6-1.5K-t-pr-virtual"
    materialized_key: str = "UKESM-G6-1.5K-icechunk"
    t_pr_materialized_key: str = "UKESM-G6-1.5K-t-pr-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/transfer/G6-1.5K"


@dataclass
class UKESM_G6_1p5K_T_PR_Config(BaseUKESM_Config):
    scenario: str = "G6-1.5K"
    catalog_key: str = "UKESM-G6-1.5K-t-pr-virtual"
    materialized_key: str = "UKESM-G6-1.5K-t-pr-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/G6-1.5K/netcdf"
    # raw source prefix for NetCDF3 files to be prepared
    s3_raw_prefix: str = "input/tensor/UKESM/netcdf_G6-1.5K-SAI"


@dataclass
class UKESM_Historical_Config(BaseUKESM_Config):
    scenario: str = "historical"
    materialized_key: str = "UKESM-historical-icechunk"
    catalog_key: str = "UKESM-historical-virtual"
    s3_input_prefix: str = "input/tensor/UKESM/netcdf/historical"
    source_base_url: str = (
        "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical"
    )
    drop_variables: list = field(
        default_factory=lambda: [
            "time_bnds",
            "lat_bnds",
            "lon_bnds",
            "height",
            "lat_bounds",
            "lon_bounds",
            "time_bounds",
        ]
    )

    ensemble_members: list = field(
        default_factory=lambda: [
            "r2i1p1f2",
            "r3i1p1f2",
            "r12i1p1f2",
        ]
    )

    process_cluster: dict = field(
        default_factory=lambda: {
            "n_workers": [4, 16],
            "worker_vm_types": ["r8g.4xlarge"],
            "scheduler_vm_types": "c8g.8xlarge",
        }
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
        "r2i1p1f2": "d20190708",
        "r3i1p1f2": "d20190708",
        "r12i1p1f2": "d20191210",
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
        if path.endswith(".nc")
        and any(f"{var}_".lower() in path.lower() for var in variables)
        and (not config.ensemble_members or any(f"_{m}_" in path for m in config.ensemble_members))
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
    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    return ds


def _preprocess_ensemble_t_pr(ds: xr.Dataset, url: str = None) -> xr.Dataset:
    """Extract ensemble member from filename pattern: PREFIX_001_var_rechunked.nc.

    Raw positional ID (e.g. '001'). update coord values
    """
    if url is None:
        raise ValueError("url parameter is required to determine ensemble member")
    filename = url.split("/")[-1]
    member_idx = filename.split("_")[1]
    ds = ds.expand_dims({"ensemble_member": [member_idx]})
    return ds


def _preprocess_ukesm(
    ds: xr.Dataset,
    config: BaseUKESM_Config,
    subset: bool = False,
) -> xr.Dataset:
    if subset:
        ds = ds.isel(time=slice(0, 365))

    if not isinstance(config, UKESM_Historical_Config):
        ds = ds.drop_duplicates(dim="time", keep="first")
    ds = to_proleptic_gregorian(ds)
    ds = ds.drop_encoding()
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])

    if isinstance(config, UKESM_Historical_Config) and hasattr(config, "time_range"):
        start_year, end_year = config.time_range.split("-")
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    ds = trim_negative_precipitation(ds)
    return ds


def _derivation_logic(config: BaseUKESM_Config) -> str:
    """For documenting how we get the ensemble_member, ie from attrs or filepath."""

    if type(config) in T_PR_SCENARIOS:
        return (
            "Extracted from filename position index: filename.split('_')[1] "
            "(e.g. '001'). Positional ID stored under 'ensemble_member' dim in separate "
            "T/PR icechunk store. Source files are private T/PR NetCDFs; CMIP6 ripf "
            "mapping unconfirmed. Update coord values once mapping confirmed "
            "(001->r12i1p1f2, 002->r2i1p1f2, 003->r3i1p1f2)."
        )
    return "Extracted from CMIP6 filename: url.split('.nc')[0].split('_gn')[0].split('_')[-1]. "


def _update_attrs(ds: xr.Dataset, var_specs: dict, config: BaseUKESM_Config) -> xr.Dataset:
    ds = update_variable_attrs(ds, var_specs)
    ds.attrs.update(
        {
            "scenario": config.scenario,
            "model": "UKESM1-0-LL",
            "Conventions": "CF-1.8",
        }
    )
    return apply_ensemble_provenance(ds, _derivation_logic(config))


# ~128MB/chunk for float32 at 144x192
_DISK_CHUNKS = {"time": 1095, "lat": 144, "lon": 192}


def _disk_chunk_encoding(ds: xr.Dataset) -> dict:
    """h5netcdf chunksizes per var, ordered by dims, ~128MB per chunk."""
    return {
        name: {"chunksizes": tuple(_DISK_CHUNKS[d] for d in da.dims)}
        for name, da in ds.data_vars.items()
    }


def _prepare_single_member(member: str, config: BaseUKESM_Config) -> None:
    """Convert one member's raw t-pr NetCDF3 files to rechunked NetCDF4 on S3.

    Pulls the NetCDF3 to local disk via fsspec simplecache. opens with scipy, normalizes dims/vars, loads to memory, writes NetCDF4
    PRECT: squeeze surface, rename precip->pr, t->time, latitude/longitude->lat/lon.
    T: squeeze ht, rename t->time, latitude/longitude->lat/lon, split
    temp/temp_1/temp_2 -> tasmax/tasmin/tas (one file per var).
    """
    import tempfile
    from pathlib import Path

    import fsspec

    write_store = from_url(f"s3://{config.s3_bucket}", region="us-west-2", **config.aws_creds)
    cache_kw = {"simplecache": {"cache_storage": "/tmp/fsspec_cache"}}
    print(member)

    # --- PRECT -> pr ---
    # Read eagerly (.load) while the simplecache file exists
    fpath = f"s3://{config.s3_bucket}/{config.s3_raw_prefix}/PRECT/PRECT_{member}.nc"
    cache = fsspec.open_local(f"simplecache::{fpath}", **cache_kw)
    ds = (
        xr.open_dataset(cache, engine="scipy")
        .squeeze("surface")
        .drop_vars("surface")
        .rename({"t": "time", "latitude": "lat", "longitude": "lon", "precip": "pr"})
        .drop_encoding()
        .load()
    )
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        ds.to_netcdf(tmp.name, engine="h5netcdf", encoding=_disk_chunk_encoding(ds))
        out = f"{config.s3_input_prefix}/PRECT/PRECT_{member}_pr_rechunked.nc"
        print(f"uploading s3://{config.s3_bucket}/{out}")
        obs.put(write_store, out, Path(tmp.name).read_bytes())

    # --- T -> tasmax/tasmin/tas (split) ---
    fpath = f"s3://{config.s3_bucket}/{config.s3_raw_prefix}/T/T_{member}.nc"
    cache = fsspec.open_local(f"simplecache::{fpath}", **cache_kw)
    ds = (
        xr.open_dataset(cache, engine="scipy")
        .squeeze("ht")
        .drop_vars("ht")
        .rename({"t": "time", "latitude": "lat", "longitude": "lon", **T_VAR_MAP})
        .drop_encoding()
        .load()
    )
    for cf_var in T_VAR_MAP.values():
        sub = ds[[cf_var]]
        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            sub.to_netcdf(tmp.name, engine="h5netcdf", encoding=_disk_chunk_encoding(sub))
            out = f"{config.s3_input_prefix}/T/T_{member}_{cf_var}_rechunked.nc"
            print(f"uploading s3://{config.s3_bucket}/{out}")
            obs.put(write_store, out, Path(tmp.name).read_bytes())


def _prepare_t_pr(config: BaseUKESM_Config) -> None:
    for member in config.ensemble_members:
        _prepare_single_member(member, config)


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
        registry = ObjectStoreRegistry({f"s3://{config.s3_bucket}": base_store})

        if is_t_pr:
            parser = HDFParser()
            netcdf_urls = _get_netcdf_urls_t_pr(config, variables)
            preprocess_fn = _preprocess_ensemble_t_pr
            loadable_variables = ["lat", "lon", "time"]
        else:
            parser = HDFParser(drop_variables=config.drop_variables)
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
    """Read virtual icechunk stores, postprocess, rechunk, shard and write to icechunk."""
    config = SCENARIO_CONFIG_MAP[scenario]()

    if coiled:
        from srm.config import ClusterConfig

        client = setup_cluster(ClusterConfig(**config.process_cluster))

    mat_cat = catalog.get(config.materialized_key)
    t_pr_mat_key = getattr(config, "t_pr_materialized_key", None)
    t_pr_mat_cat = catalog.get(t_pr_mat_key) if t_pr_mat_key else None

    var_specs = get_var_specs(mat_cat)
    if t_pr_mat_cat:
        var_specs.update(get_var_specs(t_pr_mat_cat))

    if all_variables:
        variables = [var.name for var in mat_cat.expected_vars]
        if t_pr_mat_cat:
            t_pr_expected = [var.name for var in t_pr_mat_cat.expected_vars]
            variables = variables + [v for v in t_pr_expected if v not in variables]
    elif variable:
        variables = list(variable)
    else:
        raise click.UsageError("Must specify either --variable or --all-variables")

    try:
        t_pr_key = getattr(config, "t_pr_catalog_key", None)

        for var in variables:
            use_t_pr_store = t_pr_mat_cat and (var in T_PR_VARS or var.lower() == "dtr")
            target_cat = t_pr_mat_cat if use_t_pr_store else mat_cat

            if var.lower() == "dtr":
                ds = load_dtr_from_store(target_cat.bucket, target_cat.prefix)
            else:
                source_key = t_pr_key if (var in T_PR_VARS and t_pr_key) else config.catalog_key
                ds = catalog.get(source_key).to_xarray()[[var]]
                if config.ensemble_members and "ensemble_member" in ds.dims:
                    available = [
                        m
                        for m in config.ensemble_members
                        if m in ds.coords["ensemble_member"].values
                    ]
                    missing = set(config.ensemble_members) - set(available)
                    if missing:
                        print(f"WARNING: ensemble members not in virtual store: {sorted(missing)}")
                    ds = ds.sel(ensemble_member=available)
                ds = _preprocess_ukesm(ds, config, subset=subset)

            ds = _update_attrs(ds, var_specs, config)

            repo, session = init_repo(target_cat.bucket, target_cat.prefix, readonly=False)
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
        if coiled:
            client.shutdown()


@click.command()
@click.option("--scenario", type=click.Choice(["G6-1.5K-t-pr", "SSP245-t-pr"]), required=True)
@click.option(
    "--member", multiple=True, help="Limit to specific ensemble members (e.g. --member 001)"
)
@click.option("--coiled/--local", default=False)
def prepare(scenario, member, coiled):
    """Convert raw t-pr NetCDF3 files to rechunked NetCDF4 on S3."""
    config = SCENARIO_CONFIG_MAP[scenario]()
    if member:
        config.ensemble_members = list(member)

    if coiled:
        import coiled

        remote_fn = coiled.function(
            vm_type="r8g.4xlarge",
            region="us-west-2",
            disk_size=200,
            tags={"Project": "SRM"},
        )(_prepare_single_member)
        list(remote_fn.map(config.ensemble_members, [config] * len(config.ensemble_members)))
    else:
        _prepare_t_pr(config)


cli.add_command(fetch)
cli.add_command(virtualize)
cli.add_command(process)
cli.add_command(prepare)

if __name__ == "__main__":
    cli()
