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
from srm.config import init_repo, setup_cluster, setup_local_client
from srm.input_data.etl_config import BaseETLConfig
from srm.input_data.etl_utils import (
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    trim_negative_precipitation,
    update_variable_attrs,
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 128})


CMIP6_SOURCE_BASE_URL = "https://www.jamstec.go.jp/swpub/public/CMIP6/MIROC-ES2H"
GEOMIP_SOURCE_BASE_URL = "https://www.jamstec.go.jp/swpub/public/GeoMIP"

MIROC_VARIABLES = ["hurs", "huss", "pr", "rlds", "rsds", "tas", "tasmax", "tasmin"]

CMIP6_ENSEMBLE_MEMBERS = ["r1i1p4f2", "r2i1p4f2", "r3i1p4f2"]

GEOMIP_ENSEMBLE_MEMBERS = [f"r{i:02d}" for i in range(1, 11)]
GEOMIP_ENSEMBLE_MEMBER_MAP = {f"r{i:02d}": f"r{i}i1p1f1" for i in range(1, 11)}


CMIP6_ENSEMBLE_VERSIONS: dict[str, dict[str, str]] = {
    "historical": {
        "r1i1p4f2": "v20220214",
        "r2i1p4f2": "v20220214",
        "r3i1p4f2": "v20220214",
    },
    "ssp245": {
        "r1i1p4f2": "v20220214",
        "r2i1p4f2": "v20220214",
        "r3i1p4f2": "v20220214",
    },
}

CMIP6_YEAR_RANGES: dict[str, range] = {
    "historical": range(1850, 2015),
    "ssp245": range(2015, 2101),
}


@dataclass
class BaseMIROC_ES2H_Config(BaseETLConfig):
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
        region = sesh.region_name or "us-west-2"
        self.aws_creds = {
            "aws_access_key_id": creds.access_key,
            "aws_secret_access_key": creds.secret_key,
            "aws_region": region,
        }


# --- CMIP6 scenarios (historical, ssp245) -----------------------------------


@dataclass
class BaseMIROC_CMIP6_Config(BaseMIROC_ES2H_Config):
    """Shared config for CMIP6 scenarios (historical / ssp245)."""

    source_base_url: str = CMIP6_SOURCE_BASE_URL
    ensemble_members: list = field(default_factory=lambda: list(CMIP6_ENSEMBLE_MEMBERS))

    def __post_init__(self):
        super().__post_init__()
        self.s3_input_prefix = f"input/tensor/MIROC-ES2H/{self.scenario}/netcdf"


@dataclass
class MIROC_ES2H_Historical_Config(BaseMIROC_CMIP6_Config):
    scenario: str = "historical"
    catalog_key: str = "MIROC-ES2H-historical-virtual"
    materialized_key: str = "MIROC-ES2H-historical-icechunk"
    time_range: str = "1850-2014"


@dataclass
class MIROC_ES2H_SSP245_Config(BaseMIROC_CMIP6_Config):
    scenario: str = "ssp245"
    catalog_key: str = "MIROC-ES2H-SSP245-virtual"
    materialized_key: str = "MIROC-ES2H-SSP245-icechunk"
    time_range: str = "2015-2100"


# --- GeoMIP scenarios (G6-1.5K-SAI, baseline) --------------------------------


@dataclass
class BaseMIROC_GeoMIP_Config(BaseMIROC_ES2H_Config):
    """Shared config for GeoMIP scenarios (G6-1.5K-SAI / baseline).

    Files live at:
      {GEOMIP_SOURCE_BASE_URL}/{geomip_scenario}/MIROC-ES2H/day/
    and are named:
      {var}_{geomip_scenario}_{rXX}.nc
    """

    source_base_url: str = GEOMIP_SOURCE_BASE_URL
    ensemble_members: list = field(default_factory=lambda: list(GEOMIP_ENSEMBLE_MEMBERS))

    def __post_init__(self):
        super().__post_init__()
        self.s3_input_prefix = f"input/tensor/MIROC-ES2H/{self.scenario}/netcdf"

    @property
    def source_url(self) -> str:
        return f"{self.source_base_url}/{self.geomip_scenario}/MIROC-ES2H/day/"


@dataclass
class MIROC_ES2H_G6_1p5K_Config(BaseMIROC_GeoMIP_Config):
    scenario: str = "G6-1.5K"
    geomip_scenario: str = "G6-1.5K-SAI"
    catalog_key: str = "MIROC-ES2H-G6-1.5K-virtual"
    materialized_key: str = "MIROC-ES2H-G6-1.5K-icechunk"


@dataclass
class MIROC_ES2H_Baseline_Config(BaseMIROC_GeoMIP_Config):
    scenario: str = "baseline"
    geomip_scenario: str = "baseline"
    catalog_key: str = "MIROC-ES2H-baseline-virtual"
    materialized_key: str = "MIROC-ES2H-baseline-icechunk"


SCENARIO_CONFIG_MAP = {
    "historical": MIROC_ES2H_Historical_Config,
    "ssp245": MIROC_ES2H_SSP245_Config,
    "G6-1.5K": MIROC_ES2H_G6_1p5K_Config,
    "baseline": MIROC_ES2H_Baseline_Config,
}


def _build_cmip6_urls(variables: list[str], config: BaseMIROC_CMIP6_Config) -> list[str]:
    """Build all per-year NetCDF URLs for a CMIP6 scenario without directory crawling."""
    ensemble_versions = CMIP6_ENSEMBLE_VERSIONS[config.scenario]
    urls = []
    for ens in config.ensemble_members:
        version = ensemble_versions[ens]
        for var in variables:
            for year in CMIP6_YEAR_RANGES[config.scenario]:
                t_range = f"{year}0101-{year}1231"
                folder = f"{config.source_base_url}/{config.scenario}/{ens}/day/{var}/gn/{version}"
                fname = f"{var}_day_MIROC-ES2H_{config.scenario}_{ens}_gn_{t_range}.nc"
                urls.append(f"{folder}/{fname}")
    return urls


def _build_geomip_urls(variables: list[str], config: BaseMIROC_GeoMIP_Config) -> list[str]:
    """Build GeoMIP NetCDF URLs from HTML directory listing."""
    import pandas as pd

    base = config.source_url
    df = pd.read_html(base)[0][["Name"]].iloc[2:].dropna().reset_index(drop=True)
    all_names = df["Name"].tolist()

    urls = []
    for name in all_names:
        if not name.endswith(".nc"):
            continue
        if any(name.startswith(f"{var}_") for var in variables):
            if any(f"_{ens}.nc" in name for ens in config.ensemble_members):
                urls.append(f"{base}{name}")
    return urls


def _filter_existing_urls(urls: list[str], config: BaseMIROC_ES2H_Config) -> list[str]:
    """Filter out URLs whose filenames already exist in S3 via obstore list.

    rclone --no-clobber does the same check but has to HEAD each file
    individually against the jamstec server, which is slow on a flaky remote.
    Listing the destination bucket first is waaay faster.
    """
    store = from_url(
        f"s3://{config.s3_bucket}/{config.s3_input_prefix}/",
        **config.aws_creds,
    )

    existing_names: set[str] = set()
    for batch in obs.list(store):
        for item in batch:
            existing_names.add(item["path"].split("/")[-1])

    print(f"  {len(existing_names):,} files already present in S3")

    filtered = [u for u in urls if u.split("/")[-1] not in existing_names]
    skipped = len(urls) - len(filtered)
    print(f"  Skipping {skipped:,} already-uploaded files, {len(filtered):,} to fetch")
    return filtered


def _rclone_copy_urls(urls: list[str], urls_file: str, config: BaseMIROC_ES2H_Config) -> None:
    """Write urls_file and invoke rclone copyurl to S3."""
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
        "--checkers",
        "8",
        "--transfers",
        "2",
        "--s3-chunk-size",
        "16M",
        "--s3-upload-concurrency",
        "2",
        "--buffer-size",
        "256M",
        "--fast-list",
        "--s3-no-check-bucket",
        "--disable-http2",
        "--retries",
        "20",
        "--low-level-retries",
        "40",
        "--retries-sleep",
        "30s",
        "--timeout",
        "30m",
        "--contimeout",
        "60s",
    ]
    print(" \\\n    ".join(command))
    subprocess.run(command)


def _fetch_cmip6_netcdfs(variables: list[str], config: BaseMIROC_CMIP6_Config) -> None:
    """Fetch CMIP6 NetCDFs from JAMSTEC and copy to S3."""
    import warnings

    warnings.warn(
        "Fetches raw NetCDF files from JAMSTEC and copies to S3. rclone must be installed."
    )
    urls = _build_cmip6_urls(variables, config)
    urls = _filter_existing_urls(urls, config)
    if not urls:
        print("Nothing to fetch — all files already in S3.")
        return
    urls_file = f"MIROC-ES2H-CMIP6-{config.scenario}-urls.txt"
    print(f"Queued {len(urls)} files → {urls_file}")
    _rclone_copy_urls(urls, urls_file, config)


def _fetch_geomip_netcdfs(variables: list[str], config: BaseMIROC_GeoMIP_Config) -> None:
    """Fetch GeoMIP NetCDFs from JAMSTEC HTML listing and copy to S3."""
    import warnings

    warnings.warn(
        "Fetches raw NetCDF files from a slow JAMSTEC server and copies to S3. rclone must be installed."
    )
    urls = _build_geomip_urls(variables, config)
    urls = _filter_existing_urls(urls, config)
    if not urls:
        print("Nothing to fetch — all files already in S3.")
        return
    urls_file = f"MIROC-ES2H-GeoMIP-{config.scenario}-urls.txt"
    print(f"Queued {len(urls)} files → {urls_file}")
    _rclone_copy_urls(urls, urls_file, config)


# ---------------------------------------------------------------------------
# Virtualize helpers
# ---------------------------------------------------------------------------


def _get_cmip6_netcdf_urls(variables: list[str], config: BaseMIROC_CMIP6_Config) -> list[str]:
    """List CMIP6 NetCDF files from S3, filtered by variable."""
    store = from_url(f"s3://{config.s3_bucket}", **config.aws_creds)
    stream = obs.list_with_delimiter(store, prefix=config.s3_input_prefix, return_arrow=True)
    all_paths = list(stream["objects"]["path"].to_numpy())
    return [
        f"s3://{config.s3_bucket}/{path}"
        for path in sorted(all_paths)
        if path.endswith(".nc")
        and any(path.split("/")[-1].startswith(f"{var}_") for var in variables)
    ]


def _get_geomip_netcdf_urls(variables: list[str], config: BaseMIROC_GeoMIP_Config) -> list[str]:
    """Construct GeoMIP NetCDF S3 URLs from known filename pattern."""
    return [
        f"s3://{config.s3_bucket}/{config.s3_input_prefix}/{var}_{config.geomip_scenario}_{ens}.nc"
        for var in variables
        for ens in config.ensemble_members
    ]


def _preprocess_cmip6_ensemble(ds: xr.Dataset, url: str = None) -> xr.Dataset:
    """Extract CMIP6 ensemble member (e.g. r1i1p4f2) from URL and add as dimension."""
    if url is None:
        raise ValueError("url parameter is required to determine ensemble member")
    ensemble = url.split(".nc")[0].split("_gn")[0].split("_")[-1]
    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    return ds


def _preprocess_geomip_ensemble(ds: xr.Dataset, url: str = None) -> xr.Dataset:
    """Extract GeoMIP ensemble member from URL (r01 → r1i1p1f1) and add as dimension."""
    if url is None:
        raise ValueError("url parameter is required to determine ensemble member")
    raw = url.split(".nc")[0].split("_")[-1]  # e.g. "r01"
    ensemble = GEOMIP_ENSEMBLE_MEMBER_MAP[raw]
    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    return ds


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _preprocess_miroc(
    ds: xr.Dataset, config: BaseMIROC_ES2H_Config, subset: bool = False
) -> xr.Dataset:
    ds = ds.drop_duplicates(dim="time", keep="first")
    ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False, align_on="date")
    ds = ds.drop_encoding()
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])
    ds = trim_negative_precipitation(ds)
    if subset:
        ds = ds.isel(time=slice(0, 365))
    return ds


def _update_attrs(ds: xr.Dataset, var_specs: dict, config: BaseMIROC_ES2H_Config) -> xr.Dataset:
    ds = update_variable_attrs(ds, var_specs)
    attrs = {
        "scenario": config.scenario,
        "model": "MIROC-ES2H",
        "Conventions": "CF-1.8",
    }
    if hasattr(config, "time_range"):
        attrs["time_range"] = config.time_range
    ds.attrs.update(attrs)
    return ds


# ---------------------------------------------------------------------------
# Click commands
# ---------------------------------------------------------------------------


@click.group()
def cli():
    pass


@click.command()
@click.option("--variable", multiple=True, help="Variables to fetch (defaults to all)")
@click.option(
    "--scenario",
    type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())),
    required=True,
    help="Which MIROC-ES2H scenario to fetch",
)
def fetch(variable, scenario):
    config_class = SCENARIO_CONFIG_MAP[scenario]
    config = config_class()
    variables = list(variable) if variable else MIROC_VARIABLES

    if isinstance(config, BaseMIROC_CMIP6_Config):
        _fetch_cmip6_netcdfs(variables, config)
    elif isinstance(config, BaseMIROC_GeoMIP_Config):
        _fetch_geomip_netcdfs(variables, config)
    else:
        raise ValueError(f"fetch not implemented for scenario: {scenario}")


@click.command()
@click.option("--scenario", type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())), required=True)
@click.option("--coiled/--local", default=False)
def virtualize(scenario, coiled):
    """Virtualize NetCDF files on S3 into a virtual icechunk store."""
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

    try:
        base_store = from_url(f"s3://{config.s3_bucket}", **config.aws_creds)
        registry = ObjectStoreRegistry(
            {f"s3://{config.s3_bucket}": CachingReadableStore(SplittingReadableStore(base_store))}
        )
        parser = HDFParser(reader_factory=BufferedStoreReader)

        if isinstance(config, BaseMIROC_CMIP6_Config):
            netcdf_urls = _get_cmip6_netcdf_urls(variables, config)
            preprocess_fn = _preprocess_cmip6_ensemble
        else:
            netcdf_urls = _get_geomip_netcdf_urls(variables, config)
            preprocess_fn = _preprocess_geomip_ensemble
        combined_ds = virtualize_and_combine(
            urls=netcdf_urls,
            registry=registry,
            parser=parser,
            loadable_variables=["lat", "lon", "time", "time_bnds", "lat_bnds", "lon_bnds"],
            preprocess_fn=preprocess_fn,
        )

        repo_config = icechunk.RepositoryConfig.default()
        repo_config.set_virtual_chunk_container(
            icechunk.VirtualChunkContainer(
                f"s3://{config.s3_bucket}/",
                store=icechunk.s3_store(region="us-west-2"),
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
@click.option("--all-variables", is_flag=True, help="Process all expected variables from catalog")
@click.option("--subset/--no-subset", default=False)
def process(variable, scenario, coiled, all_variables, subset):
    """Read virtual icechunk store, postprocess, rechunk/shard and write to icechunk."""
    config = SCENARIO_CONFIG_MAP[scenario]()

    if coiled:
        from srm.config import ClusterConfig

        client = setup_cluster(ClusterConfig(**config.process_cluster))
    else:
        client = setup_local_client()

    mat_cat = catalog.get(config.materialized_key)
    var_specs = get_var_specs(mat_cat)

    if all_variables:
        variables = [var.name for var in mat_cat.expected_vars]
    elif variable:
        variables = list(variable)
    else:
        raise click.UsageError("Must specify either --variable or --all-variables")

    try:
        virt_ds = catalog.get(config.catalog_key).to_xarray()

        for var in variables:
            ds = virt_ds[[var]]
            ds = _preprocess_miroc(ds, config, subset=subset)
            ds = _update_attrs(ds, var_specs, config)

            repo, session = init_repo(mat_cat.bucket, mat_cat.prefix, readonly=False)
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
