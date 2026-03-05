import subprocess
from dataclasses import dataclass, field

import boto3
import click
import obstore as obs
from obstore.store import from_url

from srm import catalog
from srm.config import setup_cluster, setup_local_client
from srm.input_data.etl_config import BaseETLConfig


CMIP6_SOURCE_BASE_URL = "https://www.jamstec.go.jp/swpub/public/CMIP6/MIROC-ES2H"
GEOMIP_SOURCE_BASE_URL = "https://www.jamstec.go.jp/swpub/public/GeoMIP"

MIROC_VARIABLES = ["hurs", "huss", "pr", "rlds", "rsds", "tas", "tasmax", "tasmin"]

# Ensemble members for CMIP6 scenarios
CMIP6_ENSEMBLE_MEMBERS = ["r1i1p4f2", "r2i1p4f2", "r3i1p4f2"]

# GeoMIP r01–r10, normalized to CMIP-style for storage
GEOMIP_ENSEMBLE_MEMBERS = [f"r{i:02d}" for i in range(1, 11)]
GEOMIP_ENSEMBLE_MEMBER_MAP = {
    f"r{i:02d}": f"r{i}i1p1f1" for i in range(1, 11)
}


CMIP6_ENSEMBLE_VERSIONS: dict[str, dict[str, str]] = {
    "historical": {
        "r1i1p4f2": "v20220214",  # TODO: verify
        "r2i1p4f2": "v20220214",  # TODO: verify
        "r3i1p4f2": "v20220214",  # TODO: verify
    },
    "ssp245": {
        "r1i1p4f2": "v20220214",  # TODO: verify
        "r2i1p4f2": "v20220214",  # TODO: verify
        "r3i1p4f2": "v20220214",  # TODO: verify
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
    materialized_key: str = "MIROC-ES2H-historical-materialized"
    time_range: str = "1850-2014"


@dataclass
class MIROC_ES2H_SSP245_Config(BaseMIROC_CMIP6_Config):
    scenario: str = "ssp245"
    catalog_key: str = "MIROC-ES2H-SSP245-virtual"
    materialized_key: str = "MIROC-ES2H-SSP245-materialized"
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
    materialized_key: str = "MIROC-ES2H-G6-1.5K-materialized"


@dataclass
class MIROC_ES2H_Baseline_Config(BaseMIROC_GeoMIP_Config):
    scenario: str = "baseline"
    geomip_scenario: str = "baseline"
    catalog_key: str = "MIROC-ES2H-baseline-virtual"
    materialized_key: str = "MIROC-ES2H-baseline-materialized"


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

    # Filter to files matching our variables and ensemble members
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
    individually against the source server, which is slow on a flaky remote.
    Listing the destination bucket first is much faster.
    """
    store = from_url(
        f"s3://{config.s3_bucket}/{config.s3_input_prefix}/",
        **config.aws_creds,
    )

    print(f"Listing existing objects in s3://{config.s3_bucket}/{config.s3_input_prefix}/ ...")
    existing_names: set[str] = set()
    for batch in obs.list(store):
        for item in batch:
            # item is a dict; "path" is relative to the store prefix
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
        "--urls", urls_file,
        target_remote,
        "--progress",
        "--no-clobber",
        "--checkers", "8",
        "--transfers", "2",
        "--s3-chunk-size", "16M",
        "--s3-upload-concurrency", "2",
        "--buffer-size", "256M",
        "--fast-list",
        "--s3-no-check-bucket",
        "--disable-http2",
        "--retries", "20",
        "--low-level-retries", "40",
        "--retries-sleep", "30s",
        "--timeout", "30m",
        "--contimeout", "60s",
    ]
    print(" \\\n    ".join(command))
    # subprocess.run(command)


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


cli.add_command(fetch)

if __name__ == "__main__":
    cli()