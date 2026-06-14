# COILED vm-type r8g.4xlarge
# COILED region us-west-2

import logging
import subprocess
from dataclasses import dataclass, field

import click
import dask
import icechunk
import obstore as obs
import xarray as xr
import zarr
from obstore.store import from_url

from srm import catalog
from srm.config import init_repo
from srm.input_data.etl_config import BaseETLConfig
from srm.input_data.etl_utils import (
    CMORIZE_hurs,
    apply_ensemble_provenance,
    get_var_specs,
    group_paths_by_member,
    open_netcdf_from_s3,
    resolve_variables,
    trim_negative_precipitation,
    update_variable_attrs,
    variable_in_store,
    write_variable_to_icechunk,
)
from srm.utils import lon_to_180, to_proleptic_gregorian

zarr.config.set({"async.concurrency": 128})
dask.config.set(scheduler="threads")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


CMIP6_SOURCE_BASE_URL = "https://www.jamstec.go.jp/swpub/public/CMIP6/MIROC-ES2H"
GEOMIP_SOURCE_BASE_URL = "https://www.jamstec.go.jp/swpub/public/GeoMIP"

MIROC_VARIABLES = ["hurs", "pr", "rsds", "tas", "tasmax", "tasmin"]

CMIP6_ENSEMBLE_MEMBERS = ["r1i1p4f2", "r2i1p4f2", "r3i1p4f2"]

GEOMIP_ENSEMBLE_MEMBERS = [f"r{i:02d}" for i in range(1, 11)]


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

UNIFIED_KEY = "MIROC-ES2H-unified-icechunk"

# GeoMIP SSP245 (baseline) starts 2020; ESGF SSP245 fills the 2015-2019 gap.
# Maps each GeoMIP member to its ESGF counterpart.
_SSP245_ESGF_MEMBER: dict[str, str] = {
    "r01": "r1i1p4f2",
    "r02": "r2i1p4f2",
    "r03": "r3i1p4f2",
    "r04": "r1i1p4f2",
    "r05": "r2i1p4f2",
    "r06": "r3i1p4f2",
    "r07": "r1i1p4f2",
    "r08": "r2i1p4f2",
    "r09": "r3i1p4f2",
    "r10": "r1i1p4f2",
}


@dataclass
class BaseMIROC_ES2H_Config(BaseETLConfig):
    # MIROC SSP245/G6-1.5K/baseline raw hurs is stored as fraction (0-1) despite units='%'
    # due to a CMOR labeling bug. Historical is correctly in %. Set True to apply ×100.
    cmorize_hurs: bool = False
    materialized_key: str = UNIFIED_KEY
    group: str = ""


# --- CMIP6 scenarios (historical, ssp245) -----------------------------------


@dataclass
class BaseMIROC_CMIP6_Config(BaseMIROC_ES2H_Config):
    """Shared config for CMIP6 scenarios (historical / ssp245)."""

    source_base_url: str = CMIP6_SOURCE_BASE_URL
    ensemble_members: list = field(default_factory=lambda: list(CMIP6_ENSEMBLE_MEMBERS))

    def __post_init__(self):
        self.s3_input_prefix = f"input/tensor/MIROC-ES2H/{self.scenario}/netcdf"


@dataclass
class MIROC_ES2H_Historical_Config(BaseMIROC_CMIP6_Config):
    scenario: str = "historical"
    group: str = "historical"
    time_range: str = "1850-2014"


@dataclass
class MIROC_ES2H_ESGF_SSP245_Config(BaseMIROC_CMIP6_Config):
    scenario: str = "ssp245"
    group: str = "esgf_ssp245"
    time_range: str = "2015-2100"


# --- GeoMIP scenarios (G6-1.5K-SAI, SSP245/baseline) --------------------------------


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
        self.s3_input_prefix = f"input/tensor/MIROC-ES2H/{self.scenario}/netcdf"

    @property
    def source_url(self) -> str:
        return f"{self.source_base_url}/{self.geomip_scenario}/MIROC-ES2H/day/"


@dataclass
class MIROC_ES2H_G6_1p5K_Config(BaseMIROC_GeoMIP_Config):
    scenario: str = "G6-1.5K"
    group: str = "g6_1p5k"
    geomip_scenario: str = "G6-1.5K-SAI"
    cmorize_hurs: bool = True


@dataclass
class MIROC_ES2H_SSP245_Config(BaseMIROC_GeoMIP_Config):
    # GeoMIP baseline run — renamed to SSP245 to match catalog naming convention.
    # Source files still live under baseline/ in S3, so s3_input_prefix is overridden.
    scenario: str = "SSP245"
    group: str = "ssp245"
    geomip_scenario: str = "baseline"
    cmorize_hurs: bool = True

    def __post_init__(self):
        self.s3_input_prefix = "input/tensor/MIROC-ES2H/baseline/netcdf"


SCENARIO_CONFIG_MAP = {
    "historical": MIROC_ES2H_Historical_Config,
    "ssp245": MIROC_ES2H_SSP245_Config,
    "esgf-ssp245": MIROC_ES2H_ESGF_SSP245_Config,
    "G6-1.5K": MIROC_ES2H_G6_1p5K_Config,
}


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


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
        region="us-west-2",
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
# Process helpers
# ---------------------------------------------------------------------------


def _get_netcdf_urls(config: BaseMIROC_ES2H_Config, variable: str) -> list[tuple[str, str]]:
    """Return (member_id, s3_path) pairs for one variable."""
    store = from_url(f"s3://{config.s3_bucket}", region="us-west-2")
    stream = obs.list_with_delimiter(store, prefix=config.s3_input_prefix, return_arrow=True)
    paths = list(stream["objects"]["path"].to_numpy())

    is_cmip6 = isinstance(config, BaseMIROC_CMIP6_Config)
    result = []
    for path in sorted(paths):
        fname = path.split("/")[-1]
        if not (fname.endswith(".nc") and fname.startswith(f"{variable}_")):
            continue
        if is_cmip6:
            # {var}_day_MIROC-ES2H_{scenario}_{ens}_gn_{trange}.nc
            member = fname.split(".nc")[0].split("_gn")[0].split("_")[-1]
        else:
            # {var}_{geomip_scenario}_{rXX}.nc
            member = fname.split(".nc")[0].split("_")[-1]
        if config.ensemble_members and member not in config.ensemble_members:
            continue
        result.append((member, path))
    return result


def _preprocess_miroc(
    ds: xr.Dataset, config: BaseMIROC_ES2H_Config, subset: bool = False
) -> xr.Dataset:
    # Keep only standard spatial/temporal coords and drop bnds variables
    # (time_bnds etc.) before calendar conversion; chunk({"time": -1}) in
    # to_proleptic_gregorian fails on multi-dim bnds variables
    keep_coords = set(ds.dims) | {"lat", "lon", "time"}
    ds = ds.drop_vars([c for c in ds.coords if c not in keep_coords], errors="ignore")
    bnds_data_vars = [v for v in ds.data_vars if any("bnds" in d for d in ds[v].dims)]
    if bnds_data_vars:
        ds = ds.drop_vars(bnds_data_vars)
    ds = ds.drop_duplicates(dim="time", keep="first")
    ds = to_proleptic_gregorian(ds)
    ds = ds.drop_encoding()
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])
    ds = trim_negative_precipitation(ds)
    if config.cmorize_hurs and "hurs" in ds.data_vars:
        ds = CMORIZE_hurs(ds, "hurs")
    if subset:
        ds = ds.isel(time=slice(0, 365))
    return ds


def _derivation_logic(config: BaseMIROC_ES2H_Config) -> str:
    """For documenting how we get the ensemble_member, ie from attrs or filepath."""
    if isinstance(config, BaseMIROC_CMIP6_Config):
        return (
            "Extracted from CMIP6 DRS filename: "
            "url.split('.nc')[0].split('_gn')[0].split('_')[-1]. "
        )
    return "Raw filename suffix: url.split('.nc')[0].split('_')[-1] (e.g. 'r01'). "


def _update_attrs(ds: xr.Dataset, var_specs: dict, config: BaseMIROC_ES2H_Config) -> xr.Dataset:
    ds = update_variable_attrs(ds, var_specs)

    global_attrs: dict = {
        "scenario": config.scenario,
        "model": "MIROC-ES2H",
        "Conventions": "CF-1.8",
    }
    if hasattr(config, "time_range"):
        global_attrs["time_range"] = config.time_range
    ds.attrs.update(global_attrs)

    return apply_ensemble_provenance(ds, _derivation_logic(config))


def _stitch_esgf_gap(geomip: xr.Dataset, variable: str, repo: icechunk.Repository) -> xr.Dataset:
    """Prepend ESGF SSP245 2015-2019 to the GeoMIP baseline run (which starts 2020).

    Reads the bridge from the unified store's ``esgf_ssp245`` group; each GeoMIP
    member gets its ESGF counterpart via _SSP245_ESGF_MEMBER. Requires
    --scenario esgf-ssp245 to have been processed first.
    """
    session = repo.readonly_session("main")
    try:
        esgf = xr.open_dataset(session.store, engine="zarr", group="esgf_ssp245", chunks="auto")[
            [variable]
        ]
    except (FileNotFoundError, KeyError) as err:
        raise RuntimeError(
            f"esgf_ssp245 group missing variable '{variable}' in the unified store; "
            "run --scenario esgf-ssp245 before --scenario ssp245 (gap-fill dependency)"
        ) from err

    bridge = esgf.sel(time=slice("2015", "2019"))
    gap_pieces = [
        bridge.sel(ensemble_member=[_SSP245_ESGF_MEMBER[m]]).assign_coords(ensemble_member=[m])
        for m in geomip.ensemble_member.values
    ]
    gap_ds = xr.concat(gap_pieces, dim="ensemble_member")
    combined = xr.concat([gap_ds, geomip], dim="time")
    combined.attrs.update(
        {
            "gap_fill_source": "esgf_ssp245 group (MIROC-ES2H ESGF SSP245)",
            "gap_fill_period": "2015-2019",
            "gap_fill_member_map": ", ".join(
                f"{m}->{e}" for m, e in sorted(_SSP245_ESGF_MEMBER.items())
            ),
        }
    )
    return combined


def _process_single_variable(
    config: BaseMIROC_ES2H_Config,
    variable: str,
    repo: icechunk.Repository,
    var_specs: dict,
    overwrite: bool,
    subset: bool,
) -> None:
    log.info("variable=%s group=%s start", variable, config.group)

    var_in_store = variable_in_store(repo, variable, group=config.group)
    if not overwrite and var_in_store:
        log.info("variable=%s skip: already in store (use --overwrite to replace)", variable)
        return

    obstore_inst = from_url(f"s3://{config.s3_bucket}", region="us-west-2")
    url_pairs = _get_netcdf_urls(config, variable)
    if not url_pairs:
        log.warning("variable=%s no files found, skipping", variable)
        return
    log.info("variable=%s found %d files", variable, len(url_pairs))

    member_paths = group_paths_by_member(url_pairs)

    member_datasets = []
    for member, paths in sorted(member_paths.items()):
        log.info("variable=%s member=%s opening %d file(s)", variable, member, len(paths))
        time_slices = [open_netcdf_from_s3(obstore_inst, p) for p in sorted(paths)]
        member_ds = (
            xr.concat(time_slices, dim="time", data_vars="minimal")
            if len(time_slices) > 1
            else time_slices[0]
        )
        member_ds = _preprocess_miroc(member_ds, config, subset=subset)
        member_ds = member_ds.expand_dims({"ensemble_member": [member]})
        member_datasets.append(member_ds)
        log.info("variable=%s member=%s shape=%s", variable, member, dict(member_ds.dims))

    ds = xr.concat(member_datasets, dim="ensemble_member")
    ds = ds[[variable]]
    log.info("variable=%s concat done shape=%s", variable, dict(ds.dims))

    if config.group == "ssp245":
        ds = _stitch_esgf_gap(ds, variable, repo)
        log.info("variable=%s gap-filled 2015-2019 from esgf_ssp245", variable)

    ds = _update_attrs(ds, var_specs, config)

    write_variable_to_icechunk(
        ds,
        repo,
        variable=variable,
        scenario=config.scenario,
        chunks=config.encoding["chunks"],
        shards=config.encoding["shards"],
        overwrite=overwrite,
        var_in_store=var_in_store,
        group=config.group,
    )
    log.info("variable=%s done", variable)


def _run_process(
    config: BaseMIROC_ES2H_Config,
    variables: list[str],
    overwrite: bool,
    subset: bool,
    store_prefix: str | None = None,
) -> None:
    log.info(
        "scenario=%s group=%s variables=%s overwrite=%s subset=%s",
        config.scenario,
        config.group,
        variables,
        overwrite,
        subset,
    )
    mat_cat = catalog.get(config.materialized_key)
    var_specs = get_var_specs(mat_cat)
    repo, _ = init_repo(mat_cat.bucket, store_prefix or mat_cat.prefix, readonly=False)
    for var in variables:
        _process_single_variable(config, var, repo, var_specs, overwrite, subset)
    log.info("scenario=%s all variables complete", config.scenario)


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
@click.option("--variable", multiple=True, help="Specific variable(s) to process")
@click.option(
    "--scenario",
    multiple=True,
    type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())),
    required=True,
    help="Scenario(s) to process; repeat to run several sequentially in order given",
)
@click.option("--all-variables", is_flag=True, help="Process all expected variables from catalog")
@click.option("--subset/--no-subset", default=False)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing variable arrays in-place (r+ mode)",
)
@click.option(
    "--store-prefix",
    default=None,
    help="Override the unified store prefix (e.g. a dev path for test runs)",
)
def process(variable, scenario, all_variables, subset, overwrite, store_prefix):
    """Open NetCDF files from S3, concat, rechunk, and write to the unified per-GCM
    icechunk store under each scenario's zarr group. One variable at a time.

    Ordering: esgf-ssp245 must complete before ssp245 (the ssp245 group
    gap-fills 2015-2019 from the esgf_ssp245 group); list them in that order.
    """
    for scen in scenario:
        config = SCENARIO_CONFIG_MAP[scen]()
        variables = resolve_variables(variable, all_variables, catalog.get(config.materialized_key))
        _run_process(config, variables, overwrite, subset, store_prefix=store_prefix)


cli.add_command(fetch)
cli.add_command(process)

if __name__ == "__main__":
    cli()
