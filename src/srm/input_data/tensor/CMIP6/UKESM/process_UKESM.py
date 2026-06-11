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
from obspec_utils.readers import EagerStoreReader
from obstore.store import from_url

from srm import catalog
from srm.config import init_repo
from srm.input_data.etl_config import BaseETLConfig
from srm.input_data.etl_utils import (
    apply_ensemble_provenance,
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    load_dtr_from_store,
    trim_negative_precipitation,
    update_variable_attrs,
    write_dataset_to_icechunk,
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


T_PR_VARS = ["pr", "tas", "tasmin", "tasmax"]


@dataclass
class BaseUKESM_Config(BaseETLConfig):
    s3_input_prefix: str = ""
    ensemble_members: list = field(default_factory=lambda: ["001", "002", "003"])


@dataclass
class UKESM_SSP245_Config(BaseUKESM_Config):
    scenario: str = "SSP245"
    materialized_key: str = "UKESM-SSP245-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/transfer/SSP2-4.5/"
    time_range: str = "2015-2099"
    t_pr_input_prefix: str = "input/tensor/UKESM/UKESM_SSP245_T_PR_NETCDF/UKESM_SSP245_T_PR"
    ensemble_members: list = field(default_factory=lambda: ["r12i1p1f2", "r2i1p1f2", "r3i1p1f2"])
    # r2 member has tas/tasmin/tasmax combined in one file
    t_pr_var_lookup: dict = field(
        default_factory=lambda: {
            "pr": ["pr_day_"],
            "tas": ["tas_day_", "tas_mean_min_max_day_"],
            "tasmin": ["tasmin_day_", "tas_mean_min_max_day_"],
            "tasmax": ["tasmax_day_", "tas_mean_min_max_day_"],
        }
    )
    # r2i1p1f1 (UKESM1-1-LL) uses legacy UM variable names instead of CMIP6 standard;
    # air_temperature cell_methods: max=tasmax, min=tasmin, mean=tas (same as G6-1.5K)
    t_pr_var_rename: dict = field(
        default_factory=lambda: {
            "precipitation_flux": "pr",
            "air_temperature": "tasmax",
            "air_temperature_0": "tasmin",
            "air_temperature_1": "tas",
        }
    )
    # pr file uses r2i1p1f1 label (UKESM1-1-LL); remap to match catalog member ID
    t_pr_member_rename: dict = field(
        default_factory=lambda: {
            "r2i1p1f1": "r2i1p1f2",
        }
    )


@dataclass
class UKESM_G6_1p5K_Config(BaseUKESM_Config):
    scenario: str = "G6-1.5K"
    materialized_key: str = "UKESM-G6-1.5K-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/transfer/G6-1.5K"
    t_pr_input_prefix: str = "input/tensor/UKESM/UKESM_G6-1.5K_T_PR_NETCDF/UKESM_G6-1.5K_T_PR"
    ensemble_members: list = field(default_factory=lambda: ["r12i1p1f2", "r2i1p1f2", "r3i1p1f2"])
    # tas_day files contain tas, tasmin, and tasmax
    t_pr_var_lookup: dict = field(
        default_factory=lambda: {
            "pr": ["pr_day_"],
            "tas": ["tas_day_"],
            "tasmin": ["tas_day_"],
            "tasmax": ["tas_day_"],
        }
    )
    # file variable name → CMOR name mapping
    # air_temperature cell_methods: max=tasmax, min=tasmin, mean=tas
    t_pr_var_rename: dict = field(
        default_factory=lambda: {
            "precipitation_flux": "pr",
            "air_temperature": "tasmax",
            "air_temperature_0": "tasmin",
            "air_temperature_1": "tas",
        }
    )
    # T_PR files span 2015-2100; clip to match hurs/rsds time range
    time_range: str = "2035-2084"
    # UM suite ID → CMIP6 ripf mapping for t-pr files
    t_pr_member_rename: dict = field(
        default_factory=lambda: {
            "u-dp583": "r2i1p1f2",
            "u-dp690": "r3i1p1f2",
            "u-dp691": "r12i1p1f2",
        }
    )


@dataclass
class UKESM_Historical_Config(BaseUKESM_Config):
    scenario: str = "historical"
    materialized_key: str = "UKESM-historical-icechunk"
    s3_input_prefix: str = "input/tensor/UKESM/netcdf/historical"
    source_base_url: str = (
        "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical"
    )
    ensemble_members: list = field(
        default_factory=lambda: [
            "r2i1p1f2",
            "r3i1p1f2",
            "r12i1p1f2",
        ]
    )


SCENARIO_CONFIG_MAP = {
    "SSP245": UKESM_SSP245_Config,
    "G6-1.5K": UKESM_G6_1p5K_Config,
    "historical": UKESM_Historical_Config,
}


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


def _get_netcdf_urls(config: BaseUKESM_Config, variable: str) -> list[tuple[str, str]]:
    """Return (member_id, s3_path) pairs for one variable."""
    store = from_url(f"s3://{config.s3_bucket}", region="us-west-2")

    if variable in T_PR_VARS and hasattr(config, "t_pr_input_prefix"):
        lookup = getattr(config, "t_pr_var_lookup", {})
        fname_prefixes = lookup.get(variable, [f"{variable}_day_"])
        stream = obs.list_with_delimiter(store, prefix=config.t_pr_input_prefix, return_arrow=True)
        paths = list(stream["objects"]["path"].to_numpy())
        result = []
        for path in paths:
            fname = path.split("/")[-1]
            if not (fname.endswith(".nc") and any(fname.startswith(p) for p in fname_prefixes)):
                continue
            member = (
                fname.split("_gn_")[0].split("_")[-1]
                if "_gn_" in fname
                else fname.split(".nc")[0].split("_")[-2]
            )
            result.append((member, path))
        return result

    stream = obs.list_with_delimiter(store, prefix=config.s3_input_prefix, return_arrow=True)
    paths = list(stream["objects"]["path"].to_numpy())
    result = []
    for path in paths:
        if not path.endswith(".nc"):
            continue
        if f"{variable}_".lower() not in path.lower():
            continue
        if config.ensemble_members and not any(f"_{m}_" in path for m in config.ensemble_members):
            continue
        try:
            member = path.split(".nc")[0].split("_gn")[0].split("_")[-1]
        except IndexError:
            member = "unknown"
        result.append((member, path))
    return result


def _open_netcdf_from_s3(store, path: str) -> xr.Dataset:
    reader = EagerStoreReader(store, path)
    return xr.open_dataset(reader, engine="h5netcdf", chunks="auto")


def _preprocess_ukesm(
    ds: xr.Dataset,
    config: BaseUKESM_Config,
    subset: bool = False,
) -> xr.Dataset:
    if subset:
        ds = ds.isel(time=slice(0, 360))

    # Keep only standard spatial/temporal coords; drop everything else before calendar
    # conversion to avoid auxiliary object-dtype coords (e.g. forecast_reference_time)
    # getting float NaN mixed in during convert_calendar, which breaks CF encoding at write
    keep_coords = set(ds.dims) | {"lat", "lon", "latitude", "longitude", "time"}
    ds = ds.drop_vars([c for c in ds.coords if c not in keep_coords], errors="ignore")
    # Drop data vars that use a bnds dimension (time_bnds, forecast_period_bnds, etc.)
    # chunk({"time": -1}) in to_proleptic_gregorian fails on multi-dim bnds variables
    bnds_data_vars = [v for v in ds.data_vars if any("bnds" in d for d in ds[v].dims)]
    if bnds_data_vars:
        ds = ds.drop_vars(bnds_data_vars)
    if not isinstance(config, UKESM_Historical_Config):
        ds = ds.drop_duplicates(dim="time", keep="first")
    ds = to_proleptic_gregorian(ds)
    ds = ds.drop_encoding()
    # CMORize: rename latitude/longitude → lat/lon if needed
    rename_map = {
        k: v
        for k, v in [("latitude", "lat"), ("longitude", "lon")]
        if k in ds.dims and v not in ds.dims
    }
    if rename_map:
        ds = ds.rename(rename_map)
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])

    if hasattr(config, "time_range"):
        start_year, end_year = config.time_range.split("-")
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    ds = trim_negative_precipitation(ds)
    return ds


def _derivation_logic(config: BaseUKESM_Config, variable: str = None) -> str:
    if variable in T_PR_VARS and hasattr(config, "t_pr_input_prefix"):
        member_rename = getattr(config, "t_pr_member_rename", {})
        if member_rename:
            mapping_str = ", ".join(f"{k}->{v}" for k, v in member_rename.items())
            return (
                f"UM suite IDs extracted from filename and remapped to CMIP6 ripf: {mapping_str}. "
                "Source files are private T/PR NetCDFs."
            )
        return (
            "Extracted from filename position index: filename.split('_')[1] "
            "(e.g. '001'). Positional ID stored under 'ensemble_member' dim. "
            "Source files are private T/PR NetCDFs."
        )
    return "Extracted from CMIP6 filename: path.split('.nc')[0].split('_gn')[0].split('_')[-1]. "


def _update_attrs(
    ds: xr.Dataset, var_specs: dict, config: BaseUKESM_Config, variable: str = None
) -> xr.Dataset:
    ds = update_variable_attrs(ds, var_specs)
    ds.attrs.update(
        {
            "scenario": config.scenario,
            "model": "UKESM1-0-LL",
            "Conventions": "CF-1.8",
        }
    )
    return apply_ensemble_provenance(ds, _derivation_logic(config, variable))


def _process_single_variable(
    config: BaseUKESM_Config,
    variable: str,
    repo: icechunk.Repository,
    var_specs: dict,
    overwrite: bool,
    subset: bool,
) -> None:
    log.info("variable=%s start", variable)

    var_in_store = False
    try:
        session = repo.readonly_session("main")
        existing = xr.open_dataset(session.store, engine="zarr", decode_times=False)
        var_in_store = variable in existing.data_vars
    except Exception:
        pass

    if not overwrite and var_in_store:
        log.info("variable=%s skip: already in store (use --overwrite to replace)", variable)
        return

    if variable.lower() == "dtr":
        log.info("variable=%s deriving dtr from icechunk store", variable)
        mat_cat = catalog.get(config.materialized_key)
        ds = load_dtr_from_store(mat_cat.bucket, mat_cat.prefix)
    else:
        obstore_inst = from_url(f"s3://{config.s3_bucket}", region="us-west-2")
        url_pairs = _get_netcdf_urls(config, variable)
        if not url_pairs:
            log.warning("variable=%s no files found, skipping", variable)
            return
        log.info("variable=%s found %d files", variable, len(url_pairs))
        for member, path in url_pairs:
            log.info("  member=%s path=%s", member, path)

        member_paths: dict[str, list[str]] = {}
        for member, path in url_pairs:
            member_paths.setdefault(member, []).append(path)

        member_rename = getattr(config, "t_pr_member_rename", {})
        if member_rename:
            member_paths = {member_rename.get(m, m): paths for m, paths in member_paths.items()}

        member_datasets = []
        for member, paths in sorted(member_paths.items()):
            log.info("variable=%s member=%s opening %d file(s)", variable, member, len(paths))
            time_slices = [_open_netcdf_from_s3(obstore_inst, p) for p in sorted(paths)]
            member_ds = (
                xr.concat(time_slices, dim="time", data_vars="minimal")
                if len(time_slices) > 1
                else time_slices[0]
            )
            member_ds = _preprocess_ukesm(member_ds, config, subset=subset)
            # Rename per-member before concat so members with mixed naming conventions
            # (e.g. UM legacy names vs CMIP6 standard) align on the same variable names.
            var_rename = getattr(config, "t_pr_var_rename", {})
            if var_rename:
                member_ds = member_ds.rename(
                    {k: v for k, v in var_rename.items() if k in member_ds}
                )
            member_ds = member_ds.expand_dims({"ensemble_member": [member]})
            member_datasets.append(member_ds)
            log.info("variable=%s member=%s shape=%s", variable, member, dict(member_ds.dims))

        ds = xr.concat(member_datasets, dim="ensemble_member")
        if variable in ds:
            ds = ds[[variable]]
        log.info("variable=%s concat done shape=%s", variable, dict(ds.dims))

    ds = _update_attrs(ds, var_specs, config, variable)

    session = repo.writable_session("main")
    if overwrite and var_in_store:
        write_mode = "r+"
    elif overwrite and not var_in_store:
        write_mode = "a"
    else:
        write_mode = determine_write_mode(repo)
    encoding = build_encoding_dict(ds, config.encoding["chunks"], config.encoding["shards"])
    log.info("variable=%s writing to icechunk write_mode=%s", variable, write_mode)
    write_dataset_to_icechunk(
        ds,
        session,
        encoding=None if overwrite else encoding,
        shards=None if overwrite else config.encoding["shards"],
        commit_message=f"{config.scenario}: {variable}" + (" (overwrite)" if overwrite else ""),
        write_mode=write_mode,
    )
    log.info("variable=%s done", variable)


def _run_process(
    config: BaseUKESM_Config,
    variables: list[str],
    overwrite: bool,
    subset: bool,
) -> None:
    log.info(
        "scenario=%s variables=%s overwrite=%s subset=%s",
        config.scenario,
        variables,
        overwrite,
        subset,
    )
    mat_cat = catalog.get(config.materialized_key)
    var_specs = get_var_specs(mat_cat)
    repo, _ = init_repo(mat_cat.bucket, mat_cat.prefix, readonly=False)
    for var in variables:
        _process_single_variable(config, var, repo, var_specs, overwrite, subset)
    log.info("scenario=%s all variables complete", config.scenario)


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
@click.option("--variable", multiple=True, help="Specific variable(s) to process")
@click.option("--scenario", type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())), required=True)
@click.option("--all-variables", is_flag=True, help="Process all expected variables from catalog")
@click.option("--subset/--no-subset", default=False)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing variable arrays in-place (r+ mode)",
)
def process(variable, scenario, all_variables, subset, overwrite):
    """Open NetCDF files from S3, concat, rechunk, and write to icechunk. One variable at a time."""
    config = SCENARIO_CONFIG_MAP[scenario]()

    if all_variables:
        mat_cat = catalog.get(config.materialized_key)
        variables = [var.name for var in mat_cat.expected_vars]
    elif variable:
        variables = list(variable)
    else:
        raise click.UsageError("Must specify either --variable or --all-variables")

    _run_process(config, variables, overwrite, subset)


cli.add_command(fetch)
cli.add_command(process)

if __name__ == "__main__":
    cli()
