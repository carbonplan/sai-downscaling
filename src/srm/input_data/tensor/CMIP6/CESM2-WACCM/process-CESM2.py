import time
from dataclasses import dataclass, field

import boto3
import click
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
    CMORIZE_hurs,
    CMORIZE_pr,
    add_cf_bounds,
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    trim_negative_precipitation,
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 64})


@dataclass
class BaseCESM_Config(BaseETLConfig):
    s3_input_prefix: str = ""

    CESM_WACCM_VARIABLE_MAPPING: dict = field(
        default_factory=lambda: {
            "FLDS": "rlds",
            "FSDS": "rsds",
            "PS": "ps",
            "TREFHT": "tas",
            "TREFHTMX": "tasmax",
            "TREFHTMN": "tasmin",
            "QREFHT": "huss",
            "RHREFHT": "hurs",
            "U10": "sfcWind",
            "PRECT": "pr",
        }
    )

    CESM_UNIT_MAPPING: dict = field(
        default_factory=lambda: {
            "pr": "kg m-2 s-1",
            "tas": "K",
            "tasmin": "K",
            "tasmax": "K",
            "hurs": "%",
            "rsds": "W m-2",
            "sfcWind": "m s-1",
            "huss": "1",
            "rlds": "W m-2",
            "ps": "Pa",
        }
    )

    cmorization_functions: dict = field(
        default_factory=lambda: {
            "pr": CMORIZE_pr,
            "hurs": CMORIZE_hurs,
        }
    )

    drop_variables: list = field(
        default_factory=lambda: [
            "gw",
            "hyam",
            "hybm",
            "P0",
            "hyai",
            "hybi",
            "ndbase",
            "nsbase",
            "nbdate",
            "nbsec",
            "mdt",
            "date",
            "datesec",
            "time_bnds",
            "date_written",
            "time_written",
            "ndcur",
            "nscur",
            "co2vmr",
            "ch4vmr",
            "n2ovmr",
            "f11vmr",
            "f12vmr",
            "sol_tsi",
            "nsteph",
        ]
    )

    has_ensemble: bool = False

    aws_creds: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        sesh = boto3.Session()
        creds = sesh.get_credentials()
        self.aws_creds = {
            "aws_access_key_id": creds.access_key,
            "aws_secret_access_key": creds.secret_key,
        }


@dataclass
class CESM_Historical_Config(BaseCESM_Config):
    scenario: str = "historical"
    catalog_key: str = "CESM2-WACCM-Historical-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-Historical/netcdf"
    has_ensemble: bool = False


@dataclass
class CESM_SSP245_Config(BaseCESM_Config):
    scenario: str = "SSP245"
    catalog_key: str = "CESM2-WACCM-SSP245-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-SSP245/netcdf"
    has_ensemble: bool = True

    subset_6: list = field(default_factory=lambda: ["001", "002", "003", "004", "005"])
    subset_7_10: list = field(default_factory=lambda: ["007", "008", "009", "010"])


@dataclass
class CESM_G6_1p5K_Config(BaseCESM_Config):
    scenario: str = "G6-1.5K"
    catalog_key: str = "CESM2-WACCM-G6-1.5K-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-G6-1.5K/netcdf"
    has_ensemble: bool = True


SCENARIO_CONFIG_MAP = {
    "historical": CESM_Historical_Config,
    "SSP245": CESM_SSP245_Config,
    "G6-1.5K": CESM_G6_1p5K_Config,
}


def _preprocess_ensemble(ds: xr.Dataset, url: str) -> xr.Dataset:
    ensemble = ds.attrs["case"].rsplit(".")[-1]
    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    return ds


def _get_cesm_var_from_cmip6(cmip6_var: str, config: BaseCESM_Config) -> str:
    reverse_mapping = {v: k for k, v in config.CESM_WACCM_VARIABLE_MAPPING.items()}
    return reverse_mapping.get(cmip6_var, cmip6_var)


def _get_netcdf_urls(config: BaseCESM_Config, variables: list[str]) -> list[str]:
    store = from_url(f"s3://{config.s3_bucket}", region="us-west-2", **config.aws_creds)

    stream = obs.list_with_delimiter(store, prefix=config.s3_input_prefix, return_arrow=True)
    netcdf_list = list(stream["objects"]["path"].to_numpy())

    cesm_vars = [_get_cesm_var_from_cmip6(var, config) for var in variables]

    filtered_urls = [
        f"s3://{config.s3_bucket}/{path}"
        for path in netcdf_list
        if path.endswith(".nc") and any(f".{cesm_var}." in path for cesm_var in cesm_vars)
    ]

    return filtered_urls


def _virtualize_netcdfs(config: BaseCESM_Config, netcdf_urls: list[str]) -> xr.Dataset:
    # https://github.com/zarr-developers/VirtualiZarr/blob/main/examples/V2/goes_with_caching_stores.py
    from obspec_utils.readers import BufferedStoreReader
    from obspec_utils.wrappers import CachingReadableStore, SplittingReadableStore

    base_store = from_url(f"s3://{config.s3_bucket}", region="us-west-2", **config.aws_creds)

    splitting_store = SplittingReadableStore(base_store)

    #  256MB
    max_size = 512 * 1024 * 1024
    caching_store = CachingReadableStore(splitting_store, max_size=max_size)

    registry = ObjectStoreRegistry({f"s3://{config.s3_bucket}": caching_store})
    parser = HDFParser(drop_variables=config.drop_variables, reader_factory=BufferedStoreReader)

    preprocess_fn = _preprocess_ensemble if config.has_ensemble else None
    ds = virtualize_and_combine(netcdf_urls, registry, parser, preprocess_fn)
    return ds.drop_duplicates(dim="time", keep="first")


def _standardize_vars(ds: xr.Dataset, config: BaseCESM_Config) -> xr.Dataset:
    for var, cmip6_var in config.CESM_WACCM_VARIABLE_MAPPING.items():
        if var in ds.data_vars:
            ds = ds.rename({var: cmip6_var})
    return ds


def _preprocess_cesm(
    ds: xr.Dataset, config: BaseCESM_Config, var: str, subset: bool = False
) -> xr.Dataset:
    ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False)
    ds = ds.drop_encoding()
    ds = ds.drop_vars(["ilev", "lev"], errors="ignore")
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])
    ds = _standardize_vars(ds, config)
    ds = trim_negative_precipitation(ds)

    if var in config.cmorization_functions:
        ds = config.cmorization_functions[var](ds, var)

    if subset:
        ds = ds.isel(time=slice(0, 365))
    return ds


def _update_attrs(ds: xr.Dataset, var_specs: dict, config: BaseCESM_Config) -> xr.Dataset:
    for var_name in ds.data_vars:
        if var_name in config.CESM_UNIT_MAPPING:
            ds[var_name].attrs["units"] = config.CESM_UNIT_MAPPING[var_name]

        if var_name in var_specs:
            spec = var_specs[var_name]
            if spec.long_name:
                ds[var_name].attrs["long_name"] = spec.long_name
            if spec.cell_methods:
                ds[var_name].attrs["cell_methods"] = spec.cell_methods

    ds = add_cf_bounds(ds)

    ds.attrs.update(
        {
            "scenario": config.scenario,
            "model": "CESM2-WACCM",
            "Conventions": "CF-1.8",
        }
    )

    return ds


def process_cesm_pipeline(
    variables: list[str],
    scenario: str = "historical",
    use_coiled: bool = False,
    subset: bool = False,
):
    config_class = SCENARIO_CONFIG_MAP.get(scenario)
    if config_class is None:
        raise ValueError(f"unknown scenario: {scenario}")

    config = config_class()

    if use_coiled:
        client = setup_cluster(config.cluster)
    else:
        client = setup_local_client()

    cesm_cat = catalog.get(config.catalog_key)
    var_specs = get_var_specs(cesm_cat)

    try:
        for var in variables:
            repo, session = init_repo(cesm_cat.bucket, cesm_cat.prefix, readonly=False)
            write_mode = determine_write_mode(repo)

            if scenario == "SSP245":
                netcdf_urls_all = _get_netcdf_urls(config, [var])

                netcdf_urls_6 = [
                    path
                    for path in netcdf_urls_all
                    if any(f"CMIP6-SSP2-4.5-WACCM.{num}." in path for num in config.subset_6)
                ]
                netcdf_urls_7_10 = [
                    path
                    for path in netcdf_urls_all
                    if any(f"CMIP6-SSP2-4.5-WACCM.{num}." in path for num in config.subset_7_10)
                ]

                ds_6 = _virtualize_netcdfs(config, netcdf_urls_6)
                ds_7_10 = _virtualize_netcdfs(config, netcdf_urls_7_10)
                ds = xr.combine_by_coords(
                    [ds_6, ds_7_10],
                    coords="minimal",
                    data_vars="minimal",
                    compat="override",
                    combine_attrs="override",
                )
            else:
                netcdf_urls = _get_netcdf_urls(config, [var])
                ds = _virtualize_netcdfs(config, netcdf_urls)

            ds = _preprocess_cesm(ds, config, var, subset=subset)

            ds = _update_attrs(ds, var_specs, config)
            encoding = build_encoding_dict(ds, config.encoding["chunks"], config.encoding["shards"])

            write_dataset_to_icechunk(
                ds,
                session,
                encoding=encoding,
                shards=config.encoding["shards"],
                commit_message=f"{scenario}: {var}",
                write_mode=write_mode,
            )

            time.sleep(5)

    finally:
        client.shutdown()


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--variable",
    multiple=True,
    required=True,
    help="CESM variables to process",
)
@click.option(
    "--scenario",
    type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())),
    required=True,
    help="which CESM scenario to process",
)
@click.option("--coiled/--local", default=False)
@click.option("--subset/--no-subset", default=False)
def cesm2(variable, scenario, coiled, subset):
    process_cesm_pipeline(
        variables=list(variable),
        scenario=scenario,
        use_coiled=coiled,
        subset=subset,
    )


cli.add_command(cesm2)


if __name__ == "__main__":
    cli()
