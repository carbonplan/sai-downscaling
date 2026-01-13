import time
from dataclasses import dataclass, field

import click
import obstore as obs
import xarray as xr
import zarr
from obstore.store import from_url

from srm import catalog
from srm.config import (
    init_repo,
    setup_cluster,
    setup_local_client,
)
from srm.input_data.etl_utils import (
    add_cf_bounds,
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    trim_negative_precipitation,
    update_variable_attrs,
    write_dataset_to_icechunk,
    compute_wind_speed,
)
from srm.input_data.etl_config import BaseETLConfig

from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 64})


@dataclass
class BaseUKESM_Config(BaseETLConfig):
    s3_input_prefix: str = ""

    ensemble_members: list = field(default_factory=lambda: ["001", "002", "003"])


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


SCENARIO_CONFIG_MAP = {
    "SSP245": UKESM_SSP245_Config,
    "G6-1.5K": UKESM_G6_1p5K_Config,
    "SSP245-SFCWIND": UKESM_SSP245_SFCWIND_Config,
    "G6-1.5K-SFCWIND": UKESM_G6_1p5K_SFCWIND_Config,
}


def _get_netcdf_urls(config: BaseUKESM_Config, variables: list[str]) -> list[str]:
    store = from_url(
        f"s3://{config.s3_bucket}", region="us-west-2", skip_signature=True
    )

    stream = obs.list_with_delimiter(
        store, prefix=config.s3_input_prefix, return_arrow=True
    )
    netcdf_list = list(stream["objects"]["path"].to_numpy())
    filtered_urls = [
        f"s3://{config.s3_bucket}/{path}"
        for path in netcdf_list
        if path.endswith(".nc") and any(f"{var}_" in path for var in variables)
    ]

    return filtered_urls


def _preprocess_ensemble(ds: xr.Dataset) -> xr.Dataset:
    ensemble = ds.encoding["source"].split(".nc")[0].split("_r")[1][0].zfill(3)
    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    return ds


def _open_ukesm_dataset(config: BaseUKESM_Config, netcdf_urls: list[str]) -> xr.Dataset:
    ds = xr.open_mfdataset(
        netcdf_urls,
        engine="h5netcdf",
        chunks="auto",
        preprocess=_preprocess_ensemble,
        parallel=True,
        coords="minimal",
        data_vars="minimal",
        compat="override",
    )
    return ds


def _preprocess_ukesm(
    ds: xr.Dataset,
    config: BaseUKESM_Config,
    derive_wind: bool = False,
    subset: bool = False,
) -> xr.Dataset:
    ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False, align_on="date")
    ds = ds.drop_encoding()
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])

    if derive_wind:
        ds["sfcWind"] = compute_wind_speed(
            ds_u=ds[["uas"]], ds_v=ds[["vas"]], u_var_name="uas", v_var_name="vas"
        )["sfcWind"]
        ds = ds.drop_vars(["uas", "vas"])

    ds = trim_negative_precipitation(ds)
    if subset:
        ds = ds.isel(time=slice(0, 365))
    return ds


def _update_attrs(
    ds: xr.Dataset, var_specs: dict, config: BaseUKESM_Config
) -> xr.Dataset:
    ds = update_variable_attrs(ds, var_specs)
    ds = add_cf_bounds(ds)

    ds.attrs.update(
        {
            "scenario": config.scenario,
            "model": "UKESM1-0-LL",
            "Conventions": "CF-1.8",
        }
    )

    return ds


def process_ukesm_pipeline(
    variables: list[str],
    scenario: str = "G6-1.5K",
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

    ukesm_cat = catalog.get(config.catalog_key)
    var_specs = get_var_specs(ukesm_cat)

    is_sfcwind_scenario = "SFCWIND" in scenario

    try:
        for var in variables:
            repo, session = init_repo(
                ukesm_cat.bucket, ukesm_cat.prefix, readonly=False
            )
            write_mode = determine_write_mode(repo)

            if is_sfcwind_scenario and var.lower() == "sfcwind":
                netcdf_urls = _get_netcdf_urls(config, ["uas", "vas"])
                ds = _open_ukesm_dataset(config, netcdf_urls)
                ds = _preprocess_ukesm(ds, config, derive_wind=True, subset=subset)
            elif is_sfcwind_scenario:
                raise ValueError(
                    f"SFCWIND scenario only supports sfcWind variable, got {var}"
                )
            else:
                if var.lower() == "sfcwind":
                    raise ValueError(
                        f"Use {scenario}-SFCWIND scenario for sfcWind variable"
                    )
                netcdf_urls = _get_netcdf_urls(config, [var])
                ds = _open_ukesm_dataset(config, netcdf_urls)
                ds = _preprocess_ukesm(ds, config, derive_wind=False, subset=subset)

            ds = _update_attrs(ds, var_specs, config)
            encoding = build_encoding_dict(
                ds, config.encoding["chunks"], config.encoding["shards"]
            )

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
def ukesm(variable, scenario, coiled, subset):
    process_ukesm_pipeline(
        variables=list(variable),
        scenario=scenario,
        use_coiled=coiled,
        subset=subset,
    )


cli.add_command(ukesm)


if __name__ == "__main__":
    cli()
