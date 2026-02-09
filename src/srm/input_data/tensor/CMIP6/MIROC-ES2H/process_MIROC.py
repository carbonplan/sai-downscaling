import subprocess
import time
from dataclasses import dataclass, field

import click
import pandas as pd
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
from srm.etl_utils import (
    add_cf_bounds,
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    trim_negative_precipitation,
    update_variable_attrs,
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.input_data.etl_config import BaseETLConfig
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 64})


class BaseMIROC_ES2H_Config(BaseETLConfig):
    source_base_url: str = "https://www.jamstec.go.jp/swpub/public/GeoMIP"
    s3_input_prefix: str = "input/tensor/MIROC-ES2H/netcdf"

    ensemble_members: list = field(
        default_factory=lambda: [str(val).zfill(2) for val in range(1, 11)]
    )

    def __post_init__(self):
        super().__post_init__()
        self._set_source_url()

    def _set_source_url(self):
        if self.scenario == "baseline":
            self.source_url = f"{self.source_base_url}/baseline/MIROC-ES2H/day/"
        else:
            self.source_url = f"{self.source_base_url}/{self.scenario}/MIROC-ES2H/day/"


@dataclass
class MIROC_ES2H_Baseline_Config(BaseMIROC_ES2H_Config):
    scenario: str = "baseline"
    catalog_key: str = "MIROC-ES2H-baseline-icechunk"
    time_range: str = "2020-2069"


@dataclass
class MIROC_ES2H_G6_1p5K_Config(BaseMIROC_ES2H_Config):
    scenario: str = "G6-1.5K-SAI"
    catalog_key: str = "MIROC-ES2H-G6-1.5K-icechunk"
    time_range: str = "2020-2069"


SCENARIO_CONFIG_MAP = {
    "baseline": MIROC_ES2H_Baseline_Config,
    "G6-1.5K-SAI": MIROC_ES2H_G6_1p5K_Config,
}


def _fetch_netcdfs(config: BaseMIROC_ES2H_Config) -> None:
    import warnings

    warnings.warn(
        "This fetches the raw netcdf files from a VERY slow server and moves them to s3. You must have rclone installed and configured."
    )

    urls_csv = f"MIROC-ES2H-{config.scenario}-urls.csv"
    df = pd.read_html(config.source_url)[0][["Name"]].iloc[2::].dropna().reset_index()[["Name"]]
    df["Name"] = config.source_url + df["Name"]
    df.to_csv(urls_csv, index=False, header=False)

    target_remote = f"aws:{config.s3_bucket}/{config.s3_input_prefix}/"
    command = [
        "rclone",
        "copyurl",
        "--urls",
        urls_csv,
        target_remote,
        "--progress",
        "--s3-no-check-bucket",
        "--no-clobber",
        "--transfers",
        "32",
        "--checkers",
        "128",
        "--s3-upload-concurrency",
        "4",
        "--s3-chunk-size",
        "8M",
    ]
    subprocess.run(command)


def _preprocess_miroc_ensemble(ds: xr.Dataset, url: str) -> xr.Dataset:
    ensemble = url.split(".nc")[0].split("_r")[1]
    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    return ds


def _virtualize_netcdfs(variables: list[str], config: BaseMIROC_ES2H_Config) -> xr.Dataset:
    store = from_url(f"s3://{config.s3_bucket}", region="us-west-2")
    registry = ObjectStoreRegistry({f"s3://{config.s3_bucket}": store})
    parser = HDFParser()

    netcdf_urls = [
        f"s3://{config.s3_bucket}/{config.s3_input_prefix}/{var}_{config.scenario}_r{ensm}.nc"
        for var in variables
        for ensm in config.ensemble_members
    ]

    return virtualize_and_combine(netcdf_urls, registry, parser, _preprocess_miroc_ensemble)


def _preprocess_cmip6(
    ds: xr.Dataset, config: BaseMIROC_ES2H_Config, subset: bool = False
) -> xr.Dataset:
    ds = ds.drop_encoding()
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])
    ds = trim_negative_precipitation(ds)
    if subset:
        ds = ds.isel(time=slice(0, 365))
    return ds


def _update_attrs(ds: xr.Dataset, var_specs: dict, config: BaseMIROC_ES2H_Config) -> xr.Dataset:
    ds = update_variable_attrs(ds, var_specs)
    ds = add_cf_bounds(ds)

    ds.attrs.update(
        {
            "scenario": config.scenario,
            "time_range": config.time_range,
            "model": "MIROC-ES2H",
            "Conventions": "CF-1.8",
        }
    )

    return ds


def process_miroc_pipeline(
    variables: list[str],
    scenario: str = "baseline",
    fetch: bool = False,
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

    miroc_cat = catalog.get(config.catalog_key)
    var_specs = get_var_specs(miroc_cat)

    try:
        for var in variables:
            repo, session = init_repo(miroc_cat.bucket, miroc_cat.prefix, readonly=False)
            write_mode = determine_write_mode(repo)

            ds = _virtualize_netcdfs([var], config)
            ds = _preprocess_cmip6(ds, config, subset=subset)
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
    help="CMIP6 variables to process",
)
@click.option(
    "--scenario",
    type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())),
    required=True,
    help="which CMIP6 scenario to process",
)
@click.option("--fetch/--no-fetch", default=False)
@click.option("--coiled/--local", default=False)
@click.option("--subset/--no-subset", default=False)
def miroc(variable, scenario, fetch, coiled, subset):
    process_miroc_pipeline(
        variables=list(variable),
        scenario=scenario,
        fetch=fetch,
        use_coiled=coiled,
        subset=subset,
    )


cli.add_command(miroc)


if __name__ == "__main__":
    cli()
