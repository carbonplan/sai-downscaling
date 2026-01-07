import subprocess
import time
from dataclasses import dataclass, field

import click
import dask
import icechunk
import pandas as pd
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk
from obstore.store import from_url
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry

from srm import catalog
from srm.config import (
    ClusterConfig,
    VarSpec,
    init_repo,
    setup_cluster,
    setup_local_client,
)
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 64})


def _get_var_specs(catalog_entry) -> dict[str, VarSpec]:
    return {var.name: var for var in catalog_entry.expected_vars}


@dataclass
class BaseMIROC_ES2H_Config:
    scenario: str
    catalog_key: str

    source_base_url: str = "https://www.jamstec.go.jp/swpub/public/GeoMIP"
    s3_bucket: str = "carbonplan-srm"
    s3_input_prefix: str = "input/tensor/MIROC-ES2H/netcdf"

    ensemble_members: list = field(
        default_factory=lambda: [str(val).zfill(2) for val in range(1, 11)]
    )

    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    def __post_init__(self):
        self.dataset_entry = catalog.get(self.catalog_key)
        self.bucket = self.dataset_entry.bucket
        self.prefix = self.dataset_entry.prefix
        self.output_path = self.dataset_entry.path
        self.all_variables = [var.name for var in self.dataset_entry.expected_vars]
        self.encoding = {
            "chunks": self.dataset_entry.expected_chunks,
            "shards": self.dataset_entry.expected_shards,
        }
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
    df = (
        pd.read_html(config.source_url)[0][["Name"]]
        .iloc[2::]
        .dropna()
        .reset_index()[["Name"]]
    )
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


def _virtualize_netcdfs(
    variables: list[str], config: BaseMIROC_ES2H_Config
) -> xr.Dataset:
    store = from_url(f"s3://{config.s3_bucket}", region="us-west-2")
    registry = ObjectStoreRegistry({f"s3://{config.s3_bucket}": store})
    parser = HDFParser()

    netcdf_urls = [
        f"s3://{config.s3_bucket}/{config.s3_input_prefix}/{var}_{config.scenario}_r{ensm}.nc"
        for var in variables
        for ensm in config.ensemble_members
    ]

    @dask.delayed
    def manifest_ds(url) -> xr.Dataset:
        ensemble = url.split(".nc")[0].split("_r")[1]
        manifest_store = parser(url=url, registry=registry)
        loadable_ds = xr.open_zarr(manifest_store, consolidated=False, zarr_format=3)
        loadable_ds = loadable_ds.expand_dims({"ensemble_member": [ensemble]})
        return loadable_ds

    delayed_vds = [manifest_ds(url) for url in netcdf_urls]
    vds_list = dask.compute(delayed_vds)[0]
    return xr.combine_by_coords(
        vds_list, coords="minimal", data_vars="minimal", compat="override"
    )


def _trim_negative(ds: xr.Dataset) -> xr.Dataset:
    if "pr" in ds.data_vars:
        ds["pr"] = ds["pr"].clip(min=0)
    return ds


def _preprocess_cmip6(
    ds: xr.Dataset, config: BaseMIROC_ES2H_Config, subset: bool = False
) -> xr.Dataset:
    ds = ds.drop_encoding()
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])
    ds = _trim_negative(ds)
    # quick way to test the pipeline with a year subset
    if subset:
        ds = ds.isel(time=slice(0, 365))
    return ds


def _encoding(ds: xr.Dataset, config: BaseMIROC_ES2H_Config):
    encoding = {}
    target_vars = [
        str(v) for v in ds.data_vars if not str(v).endswith(("_bounds", "_bnds"))
    ]

    for var_name in target_vars:
        var = ds[var_name]
        var_chunks = tuple(config.encoding["chunks"][str(d)] for d in var.dims)
        var_shards = tuple(config.encoding["shards"][str(d)] for d in var.dims)
        encoding[var_name] = {
            "chunks": var_chunks,
            "shards": var_shards,
        }
    return encoding


def _update_attrs(
    ds: xr.Dataset, var_specs: dict[str, VarSpec], config: BaseMIROC_ES2H_Config
) -> xr.Dataset:
    import cf_xarray  # noqa ignore

    for var_name, spec in var_specs.items():
        if var_name in ds.data_vars:
            ds[var_name].attrs["units"] = spec.units
            if spec.long_name:
                ds[var_name].attrs["long_name"] = spec.long_name
            if spec.cell_methods:
                ds[var_name].attrs["cell_methods"] = spec.cell_methods

    bnds_to_drop = [
        v for v in list(ds.data_vars) + list(ds.coords) if v.endswith("_bnds")
    ]
    ds = ds.drop_vars(bnds_to_drop, errors="ignore")

    ds = ds.cf.add_bounds("time")
    ds = ds.cf.add_bounds("lat")
    ds = ds.cf.add_bounds("lon")

    bounds_vars = [v for v in ds.data_vars if v.endswith("_bounds")]
    if bounds_vars:
        ds = ds.set_coords(bounds_vars)

    ds.attrs.update(
        {
            "scenario": config.scenario,
            "time_range": config.time_range,
            "model": "MIROC-ES2H",
            "Conventions": "CF-1.8",
        }
    )

    return ds


def write_to_icechunk(
    ds: xr.Dataset,
    session,
    encoding: dict,
    commit_message: str,
    write_mode: str,
    config: BaseMIROC_ES2H_Config,
):
    ds = ds.chunk(config.encoding["shards"])
    to_icechunk(ds, session, encoding=encoding, mode=write_mode)
    session.commit(commit_message)


def _determine_mode_based_on_ancestry(
    repo: icechunk.Repository, branch: str = "main"
) -> str:
    history = list(repo.ancestry(branch=branch))
    if len(history) <= 1:
        return "w"
    else:
        return "a"


def process_cmip6_pipeline(
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

    cmip6_cat = catalog.get(config.catalog_key)
    var_specs = _get_var_specs(cmip6_cat)

    try:
        for var in variables:
            repo, session = init_repo(
                cmip6_cat.bucket, cmip6_cat.prefix, readonly=False
            )
            write_mode = _determine_mode_based_on_ancestry(repo)

            ds = _virtualize_netcdfs([var], config)
            ds = _preprocess_cmip6(ds, config, subset=subset)
            ds = _update_attrs(ds, var_specs, config)
            encoding = _encoding(ds, config)

            write_to_icechunk(
                ds,
                session,
                encoding=encoding,
                commit_message=f"{scenario}: {var}",
                write_mode=write_mode,
                config=config,
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
def cmip6(variable, scenario, fetch, coiled, subset):
    process_cmip6_pipeline(
        variables=list(variable),
        scenario=scenario,
        fetch=fetch,
        use_coiled=coiled,
        subset=subset,
    )


cli.add_command(cmip6)


if __name__ == "__main__":
    cli()
