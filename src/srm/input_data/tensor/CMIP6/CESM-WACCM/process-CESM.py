import time
from dataclasses import dataclass, field

import boto3
import click
import dask
import icechunk
import obstore as obs
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
class BaseCESM_Config:
    scenario: str
    catalog_key: str

    s3_bucket: str = "carbonplan-srm"
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

    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    aws_creds: dict = field(default_factory=dict)

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


def _preprocess_ensemble(ds: xr.Dataset) -> xr.Dataset:
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
    store = from_url(f"s3://{config.s3_bucket}", region="us-west-2", **config.aws_creds)
    registry = ObjectStoreRegistry({f"s3://{config.s3_bucket}": store})
    parser = HDFParser(drop_variables=config.drop_variables)

    @dask.delayed
    def manifest_ds(url) -> xr.Dataset:
        manifest_store = parser(url=url, registry=registry)
        loadable_ds = xr.open_zarr(manifest_store, consolidated=False, zarr_format=3)
        if config.has_ensemble:
            loadable_ds = _preprocess_ensemble(loadable_ds)
        return loadable_ds

    delayed_vds = [manifest_ds(url) for url in netcdf_urls]
    vds_list = dask.compute(delayed_vds)[0]
    ds = xr.combine_by_coords(
        vds_list,
        coords="minimal",
        data_vars="minimal",
        compat="override",
        combine_attrs="override",
    )
    return ds.drop_duplicates(dim="time", keep="first")


def _standardize_vars(ds: xr.Dataset, config: BaseCESM_Config) -> xr.Dataset:
    for var, cmip6_var in config.CESM_WACCM_VARIABLE_MAPPING.items():
        if var in ds.data_vars:
            ds = ds.rename({var: cmip6_var})
    return ds


def _trim_negative(ds: xr.Dataset) -> xr.Dataset:
    if "pr" in ds.data_vars:
        ds["pr"] = ds["pr"].clip(min=0)
    return ds


def _preprocess_cesm(ds: xr.Dataset, config: BaseCESM_Config, subset: bool = False) -> xr.Dataset:
    ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False)
    ds = ds.drop_encoding()
    ds = ds.drop_vars(["ilev", "lev"], errors="ignore")
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])
    ds = _standardize_vars(ds, config)
    ds = _trim_negative(ds)
    if subset:
        ds = ds.isel(time=slice(0, 365))
    return ds


def _encoding(ds: xr.Dataset, config: BaseCESM_Config):
    encoding = {}
    for var_name in ds.data_vars:
        if var_name.endswith("_bounds") or var_name.endswith("_bnds"):
            continue
        var = ds[var_name]
        var_chunks = tuple(config.encoding["chunks"][d] for d in var.dims)
        var_shards = tuple(config.encoding["shards"][d] for d in var.dims)
        encoding[var_name] = {
            "chunks": var_chunks,
            "shards": var_shards,
        }
    return encoding


def _update_attrs(
    ds: xr.Dataset, var_specs: dict[str, VarSpec], config: BaseCESM_Config
) -> xr.Dataset:
    import cf_xarray  # noqa ignore

    for var_name in ds.data_vars:
        if var_name in config.CESM_UNIT_MAPPING:
            ds[var_name].attrs["units"] = config.CESM_UNIT_MAPPING[var_name]

        if var_name in var_specs:
            spec = var_specs[var_name]
            if spec.long_name:
                ds[var_name].attrs["long_name"] = spec.long_name
            if spec.cell_methods:
                ds[var_name].attrs["cell_methods"] = spec.cell_methods

    bnds_to_drop = [v for v in list(ds.data_vars) + list(ds.coords) if v.endswith("_bnds")]
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
            "model": "CESM2-WACCM",
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
    config: BaseCESM_Config,
):
    ds = ds.chunk(config.encoding["shards"])
    to_icechunk(ds, session, encoding=encoding, mode=write_mode)
    session.commit(commit_message)


def _determine_mode_based_on_ancestry(repo: icechunk.Repository, branch: str = "main") -> str:
    history = list(repo.ancestry(branch=branch))
    if len(history) <= 1:
        return "w"
    else:
        return "a"


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
    var_specs = _get_var_specs(cesm_cat)

    try:
        for var in variables:
            repo, session = init_repo(cesm_cat.bucket, cesm_cat.prefix, readonly=False)
            write_mode = _determine_mode_based_on_ancestry(repo)

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
                # ds = ds.drop_duplicates(dim="time", keep="first")
            else:
                netcdf_urls = _get_netcdf_urls(config, [var])
                ds = _virtualize_netcdfs(config, netcdf_urls)

            ds = _preprocess_cesm(ds, config, subset=subset)
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
def cesm(variable, scenario, coiled, subset):
    process_cesm_pipeline(
        variables=list(variable),
        scenario=scenario,
        use_coiled=coiled,
        subset=subset,
    )


cli.add_command(cesm)


if __name__ == "__main__":
    cli()
