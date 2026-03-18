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
    ENSEMBLE_MEMBER_MAPPING,
    CMORIZE_hurs,
    CMORIZE_pr,
    add_cf_bounds,
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    load_dtr_from_store,
    remap_ensemble_members,
    trim_negative_precipitation,
    update_variable_attrs,
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 128})


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

    virtualize_cluster: dict = field(
        default_factory=lambda: {
            "n_workers": [6, 24],
            "worker_vm_types": ["r8g.4xlarge"],
            "scheduler_vm_types": "c8g.2xlarge",
        }
    )

    process_cluster: dict = field(
        default_factory=lambda: {
            "n_workers": [4, 16],
            "worker_vm_types": ["m8g.2xlarge"],
            "scheduler_vm_types": "c8g.xlarge",
        }
    )

    def __post_init__(self):
        # super().__post_init__()
        sesh = boto3.Session()
        creds = sesh.get_credentials()
        self.aws_creds = {
            "aws_access_key_id": creds.access_key,
            "aws_secret_access_key": creds.secret_key,
        }


@dataclass
class CESM_Historical_Config(BaseCESM_Config):
    scenario: str = "historical"
    catalog_key: str = "CESM2-WACCM-historical-virtual"
    materialized_key: str = "CESM2-WACCM-historical-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-Historical/netcdf"
    has_ensemble: bool = False


@dataclass
class CESM_SSP245_Config(BaseCESM_Config):
    scenario: str = "SSP245"
    catalog_key: str = "CESM2-WACCM-SSP245-001-005-virtual"
    catalog_key_5: str = "CESM2-WACCM-SSP245-001-005-virtual"
    catalog_key_7_10: str = "CESM2-WACCM-SSP245-007-010-virtual"
    materialized_key: str = "CESM2-WACCM-SSP245-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-SSP245/netcdf"
    has_ensemble: bool = True

    subset_5: list = field(default_factory=lambda: ["001", "002", "003", "004", "005"])
    subset_7_10: list = field(default_factory=lambda: ["007", "008", "009", "010"])


@dataclass
class CESM_G6_1_5K_Config(BaseCESM_Config):
    scenario: str = "G6-1.5K"
    catalog_key: str = "CESM2-WACCM-G6-1.5K-virtual"
    materialized_key: str = "CESM2-WACCM-G6-1.5K-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-G6-1.5K/netcdf"
    has_ensemble: bool = True


SCENARIO_CONFIG_MAP = {
    "historical": CESM_Historical_Config,
    "SSP245": CESM_SSP245_Config,
    "G6-1.5K": CESM_G6_1_5K_Config,
}


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


def _preprocess_ensemble(ds: xr.Dataset, url: str = None) -> xr.Dataset:
    if url is None:
        raise ValueError("url parameter is required to determine ensemble member")
    ensemble = ds.attrs["case"].rsplit(".")[-1]
    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    return ds


def _preprocess_cesm(
    ds: xr.Dataset,
    config: BaseCESM_Config,
    var: str,
    subset: bool = False,
) -> xr.Dataset:
    ds = ds.drop_duplicates(dim="time", keep="first")
    ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False, align_on="date")
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


def _standardize_vars(ds: xr.Dataset, config: BaseCESM_Config) -> xr.Dataset:
    for var, cmip6_var in config.CESM_WACCM_VARIABLE_MAPPING.items():
        if var in ds.data_vars:
            ds = ds.rename({var: cmip6_var})
    return ds


def _update_attrs(ds: xr.Dataset, var_specs: dict, config: BaseCESM_Config) -> xr.Dataset:
    for var_name in ds.data_vars:
        if var_name in config.CESM_UNIT_MAPPING:
            ds[var_name].attrs["units"] = config.CESM_UNIT_MAPPING[var_name]

    ds = update_variable_attrs(ds, var_specs)
    ds = add_cf_bounds(ds)

    ds.attrs.update(
        {
            "scenario": config.scenario,
            "model": "CESM2-WACCM",
            "Conventions": "CF-1.8",
        }
    )

    return ds


@click.group()
def cli():
    pass


@click.command()
@click.option("--scenario", type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())), required=True)
@click.option("--coiled/--local", default=False)
def virtualize(scenario, coiled):
    from obspec_utils.readers import BufferedStoreReader
    from obspec_utils.wrappers import CachingReadableStore, SplittingReadableStore

    config = SCENARIO_CONFIG_MAP[scenario]()

    if coiled:
        from srm.config import ClusterConfig

        client = setup_cluster(ClusterConfig(**config.virtualize_cluster))
    else:
        client = setup_local_client()

    try:
        base_store = from_url(f"s3://{config.s3_bucket}", region="us-west-2", **config.aws_creds)
        registry = ObjectStoreRegistry(
            {f"s3://{config.s3_bucket}": CachingReadableStore(SplittingReadableStore(base_store))}
        )
        parser = HDFParser(drop_variables=config.drop_variables, reader_factory=BufferedStoreReader)

        preprocess_fn = _preprocess_ensemble if config.has_ensemble else None
        loadable_variables = ["lat", "lon", "time"]

        if scenario == "SSP245":
            virt_cat_5 = catalog.get(config.catalog_key_5)
            virt_cat_7_10 = catalog.get(config.catalog_key_7_10)
            variables_5 = [var.name for var in virt_cat_5.expected_vars]
            variables_7_10 = [var.name for var in virt_cat_7_10.expected_vars]

            netcdf_urls_all = _get_netcdf_urls(config, variables_5)

            netcdf_urls_5 = [
                path
                for path in netcdf_urls_all
                if any(f"CMIP6-SSP2-4.5-WACCM.{num}." in path for num in config.subset_5)
            ]
            netcdf_urls_7_10 = [
                path
                for path in netcdf_urls_all
                if any(f"CMIP6-SSP2-4.5-WACCM.{num}." in path for num in config.subset_7_10)
            ]

            combined_ds_5 = virtualize_and_combine(
                urls=netcdf_urls_5,
                registry=registry,
                parser=parser,
                loadable_variables=loadable_variables,
                drop_variables=["ilev", "lev"],
                preprocess_fn=preprocess_fn,
            )
            print(f"ds6, {combined_ds_5}")

            repo_config = icechunk.RepositoryConfig.default()
            repo_config.set_virtual_chunk_container(
                icechunk.VirtualChunkContainer(
                    f"s3://{config.s3_bucket}/", store=icechunk.s3_store(region="us-west-2")
                )
            )

            storage_5 = icechunk.s3_storage(
                bucket=virt_cat_5.bucket, prefix=virt_cat_5.prefix, region="us-west-2"
            )
            repo_5 = icechunk.Repository.open_or_create(storage_5, repo_config)
            session_5 = repo_5.writable_session("main")

            combined_ds_5.vz.to_icechunk(session_5.store)
            session_5.commit(f"{scenario}: virtualized 001-005 variables {variables_5}")
            repo_5.save_config()

            combined_ds_7_10 = virtualize_and_combine(
                urls=netcdf_urls_7_10,
                registry=registry,
                parser=parser,
                loadable_variables=loadable_variables,
                preprocess_fn=preprocess_fn,
                drop_variables=["ilev", "lev"],
            )
            print(f"ds7_10, {combined_ds_7_10}")
            storage_7_10 = icechunk.s3_storage(
                bucket=virt_cat_7_10.bucket, prefix=virt_cat_7_10.prefix, region="us-west-2"
            )
            repo_7_10 = icechunk.Repository.open_or_create(storage_7_10, repo_config)
            session_7_10 = repo_7_10.writable_session("main")

            combined_ds_7_10.vz.to_icechunk(session_7_10.store)
            session_7_10.commit(f"{scenario}: virtualized 007-010 variables {variables_7_10}")
            repo_7_10.save_config()

        else:
            virt_cat = catalog.get(config.catalog_key)
            variables = [var.name for var in virt_cat.expected_vars]

            netcdf_urls = _get_netcdf_urls(config, variables)
            combined_ds = virtualize_and_combine(
                urls=netcdf_urls,
                registry=registry,
                parser=parser,
                loadable_variables=loadable_variables,
                drop_variables=["ilev", "lev"],
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
    config = SCENARIO_CONFIG_MAP[scenario]()

    if coiled:
        from srm.config import ClusterConfig

        client = setup_cluster(ClusterConfig(**config.process_cluster))
    else:
        client = setup_local_client()

    mat_key = getattr(config, "materialized_key", f"CESM2-WACCM-{scenario}-icechunk")
    materialized_cat = catalog.get(mat_key)
    var_specs = get_var_specs(materialized_cat)

    if all_variables:
        variables = [var.name for var in materialized_cat.expected_vars]
        variables = [_get_cesm_var_from_cmip6(var, config) for var in variables]

    elif variable:
        variables = list(variable)
    else:
        raise click.UsageError("Must specify either --variable or --all-variables")
    try:
        for var in variables:
            if var.lower() == "dtr":
                ds = load_dtr_from_store(
                    materialized_cat.bucket, materialized_cat.prefix, config.encoding["shards"]
                )
            else:
                if scenario == "SSP245":
                    virt_ds_5 = catalog.get(config.catalog_key_5).to_xarray()
                    virt_ds_7_10 = catalog.get(config.catalog_key_7_10).to_xarray()

                    ds_5 = virt_ds_5[[var]]
                    print(f"ds_5: {ds_5}")
                    ds_7_10 = virt_ds_7_10[[var]]
                    print(f"ds_7_10: {ds_7_10}")

                    ds = xr.combine_by_coords(
                        [ds_5, ds_7_10],
                        coords="minimal",
                        data_vars="minimal",
                        compat="override",
                        combine_attrs="override",
                    )
                    print(f"ds_combined: {ds}")
                else:
                    virt_ds = catalog.get(config.catalog_key).to_xarray()
                    ds = virt_ds[[var]]
                ds = _preprocess_cesm(ds, config, var, subset=subset)

            ds = _update_attrs(ds, var_specs, config)

            repo, session = init_repo(
                materialized_cat.bucket, materialized_cat.prefix, readonly=False
            )
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


@click.command()
@click.option("--scenario", type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())), required=True)
def remap_ensemble(scenario):
    config = SCENARIO_CONFIG_MAP[scenario]()
    if not config.has_ensemble:
        click.echo(f"{scenario} has no ensemble members, skipping.")
        return
    mat_cat = catalog.get(config.materialized_key)
    remap_ensemble_members(mat_cat.bucket, mat_cat.prefix, ENSEMBLE_MEMBER_MAPPING)


cli.add_command(virtualize)
cli.add_command(process)
cli.add_command(remap_ensemble)
if __name__ == "__main__":
    cli()
