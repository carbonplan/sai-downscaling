import json
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
from srm.config import ClusterConfig, init_repo, setup_cluster, setup_local_client
from srm.input_data.etl_config import BaseETLConfig
from srm.input_data.etl_utils import (
    CMORIZE_hurs,
    CMORIZE_pr,
    apply_ensemble_provenance,
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    load_dtr_from_store,
    trim_negative_precipitation,
    update_variable_attrs,
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180, to_proleptic_gregorian

zarr.config.set({"async.concurrency": 128})

SHARED_VARIABLES = ["tas", "rsds", "hurs", "pr", "tasmax", "tasmin", "dtr"]
SHARED_ENSEMBLE_MEMBERS = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]


@dataclass
class BaseCESM_Config(BaseETLConfig):
    has_ensemble: bool = True
    CESM_WACCM_VARIABLE_MAPPING: dict = field(
        default_factory=lambda: {
            "FSDS": "rsds",
            "TREFHT": "tas",
            "TREFHTMX": "tasmax",
            "TREFHTMN": "tasmin",
            "RHREFHT": "hurs",
            "PRECT": "pr",
        }
    )

    CESM_UNIT_MAPPING: dict = field(
        default_factory=lambda: {
            "pr": "kg m-2 s-1",
            "tas": "K",
            "tasmax": "K",
            "tasmin": "K",
            "hurs": "%",
            "rsds": "W m-2",
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
            "worker_vm_types": ["r8g.4xlarge"],
            "scheduler_vm_types": "c8g.xlarge",
        }
    )

    def __post_init__(self):
        sesh = boto3.Session()
        creds = sesh.get_credentials()
        self.region = sesh.region_name
        self.aws_creds = {
            "aws_access_key_id": creds.access_key,
            "aws_secret_access_key": creds.secret_key,
        }


@dataclass
class Pangeo_CESM_Historical_Config(BaseCESM_Config):
    scenario: str = "pangeo-historical"
    materialized_key: str = "pangeo-CESM2-WACCM-historical-icechunk"


@dataclass
class CESM_Historical_Config(BaseCESM_Config):
    scenario: str = "historical"
    catalog_key: str = "CESM2-WACCM-historical-virtual-icechunk"
    materialized_key: str = "CESM2-WACCM-historical-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-Historical/netcdf"
    has_ensemble: bool = True
    # NOTE:  email confirmation that this ensemble member maps to 001
    ensemble_members: list = field(default_factory=lambda: ["001"])


@dataclass
class CESM_SSP245_Config(BaseCESM_Config):
    scenario: str = "SSP245"
    catalog_key: str = "CESM2-WACCM-SSP245-001-005-virtual"
    catalog_key_5: str = "CESM2-WACCM-SSP245-001-005-virtual"
    catalog_key_6: str = "CESM2-WACCM-SSP245-006-virtual"
    catalog_key_7_10: str = "CESM2-WACCM-SSP245-007-010-virtual"
    materialized_key: str = "CESM2-WACCM-SSP245-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-SSP245/netcdf"
    has_ensemble: bool = True
    ensemble_members: list = field(
        default_factory=lambda: [
            "001",
            "002",
            "003",
            "004",
            "005",
            "006",
            "007",
            "008",
            "009",
            "010",
        ]
    )
    variables_6_10_only: list = field(default_factory=lambda: ["tasmax", "tasmin"])

    subset_5: list = field(default_factory=lambda: ["001", "002", "003", "004", "005"])
    subset_6: list = field(default_factory=lambda: ["006"])
    subset_7_10: list = field(default_factory=lambda: ["007", "008", "009", "010"])


@dataclass
class CESM_G6_1_5K_Config(BaseCESM_Config):
    scenario: str = "G6-1.5K"
    catalog_key: str = "CESM2-WACCM-G6-1.5K-virtual"
    materialized_key: str = "CESM2-WACCM-G6-1.5K-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-G6-1.5K/netcdf"
    ensemble_members: list = field(default_factory=lambda: ["001", "002", "003"])


SCENARIO_CONFIG_MAP = {
    "pangeo-historical": Pangeo_CESM_Historical_Config,
    "historical": CESM_Historical_Config,
    "ssp245": CESM_SSP245_Config,
    "G6-1.5K": CESM_G6_1_5K_Config,
}

# --- PROVENANCE HELPERS ---


def _attach_source_manifest(ds: xr.Dataset, url: str) -> xr.Dataset:
    prov = _capture_provenance(ds, url)
    ds.attrs["_source_manifest"] = json.dumps(prov)
    return ds


def _attach_ensemble_member(ds: xr.Dataset, url: str, enabled: bool) -> xr.Dataset:
    if not enabled:
        return ds

    ensemble = url.split(".nc")[0].split("_")[-1]
    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    ds.attrs["ensemble_member_source"] = "url"
    return ds


def preprocess_with_provenance(ds: xr.Dataset, url: str, has_ensemble: bool):
    ds = _attach_source_manifest(ds, url)
    ds = _attach_ensemble_member(ds, url, has_ensemble)
    return ds


def _capture_provenance(ds: xr.Dataset, url: str) -> dict:
    """Extracts identification markers and records exactly which keys were found."""
    found_in = []
    info = {"source_url": url}

    # Check attributes for ensemble/variant info
    for key in ["variant_label", "case", "parent_variant_label", "parent_id", "tracking_id"]:
        val = ds.attrs.get(key)
        if val:
            info[key] = val
            found_in.append(f"attr:{key}")

    info["provenance_sources"] = list(set(found_in))
    return info


def _preprocess_ensemble(ds: xr.Dataset, url: str = None) -> xr.Dataset:
    if url is None:
        raise ValueError("url parameter is required to determine ensemble member")
    case = ds.attrs.get("case", "")
    ensemble = case.rsplit(".")[-1] if case else "unknown"
    prov = _capture_provenance(ds, url)
    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    ds.attrs["_source_manifest"] = json.dumps(prov)
    return ds


def _make_fixed_ensemble_preprocess(member: str):
    def fn(ds: xr.Dataset, url: str) -> xr.Dataset:
        prov = _capture_provenance(ds, url)
        ds = ds.expand_dims({"ensemble_member": [member]})
        ds.attrs["_source_manifest"] = json.dumps(prov)
        return ds

    return fn


def _finalize_metadata(ds: xr.Dataset, config: BaseCESM_Config) -> xr.Dataset:
    """Records the parsing and the lineage chain"""

    if config.has_ensemble:
        derivation_logic = "Ensemble member derived from url or variant_label"
    else:
        derivation_logic = "No ensemble dimension (single realization dataset)"

    parent_exp = ds.attrs.get("parent_experiment_id", "unknown_parent")
    lineage = f"{parent_exp} -> {config.scenario}"

    etl_attrs = {
        "scenario": config.scenario,
        "model": "CESM2-WACCM",
        "experiment_lineage": lineage,
        "ensemble_derivation_logic": derivation_logic,  # ie, did it come from attrs / parsing the filepath.
        "processing_steps": (
            "time_drop_duplicates, lon_to_180, lat_lon_sort, trim_negative_precip"
        ),
    }

    if any(v in config.cmorization_functions for v in ds.data_vars):
        etl_attrs["processing_steps"] += ", cmorization_unit_conversion"

    ds.attrs.update(etl_attrs)

    if "_source_manifest" in ds.attrs:
        del ds.attrs["_source_manifest"]

    return apply_ensemble_provenance(ds, derivation_logic)


def get_CESM_WACCM_ds(scenario: str) -> xr.Dataset:
    """Fetches Pangeo Zarr stores and builds a comprehensive provenance manifest."""
    import intake
    from zarr.storage import ObjectStore

    experiment_id = scenario.removeprefix("pangeo-")
    cat = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    subset = cat.search(
        source_id=["CESM2-WACCM"],
        experiment_id=experiment_id,
        variable_id=SHARED_VARIABLES,
        member_id=SHARED_ENSEMBLE_MEMBERS,
        table_id="day",
    )

    datasets = []
    #  store the full audit trail for every member
    full_manifest = {}

    for zstore_url in subset.df.zstore:
        gcs_store = from_url(zstore_url, skip_signature=True)
        zarr_store = ObjectStore(gcs_store)

        ds = xr.open_dataset(
            zarr_store, engine="zarr", consolidated=True, chunks="auto"
        ).drop_encoding()

        member_id = ds.attrs.get("variant_label", "unknown")
        # Store all raw attributes and the URL for this specific member
        full_manifest[member_id] = _capture_provenance(ds, zstore_url)

        ds = _attach_source_manifest(ds, zstore_url)
        ds = ds.expand_dims({"ensemble_member": [member_id]})
        ds.attrs["ensemble_member_source"] = "variant_label"
        datasets.append(ds)

    # Combine all members into a single dataset
    combined = xr.combine_by_coords(
        datasets,
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )

    # Store the full audit trail as  JSON
    combined.ensemble_member.attrs["member_specific_provenance"] = json.dumps(full_manifest)

    return combined


# --- UTILS & CLI ---


def _get_cesm_var_from_cmip6(cmip6_var: str, config: BaseCESM_Config) -> str:
    reverse_mapping = {v: k for k, v in config.CESM_WACCM_VARIABLE_MAPPING.items()}
    return reverse_mapping.get(cmip6_var, cmip6_var)


def _get_netcdf_urls(config: BaseCESM_Config, variables: list[str]) -> list[str]:
    store = from_url(f"s3://{config.s3_bucket}", region=config.region, **config.aws_creds)
    stream = obs.list_with_delimiter(store, prefix=config.s3_input_prefix, return_arrow=True)
    netcdf_list = list(stream["objects"]["path"].to_numpy())
    cesm_vars = [_get_cesm_var_from_cmip6(var, config) for var in variables]

    return [
        f"s3://{config.s3_bucket}/{path}"
        for path in netcdf_list
        if path.endswith(".nc") and any(f".{cesm_var}." in path for cesm_var in cesm_vars)
    ]


def _standardize_vars(ds: xr.Dataset, config: BaseCESM_Config) -> xr.Dataset:
    for var, cmip6_var in config.CESM_WACCM_VARIABLE_MAPPING.items():
        if var in ds.data_vars:
            ds = ds.rename({var: cmip6_var})
    return ds


def _is_pangeo_scenario(scenario: str) -> bool:
    return scenario.startswith("pangeo-")


def _preprocess_cesm(
    ds: xr.Dataset, config: BaseCESM_Config, var: str, subset: bool = False
) -> xr.Dataset:
    ds = ds.drop_duplicates(dim="time", keep="first")
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

    ds = update_variable_attrs(ds, var_specs)
    ds = _finalize_metadata(ds, config)
    return ds


@click.group()
def cli():
    pass


@click.command()
@click.option("--scenario", type=click.Choice(list(SCENARIO_CONFIG_MAP.keys())), required=True)
@click.option("--coiled/--local", default=False)
def virtualize(scenario, coiled):
    """Virtualize netcdf files into icechunk."""
    from obspec_utils.readers import BufferedStoreReader
    from obspec_utils.wrappers import CachingReadableStore, SplittingReadableStore

    config = SCENARIO_CONFIG_MAP[scenario]()
    client = (
        setup_cluster(ClusterConfig(**config.virtualize_cluster))
        if coiled
        else setup_local_client()
    )
    try:
        base_store = from_url(f"s3://{config.s3_bucket}", region=config.region, **config.aws_creds)
        registry = ObjectStoreRegistry(
            {f"s3://{config.s3_bucket}": CachingReadableStore(SplittingReadableStore(base_store))}
        )
        parser = HDFParser(drop_variables=config.drop_variables, reader_factory=BufferedStoreReader)

        if not config.has_ensemble:
            preprocess_fn = None
        elif len(config.ensemble_members) == 1:
            preprocess_fn = _make_fixed_ensemble_preprocess(config.ensemble_members[0])
        else:
            preprocess_fn = _preprocess_ensemble
        loadable_variables = ["lat", "lon", "time"]

        if scenario == "ssp245":
            virt_cat_5 = catalog.get(config.catalog_key_5)
            virt_cat_6 = catalog.get(config.catalog_key_6)
            virt_cat_7_10 = catalog.get(config.catalog_key_7_10)
            variables_5 = [var.name for var in virt_cat_5.expected_vars]
            variables_6_10 = [var.name for var in virt_cat_7_10.expected_vars]

            netcdf_urls_all_5 = _get_netcdf_urls(config, variables_5)
            netcdf_urls_all_6_10 = _get_netcdf_urls(config, variables_6_10)

            netcdf_urls_5 = [
                path
                for path in netcdf_urls_all_5
                if any(f"CMIP6-SSP2-4.5-WACCM.{num}." in path for num in config.subset_5)
            ]
            netcdf_urls_6 = [
                path
                for path in netcdf_urls_all_6_10
                if any(f"CMIP6-SSP2-4.5-WACCM.{num}." in path for num in config.subset_6)
            ]
            netcdf_urls_7_10 = [
                path
                for path in netcdf_urls_all_6_10
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
            print(f"ds5, {combined_ds_5}")

            repo_config = icechunk.RepositoryConfig.default()
            repo_config.set_virtual_chunk_container(
                icechunk.VirtualChunkContainer(
                    f"s3://{config.s3_bucket}/", store=icechunk.s3_store(region=config.region)
                )
            )

            storage_5 = icechunk.s3_storage(
                bucket=virt_cat_5.bucket, prefix=virt_cat_5.prefix, region=config.region
            )
            repo_5 = icechunk.Repository.open_or_create(storage_5, repo_config)
            session_5 = repo_5.writable_session("main")

            combined_ds_5.vz.to_icechunk(session_5.store)
            session_5.commit(f"{scenario}: virtualized 001-005 variables {variables_5}")
            repo_5.save_config()

            # member 006 virtualized separately — ends 20691230, one day short of 007-010
            combined_ds_6 = virtualize_and_combine(
                urls=netcdf_urls_6,
                registry=registry,
                parser=parser,
                loadable_variables=loadable_variables,
                preprocess_fn=preprocess_fn,
                drop_variables=["ilev", "lev"],
            )
            print(f"ds6, {combined_ds_6}")
            storage_6 = icechunk.s3_storage(
                bucket=virt_cat_6.bucket, prefix=virt_cat_6.prefix, region=config.region
            )
            repo_6 = icechunk.Repository.open_or_create(storage_6, repo_config)
            session_6 = repo_6.writable_session("main")

            combined_ds_6.vz.to_icechunk(session_6.store)
            session_6.commit(f"{scenario}: virtualized 006 variables {variables_6_10}")
            repo_6.save_config()

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
                bucket=virt_cat_7_10.bucket, prefix=virt_cat_7_10.prefix, region=config.region
            )
            repo_7_10 = icechunk.Repository.open_or_create(storage_7_10, repo_config)
            session_7_10 = repo_7_10.writable_session("main")

            combined_ds_7_10.vz.to_icechunk(session_7_10.store)
            session_7_10.commit(f"{scenario}: virtualized 007-010 variables {variables_6_10}")
            repo_7_10.save_config()

        elif _is_pangeo_scenario(scenario):
            raise ValueError(
                "The pangeo historical data for CESM2-WACCM are already zarr and store in the pangeo intake-esm catalog"
            )

        else:  # not pangeo or ssp245 - ie ncar-provided historical + G6-1.5k
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
@click.option("--all-variables", is_flag=True, help="Process all shared variables")
@click.option("--subset/--no-subset", default=False, help="Subset (G6 only)")
def process(variable, scenario, coiled, all_variables, subset):
    config = SCENARIO_CONFIG_MAP[scenario]()
    client = (
        setup_cluster(ClusterConfig(**config.process_cluster)) if coiled else setup_local_client()
    )
    materialized_cat = catalog.get(config.materialized_key)
    var_specs = get_var_specs(materialized_cat)
    variables = SHARED_VARIABLES if all_variables else list(variable)

    try:
        if _is_pangeo_scenario(scenario):
            # (Zarr to Icechunk)
            ds = get_CESM_WACCM_ds(scenario)
            available = [v for v in variables if v in ds]
            ds = ds[available]
            ds = to_proleptic_gregorian(ds)
            ds = trim_negative_precipitation(ds)
            ds = lon_to_180(ds, lon_name="lon")
            ds = ds.sortby(["lat", "lon"])
            ds = _update_attrs(ds, var_specs, config)

            repo, session = init_repo(
                materialized_cat.bucket, materialized_cat.prefix, readonly=False
            )
            encoding = build_encoding_dict(ds, config.encoding["chunks"], config.encoding["shards"])
            write_dataset_to_icechunk(
                ds,
                session,
                encoding=encoding,
                shards=config.encoding["shards"],
                commit_message=f"{scenario}: Bulk write of {variables}",
                write_mode=determine_write_mode(repo),
            )
        else:
            # --- (Virtual store to Icechunk chunks) ncar provided historical, ssp245 and g6-1.5k---
            virtual_keys = [config.catalog_key]
            if hasattr(config, "catalog_key_6"):
                virtual_keys.append(config.catalog_key_6)
            if hasattr(config, "catalog_key_7_10"):
                virtual_keys.append(config.catalog_key_7_10)

            canonical_time = (
                catalog.get(config.catalog_key).to_xarray().time
                if hasattr(config, "catalog_key_7_10")
                else None
            )

            for var in variables:
                if var.lower() == "dtr":
                    # tasmax/tasmin only exist for members 006-010; dtr inherits that partial coverage
                    ds = load_dtr_from_store(materialized_cat.bucket, materialized_cat.prefix)
                else:
                    cesm_var = _get_cesm_var_from_cmip6(var, config)
                    var_keys = (
                        [config.catalog_key_6, config.catalog_key_7_10]
                        if hasattr(config, "variables_6_10_only")
                        and var in config.variables_6_10_only
                        else virtual_keys
                    )

                    subsets = []
                    for cat_key in var_keys:
                        _ds = catalog.get(cat_key).to_xarray()[[cesm_var]]
                        _ds = _preprocess_cesm(_ds, config, cesm_var, subset=subset)
                        subsets.append(_ds)

                    ds = (
                        xr.concat(subsets, dim="ensemble_member", join="outer")
                        if len(subsets) > 1
                        else subsets[0]
                    )
                    if hasattr(config, "ensemble_members") and "ensemble_member" in ds.dims:
                        ds = ds.reindex(ensemble_member=config.ensemble_members)
                    if canonical_time is not None and len(ds.time) < len(canonical_time):
                        ds = ds.reindex(time=canonical_time)
                ds = _update_attrs(ds, var_specs, config)

                repo, session = init_repo(
                    materialized_cat.bucket, materialized_cat.prefix, readonly=False
                )
                encoding = build_encoding_dict(
                    ds, config.encoding["chunks"], config.encoding["shards"]
                )
                write_dataset_to_icechunk(
                    ds,
                    session,
                    encoding=encoding,
                    shards=config.encoding["shards"],
                    commit_message=f"{scenario}: {var}",
                    write_mode=determine_write_mode(repo),
                )
    finally:
        client.shutdown()


cli.add_command(virtualize)
cli.add_command(process)

if __name__ == "__main__":
    cli()
