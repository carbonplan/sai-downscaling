import json
import re
from dataclasses import dataclass, field
from typing import Literal

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
    add_cf_bounds,
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    trim_negative_precipitation,
    update_variable_attrs,
    virtualize_and_combine,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 128})

SHARED_VARIABLES = ["tas", "rsds", "hurs", "pr"]
SHARED_ENSEMBLE_MEMBERS = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]


@dataclass
class BaseCESM_Config(BaseETLConfig):
    CESM_WACCM_VARIABLE_MAPPING: dict = field(
        default_factory=lambda: {
            "FSDS": "rsds",
            "TREFHT": "tas",
            "RHREFHT": "hurs",
            "PRECT": "pr",
        }
    )

    CESM_UNIT_MAPPING: dict = field(
        default_factory=lambda: {
            "pr": "kg m-2 s-1",
            "tas": "K",
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
            "worker_vm_types": ["m8g.2xlarge"],
            "scheduler_vm_types": "c8g.xlarge",
        }
    )

    def __post_init__(self):
        sesh = boto3.Session()
        creds = sesh.get_credentials()
        self.aws_creds = {
            "aws_access_key_id": creds.access_key,
            "aws_secret_access_key": creds.secret_key,
        }


@dataclass
class CESM_Historical_Config(BaseCESM_Config):
    scenario: str = "historical"
    materialized_key: str = "CESM2-WACCM-historical-icechunk"


@dataclass
class CESM_SSP245_Config(BaseCESM_Config):
    scenario: str = "ssp245"
    materialized_key: str = "CESM2-WACCM-SSP245-icechunk"


@dataclass
class CESM_G6_1_5K_Config(BaseCESM_Config):
    scenario: str = "G6-1.5K"
    catalog_key: str = "CESM2-WACCM-G6-1.5K-virtual"
    materialized_key: str = "CESM2-WACCM-G6-1.5K-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-G6-1.5K/netcdf"


SCENARIO_CONFIG_MAP = {
    "historical": CESM_Historical_Config,
    "ssp245": CESM_SSP245_Config,
    "G6-1.5K": CESM_G6_1_5K_Config,
}

PANGEO_SCENARIOS = {"historical", "ssp245"}

# --- PROVENANCE HELPERS ---


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

    # Check if we have to fall back to path parsing
    url_match = re.search(r"[._/](r\d+i\d+p\d+f\d+|\d{3})[._/]", url)
    if url_match:
        info["url_extracted_code"] = url_match.group(1)
        found_in.append("filepath_regex")

    info["provenance_sources"] = list(set(found_in))
    return info


def _preprocess_pangeo(ds: xr.Dataset, url: str) -> xr.Dataset:
    ensemble_member = ds.attrs.get("variant_label", "unknown")
    prov = _capture_provenance(ds, url)
    ds = ds.expand_dims({"ensemble_member": [ensemble_member]})
    ds.attrs["_source_manifest"] = json.dumps(prov)
    return ds


def _preprocess_ensemble(ds: xr.Dataset, url: str = None) -> xr.Dataset:
    if url is None:
        raise ValueError("url parameter is required to determine ensemble member")
    case = ds.attrs.get("case", "")
    ensemble = case.rsplit(".")[-1] if case else "unknown"
    prov = _capture_provenance(ds, url)
    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    ds.attrs["_source_manifest"] = json.dumps(prov)
    return ds


def _finalize_metadata(ds: xr.Dataset, config: BaseCESM_Config) -> xr.Dataset:
    """Records the logic and the lineage chain for the entire dataset."""

    # 1. Technical Derivation Logic
    if config.scenario in PANGEO_SCENARIOS:
        derivation_logic = "Extracted from dataset attribute 'variant_label'"
    else:
        derivation_logic = "Extracted from suffix of dataset attribute 'case' via regex"

    # 2. Build Lineage Chain (e.g., piControl -> historical -> ssp245)
    parent_exp = ds.attrs.get("parent_experiment_id", "unknown_parent")
    lineage = f"{parent_exp} -> {config.scenario}"

    # 3. Consolidate Global Attributes
    etl_attrs = {
        "scenario": config.scenario,
        "model": "CESM2-WACCM",
        "experiment_lineage": lineage,
        "ensemble_derivation_logic": derivation_logic,
        "processing_steps": (
            "time_drop_duplicates, calendar_convert:proleptic_gregorian, "
            "lon_to_180, lat_lon_sort, trim_negative_precip"
        ),
    }

    # Add CMORization to steps if relevant to the variables in the DS
    if any(v in config.cmorization_functions for v in ds.data_vars):
        etl_attrs["processing_steps"] += ", cmorization_unit_conversion"

    ds.attrs.update(etl_attrs)

    # 4. Cleanup temporary global tags passed from preprocess
    if "_source_manifest" in ds.attrs:
        del ds.attrs["_source_manifest"]

    # 5. Document the Coordinate with the full audit trail
    # Note: member_specific_provenance is already attached in get_CESM_WACCM_ds
    ds.ensemble_member.attrs.update(
        {
            "long_name": "Ensemble Member Identifier",
            "derivation_method": derivation_logic,
            "description": (
                "Unique identifier for model runs. The 'member_specific_provenance' attribute "
                "contains a JSON map of tracking_ids and source_urls for every member in this dimension."
            ),
        }
    )

    return ds


# --- DATA FETCHING ---


def get_CESM_WACCM_ds(experiment_id: Literal["historical", "ssp245"]) -> xr.Dataset:
    """Fetches Pangeo Zarr stores and builds a comprehensive provenance manifest."""
    import intake
    from zarr.storage import ObjectStore

    cat = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    subset = cat.search(
        source_id=["CESM2-WACCM"],
        experiment_id=experiment_id,
        variable_id=SHARED_VARIABLES,
        member_id=SHARED_ENSEMBLE_MEMBERS,
        table_id="day",
    )

    datasets = []
    # This will store the full audit trail for every member
    full_manifest = {}

    for zstore_url in subset.df.zstore:
        gcs_store = from_url(zstore_url, skip_signature=True)
        zarr_store = ObjectStore(gcs_store)

        # Open dataset and capture its unique IDs before they are merged/lost
        ds = xr.open_dataset(
            zarr_store, engine="zarr", consolidated=True, chunks="auto"
        ).drop_encoding()

        member_id = ds.attrs.get("variant_label", "unknown")
        # Store all raw attributes and the URL for this specific member
        full_manifest[member_id] = _capture_provenance(ds, zstore_url)

        # Standardize for merging
        ds = _preprocess_pangeo(ds, zstore_url)
        datasets.append(ds)

    # Combine all members into a single dataset
    combined = xr.combine_by_coords(
        datasets,
        coords="minimal",
        compat="override",
        combine_attrs="drop_conflicts",
    )

    # Store the full audit trail as a JSON string on the coordinate attributes
    combined.ensemble_member.attrs["member_specific_provenance"] = json.dumps(full_manifest)

    return combined


# --- UTILS & CLI ---


def _get_cesm_var_from_cmip6(cmip6_var: str, config: BaseCESM_Config) -> str:
    reverse_mapping = {v: k for k, v in config.CESM_WACCM_VARIABLE_MAPPING.items()}
    return reverse_mapping.get(cmip6_var, cmip6_var)


def _get_netcdf_urls(config: CESM_G6_1_5K_Config, variables: list[str]) -> list[str]:
    store = from_url(f"s3://{config.s3_bucket}", region="us-west-2", **config.aws_creds)
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


def _preprocess_cesm(
    ds: xr.Dataset, config: BaseCESM_Config, var: str, subset: bool = False
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


def _update_attrs(ds: xr.Dataset, var_specs: dict, config: BaseCESM_Config) -> xr.Dataset:
    for var_name in ds.data_vars:
        if var_name in config.CESM_UNIT_MAPPING:
            ds[var_name].attrs["units"] = config.CESM_UNIT_MAPPING[var_name]

    ds = update_variable_attrs(ds, var_specs)
    ds = add_cf_bounds(ds)
    ds = _finalize_metadata(ds, config)
    return ds


@click.group()
def cli():
    pass


@click.command()
@click.option("--coiled/--local", default=False)
def virtualize(coiled):
    """Virtualize G6-1.5K netcdf files into icechunk."""
    from obspec_utils.readers import BufferedStoreReader
    from obspec_utils.wrappers import CachingReadableStore, SplittingReadableStore

    config = CESM_G6_1_5K_Config()
    client = (
        setup_cluster(ClusterConfig(**config.virtualize_cluster))
        if coiled
        else setup_local_client()
    )

    try:
        base_store = from_url(f"s3://{config.s3_bucket}", region="us-west-2", **config.aws_creds)
        registry = ObjectStoreRegistry(
            {f"s3://{config.s3_bucket}": CachingReadableStore(SplittingReadableStore(base_store))}
        )
        parser = HDFParser(drop_variables=config.drop_variables, reader_factory=BufferedStoreReader)

        virt_cat = catalog.get(config.catalog_key)
        cesm_vars = [_get_cesm_var_from_cmip6(v, config) for v in SHARED_VARIABLES]
        netcdf_urls = _get_netcdf_urls(config, cesm_vars)

        combined_ds = virtualize_and_combine(
            urls=netcdf_urls,
            registry=registry,
            parser=parser,
            loadable_variables=["lat", "lon", "time"],
            drop_variables=["ilev", "lev"],
            preprocess_fn=_preprocess_ensemble,
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
        session.commit("G6-1.5K: virtualized variables")
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
        if scenario in PANGEO_SCENARIOS:
            # --- BULK PROCESSING PATH (Zarr to Icechunk) ---
            ds = get_CESM_WACCM_ds(scenario)[variables]
            ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False, align_on="date")
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
            # --- ITERATIVE PROCESSING PATH (Virtual to Icechunk) ---
            virt_ds = catalog.get(config.catalog_key).to_xarray()
            for var in variables:
                cesm_var = _get_cesm_var_from_cmip6(var, config)
                ds = virt_ds[[cesm_var]]
                ds = _preprocess_cesm(ds, config, cesm_var, subset=subset)
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
