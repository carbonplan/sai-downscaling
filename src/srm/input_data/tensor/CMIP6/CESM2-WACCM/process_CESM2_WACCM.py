# COILED vm-type r8g.8xlarge
# COILED region us-west-2

import json
import logging
import re
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
    CMORIZE_pr,
    apply_ensemble_provenance,
    build_encoding_dict,
    determine_write_mode,
    get_var_specs,
    group_paths_by_member,
    load_dtr_from_store,
    open_netcdf_from_s3,
    resolve_variables,
    trim_negative_precipitation,
    update_variable_attrs,
    variable_in_store,
    write_dataset_to_icechunk,
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


SHARED_VARIABLES = ["tas", "rsds", "hurs", "pr", "tasmax", "tasmin", "dtr"]
SHARED_ENSEMBLE_MEMBERS = ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]

# CESM case names embed the member as a 3-digit segment, e.g.
# b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001.cam.h1....nc
MEMBER_PATTERN = re.compile(r"\.(\d{3})\.cam\.")


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
            "ilev",
            "lev",
        ]
    )


@dataclass
class Pangeo_CESM_Historical_Config(BaseCESM_Config):
    scenario: str = "pangeo-historical"
    materialized_key: str = "pangeo-CESM2-WACCM-historical-dev-icechunk"


@dataclass
class CESM_Historical_Config(BaseCESM_Config):
    scenario: str = "historical"
    materialized_key: str = "CESM2-WACCM-historical-dev-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-Historical/netcdf"
    # NOTE:  email confirmation that this ensemble member maps to 001
    ensemble_members: list = field(default_factory=lambda: ["001"])


@dataclass
class CESM_SSP245_Config(BaseCESM_Config):
    scenario: str = "SSP245"
    materialized_key: str = "CESM2-WACCM-SSP245-dev-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-SSP245/netcdf"
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


@dataclass
class CESM_G6_1_5K_Config(BaseCESM_Config):
    scenario: str = "G6-1.5K"
    materialized_key: str = "CESM2-WACCM-G6-1.5K-dev-icechunk"
    s3_input_prefix: str = "input/tensor/CESM2/CESM2-WACCM-G6-1.5K/netcdf"
    ensemble_members: list = field(default_factory=lambda: ["001", "002", "003"])


SCENARIO_CONFIG_MAP = {
    "pangeo-historical": Pangeo_CESM_Historical_Config,
    "historical": CESM_Historical_Config,
    "ssp245": CESM_SSP245_Config,
    "G6-1.5K": CESM_G6_1_5K_Config,
}

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

    info["provenance_sources"] = list(set(found_in))
    return info


def _attach_source_manifest(ds: xr.Dataset, url: str) -> xr.Dataset:
    prov = _capture_provenance(ds, url)
    ds.attrs["_source_manifest"] = json.dumps(prov)
    return ds


def _finalize_metadata(ds: xr.Dataset, config: BaseCESM_Config) -> xr.Dataset:
    """Records the parsing and the lineage chain"""

    if config.has_ensemble:
        derivation_logic = "Ensemble member derived from filename case segment or variant_label"
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
            "time_drop_duplicates, lon_to_180, lat_lon_sort, trim_negative_precip, convert_calendar_to_proleptic_gregorian"
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


# --- PROCESS HELPERS ---


def _get_cesm_var_from_cmip6(cmip6_var: str, config: BaseCESM_Config) -> str:
    reverse_mapping = {v: k for k, v in config.CESM_WACCM_VARIABLE_MAPPING.items()}
    return reverse_mapping.get(cmip6_var, cmip6_var)


def _member_from_path(path: str, config: BaseCESM_Config) -> str:
    match = MEMBER_PATTERN.search(path.split("/")[-1])
    if match:
        return match.group(1)
    # NCAR historical files carry no member segment; config pins the member
    if len(config.ensemble_members) == 1:
        return config.ensemble_members[0]
    return "unknown"


def _get_netcdf_urls(config: BaseCESM_Config, variable: str) -> list[tuple[str, str]]:
    """Return (member_id, s3_path) pairs for one variable."""
    store = from_url(f"s3://{config.s3_bucket}", region="us-west-2")
    stream = obs.list_with_delimiter(store, prefix=config.s3_input_prefix, return_arrow=True)
    paths = list(stream["objects"]["path"].to_numpy())
    cesm_var = _get_cesm_var_from_cmip6(variable, config)

    result = []
    for path in sorted(paths):
        if not (path.endswith(".nc") and f".{cesm_var}." in path):
            continue
        member = _member_from_path(path, config)
        if config.ensemble_members and member not in config.ensemble_members:
            continue
        result.append((member, path))
    return result


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
    keep_coords = set(ds.dims) | {"lat", "lon", "time"}
    ds = ds.drop_vars([c for c in ds.coords if c not in keep_coords], errors="ignore")
    ds = ds.drop_duplicates(dim="time", keep="first")
    ds = to_proleptic_gregorian(ds)
    ds = ds.drop_encoding()
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


def _process_single_variable(
    config: BaseCESM_Config,
    variable: str,
    repo: icechunk.Repository,
    var_specs: dict,
    overwrite: bool,
    subset: bool,
) -> None:
    log.info("variable=%s start", variable)

    var_in_store = variable_in_store(repo, variable)
    if not overwrite and var_in_store:
        log.info("variable=%s skip: already in store (use --overwrite to replace)", variable)
        return

    if variable.lower() == "dtr":
        # dtr derived from tasmax/tasmin already written to the store
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

        member_paths = group_paths_by_member(url_pairs)

        member_datasets = []
        member_manifest = {}
        for member, paths in sorted(member_paths.items()):
            log.info("variable=%s member=%s opening %d file(s)", variable, member, len(paths))
            time_slices = [
                open_netcdf_from_s3(obstore_inst, p, config.drop_variables) for p in sorted(paths)
            ]
            member_manifest[member] = _capture_provenance(time_slices[0], paths[0])
            member_ds = (
                xr.concat(time_slices, dim="time", data_vars="minimal")
                if len(time_slices) > 1
                else time_slices[0]
            )
            member_ds = _preprocess_cesm(member_ds, config, variable, subset=subset)
            member_ds = member_ds.expand_dims({"ensemble_member": [member]})
            member_datasets.append(member_ds)
            log.info("variable=%s member=%s shape=%s", variable, member, dict(member_ds.dims))

        # join="outer" unions time axes: SSP245 member 006 ends one day short
        # of 007-010 and gets NaN for the missing final day
        ds = (
            xr.concat(member_datasets, dim="ensemble_member", join="outer")
            if len(member_datasets) > 1
            else member_datasets[0]
        )
        ds = ds[[variable]]
        # reindex to full member list; fills any missing members with NaN
        if config.ensemble_members and "ensemble_member" in ds.dims:
            ds = ds.reindex(ensemble_member=config.ensemble_members)
        ds.ensemble_member.attrs["member_specific_provenance"] = json.dumps(member_manifest)
        log.info("variable=%s concat done shape=%s", variable, dict(ds.dims))

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
    )
    log.info("variable=%s done", variable)


def _run_process(
    config: BaseCESM_Config,
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


def _run_pangeo_process(config: BaseCESM_Config, variables: list[str]) -> None:
    """Pangeo Zarr to Icechunk bulk write."""
    mat_cat = catalog.get(config.materialized_key)
    var_specs = get_var_specs(mat_cat)

    ds = get_CESM_WACCM_ds(config.scenario)
    available = [v for v in variables if v in ds]
    ds = ds[available]
    ds = to_proleptic_gregorian(ds)
    ds = trim_negative_precipitation(ds)
    ds = lon_to_180(ds, lon_name="lon")
    ds = ds.sortby(["lat", "lon"])
    ds = _update_attrs(ds, var_specs, config)

    repo, session = init_repo(mat_cat.bucket, mat_cat.prefix, readonly=False)
    encoding = build_encoding_dict(ds, config.encoding["chunks"], config.encoding["shards"])
    write_dataset_to_icechunk(
        ds,
        session,
        encoding=encoding,
        shards=config.encoding["shards"],
        commit_message=f"{config.scenario}: Bulk write of {available}",
        write_mode=determine_write_mode(repo),
    )


# --- CLI ---


@click.group()
def cli():
    pass


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
    variables = resolve_variables(variable, all_variables, catalog.get(config.materialized_key))

    if _is_pangeo_scenario(scenario):
        _run_pangeo_process(config, variables)
    else:
        _run_process(config, variables, overwrite, subset)


cli.add_command(process)

if __name__ == "__main__":
    cli()
