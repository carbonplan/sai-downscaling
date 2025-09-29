# Easily run on an m8g.4xlarge

import xarray as xr
import zarr
from obstore.store import from_url
import obstore as obs
from virtualizarr import open_virtual_mfdataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry
import icechunk
from icechunk.xarray import to_icechunk
import coiled
import boto3

sess = boto3.Session()
creds = sess.get_credentials()
zarr.config.set({"async.concurrency": 128})


coiled_cluster_args = {
    "name": "virtual_cesm",
    "n_workers": [10, 20],
    "spot_policy": "spot_with_fallback",
    "worker_vm_types": "m8g.2xlarge",
    "scheduler_vm_types": "c8g.8xlarge",
    "tags": {"Project": "OCR"},
    "idle_timeout": "120 minutes",
    "use_best_zone": True,
}
cluster = coiled.Cluster(**coiled_cluster_args)

client = cluster.get_client()
print(cluster.dashboard_link)
print(cluster._dashboard_address)

# setup VZ config + bucket and prefix info
bucket = "s3://carbonplan-srm/"
prefix = "input/tensor/CESM2-WACCM-SSP245/netcdf"
virtual_ic_prefix_7_10 = (
    "input/tensor/CESM2-WACCM-SSP245/icechunk/virtual_icechunk_007_010"
)
virtual_ic_prefix_6 = "input/tensor/CESM2-WACCM-SSP245/icechunk/virtual_icechunk_006"
ic_prefix = "input/tensor/CESM2-WACCM-SSP245/icechunk/icechunk"


store = from_url(
    bucket,
    region="us-west-2",
    aws_access_key_id=creds.access_key,
    aws_secret_access_key=creds.secret_key,
)


registry = ObjectStoreRegistry({bucket: store})
drop_variables = [
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
parser = HDFParser(drop_variables=drop_variables)


def preprocess(ds):
    """
    get ensemble member from ds attrs filename
    """
    ensemble = ds.attrs["case"].rsplit(".")[-1]

    ds = ds.expand_dims({"ensemble_member": [ensemble]})
    return ds


# get all NetCDF file urls
stream = obs.list_with_delimiter(store, prefix=prefix, return_arrow=True)
netcdf_list = list(stream["objects"]["path"].to_numpy())
netcdf_list.remove(prefix)
netcdf_urls = [bucket + netcdf_path for netcdf_path in netcdf_list]


subset_7_10 = ["001", "002", "003", "004", "005", "006"]
subset_6 = ["001", "002", "003", "004", "005", "007", "008", "009", "010"]

netcdf_urls_7_10 = [
    path
    for path in netcdf_urls
    if not any(f"CMIP6-SSP2-4.5-WACCM.{num}." in path for num in subset_7_10)
]
netcdf_urls_6 = [
    path
    for path in netcdf_urls
    if not any(f"CMIP6-SSP2-4.5-WACCM.{num}." in path for num in subset_6)
]


vds_6 = open_virtual_mfdataset(
    netcdf_urls_6,
    registry=registry,
    parser=parser,
    preprocess=preprocess,
    combine="by_coords",
    combine_attrs="drop_conflicts",
    loadable_variables=["lat", "lev", "ilev", "time", "nbnd", "lon"],
    parallel="dask",
)
vds_6 = vds_6.drop_vars(["ilev", "lev"])

config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        "s3://carbonplan-srm/",
        store=icechunk.s3_store(region="us-west-2"),
    ),
)


storage = icechunk.s3_storage(
    bucket="carbonplan-srm", prefix=virtual_ic_prefix_6, from_env=True
)
repo = icechunk.Repository.open_or_create(storage, config)
session = repo.writable_session("main")

vds_6.vz.to_icechunk(session.store)
snapshot_id = session.commit("virtual_CESM2-WACCM-SSP245_006")
print(snapshot_id)
repo.save_config()

vds_7_10 = open_virtual_mfdataset(
    netcdf_urls_7_10,
    registry=registry,
    parser=parser,
    preprocess=preprocess,
    combine="by_coords",
    combine_attrs="drop_conflicts",
    loadable_variables=["lat", "lev", "ilev", "time", "nbnd", "lon"],
    parallel="dask",
)
vds_7_10 = vds_7_10.drop_vars(["ilev", "lev"])

config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        "s3://carbonplan-srm/",
        store=icechunk.s3_store(region="us-west-2"),
    ),
)


storage = icechunk.s3_storage(
    bucket="carbonplan-srm", prefix=virtual_ic_prefix_7_10, from_env=True
)
repo = icechunk.Repository.open_or_create(storage, config)
session = repo.writable_session("main")

vds_7_10.vz.to_icechunk(session.store)
snapshot_id = session.commit("virtual_CESM2-WACCM-SSP245_007_010")
print(snapshot_id)
repo.save_config()

## Write Virtual to Icechunk


config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        "s3://carbonplan-srm/",
        store=icechunk.s3_store(region="us-west-2"),
    ),
)

credentials = icechunk.containers_credentials(
    {
        "s3://carbonplan-srm": icechunk.s3_credentials(),
    }
)
storage_6 = icechunk.s3_storage(
    bucket="carbonplan-srm", prefix=virtual_ic_prefix_6, from_env=True
)
storage_7_10 = icechunk.s3_storage(
    bucket="carbonplan-srm", prefix=virtual_ic_prefix_7_10, from_env=True
)
vz_repo_6 = icechunk.Repository.open(
    storage=storage_6,
    config=config,
    authorize_virtual_chunk_access=credentials,
)
vz_repo_7_10 = icechunk.Repository.open(
    storage=storage_7_10,
    config=config,
    authorize_virtual_chunk_access=credentials,
)
vz_session_6 = vz_repo_6.readonly_session("main")
vz_session_7_10 = vz_repo_7_10.readonly_session("main")


ds_6 = xr.open_zarr(
    vz_session_6.store,
    zarr_format=3,
    consolidated=False,
    chunks={},
)
ds_7_10 = xr.open_zarr(
    vz_session_7_10.store,
    zarr_format=3,
    consolidated=False,
    chunks={},
)

merge_ds = xr.combine_by_coords([ds_6, ds_7_10])
merge_ds.coords["lon"] = (merge_ds.coords["lon"] + 180) % 360 - 180
merge_ds = merge_ds.sortby(merge_ds.lon)
merge_ds = merge_ds.chunk({"time": -1, "lat": 32, "lon": 48})
merge_ds = merge_ds.drop_encoding()

write_storage_config = icechunk.s3_storage(bucket="carbonplan-srm", prefix=ic_prefix)
write_repo = icechunk.Repository.open_or_create(write_storage_config)
write_session = write_repo.writable_session("main")
to_icechunk(merge_ds, write_session)

first_snapshot = write_session.commit(
    "CESM245 - 006-010: Create spatially chunked store: {'time':-1,'lat':32,'lon':48}"
)
