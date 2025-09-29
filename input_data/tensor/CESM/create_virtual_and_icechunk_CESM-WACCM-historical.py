import xarray as xr
import zarr
from obstore.store import from_url
import obstore as obs
from virtualizarr import open_virtual_mfdataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry
import icechunk
import coiled
from icechunk.xarray import to_icechunk


zarr.config.set({"async.concurrency": 128})


cluster = coiled.Cluster(
    name="CESM-historical",
    n_workers=[10, 100],
    region="us-west-2",
    worker_vm_types="r8g.medium",
    scheduler_vm_types=["c8g.4xlarge"],
    spot_policy="spot_with_fallback",
    tags={"Project": "SRM"},
)


client = cluster.get_client()
print(cluster.dashboard_link)
print(cluster._dashboard_address)

bucket = "s3://carbonplan-srm/"
prefix = "input/tensor/CESM2-WACCM-Historical/netcdf"
virtual_ic_prefix = "input/tensor/CESM2-WACCM-Historical/icechunk/virtual_icechunk"
ic_prefix = "input/tensor/CESM2-WACCM-Historical/icechunk/icechunk"
store = from_url(bucket, region="us-west-2")
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

stream = obs.list_with_delimiter(store, prefix=prefix, return_arrow=True)
netcdf_list = list(stream["objects"]["path"].to_numpy())
netcdf_list.remove(prefix)
netcdf_urls = [bucket + netcdf_path for netcdf_path in netcdf_list]

combined_vds = open_virtual_mfdataset(
    netcdf_urls,
    registry=registry,
    parser=parser,
    combine="by_coords",
    combine_attrs="drop_conflicts",
    loadable_variables=["lat", "lev", "ilev", "time", "nbnd", "lon"],
    parallel="dask",
)
combined_vds = combined_vds.drop_vars(["ilev", "lev"])

config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        "s3://carbonplan-srm/",
        store=icechunk.s3_store(region="us-west-2"),
    ),
)


storage = icechunk.s3_storage(
    bucket="carbonplan-srm", prefix=virtual_ic_prefix, from_env=True
)
repo = icechunk.Repository.open_or_create(storage, config)
session = repo.writable_session("main")

combined_vds.vz.to_icechunk(session.store)
snapshot_id = session.commit("virtual_CESM2-WACCM-Historical")
print(snapshot_id)
repo.save_config()


credentials = icechunk.containers_credentials(
    {
        "s3://carbonplan-srm": icechunk.s3_credentials(),
    }
)

vz_repo = icechunk.Repository.open(
    storage=storage,
    config=config,
    authorize_virtual_chunk_access=credentials,
)
vz_session = vz_repo.readonly_session("main")

ds = xr.open_zarr(
    vz_session.store,
    zarr_format=3,
    consolidated=False,
    chunks={},
)

write_storage_config = icechunk.s3_storage(bucket="carbonplan-srm", prefix=ic_prefix)
write_repo = icechunk.Repository.open_or_create(write_storage_config)
write_session = write_repo.writable_session("main")

ds = ds.chunk({"time": -1, "lat": 32, "lon": 48})
ds = ds.drop_encoding()

to_icechunk(ds, write_session)

first_snapshot = write_session.commit(
    "create spatially chunked store: {'time':-1,'lat':32,'lon':48}"
)
