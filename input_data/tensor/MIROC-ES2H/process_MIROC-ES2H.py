import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import click
import coiled
import dask
import icechunk
import pandas as pd
import xarray as xr
import zarr
from distributed import Client
from icechunk.xarray import to_icechunk
from obstore.store import from_url
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry


@dataclass
class BaseModelETL(ABC):
    dataset: xr.Dataset | None = field(default=None, init=False)
    dataset_opt1: xr.Dataset | None = field(default=None, init=False)
    dataset_opt2: xr.Dataset | None = field(default=None, init=False)
    client: Client | None = field(default=None, init=False)

    @abstractmethod
    def fetch_netcdfs(self) -> None:
        pass

    @abstractmethod
    def virtualize(self) -> xr.Dataset:
        pass

    @abstractmethod
    def process(self) -> xr.Dataset:
        pass

    @abstractmethod
    def rechunk(self) -> tuple[xr.Dataset, xr.Dataset]:
        pass

    @abstractmethod
    def write(self) -> None:
        pass

    def setup_cluster(self, cluster_args: dict) -> Client:
        zarr.config.set(
            {
                "async.concurrency": 4,
                "threading.max_workers": 4,
            }
        )
        try:
            self.cluster = coiled.Cluster(**cluster_args)
            self.client = self.cluster.get_client()
            return self.client
        except Exception as exc:
            raise Exception(f"Error setting up cluster: {exc}")

    def setup_local_dask(self, n_workers: int = 64) -> None:
        self.client = Client(n_workers=n_workers)
        zarr.config.set(
            {
                "async.concurrency": 4,
                "threading.max_workers": 4,
            }
        )

    def cleanup(self) -> None:
        if self.client:
            self.client.shutdown()
        if hasattr(self, "cluster"):
            self.cluster.shutdown()

    def setup_repository(self, storage_config: dict) -> tuple:
        storage = icechunk.s3_storage(
            bucket=storage_config["bucket"],
            prefix=storage_config["prefix"],
            region=storage_config["region"],
        )
        repo = icechunk.Repository.open_or_create(storage)
        session = repo.writable_session("main")
        return repo, session

    def run_pipeline(
        self, skip_fetch: bool = True, use_coiled: bool = False, cluster_args: dict = None
    ) -> None:
        if use_coiled:
            self.setup_cluster(cluster_args or self.get_default_cluster_args())
        else:
            self.setup_local_dask()

        if not skip_fetch:
            self.fetch_netcdfs()
        self.dataset = self.virtualize()
        self.dataset = self.process()
        self.dataset_opt1, self.dataset_opt2 = self.rechunk()
        self.write()
        self.cleanup()

    def get_default_cluster_args(self) -> dict:
        return {
            "region": "us-west-2",
            "n_workers": [2, 50],
            "tags": {"Project": "SRM"},
            "worker_vm_types": ["r8g.medium", "m8g.large", "r8g.large"],
            "scheduler_vm_types": "m8g.4xlarge",
            "spot_policy": "spot_with_fallback",
            "use_best_zone": True,
            "idle_timeout": "2 hours",
            "no_client_timeout": None,
        }


@dataclass
class MIROC_ES2H_G6_1_5K(BaseModelETL):
    source_url: str = "https://www.jamstec.go.jp/swpub/public/GeoMIP/G6-1.5K-SAI/MIROC-ES2H/day/"
    s3_bucket: str = "s3://carbonplan-srm/"
    s3_prefix: str = "input/tensor/MIROC-ES2H/netcdf"
    ic_prefix: str = "input/tensor/MIROC-ES2H/MIROC-ES2H-G6-1.5K/MIROC-ES2H-G6-1.5K.icechunk"
    scenario: str = "G6-1.5K-SAI"
    variables: list = field(
        default_factory=lambda: ["hurs", "huss", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]
    )
    ensemble_members: list = field(
        default_factory=lambda: [str(val).zfill(2) for val in range(1, 11)]
    )

    def fetch_netcdfs(self) -> None:
        urls_csv = f"MIROC-ES2H-{self.scenario}-urls.csv"
        df = pd.read_html(self.source_url)[0][["Name"]].iloc[2::].dropna().reset_index()[["Name"]]
        df["Name"] = self.source_url + df["Name"]
        df.to_csv(urls_csv, index=False, header=False)

        target_remote = "aws:carbonplan-srm/input/tensor/MIROC-ES2H/"
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

    def virtualize(self) -> xr.Dataset:
        store = from_url(self.s3_bucket, region="us-west-2")  # ,skip_signature=True)
        registry = ObjectStoreRegistry({self.s3_bucket: store})
        parser = HDFParser()

        netcdf_urls = [
            f"{self.s3_bucket}{self.s3_prefix}/{var}_{self.scenario}_r{ensm}.nc"
            for var in self.variables
            for ensm in self.ensemble_members
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
        mds = xr.combine_by_coords(
            vds_list, coords="minimal", data_vars="minimal", compat="override"
        )

        return mds

    def process(self) -> xr.Dataset:
        from srm.utils import lon_to_180

        ds = lon_to_180(self.dataset, lon_name="lon")
        ds = ds.drop_vars(["lon_bnds", "lat_bnds", "time_bnds"])
        ds = ds.drop_encoding()

        return ds

    def rechunk(self) -> xr.Dataset:
        ds = self.dataset.chunk({"time": -1, "ensemble_member": 1, "lat": 8, "lon": 16})
        # opt2 = self.dataset.chunk({'time':100,'ensemble_member':1,'lat':-1,'lon':-1})
        return ds

    def write(self) -> None:
        storage_config = {
            "bucket": "carbonplan-srm",
            "prefix": self.ic_prefix,
            "region": "us-west-2",
        }

        _, session = self.setup_repository(storage_config)
        to_icechunk(self.dataset_opt1, session)
        session.commit("time-chunked")
        # # icechunk.IcechunkError:   × ref not found `G6-1.5K-SAI-space-optimized`
        # branch_name = "space-chunked"
        # main_branch_snapshot_id = repo.lookup_branch("main")
        # repo.create_branch(branch_name, snapshot_id=main_branch_snapshot_id)
        # session = repo.writable_session(branch_name)
        # to_icechunk(self.dataset_opt2, session)
        # snapshot_id = session.commit(branch_name)


@dataclass
class MIROC_ES2H_Baseline(MIROC_ES2H_G6_1_5K):
    source_url: str = "https://www.jamstec.go.jp/swpub/public/GeoMIP/baseline/MIROC-ES2H/day/"
    scenario: str = "baseline"
    ic_prefix: str = "input/tensor/MIROC-ES2H/MIROC-ES2H-baseline/MIROC-ES2H-baseline.icechunk"


MODEL_MAP = {
    "1-5k": MIROC_ES2H_G6_1_5K,
    "baseline": MIROC_ES2H_Baseline,
}


@click.group()
def cli():
    pass


@cli.command()
@click.option("--model", type=click.Choice(list(MODEL_MAP.keys())), required=True)
@click.option(
    "--fetch/--no-fetch", default=False, help="download netcdf files from source to s3 with rclone"
)
@click.option("--coiled/--no-coiled", default=False, help="use coiled cluster")
def run(model: str, fetch: bool, coiled: bool):
    pipeline = MODEL_MAP[model]()
    pipeline.run_pipeline(skip_fetch=not fetch, use_coiled=coiled)


if __name__ == "__main__":
    cli()
