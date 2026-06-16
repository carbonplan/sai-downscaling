# datasets.py

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dask.system
from cloudpathlib import CloudPath

from srm.config import VarSpec, VarStandards

if TYPE_CHECKING:
    import xarray as xr


@dataclass
class VirtualChunkContainerConfig:
    uri: str
    anonymous: bool = True
    region: str = "us-west-2"


@dataclass(kw_only=True)
class BaseCatalogEntry:
    name: str


@dataclass(kw_only=True)
class BaseDataset(BaseCatalogEntry):
    format: typing.Literal["zarr", "icechunk"]
    region: str = "us-west-2"
    expected_vars: list[VarSpec] | None = None
    ensemble_members: list[str] | None = None
    ensemble_member: list[str] | None = None

    def _open_icechunk(
        self,
        prefix: str,
        is_virtual: bool,
        virtual_chunk_container: VirtualChunkContainerConfig | None = None,
        group: str | None = None,
    ) -> xr.Dataset:
        import icechunk
        import xarray as xr

        # if dask.system.CPU_COUNT > 128 set max_concurrent_requests to 128 to avoid overwhelming the system, otherwise use the number of CPUs

        config = icechunk.RepositoryConfig(max_concurrent_requests=min(dask.system.CPU_COUNT, 128))
        storage = icechunk.s3_storage(bucket=self.bucket, prefix=prefix, from_env=True)

        if is_virtual:
            if virtual_chunk_container is not None:
                container_uri = virtual_chunk_container.uri
                anon = virtual_chunk_container.anonymous
                config.set_virtual_chunk_container(
                    icechunk.VirtualChunkContainer(
                        container_uri,
                        store=icechunk.s3_store(region=virtual_chunk_container.region),
                    )
                )
            else:
                container_uri = self.bucket_uri
                anon = False

            credentials = icechunk.containers_credentials(
                {container_uri: icechunk.s3_credentials(anonymous=anon)}
            )
            repo = icechunk.Repository.open(
                storage, authorize_virtual_chunk_access=credentials, config=config
            )

        else:
            repo = icechunk.Repository.open(storage, config=config)

        session = repo.readonly_session("main")
        return xr.open_dataset(
            session.store,
            engine="zarr",
            chunks="auto",
            consolidated=False,
            zarr_format=3,
            group=group,
        )

    def get_chunking_dict(self) -> dict[str, int]:
        ds = self.to_xarray()
        first_var = list(ds.data_vars)[0]
        chunks = ds[first_var].chunksizes
        if not chunks:
            return {}
        return {dim: c[0] for dim, c in chunks.items()}


@dataclass(kw_only=True)
class Dataset(BaseDataset):
    path: str | CloudPath
    expected_chunks: dict[str, int] | None = None
    expected_shards: dict[str, int] | None = None

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = CloudPath(self.path)

    @property
    def encoding(self) -> dict:
        return {
            "chunks": self.expected_chunks,
            "shards": self.expected_shards,
        }

    @property
    def bucket(self) -> str:
        return str(self.path.bucket)

    @property
    def prefix(self) -> str:
        return str(self.path.key)

    @property
    def bucket_uri(self) -> str:
        return f"{self.path.cloud_prefix}{self.bucket}/"

    def to_xarray(self, group: str | None = None) -> xr.Dataset:
        """Open the dataset as an xarray Dataset with time normalized to proleptic_gregorian."""
        if self.format == "icechunk":
            ds = self._open_icechunk(self.prefix, is_virtual=False, group=group)
        elif self.format == "zarr":
            import xarray as xr

            ds = xr.open_zarr(self.path)
        else:
            raise ValueError(f"Unknown format: {self.format}")

        return ds


@dataclass(kw_only=True)
class Datatree(Dataset):
    format: typing.Literal["icechunk"] = "icechunk"

    def to_xarray(self) -> xr.DataTree:
        import icechunk
        import xarray as xr

        config = icechunk.RepositoryConfig(max_concurrent_requests=min(dask.system.CPU_COUNT, 128))
        storage = icechunk.s3_storage(bucket=self.bucket, prefix=self.prefix, from_env=True)
        repo = icechunk.Repository.open(storage, config=config)
        session = repo.readonly_session("main")
        return xr.open_datatree(
            session.store,
            engine="zarr",
            chunks="auto",
            consolidated=False,
            zarr_format=3,
        )


@dataclass(kw_only=True)
class VectorDataset(BaseCatalogEntry):
    """A vector dataset stored as a GeoParquet file (S3 or local)."""

    path: str | CloudPath
    format: typing.Literal["geoparquet"] = "geoparquet"

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = CloudPath(self.path)

    def to_geodataframe(self):
        import geopandas as gpd

        return gpd.read_parquet(str(self.path))


@dataclass(kw_only=True)
class VirtualDataset(BaseDataset):
    virtual_path: str | CloudPath
    virtual_chunk_container: VirtualChunkContainerConfig | None = None

    def __post_init__(self):
        if isinstance(self.virtual_path, str):
            self.virtual_path = CloudPath(self.virtual_path)

    @property
    def bucket(self) -> str:
        return str(self.virtual_path.bucket)

    @property
    def prefix(self) -> str:
        return str(self.virtual_path.key)

    @property
    def bucket_uri(self) -> str:
        return f"{self.virtual_path.cloud_prefix}{self.bucket}/"

    def to_xarray(self, group: str | None = None) -> xr.Dataset:
        if self.format == "icechunk":
            return self._open_icechunk(
                self.prefix,
                is_virtual=True,
                virtual_chunk_container=self.virtual_chunk_container,
                group=group,
            )
        else:
            raise ValueError(f"icechunk only for virtual datasets {self.format}")


class Catalog:
    def __init__(self):
        self.standard_vars = [
            VarStandards.RSDS,
            VarStandards.TAS,
            VarStandards.TASMAX,
            VarStandards.TASMIN,
            VarStandards.HURS,
            VarStandards.PR,
        ]
        self.datasets: dict[str, BaseDataset] = {
            # Unified per-GCM stores: scenarios live as zarr groups
            # (historical / ssp245 / g6_1p5k, plus esgf_ssp245 for MIROC).
            # dtr is not stored; derive it via srm.utils.get_variable.
            "CESM2-WACCM-unified-icechunk": Datatree(
                name="CESM2-WACCM-unified-icechunk",
                path="s3://carbonplan-srm/input/tensor/cesm2_waccm.icechunk",
                expected_chunks={"ensemble_member": 1, "time": 30, "lat": 192, "lon": 288},
                expected_shards={"ensemble_member": 1, "time": 480, "lat": 192, "lon": 288},
                ensemble_members=[
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
                    "r1i1p1f1",
                    "r2i1p1f1",
                    "r3i1p1f1",
                ],
                expected_vars=self.standard_vars,
            ),
            "MIROC-ES2H-unified-icechunk": Datatree(
                name="MIROC-ES2H-unified-icechunk",
                path="s3://carbonplan-srm/input/tensor/miroc_es2h.icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 128, "lon": 256},
                expected_shards={"ensemble_member": 1, "time": 960, "lat": 128, "lon": 256},
                ensemble_members=[
                    "r1i1p4f2",
                    "r2i1p4f2",
                    "r3i1p4f2",
                    "r01",
                    "r02",
                    "r03",
                    "r04",
                    "r05",
                    "r06",
                    "r07",
                    "r08",
                    "r09",
                    "r10",
                ],
                expected_vars=self.standard_vars,
            ),
            "UKESM-unified-icechunk": Datatree(
                name="UKESM-unified-icechunk",
                path="s3://carbonplan-srm/input/tensor/ukesm.icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_shards={"ensemble_member": 1, "time": 960, "lat": 144, "lon": 192},
                ensemble_members=["r2i1p1f2", "r3i1p1f2", "r12i1p1f2"],
                expected_vars=self.standard_vars,
            ),
            "ERA5": Dataset(
                name="ERA5",
                path="s3://carbonplan-srm/input/tensor/ERA5/ERA5_pancakes.icechunk",
                format="icechunk",
                expected_chunks={"time": 1, "lat": 721, "lon": 1440},
                expected_shards={"time": 30, "lat": 721, "lon": 1440},
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.RLDS,
                    VarStandards.RSDS,
                    VarStandards.TASMAX,
                    VarStandards.TAS,
                    VarStandards.PS,
                    VarStandards.TASMIN,
                    VarStandards.DTR,
                ],
            ),
            "NASA-NEX-SSP245": VirtualDataset(
                name="NASA-NEX-SSP245",
                virtual_path="s3://carbonplan-srm/input/tensor/nasa-nex/ssp245/virtual.icechunk",
                format="icechunk",
                virtual_chunk_container=VirtualChunkContainerConfig(
                    uri="s3://nex-gddp-cmip6/", anonymous=True
                ),
                expected_vars=[VarStandards.TAS],
            ),
            "NASA-NEX-historical": VirtualDataset(
                name="NASA-NEX-historical",
                virtual_path="s3://carbonplan-srm/input/tensor/nasa-nex/historical/virtual.icechunk",
                virtual_chunk_container=VirtualChunkContainerConfig(
                    uri="s3://nex-gddp-cmip6/", anonymous=True
                ),
                format="icechunk",
                expected_vars=[VarStandards.TAS],
            ),
            "GDEX-GMF": Dataset(
                name="GDEX-GMF",
                path="s3://carbonplan-srm/input/tensor/NCAR/GDEX-GMF.icechunk",
                format="icechunk",
                expected_chunks={"time": 1, "lat": 720, "lon": 1440},
                expected_shards={"time": 30, "lat": 720, "lon": 1440},
                expected_vars=[VarStandards.TAS],
            ),
            "ocean-mask": VectorDataset(
                name="ocean-mask",
                path="s3://carbonplan-srm/input/vector/GSHHS/GSHHS.parquet",
                format="geoparquet",
            ),
        }

    def get(self, name: str) -> BaseDataset | VectorDataset:
        if name not in self.datasets:
            raise KeyError(f"Dataset {name} not found.")
        return self.datasets[name]

    def list(self) -> list[str]:
        return list(self.datasets.keys())

    def variable_metadata(self):
        from tabulate import tabulate

        for dataset_name, ds_info in self.datasets.items():
            ds = ds_info.to_xarray()
            specs = {s.name: s for s in (ds_info.expected_vars or [])}
            table_data = []

            for var_name in ds.data_vars:
                actual_units = ds[var_name].attrs.get("units", "")
                spec = specs.get(var_name)
                unit_str = actual_units
                if spec and actual_units != spec.units:
                    unit_str = f"MISMATCH WARNING! {actual_units} (Expected: {spec.units})"

                table_data.append(
                    [
                        var_name,
                        unit_str,
                        ds[var_name].attrs.get("long_name", ""),
                        ds[var_name].attrs.get("cell_methods", ""),
                    ]
                )

            headers = ["Variable", "Units", "Long Name", "Cell Methods"]
            print(f"\n{dataset_name}")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def __str__(self) -> str:
        from tabulate import tabulate

        table_data = [
            [
                ds.name,
                ds.__class__.__name__,
                ds.format,
                getattr(ds, "prefix", None),
                getattr(ds, "expected_chunks", None),
            ]
            for ds in self.datasets.values()
        ]
        headers = ["Name", "Type", "Format", "Prefix/Key", "Expected Chunks"]
        return f"Dataset Catalog ({len(self.datasets)} datasets)\n" + tabulate(
            table_data, headers=headers, tablefmt="grid"
        )

    def __repr__(self) -> str:
        return self.__str__()


catalog = Catalog()
