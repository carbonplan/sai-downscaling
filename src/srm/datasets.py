from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from cloudpathlib import CloudPath

# Importing your new config structures
from srm.config import VarSpec, VarStandards

if TYPE_CHECKING:
    import xarray as xr


@dataclass(kw_only=True)
class BaseDataset(ABC):
    """
    Abstract base class for all datasets.
    Ensures a consistent interface for ETL and analysis tools.
    """

    name: str
    format: typing.Literal["zarr", "icechunk"]
    expected_chunks: dict[str, int] | None = None
    expected_shards: dict[str, int] | None = None
    expected_vars: list[VarSpec] | None = None

    @property
    @abstractmethod
    def bucket(self) -> str:
        """Return bucket name"""
        pass

    @property
    @abstractmethod
    def bucket_uri(self) -> str:
        """Return cloud provider URI for the bucket (e.g., 's3://bucket-name/')"""
        pass

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Return the S3 key prefix for the data or virtual index"""
        pass

    @abstractmethod
    def to_xarray(self) -> xr.Dataset:
        """Open dataset as xarray Dataset"""
        pass

    def _open_icechunk(self, prefix: str, is_virtual: bool) -> xr.Dataset:
        """Internal helper for icechunk opening logic"""
        import icechunk
        import xarray as xr

        storage = icechunk.s3_storage(bucket=self.bucket, prefix=prefix, from_env=True)

        if is_virtual:
            credentials = icechunk.containers_credentials(
                {self.bucket_uri: icechunk.s3_credentials()}
            )
            repo = icechunk.Repository.open(storage, authorize_virtual_chunk_access=credentials)
        else:
            repo = icechunk.Repository.open(storage)

        session = repo.readonly_session("main")
        return xr.open_dataset(session.store, engine='zarr', chunks=self.encoding["shards"],consolidated=False, zarr_format=3)

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

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = CloudPath(self.path)

    @property 
    def encoding(self) ->dict:
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

    def to_xarray(self) -> xr.Dataset:
        if self.format == "icechunk":
            return self._open_icechunk(self.prefix, is_virtual=False)
        elif self.format == "zarr":
            import xarray as xr
            return xr.open_zarr(self.path)
        else:
            raise ValueError(f"Unknown format: {self.format}")


@dataclass(kw_only=True)
class VirtualDataset(BaseDataset):
    """Virtual dataset"""

    virtual_path: str | CloudPath

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

    def to_xarray(self) -> xr.Dataset:
        if self.format == "icechunk":
            return self._open_icechunk(self.prefix, is_virtual=True)
        else:
            raise ValueError(f"icechunk only for virtual datasets {self.format}")


class Catalog:
    def __init__(self):
        all_standards = [f.default for f in fields(VarStandards)]

        self.datasets: dict[str, BaseDataset] = {
            "CESM2-WACCM-Historical-icechunk": Dataset(
                name="CESM2-WACCM-Historical-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-Historical/icechunk/CESM2-WACCM-Historical_pancakes.icechunk",
                format="icechunk",
                expected_chunks={"time": 30, "lat": 192, "lon": 288},
                expected_shards={"time": 1920, "lat": 192, "lon": 288},
                expected_vars=[
                    VarStandards.HURS, VarStandards.HUSS, VarStandards.PR, VarStandards.PS, VarStandards.RLDS, VarStandards.RSDS,VarStandards.SFCWIND,
                    VarStandards.TAS,VarStandards.TASMAX, VarStandards.TASMIN,
                ],            ),


            "MIROC-ES2H-G6-1.5K-icechunk": Dataset(
                name="MIROC-ES2H-G6-1.5K-icechunk",
                path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/MIROC-ES2H-G6-1.5K/updated_MIROC-ES2H-G6-1.5K.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 18263, "lat": 8, "lon": 16},
                expected_shards={"ensemble_member": 1, "time": 18263, "lat": 64, "lon": 128},
                expected_vars=[
                    VarStandards.PR, VarStandards.RSDS, VarStandards.TASMAX,
                    VarStandards.TAS, VarStandards.SFCWIND, VarStandards.TASMIN,
                ],
            ),
            "UKESM-G6-1.5K-icechunk": Dataset(
                name="UKESM-G6-1.5K-icechunk",
                path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-G6-1.5K/UKESM-G6-1.5K.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_shards={"ensemble_member": 1, "time": 960, "lat": 144, "lon": 192},
                expected_vars=[VarStandards.HURS, VarStandards.HUSS, VarStandards.PS, VarStandards.RSDS, VarStandards.RLDS],
            ),
            "UKESM-G6-1.5K-virtual": VirtualDataset(
                name="UKESM-G6-1.5K-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-G6-1.5K/UKESM-G6-1.5K-virtual.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_shards={"ensemble_member": 1, "time": 960, "lat": 144, "lon": 192},
                expected_vars=[VarStandards.HURS, VarStandards.HUSS, VarStandards.PS, VarStandards.RSDS, VarStandards.RLDS],
            ),
            "UKESM-SSP245-icechunk": Dataset(
                name="UKESM-SSP245-icechunk",
                path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-SSP245/UKESM-SSP245.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_shards={"ensemble_member": 1, "time": 960, "lat": 144, "lon": 192},
                expected_vars=[VarStandards.HURS, VarStandards.HUSS, VarStandards.PS, VarStandards.RSDS, VarStandards.RLDS, VarStandards.SFCWIND],
            ),
            "UKESM-SSP245-virtual": VirtualDataset(
                name="UKESM-SSP245-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-SSP245/UKESM-SSP245-virtual.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_shards={"ensemble_member": 1, "time": 960, "lat": 144, "lon": 192},
                expected_vars=[VarStandards.HURS, VarStandards.HUSS, VarStandards.PS, VarStandards.RSDS, VarStandards.RLDS],
            ),
            "UKESM-SSP245-uas-virtual": VirtualDataset(
                name="UKESM-SSP245-uas-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-SSP245-WIND/UKESM-SSP245-uas-virtual.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_vars=[VarStandards.UAS],
            ),
            "UKESM-SSP245-vas-virtual": VirtualDataset(
                name="UKESM-SSP245-vas-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-SSP245-WIND/UKESM-SSP245-vas-virtual.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 145, "lon": 192},
                expected_vars=[VarStandards.VAS],
            ),
            "UKESM-G6-1.5K-uas-virtual": VirtualDataset(
                name="UKESM-G6-1.5K-uas-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-G6-1.5K-WIND/UKESM-G6-1.5K-uas-virtual.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_vars=[VarStandards.UAS],
            ),
            "UKESM-G6-1.5K-vas-virtual": VirtualDataset(
                name="UKESM-G6-1.5K-vas-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-G6-1.5K-WIND/UKESM-G6-1.5K-vas-virtual.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 145, "lon": 192},
                expected_vars=[VarStandards.VAS],
            ),
            "ERA5": Dataset(
                name="ERA5",
                path="s3://carbonplan-srm/input/tensor/ERA5/ERA5_pancakes.icechunk",
                format="icechunk",
                expected_chunks={"time": 1, "lat": 721, "lon": 1440},
                expected_shards={"time": 30, "lat": 721, "lon": 1440},
                expected_vars=[
                    VarStandards.PR, VarStandards.RLDS, VarStandards.RSDS,
                    VarStandards.TASMAX, VarStandards.TAS, VarStandards.SFCWIND,
                    VarStandards.PS, VarStandards.TASMIN,
                ],
            ),
        }

    def get(self, name: str) -> BaseDataset:
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

                table_data.append([
                    var_name, unit_str, 
                    ds[var_name].attrs.get("long_name", ""),
                    ds[var_name].attrs.get("cell_methods", "")
                ])

            headers = ["Variable", "Units", "Long Name", "Cell Methods"]
            print(f"\n{dataset_name}")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def __str__(self) -> str:
        from tabulate import tabulate
        table_data = [
            [ds.name, ds.__class__.__name__, ds.format, ds.prefix, ds.expected_chunks] 
            for ds in self.datasets.values()
        ]
        headers = ["Name", "Type", "Format", "Prefix/Key", "Expected Chunks"]
        return f"Dataset Catalog ({len(self.datasets)} datasets)\n" + tabulate(
            table_data, headers=headers, tablefmt="grid"
        )

    def __repr__(self) -> str:
        return self.__str__()


catalog = Catalog()