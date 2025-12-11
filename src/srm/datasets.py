from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cloudpathlib import CloudPath

if TYPE_CHECKING:
    import xarray as xr


@dataclass
class Dataset:
    """base class for dataset object. Keeping it simple"""

    name: str
    path: str | CloudPath
    format: typing.Literal["zarr", "icechunk"]
    expected_chunks: dict[str, int] | None = None  # ex: {'time': 13522, 'lat': 32, 'lon': 48}

    @property
    def uri(self) -> str:
        return str(self.path)

    @property
    def bucket(self) -> str:
        return str(self.path.bucket)

    @property
    def prefix(self) -> str:
        return str(self.path.key)

    def get_chunking_dict(self) -> dict[str, int]:
        """Get a chunking dictionary. Note: Just the first var,
        so we should update if we for some reason have differant chunking between vars"""
        ds = self.to_xarray()
        first_var = list(ds.data_vars)[0]
        return {dim: c[0] for dim, c in ds[first_var].chunksizes.items()}

    def to_xarray(self) -> xr.Dataset:
        import xarray as xr

        if self.format == "icechunk":
            import icechunk

            storage = icechunk.s3_storage(bucket=self.bucket, prefix=self.prefix, from_env=True)
            repo = icechunk.Repository.open(storage)
            session = repo.readonly_session("main")
            return xr.open_zarr(session.store, consolidated=False)
        elif self.format == "zarr":
            return xr.open_zarr(self.path)
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def __post_init__(self):
        """Validate and convert path to CloudPathLib"""
        if isinstance(self.path, str):
            self.path = CloudPath(self.path)


class Catalog:
    def __init__(self):
        self.datasets = {
            # -----------------------------------------------------------------------------------------------
            "CESM2-WACCM-Historical-icechunk": Dataset(
                name="CESM2-WACCM-Historical-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-Historical/icechunk/icechunk",
                format="icechunk",
                expected_chunks={"time": 13522, "lat": 32, "lon": 48},
            ),
            # -----------------------------------------------------------------------------------------------
            "CESM2-WACCM-G6-1.5K-icechunk": Dataset(
                name="CESM2-WACCM-G6-1.5K-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-G6-1.5K/icechunk/icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 18251, "lat": 32, "lon": 48},
            ),
            # -----------------------------------------------------------------------------------------------
            "CESM2-WACCM-SSP245-icechunk": Dataset(
                name="CESM2-WACCM-SSP245-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-SSP245/icechunk/icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 20076, "lat": 32, "lon": 48},
            ),
            # -----------------------------------------------------------------------------------------------
            "MIROC-ES2H-G6-1.5K-icechunk": Dataset(
                name="MIROC-ES2H-G6-1.5K-icechunk",
                path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/MIROC-ES2H-G6-1.5K/MIROC-ES2H-G6-1.5K.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 18263, "lat": 8, "lon": 16},
            ),
            # -----------------------------------------------------------------------------------------------
            "MIROC-ES2H-baseline-icechunk": Dataset(
                name="MIROC-ES2H-baseline-icechunk",
                path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/MIROC-ES2H-baseline/MIROC-ES2H-baseline.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 23742, "lat": 8, "lon": 16},
            ),
            # -----------------------------------------------------------------------------------------------
            "ERA5": Dataset(
                name="ERA5",
                path="s3://carbonplan-srm/input/tensor/ERA5/era5_rechunked_resampled.icechunk",
                format="icechunk",
                expected_chunks={"time": 730, "lat": 144, "lon": 288},
            ),
            # -----------------------------------------------------------------------------------------------
        }

    def get(self, name: str) -> Dataset:
        """get a dataset by name"""
        if name not in self.datasets:
            raise KeyError(f"Dataset {name} not found.")
        return self.datasets[name]

    def get_path(self, name: str) -> CloudPath:
        """Get UPath for dataset"""
        return self.get(name).path

    def get_uri(self, name: str) -> str:
        """create a uri"""
        return self.get(name).uri

    def list(self) -> list[str]:
        """list all dataset names"""
        return list(self.datasets.keys())

    def __str__(self) -> str:
        """create a table-like view"""
        lines = [f"Dataset Catalog ({len(self.datasets)} datasets)"]
        lines.append("-" * 80)
        for ds in self.datasets.values():
            lines.append(f"{ds.name:<20} | {ds.format:<10} | {ds.uri}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


catalog = Catalog()
