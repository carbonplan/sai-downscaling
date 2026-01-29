from __future__ import annotations

import typing
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from cloudpathlib import CloudPath

# Importing your new config structures
from srm.config import VarSpec, VarStandards

if TYPE_CHECKING:
    import xarray as xr


@dataclass
class Dataset:
    """base class for dataset object. Keeping it simple-ish for now"""

    name: str
    path: str | CloudPath
    format: typing.Literal["zarr", "icechunk"]
    expected_chunks: dict[str, int] | None = None
    expected_shards: dict[str, int] | None = None
    expected_vars: list[VarSpec] | None = None  # Now uses VarSpec objects in config.py

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
        all_standards = [f.default for f in fields(VarStandards)]

        self.datasets = {
            "CESM2-WACCM-Historical-icechunk": Dataset(
                name="CESM2-WACCM-Historical-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-Historical/icechunk/CESM2-WACCM-Historical_pancake.icechunk",
                format="icechunk",
                # expected_chunks={"time": 13521, "lat": 8, "lon": 16},
                # expected_shards={"time": 13521, "lat": 32, "lon": 64},
                expected_chunks={"time": 30, "lat": 192, "lon": 288},  # ~6MB ~ 30 days
                expected_shards={"time": 365, "lat": 192, "lon": 288},  #  ~77MB 1 year
                expected_vars=all_standards,
            ),
            "CESM2-WACCM-G6-1.5K-icechunk": Dataset(
                name="CESM2-WACCM-G6-1.5K-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-G6-1.5K/icechunk/CESM2-WACCM-G6-1.5k_pancake.icechunk",
                format="icechunk",
                expected_chunks={
                    "ensemble_member": 1,
                    "time": 30,
                    "lat": 192,
                    "lon": 288,
                },  # ~6MB ~ 30 days
                expected_shards={
                    "ensemble_member": 1,
                    "time": 365,
                    "lat": 192,
                    "lon": 288,
                },  #  ~77MB 1 year
                # expected_chunks={
                #     "ensemble_member": 1,
                #     "time": 18251,
                #     "lat": 8,
                #     "lon": 16,
                # },
                # expected_shards={
                #     "ensemble_member": 1,
                #     "time": 18251,
                #     "lat": 32,
                #     "lon": 64,
                # },
                expected_vars=all_standards,
            ),
            "CESM2-WACCM-SSP245-icechunk": Dataset(
                name="CESM2-WACCM-SSP245-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-SSP245/icechunk/CESM2-WACCM-SSP245_pancake.icechunk",
                format="icechunk",
                expected_chunks={
                    "ensemble_member": 1,
                    "time": 30,
                    "lat": 192,
                    "lon": 288,
                },  # ~6MB ~ 30 days
                expected_shards={
                    "ensemble_member": 1,
                    "time": 365,
                    "lat": 192,
                    "lon": 288,
                },  #  ~77MB 1 year
                # expected_chunks={
                #     "ensemble_member": 1,
                #     "time": 20076,
                #     "lat": 8,
                #     "lon": 16,
                # },
                # expected_shards={
                #     "ensemble_member": 1,
                #     "time": 20076,
                #     "lat": 32,
                #     "lon": 64,
                # },
                expected_vars=all_standards,
            ),
            "MIROC-ES2H-G6-1.5K-icechunk": Dataset(
                name="MIROC-ES2H-G6-1.5K-icechunk",
                path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/MIROC-ES2H-G6-1.5K/updated_MIROC-ES2H-G6-1.5K.icechunk",
                format="icechunk",
                expected_chunks={
                    "ensemble_member": 1,
                    "time": 18263,
                    "lat": 8,
                    "lon": 16,
                },
                expected_shards={
                    "ensemble_member": 1,
                    "time": 18263,
                    "lat": 64,
                    "lon": 128,
                },
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TASMAX,
                    VarStandards.TAS,
                    VarStandards.SFCWIND,
                    VarStandards.TASMIN,
                ],
            ),
            "MIROC-ES2H-baseline-icechunk": Dataset(
                name="MIROC-ES2H-baseline-icechunk",
                path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/MIROC-ES2H-baseline/updated_MIROC-ES2H-baseline.icechunk",
                format="icechunk",
                expected_chunks={
                    "ensemble_member": 1,
                    "time": 23742,
                    "lat": 8,
                    "lon": 16,
                },
                expected_shards={
                    "ensemble_member": 1,
                    "time": 23742,
                    "lat": 64,
                    "lon": 128,
                },
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TASMAX,
                    VarStandards.TAS,
                    VarStandards.SFCWIND,
                    VarStandards.TASMIN,
                ],
            ),
            "UKESM-SSP245-icechunk": Dataset(
                name="UKESM-SSP245-icechunk",
                path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-SSP245/UKESM-SSP245.icechunk",
                format="icechunk",
                expected_chunks={
                    "ensemble_member": 1,
                    "time": 30451,
                    "lat": 8,
                    "lon": 16,
                },
                expected_shards={
                    "ensemble_member": 1,
                    "time": 30451,
                    "lat": 64,
                    "lon": 128,
                },
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.HUSS,
                    VarStandards.PS,
                    VarStandards.RSDS,
                    VarStandards.RLDS,
                ],
            ),
            "UKESM-G6-1.5K-icechunk": Dataset(
                name="UKESM-G6-1.5K-icechunk",
                path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-G6-1.5K/UKESM-G6-1.5K.icechunk",
                format="icechunk",
                expected_chunks={
                    "ensemble_member": 1,
                    "time": 17913,
                    "lat": 8,
                    "lon": 16,
                },
                expected_shards={
                    "ensemble_member": 1,
                    "time": 17913,
                    "lat": 64,
                    "lon": 128,
                },
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.HUSS,
                    VarStandards.PS,
                    VarStandards.RSDS,
                    VarStandards.RLDS,
                ],
            ),
            "UKESM-SSP245-SFCWIND-icechunk": Dataset(
                name="UKESM-SSP245-SFCWIND-icechunk",
                path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-SSP245-SFCWIND/UKESM-SSP245-SFCWIND.icechunk",
                format="icechunk",
                expected_chunks={
                    "ensemble_member": 1,
                    "time": 30451,
                    "lat": 12,
                    "lon": 16,
                },
                expected_shards={
                    "ensemble_member": 1,
                    "time": 30451,
                    "lat": 144,
                    "lon": 192,
                },
                expected_vars=[
                    VarStandards.SFCWIND,
                ],
            ),
            "UKESM-G6-1.5K-SFCWIND-icechunk": Dataset(
                name="UKESM-G6-1.5K-SFCWIND-icechunk",
                path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-G6-1.5K-SFCWIND/UKESM-G6-1.5K-SFCWIND.icechunk",
                format="icechunk",
                expected_chunks={
                    "ensemble_member": 1,
                    "time": 17913,
                    "lat": 8,
                    "lon": 16,
                },
                expected_shards={
                    "ensemble_member": 1,
                    "time": 17913,
                    "lat": 144,
                    "lon": 192,
                },
                expected_vars=[
                    VarStandards.SFCWIND,
                ],
            ),
            "ERA5": Dataset(
                name="ERA5",
                # path="s3://carbonplan-srm/input/tensor/ERA5/ERA5.icechunk",
                path="s3://carbonplan-srm/input/tensor/ERA5/ERA5_pancakes.icechunk",
                format="icechunk",
                # expected_chunks={"time": 23741, "lat": 7, "lon": 14},  # ~9.5MB chunks
                # expected_shards={
                #     "time": 23741,
                #     "lat": 35,
                #     "lon": 70,
                # },  # ~ 221.88 MiB shard! # 441 chunks in 5 graph layers
                expected_chunks={
                    "time": 1,
                    "lat": 721,
                    "lon": 1440,
                },  # ~4 MB chunks # maps / pancakes
                expected_shards={
                    "time": 30,
                    "lat": 721,
                    "lon": 1440,
                },  # ~119 MB chunks # maps / pancakes
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.RLDS,
                    VarStandards.RSDS,
                    VarStandards.TASMAX,
                    VarStandards.TAS,
                    VarStandards.SFCWIND,
                    VarStandards.PS,
                    VarStandards.TASMIN,
                ],
            ),
        }

    def get(self, name: str) -> Dataset:
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
            [ds.name, ds.format, str(ds.path), ds.expected_chunks] for ds in self.datasets.values()
        ]
        headers = ["Name", "Format", "Path", "Expected Chunks"]
        return f"Dataset Catalog ({len(self.datasets)} datasets)\n" + tabulate(
            table_data, headers=headers, tablefmt="grid"
        )

    def __repr__(self) -> str:
        return self.__str__()


catalog = Catalog()
