import typing
from dataclasses import dataclass

from cloudpathlib import CloudPath


@dataclass
class Dataset:
    """base class for dataset object. Keeping it simple"""

    name: str
    path: str | CloudPath
    format: typing.Literal["zarr", "icechunk"]

    def __post_init__(self):
        """Validate and convert path to CloudPathLib"""
        if isinstance(self.path, str):
            self.path = CloudPath(self.path)

    @property
    def uri(self) -> str:
        return str(self.path)

    @property
    def bucket(self) -> str:
        return str(self.path.bucket)

    @property
    def prefix(self) -> str:
        return str(self.path.key)


class Catalog:
    def __init__(self):
        self.datasets = {
            # -----------------------------------------------------------------------------------------------
            "CESM2-WACCM-Historical-icechunk": Dataset(
                name="CESM2-WACCM-Historical-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-Historical/icechunk/icechunk",
                format="icechunk",
            ),
            # -----------------------------------------------------------------------------------------------
            "CESM2-WACCM-G6-1.5K-icechunk": Dataset(
                name="CESM2-WACCM-G6-1.5K-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-G6-1.5K/icechunk/icechunk",
                format="icechunk",
            ),
            # -----------------------------------------------------------------------------------------------
            "CESM2-WACCM-SSP245-icechunk": Dataset(
                name="CESM2-WACCM-SSP245-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-SSP245/icechunk/icechunk",
                format="icechunk",
            ),
            # -----------------------------------------------------------------------------------------------
            "ERA5": Dataset(
                name="ERA5",
                path="s3://carbonplan-srm/input/tensor/ERA5/era5_rechunked_resampled.icechunk",
                format="icechunk",
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
