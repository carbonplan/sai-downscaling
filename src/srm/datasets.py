# datasets.py

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
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
class BaseDataset(BaseCatalogEntry, ABC):
    format: typing.Literal["zarr", "icechunk"]
    region: str = "us-west-2"
    expected_vars: list[VarSpec] | None = None
    ensemble_members: list[str] | None = None
    ensemble_member: list[str] | None = None
    license: str | None = None
    citation: str | None = None

    @property
    @abstractmethod
    def bucket(self) -> str:
        pass

    @property
    @abstractmethod
    def bucket_uri(self) -> str:
        pass

    @property
    @abstractmethod
    def prefix(self) -> str:
        pass

    @abstractmethod
    def to_xarray(self) -> xr.Dataset:
        pass

    def _open_icechunk(
        self,
        prefix: str,
        is_virtual: bool,
        virtual_chunk_container: VirtualChunkContainerConfig | None = None,
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
            chunks = {}

        else:
            repo = icechunk.Repository.open(storage, config=config)
            chunks = self.encoding["shards"]

        session = repo.readonly_session("main")
        return xr.open_dataset(
            session.store, engine="zarr", chunks=chunks, consolidated=False, zarr_format=3
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

    def to_xarray(self) -> xr.Dataset:
        from srm.utils import to_proleptic_gregorian

        if self.format == "icechunk":
            return to_proleptic_gregorian(self._open_icechunk(self.prefix, is_virtual=False))
        elif self.format == "zarr":
            import xarray as xr

            return to_proleptic_gregorian(xr.open_zarr(self.path))
        else:
            raise ValueError(f"Unknown format: {self.format}")


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

    def to_xarray(self) -> xr.Dataset:
        if self.format == "icechunk":
            return self._open_icechunk(
                self.prefix, is_virtual=True, virtual_chunk_container=self.virtual_chunk_container
            )
        else:
            raise ValueError(f"icechunk only for virtual datasets {self.format}")


class Catalog:
    def __init__(self):
        self.datasets: dict[str, BaseDataset] = {
            "CESM2-WACCM-historical-icechunk-NCAR-provided": Dataset(
                name="CESM2-WACCM-historical-icechunk-NCAR-provided",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-Historical/icechunk/CESM2-WACCM-historical-NCAR-provided.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 30, "lat": 192, "lon": 288},
                expected_shards={"ensemble_member": 1, "time": 480, "lat": 192, "lon": 288},
                license="CC-BY-4.0",
                citation="Danabasoglu, Gokhan (2019). NCAR CESM2-WACCM model output prepared for CMIP6 CMIP historical. Version 20200206. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.10071",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.HUSS,
                    VarStandards.PR,
                    VarStandards.PS,
                    VarStandards.RLDS,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    VarStandards.TASMAX,
                    VarStandards.TASMIN,
                    VarStandards.DTR,
                ],
            ),
            "CESM2-WACCM-historical-icechunk": Dataset(
                name="CESM2-WACCM-historical-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-Historical/icechunk/CESM2_WACCM_Historical.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 30, "lat": 192, "lon": 288},
                expected_shards={"ensemble_member": 1, "time": 480, "lat": 192, "lon": 288},
                license="CC-BY-4.0",
                citation="Danabasoglu, Gokhan (2019). NCAR CESM2-WACCM model output prepared for CMIP6 CMIP historical. Version 20200206. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.10071",
                ensemble_members=["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                ],
            ),
            "CESM2-WACCM-historical-virtual-NCAR-provided": VirtualDataset(
                name="CESM2-WACCM-historical-virtual-NCAR-provided",
                virtual_path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-Historical/icechunk/CESM2-WACCM-historical-virtual-NCAR-provided.icechunk",
                format="icechunk",
                license="CC-BY-4.0",
                citation="Danabasoglu, Gokhan (2019). NCAR CESM2-WACCM model output prepared for CMIP6 CMIP historical. Version 20200206. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.10071",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.HUSS,
                    VarStandards.PR,
                    VarStandards.PS,
                    VarStandards.RLDS,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    VarStandards.TASMAX,
                    VarStandards.TASMIN,
                ],
            ),
            "CESM2-WACCM-G6-1.5K-icechunk-NCAR-provided": Dataset(
                name="CESM2-WACCM-G6-1.5K-icechunk-NCAR-provided",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-G6-1.5K/icechunk/CESM2-WACCM-G6-1.5k-NCAR-provided.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 30, "lat": 192, "lon": 288},
                expected_shards={"ensemble_member": 1, "time": 480, "lat": 192, "lon": 288},
                ensemble_members=["001", "002", "003"],
                citation="Lee, W. R., Visioni, D., Wagman, B. M., Wentland, C. R., Kravitz, B., Watanabe, S., Sekiya, T., Jones, A., Haywood, J., Henry, M., and Bednarz, E. M. (2025). G6-1.5K-SAI and G6sulfur: changes in impacts and uncertainty depending on stratospheric aerosol injection strategy in the Geoengineering Model Intercomparison Project. EGUsphere [preprint]. https://doi.org/10.5194/egusphere-2025-5742. Simulations run by Walker Lee using CESM2 (https://doi.org/10.5065/D67H1H0V). Walker Lee granted permission to share this data.",
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.TAS,
                    VarStandards.TASMIN,
                    VarStandards.TASMAX,
                    VarStandards.HURS,
                    VarStandards.RSDS,
                    VarStandards.HUSS,
                    VarStandards.RLDS,
                    VarStandards.PS,
                    VarStandards.DTR,
                ],
            ),
            "CESM2-WACCM-G6-1.5K-icechunk": Dataset(
                name="CESM2-WACCM-G6-1.5K-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-G6-1.5K/icechunk/CESM2-WACCM-G6-1.5k_pancakes.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 30, "lat": 192, "lon": 288},
                expected_shards={
                    "ensemble_member": 1,
                    "time": 480,
                    "lat": 192,
                    "lon": 288,
                },
                ensemble_members=["001", "002", "003"],
                citation="Lee, W. R., Visioni, D., Wagman, B. M., Wentland, C. R., Kravitz, B., Watanabe, S., Sekiya, T., Jones, A., Haywood, J., Henry, M., and Bednarz, E. M. (2025). G6-1.5K-SAI and G6sulfur: changes in impacts and uncertainty depending on stratospheric aerosol injection strategy in the Geoengineering Model Intercomparison Project. EGUsphere [preprint]. https://doi.org/10.5194/egusphere-2025-5742. Simulations run by Walker Lee using CESM2 (https://doi.org/10.5065/D67H1H0V). Walker Lee granted permission to share this data.",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    # tasmin/tasmax not available for CESM2-WACCM in CMIP6 (confirmed: NASA NEX, CEDA, Metagrid all missing)
                ],
            ),
            "CESM2-WACCM-G6-1.5K-virtual-NCAR-provided": VirtualDataset(
                name="CESM2-WACCM-G6-1.5K-virtual-NCAR-provided",
                virtual_path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-G6-1.5K/icechunk/CESM2-WACCM-G6-1.5K-virtual-NCAR-provided.icechunk",
                format="icechunk",
                citation="Lee, W. R., Visioni, D., Wagman, B. M., Wentland, C. R., Kravitz, B., Watanabe, S., Sekiya, T., Jones, A., Haywood, J., Henry, M., and Bednarz, E. M. (2025). G6-1.5K-SAI and G6sulfur: changes in impacts and uncertainty depending on stratospheric aerosol injection strategy in the Geoengineering Model Intercomparison Project. EGUsphere [preprint]. https://doi.org/10.5194/egusphere-2025-5742. Simulations run by Walker Lee using CESM2 (https://doi.org/10.5065/D67H1H0V). Walker Lee granted permission to share this data.",
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.TAS,
                    VarStandards.HURS,
                    VarStandards.RSDS,
                ],
            ),
            "CESM2-WACCM-G6-1.5K-virtual": VirtualDataset(
                name="CESM2-WACCM-G6-1.5K-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-G6-1.5K/icechunk/CESM2-WACCM-G6-1.5K-virtual.icechunk",
                format="icechunk",
                citation="Lee, W. R., Visioni, D., Wagman, B. M., Wentland, C. R., Kravitz, B., Watanabe, S., Sekiya, T., Jones, A., Haywood, J., Henry, M., and Bednarz, E. M. (2025). G6-1.5K-SAI and G6sulfur: changes in impacts and uncertainty depending on stratospheric aerosol injection strategy in the Geoengineering Model Intercomparison Project. EGUsphere [preprint]. https://doi.org/10.5194/egusphere-2025-5742. Simulations run by Walker Lee using CESM2 (https://doi.org/10.5065/D67H1H0V). Walker Lee granted permission to share this data.",
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.TAS,
                    VarStandards.HURS,
                    VarStandards.RSDS,
                    # tasmin/tasmax not available for CESM2-WACCM in CMIP6 (confirmed: NASA NEX, CEDA, Metagrid all missing)
                ],
            ),
            "CESM2-WACCM-SSP245-icechunk-NCAR-provided": Dataset(
                name="CESM2-WACCM-SSP245-icechunk-NCAR-provided",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-SSP245/icechunk/CESM2-WACCM-SSP245-NCAR-provided.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 30, "lat": 192, "lon": 288},
                expected_shards={"ensemble_member": 1, "time": 480, "lat": 192, "lon": 288},
                license="CC-BY-4.0",
                citation="Danabasoglu, Gokhan (2019). NCAR CESM2-WACCM model output prepared for CMIP6 ScenarioMIP ssp245. Version 20200206. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.10101",
                ensemble_members=[
                    "r1i1p1f1",
                    "r2i1p1f1",
                    "r3i1p1f1",
                    "r4i1p1f1",
                    "r5i1p1f1",
                    "r7i1p1f1",
                    "r8i1p1f1",
                    "r9i1p1f1",
                    "r10i1p1f1",
                ],
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.TAS,
                    VarStandards.TASMIN,
                    VarStandards.TASMAX,
                    VarStandards.HURS,
                    VarStandards.RSDS,
                    VarStandards.HUSS,
                    VarStandards.RLDS,
                    VarStandards.PS,
                    VarStandards.DTR,
                ],
            ),
            "CESM2-WACCM-SSP245-icechunk": Dataset(
                name="CESM2-WACCM-SSP245-icechunk",
                path="s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-SSP245/icechunk/CESM2_WACCM_SSP245.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 30, "lat": 192, "lon": 288},
                expected_shards={
                    "ensemble_member": 1,
                    "time": 480,
                    "lat": 192,
                    "lon": 288,
                },
                license="CC-BY-4.0",
                citation="Danabasoglu, Gokhan (2019). NCAR CESM2-WACCM model output prepared for CMIP6 ScenarioMIP ssp245. Version 20200206. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.10101",
                ensemble_members=["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    # tasmin/tasmax not available for CESM2-WACCM in CMIP6 (confirmed: NASA NEX, CEDA, Metagrid all missing)
                ],
            ),
            "MIROC-ES2H-historical-virtual": VirtualDataset(
                name="MIROC-ES2H-historical-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/historical/icechunk/MIROC-ES2H-historical-virtual.icechunk",
                format="icechunk",
                ensemble_members=["r1i1p4f2", "r2i1p4f2", "r3i1p4f2"],
                license="CC-BY-4.0",
                citation="Watanabe, Shingo; Hajima, Tomohiro; Sudo, Kengo; Abe, Manabu; Arakawa, Osamu; Ogochi, Koji; Arakawa, Takashi; Tatebe, Hiroaki; Ito, Akihiko; Ito, Akinori; Komuro, Yoshiki; Nitta, Tomoko; Noguchi, Maki A.; Ogura, Tomoo; Ohgaito, Rumi; Sekiguchi, Miho; Suzuki, Tatsuo; Tachiiri, Kaoru; Takata, Kumiko; Takemura, Toshihiko; Watanabe, Michio; Yamamoto, Akitomo; Yamazaki, Dai; Yoshimura, Kei; Kawamiya, Michio (2021). MIROC MIROC-ES2H model output prepared for CMIP6 CMIP historical. Version 20220610. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.5601",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    VarStandards.TASMAX,
                    VarStandards.TASMIN,
                ],
            ),
            "MIROC-ES2H-historical-icechunk": Dataset(
                name="MIROC-ES2H-historical-icechunk",
                path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/historical/icechunk/MIROC-ES2H-historical.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 128, "lon": 256},
                expected_shards={
                    "ensemble_member": 1,
                    "time": 960,
                    "lat": 128,
                    "lon": 256,
                },
                ensemble_members=["r1i1p4f2", "r2i1p4f2", "r3i1p4f2"],
                license="CC-BY-4.0",
                citation="Watanabe, Shingo; Hajima, Tomohiro; Sudo, Kengo; Abe, Manabu; Arakawa, Osamu; Ogochi, Koji; Arakawa, Takashi; Tatebe, Hiroaki; Ito, Akihiko; Ito, Akinori; Komuro, Yoshiki; Nitta, Tomoko; Noguchi, Maki A.; Ogura, Tomoo; Ohgaito, Rumi; Sekiguchi, Miho; Suzuki, Tatsuo; Tachiiri, Kaoru; Takata, Kumiko; Takemura, Toshihiko; Watanabe, Michio; Yamamoto, Akitomo; Yamazaki, Dai; Yoshimura, Kei; Kawamiya, Michio (2021). MIROC MIROC-ES2H model output prepared for CMIP6 CMIP historical. Version 20220610. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.5601",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    VarStandards.TASMAX,
                    VarStandards.TASMIN,
                    VarStandards.DTR,
                ],
            ),
            "MIROC-ES2H-SSP245-virtual": VirtualDataset(
                name="MIROC-ES2H-SSP245-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/ssp245/icechunk/MIROC-ES2H-SSP245-virtual.icechunk",
                format="icechunk",
                ensemble_members=["r1i1p4f2", "r2i1p4f2", "r3i1p4f2"],
                license="CC-BY-4.0",
                citation="Watanabe, Shingo; Hajima, Tomohiro; Sudo, Kengo; Abe, Manabu; Arakawa, Osamu; Ogochi, Koji; Arakawa, Takashi; Tatebe, Hiroaki; Ito, Akihiko; Ito, Akinori; Komuro, Yoshiki; Nitta, Tomoko; Noguchi, Maki A.; Ogura, Tomoo; Ohgaito, Rumi; Sekiguchi, Miho; Suzuki, Tatsuo; Tachiiri, Kaoru; Takata, Kumiko; Takemura, Toshihiko; Watanabe, Michio; Yamamoto, Akitomo; Yamazaki, Dai; Yoshimura, Kei; Kawamiya, Michio (2023). MIROC MIROC-ES2H model output prepared for CMIP6 ScenarioMIP ssp245. Version 20220610. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.15658",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    VarStandards.TASMAX,
                    VarStandards.TASMIN,
                ],
            ),
            "MIROC-ES2H-SSP245-icechunk": Dataset(
                name="MIROC-ES2H-SSP245-icechunk",
                path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/ssp245/icechunk/MIROC-ES2H-SSP245.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 128, "lon": 256},
                expected_shards={
                    "ensemble_member": 1,
                    "time": 960,
                    "lat": 128,
                    "lon": 256,
                },
                ensemble_members=["r1i1p4f2", "r2i1p4f2", "r3i1p4f2"],
                license="CC-BY-4.0",
                citation="Watanabe, Shingo; Hajima, Tomohiro; Sudo, Kengo; Abe, Manabu; Arakawa, Osamu; Ogochi, Koji; Arakawa, Takashi; Tatebe, Hiroaki; Ito, Akihiko; Ito, Akinori; Komuro, Yoshiki; Nitta, Tomoko; Noguchi, Maki A.; Ogura, Tomoo; Ohgaito, Rumi; Sekiguchi, Miho; Suzuki, Tatsuo; Tachiiri, Kaoru; Takata, Kumiko; Takemura, Toshihiko; Watanabe, Michio; Yamamoto, Akitomo; Yamazaki, Dai; Yoshimura, Kei; Kawamiya, Michio (2023). MIROC MIROC-ES2H model output prepared for CMIP6 ScenarioMIP ssp245. Version 20220610. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.15658",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    VarStandards.TASMAX,
                    VarStandards.TASMIN,
                    VarStandards.DTR,
                ],
            ),
            "MIROC-ES2H-G6-1.5K-virtual": VirtualDataset(
                name="MIROC-ES2H-G6-1.5K-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/G6-1.5K/icechunk/MIROC-ES2H-G6-1.5K-virtual.icechunk",
                format="icechunk",
                ensemble_members=[
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
                license="CC-BY-4.0",
                citation="MIROC-ES2H simulations were performed using the Earth Simulator at JAMSTEC. Shingo Watanabe granted permission to share this data.",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    VarStandards.TASMAX,
                    VarStandards.TASMIN,
                ],
            ),
            "MIROC-ES2H-G6-1.5K-icechunk": Dataset(
                name="MIROC-ES2H-G6-1.5K-icechunk",
                path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/G6-1.5K/icechunk/MIROC-ES2H-G6-1.5K.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 128, "lon": 256},
                expected_shards={
                    "ensemble_member": 1,
                    "time": 960,
                    "lat": 128,
                    "lon": 256,
                },
                ensemble_members=[
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
                license="CC-BY-4.0",
                citation="MIROC-ES2H simulations were performed using the Earth Simulator at JAMSTEC. Shingo Watanabe granted permission to share this data.",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    VarStandards.TASMAX,
                    VarStandards.TASMIN,
                    VarStandards.DTR,
                ],
            ),
            "MIROC-ES2H-baseline-virtual": VirtualDataset(
                name="MIROC-ES2H-baseline-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/baseline/icechunk/MIROC-ES2H-baseline-virtual.icechunk",
                format="icechunk",
                ensemble_members=[
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
                license="CC-BY-4.0",
                citation="MIROC-ES2H simulations were performed using the Earth Simulator at JAMSTEC. Shingo Watanabe granted permission to share this data.",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    VarStandards.TASMAX,
                    VarStandards.TASMIN,
                ],
            ),
            "MIROC-ES2H-baseline-icechunk": Dataset(
                name="MIROC-ES2H-baseline-icechunk",
                path="s3://carbonplan-srm/input/tensor/MIROC-ES2H/baseline/icechunk/MIROC-ES2H-baseline.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 128, "lon": 256},
                expected_shards={
                    "ensemble_member": 1,
                    "time": 960,
                    "lat": 128,
                    "lon": 256,
                },
                ensemble_members=[
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
                license="CC-BY-4.0",
                citation="MIROC-ES2H simulations were performed using the Earth Simulator at JAMSTEC. Shingo Watanabe granted permission to share this data.",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.PR,
                    VarStandards.RSDS,
                    VarStandards.TAS,
                    VarStandards.TASMAX,
                    VarStandards.TASMIN,
                    VarStandards.DTR,
                ],
            ),
            "UKESM-historical-icechunk": Dataset(
                name="UKESM-historical-icechunk",
                path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-historical/UKESM-historical.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_shards={
                    "ensemble_member": 1,
                    "time": 960,
                    "lat": 144,
                    "lon": 192,
                },
                license="CC-BY-4.0",
                citation="Tang, Yongming; Rumbold, Steve; Ellis, Rich; Kelley, Douglas; Mulcahy, Jane; Sellar, Alistair; Walton, Jeremy; Jones, Colin (2019). MOHC UKESM1.0-LL model output prepared for CMIP6 CMIP historical. Version 20191209. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.6113",
                ensemble_members=["r2i1p1f2", "r3i1p1f2", "r12i1p1f2"],
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.TAS,
                    VarStandards.TASMIN,
                    VarStandards.TASMAX,
                    VarStandards.HURS,
                    VarStandards.RSDS,
                    VarStandards.DTR,
                ],
            ),
            "UKESM-historical-virtual": VirtualDataset(
                name="UKESM-historical-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-historical/UKESM-historical-virtual.icechunk",
                format="icechunk",
                license="CC-BY-4.0",
                citation="Tang, Yongming; Rumbold, Steve; Ellis, Rich; Kelley, Douglas; Mulcahy, Jane; Sellar, Alistair; Walton, Jeremy; Jones, Colin (2019). MOHC UKESM1.0-LL model output prepared for CMIP6 CMIP historical. Version 20191209. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.6113",
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.TAS,
                    VarStandards.TASMIN,
                    VarStandards.TASMAX,
                    VarStandards.HURS,
                    VarStandards.RSDS,
                ],
            ),
            "UKESM-SSP245-icechunk": Dataset(
                name="UKESM-SSP245-icechunk",
                path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-SSP245/UKESM_SSP245.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_shards={
                    "ensemble_member": 1,
                    "time": 960,
                    "lat": 144,
                    "lon": 192,
                },
                ensemble_members=["r12i1p1f2", "r2i1p1f2", "r3i1p1f2"],
                license="OGLv3",
                citation="Simulations run by Andy Jones in collaboration with Jim Haywood and Matthew Henry. The UK Earth System Model is developed and maintained by the Met Office Hadley Centre.",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.RSDS,
                ],
            ),
            "UKESM-SSP245-t-pr-icechunk": Dataset(
                name="UKESM-SSP245-t-pr-icechunk",
                path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-SSP245-T-PR/UKESM-SSP245-t-pr.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_shards={
                    "ensemble_member": 1,
                    "time": 960,
                    "lat": 144,
                    "lon": 192,
                },
                ensemble_members=["001", "002", "003"],
                license="OGLv3",
                citation="Simulations run by Andy Jones in collaboration with Jim Haywood and Matthew Henry. The UK Earth System Model is developed and maintained by the Met Office Hadley Centre.",
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.TAS,
                    VarStandards.TASMIN,
                    VarStandards.TASMAX,
                    VarStandards.DTR,
                ],
            ),
            "UKESM-SSP245-virtual": VirtualDataset(
                name="UKESM-SSP245-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-SSP245/UKESM-SSP245-virtual.icechunk",
                format="icechunk",
                license="OGLv3",
                citation="Simulations run by Andy Jones in collaboration with Jim Haywood and Matthew Henry. The UK Earth System Model is developed and maintained by the Met Office Hadley Centre.",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.RSDS,
                ],
            ),
            "UKESM-SSP245-t-pr-virtual": VirtualDataset(
                name="UKESM-SSP245-t-pr-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-SSP245-T-PR/UKESM-SSP245-t-pr-virtual.icechunk",
                format="icechunk",
                license="OGLv3",
                citation="Simulations run by Andy Jones in collaboration with Jim Haywood and Matthew Henry. The UK Earth System Model is developed and maintained by the Met Office Hadley Centre.",
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.TAS,
                    VarStandards.TASMIN,
                    VarStandards.TASMAX,
                ],
            ),
            "UKESM-G6-1.5K-icechunk": Dataset(
                name="UKESM-G6-1.5K-icechunk",
                path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-G6-1.5K/UKESM_G6_1.5K.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_shards={
                    "ensemble_member": 1,
                    "time": 960,
                    "lat": 144,
                    "lon": 192,
                },
                ensemble_members=["r12i1p1f2", "r2i1p1f2", "r3i1p1f2"],
                license="OGLv3",
                citation="Simulations run by Andy Jones in collaboration with Jim Haywood and Matthew Henry. The UK Earth System Model is developed and maintained by the Met Office Hadley Centre.",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.RSDS,
                ],
            ),
            "UKESM-G6-1.5K-virtual": VirtualDataset(
                name="UKESM-G6-1.5K-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-G6-1.5K/UKESM_G6_1.5K_virtual.icechunk",
                format="icechunk",
                license="OGLv3",
                citation="Simulations run by Andy Jones in collaboration with Jim Haywood and Matthew Henry. The UK Earth System Model is developed and maintained by the Met Office Hadley Centre.",
                expected_vars=[
                    VarStandards.HURS,
                    VarStandards.RSDS,
                ],
            ),
            "UKESM-G6-1.5K-t-pr-virtual": VirtualDataset(
                name="UKESM-G6-1.5K-t-pr-virtual",
                virtual_path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-G6-1.5K-T-PR/UKESM_G6_1.5K_t_pr_virtual.icechunk",
                format="icechunk",
                license="OGLv3",
                citation="Simulations run by Andy Jones in collaboration with Jim Haywood and Matthew Henry. The UK Earth System Model is developed and maintained by the Met Office Hadley Centre.",
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.TAS,
                    VarStandards.TASMIN,
                    VarStandards.TASMAX,
                ],
            ),
            "UKESM-G6-1.5K-t-pr-icechunk": Dataset(
                name="UKESM-G6-1.5K-t-pr-icechunk",
                path="s3://carbonplan-srm/input/tensor/UKESM/UKESM-G6-1.5K-T-PR/UKESM-G6-1.5K-t-pr.icechunk",
                format="icechunk",
                expected_chunks={"ensemble_member": 1, "time": 60, "lat": 144, "lon": 192},
                expected_shards={
                    "ensemble_member": 1,
                    "time": 960,
                    "lat": 144,
                    "lon": 192,
                },
                ensemble_members=["001", "002", "003"],
                license="OGLv3",
                citation="Simulations run by Andy Jones in collaboration with Jim Haywood and Matthew Henry. The UK Earth System Model is developed and maintained by the Met Office Hadley Centre.",
                expected_vars=[
                    VarStandards.PR,
                    VarStandards.TAS,
                    VarStandards.TASMIN,
                    VarStandards.TASMAX,
                    VarStandards.DTR,
                ],
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
