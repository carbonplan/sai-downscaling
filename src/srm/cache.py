"""
Artifact caching system for BCSD pipeline.

Manages S3-based cache storage with dependency tracking and automatic
cache validation. Supports three stages:
- obs_regridded: Observation data regridded to GCM grid
- historical: Downscaled historical period
- scenario: Downscaled future scenario
"""

from __future__ import annotations

import logging
from pathlib import Path

import fsspec

from srm.bcsd_config import BCSDConfig, VariableConfig

logger = logging.getLogger(__name__)


class ArtifactCache:
    """
    S3-based cache manager with dependency tracking.

    Ensures efficient reuse of intermediate artifacts across BCSD runs.
    Cache paths are deterministic based on configuration parameters.
    """

    def __init__(
        self,
        base_path: str = "s3://carbonplan-scratch/srm/cache/",
        environment: str = "qa",
        version: str = "v1",
        output_dir: str | None = None,
    ):
        """
        Initialize cache manager.

        Parameters
        ----------
        base_path : str
            Base S3 or local path for cache storage (intermediate artifacts)
        environment : str
            Environment name (qa, staging, production) for cache namespace isolation
        version : str
            Version identifier included in all paths (e.g. 'v1', 'v2'). Bump to
            invalidate all cached artifacts without changing environment.
        output_dir : str, optional
            Directory for final scenario outputs. If None, scenarios go to cache.
        """
        self.base_path = base_path.rstrip("/")
        self.environment = environment
        self.version = version
        self.output_dir = output_dir.rstrip("/") if output_dir else None
        self.config: BCSDConfig | None = None

        # Initialize filesystem (works for s3:// and local paths)
        if base_path.startswith("s3://"):
            self.fs = fsspec.filesystem("s3")
        else:
            self.fs = fsspec.filesystem("local")

    @classmethod
    def from_config(cls, config: BCSDConfig) -> ArtifactCache:
        """
        Create ArtifactCache instance from BCSDConfig.

        Parameters
        ----------
        config : BCSDConfig
            Configuration object containing cache parameters

        Returns
        -------
        ArtifactCache
            Initialized cache manager
        """
        cache = cls(
            base_path=config.cache_dir,
            environment=config.environment,
            version=config.version,
            output_dir=config.output_dir,
        )
        cache.config = config
        return cache

    @staticmethod
    def _get_subset_id(subset_bounds: tuple[float, float, float, float] | None) -> str:
        """
        Generate a unique identifier for spatial subset bounds.

        Parameters
        ----------
        subset_bounds : tuple or None
            Spatial bounds (lat_min, lat_max, lon_min, lon_max)

        Returns
        -------
        str
            Human-readable subset identifier
        """
        if subset_bounds is None:
            return "global"
        lat_min, lat_max, lon_min, lon_max = subset_bounds
        return f"lat{lat_min}to{lat_max}_lon{lon_min}to{lon_max}"

    @staticmethod
    def _get_varconfig_id(variable_config: VariableConfig, mapping_type: str) -> str:
        """
        Generate a human-readable path segment encoding VariableConfig and mapping type.

        Included in historical and scenario cache paths so that runs with non-default
        VariableConfig or mapping_type never collide with standard runs.

        Parameters
        ----------
        variable_config : VariableConfig
            Variable-specific BCSD parameters.
        mapping_type : str
            Quantile mapping method ('parametric' or 'nonparametric').

        Returns
        -------
        str
            Path segment, e.g. ``dt1-win1-dsadditive-dscfft-dtmadditive-parametric``.
        """
        return f"{variable_config.to_path_id()}-{mapping_type}"

    def _require_config(self) -> BCSDConfig:
        if self.config is None:
            raise RuntimeError(
                "No config bound to this cache. Use ArtifactCache.from_config(config) "
                "or pass config explicitly to the path methods."
            )
        return self.config

    @property
    def obs_path(self) -> str:
        """Path to the obs regridding artifact for the bound config."""
        return self.get_obs_path(self._require_config())

    @property
    def historical_path(self) -> str:
        """Path to the historical downscaling artifact for the bound config."""
        return self.get_historical_path(self._require_config())

    @property
    def scenario_path(self) -> str:
        """Path to the scenario downscaling artifact for the bound config."""
        return self.get_scenario_path(self._require_config())

    def get_obs_path(self, config: BCSDConfig) -> str:
        """
        Get path to cached observation regridding artifact.

        Parameters
        ----------
        config : BCSDConfig
            Run configuration

        Returns
        -------
        str
            S3 or local path to zarr store
        """
        subset_id = self._get_subset_id(config.subset_bounds)
        return f"{self.base_path}/{self.environment}/{self.version}/obs/{config.gcm}_{config.variable}_{subset_id}_obs_regridded.icechunk"

    def get_historical_path(self, config: BCSDConfig) -> str:
        """
        Get path to cached historical downscaling artifact.

        Parameters
        ----------
        config : BCSDConfig
            Run configuration

        Returns
        -------
        str
            S3 or local path to zarr store
        """
        subset_id = self._get_subset_id(config.subset_bounds)
        varconfig_id = self._get_varconfig_id(config.variable_config, config.mapping_type)
        if self.output_dir:
            return f"{self.output_dir}/{self.environment}/{self.version}/historical/{config.gcm}_{config.variable}_{config.ensemble_member}_{subset_id}_{varconfig_id}_historical.icechunk"
        else:
            return (
                f"{self.base_path}/{self.environment}/{self.version}/historical/"
                f"{config.gcm}_{config.variable}_{config.ensemble_member}_{subset_id}_{varconfig_id}_historical.icechunk"
            )

    def get_scenario_path(self, config: BCSDConfig) -> str:
        """
        Get path to scenario downscaling output.

        Final scenario outputs are written to output_dir (if specified) rather than
        cache_dir, since they are the final deliverable products.

        Parameters
        ----------
        config : BCSDConfig
            Run configuration

        Returns
        -------
        str
            S3 or local path to zarr store
        """
        subset_id = self._get_subset_id(config.subset_bounds)
        varconfig_id = self._get_varconfig_id(config.variable_config, config.mapping_type)
        scenario_lower = config.scenario.lower()

        # Use output_dir for final scenarios if specified, otherwise cache
        if self.output_dir:
            return f"{self.output_dir}/{self.environment}/{self.version}/{scenario_lower}/{config.gcm}_{config.variable}_{config.ensemble_member}_{subset_id}_{varconfig_id}_{scenario_lower}.icechunk"
        else:
            return (
                f"{self.base_path}/{self.environment}/{self.version}/{scenario_lower}/"
                f"{config.gcm}_{config.variable}_{config.ensemble_member}_{subset_id}_{varconfig_id}_{scenario_lower}.icechunk"
            )

    def exists(self, path: str) -> bool:
        """
        Check if artifact exists in cache.

        Uses icechunk repository ancestry to verify a successful write, ensuring
        the store is complete and not partially written.

        Parameters
        ----------
        path : str
            Full path to icechunk store

        Returns
        -------
        bool
            True if the store exists and has a 'write complete' commit in its ancestry
        """
        try:
            import icechunk

            if path.startswith("s3://"):
                path_no_scheme = path[len("s3://") :]
                bucket, _, prefix = path_no_scheme.partition("/")
                storage = icechunk.s3_storage(bucket=bucket, prefix=prefix)
            else:
                storage = icechunk.local_filesystem_storage(path=path)

            repo = icechunk.Repository.open(storage)
            messages = [c.message for c in repo.ancestry(branch="main")]
            result = "write complete" in messages
            if result:
                logger.debug(f"Cache hit: {path}")
            else:
                logger.debug(f"Cache miss (no write complete commit): {path}")
            return result
        except Exception as e:
            logger.debug(f"Cache miss: {path}: {e}")
            return False

    def check_dependencies(self, stage: str, config: BCSDConfig) -> dict[str, tuple[bool, str]]:
        """
        Check if all dependencies for a stage exist.

        Parameters
        ----------
        stage : str
            Pipeline stage: 'prepare_observations', 'fit_historical', or 'transform_scenario'
        config : BCSDConfig
            Configuration for the run

        Returns
        -------
        dict[str, tuple[bool, str]]
            Mapping of dependency name to (exists, path) tuple
        """
        if stage == "prepare_observations":
            return {}  # No dependencies

        elif stage == "fit_historical":
            obs_path = self.get_obs_path(config)
            return {"obs_regridded": (self.exists(obs_path), obs_path)}

        elif stage == "transform_scenario":
            obs_path = self.get_obs_path(config)
            hist_path = self.get_historical_path(config)
            return {
                "obs_regridded": (self.exists(obs_path), obs_path),
                "historical": (self.exists(hist_path), hist_path),
            }

        else:
            raise ValueError(f"Unknown stage: {stage}")

    def validate_dependencies(self, stage: str, config: BCSDConfig) -> None:
        """
        Validate that all dependencies exist, raising error if missing.

        Parameters
        ----------
        stage : str
            Pipeline stage
        config : BCSDConfig
            Configuration for the run

        Raises
        ------
        ValueError
            If any required dependencies are missing
        """
        deps = self.check_dependencies(stage, config)
        missing = {name: path for name, (exists, path) in deps.items() if not exists}

        if missing:
            dep_list = "\n  ".join([f"{name}: {path}" for name, path in missing.items()])
            raise ValueError(
                f"Missing dependencies for stage '{stage}':\n  {dep_list}\n"
                f"Run the required upstream stages first."
            )

    def get_output_path(self, stage: str, config: BCSDConfig) -> str:
        """
        Get output path for a given stage and config.

        Parameters
        ----------
        stage : str
            Pipeline stage
        config : BCSDConfig
            Configuration for the run

        Returns
        -------
        str
            Full path to output artifact
        """
        if stage == "prepare_observations":
            return self.get_obs_path(config)

        elif stage == "fit_historical":
            return self.get_historical_path(config)

        elif stage == "transform_scenario":
            if config.scenario is None:
                raise ValueError("scenario must be specified for transform_scenario stage")
            return self.get_scenario_path(config)

        else:
            raise ValueError(f"Unknown stage: {stage}")

    def clear_cache(
        self,
        stage: str | None = None,
        gcm: str | None = None,
        variable: str | None = None,
    ) -> int:
        """
        Clear cached artifacts matching filters.

        Parameters
        ----------
        stage : str, optional
            Clear only specific stage ('obs', 'historical', 'scenarios')
        gcm : str, optional
            Clear only specific GCM
        variable : str, optional
            Clear only specific variable

        Returns
        -------
        int
            Number of artifacts deleted
        """
        deleted_count = 0

        # Build search patterns
        if stage:
            search_base = f"{self.base_path}/{self.environment}/{self.version}/{stage}/"
        else:
            search_base = f"{self.base_path}/{self.environment}/{self.version}/"

        try:
            # List all zarr stores
            if self.base_path.startswith("s3://"):
                search_base_no_scheme = search_base.replace("s3://", "")
                all_paths = self.fs.glob(f"{search_base_no_scheme}**/*.icechunk")
                all_paths = [f"s3://{p}" for p in all_paths]
            else:
                all_paths = list(Path(search_base).rglob("*.icechunk"))
                all_paths = [str(p) for p in all_paths]

            # Filter by GCM and variable if specified
            for path in all_paths:
                path_parts = Path(path).name.split("_")

                if gcm and not path_parts[0] == gcm:
                    continue
                if variable and len(path_parts) > 1 and not path_parts[1] == variable:
                    continue

                # Delete the zarr store
                if self.base_path.startswith("s3://"):
                    path_no_scheme = path.replace("s3://", "")
                    self.fs.rm(path_no_scheme, recursive=True)
                else:
                    import shutil

                    shutil.rmtree(path)

                deleted_count += 1
                logger.info(f"Deleted cache: {path}")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

        return deleted_count

    def list_artifacts(
        self,
        stage: str | None = None,
        gcm: str | None = None,
        variable: str | None = None,
    ) -> list[str]:
        """
        List cached artifacts matching filters.

        Uses efficient S3 prefix listing instead of recursive globbing.

        Parameters
        ----------
        stage : str, optional
            List only specific stage
        gcm : str, optional
            List only specific GCM
        variable : str, optional
            List only specific variable

        Returns
        -------
        list[str]
            List of artifact paths
        """
        artifacts = set()

        # Determine which stages to search
        if stage:
            stages = [stage]
        else:
            stages = ["obs", "historical", "scenarios"]

        try:
            for stage_name in stages:
                # Scenarios go to output_dir if specified, others to cache
                if stage_name == "scenarios" and self.output_dir:
                    search_base = f"{self.output_dir}/{self.environment}/{self.version}/"
                else:
                    search_base = (
                        f"{self.base_path}/{self.environment}/{self.version}/{stage_name}/"
                    )

                if self.base_path.startswith("s3://"):
                    search_base_no_scheme = search_base.replace("s3://", "")

                    # List all objects under the prefix (non-recursive, just first level)
                    # This is much faster than recursive globbing
                    try:
                        all_files = self.fs.ls(search_base_no_scheme, detail=False)
                    except FileNotFoundError:
                        continue

                    # Look for icechunk stores (directories ending in .icechunk)
                    for path in all_files:
                        if path.endswith(".icechunk"):
                            full_path = f"s3://{path}"

                            # Apply filters
                            path_parts = Path(path).name.split("_")
                            if gcm and path_parts[0] != gcm:
                                continue
                            if variable and len(path_parts) > 1 and path_parts[1] != variable:
                                continue

                            # Verify it's a valid zarr store
                            if self.exists(full_path):
                                artifacts.add(full_path)
                else:
                    # Local filesystem
                    search_path = Path(search_base)
                    if not search_path.exists():
                        continue

                    for path in search_path.glob("*.icechunk"):
                        # Apply filters
                        path_parts = path.name.split("_")
                        if gcm and path_parts[0] != gcm:
                            continue
                        if variable and len(path_parts) > 1 and path_parts[1] != variable:
                            continue

                        if self.exists(str(path)):
                            artifacts.add(str(path))

        except Exception as e:
            logger.error(f"Error listing artifacts: {e}")

        return sorted(list(artifacts))
