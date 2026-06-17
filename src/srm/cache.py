"""
Artifact caching system for BCSD pipeline.

Manages icechunk-based cache storage with ancestry-based existence checks.
Supports three pipeline stages: obs_regridded, historical, and scenario,
plus intermediate artifacts when save_intermediate is enabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from srm.bcsd_config import BCSDConfig, MappingType, PipelineOptions, VariableConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoreLocation:
    """Pointer to one zarr group within an icechunk repository.

    Parameters
    ----------
    store_path : str
        S3 URI or local path to the icechunk repository.
    group : str
        Zarr group path within the repository (e.g. ``"obs/tas"``).
    """

    store_path: str
    group: str


class CacheCheckError(Exception):
    """Raised when an infrastructure error prevents verifying cache existence.

    Distinguishes real errors (S3 auth failure, permission denied, network error)
    from a simple cache miss. Callers can catch this to decide fallback behavior.
    """


class ArtifactCache:
    """
    Icechunk-based cache manager with ancestry-based existence checks.

    Cache locations are deterministic based on configuration parameters.
    Each artifact is identified by a ``StoreLocation`` that encodes both
    the icechunk repository path and the zarr group within it.
    """

    def __init__(
        self,
        scratch_dir: str = "s3://carbonplan-scratch/srm/cache/",
        environment: str = "qa",
        version: str = "v1",
        output_dir: str | None = None,
    ):
        """
        Initialize cache manager.

        Parameters
        ----------
        scratch_dir : str
            Base S3 or local path for scratch storage (intermediate artifacts).
        environment : str
            Environment name (qa, production) for cache namespace isolation.
        version : str
            Version identifier included in all paths (e.g. "v1", "v2"). Bump to
            invalidate all cached artifacts without changing environment.
        output_dir : str, optional
            Directory for final scenario outputs. If None, scenarios go to scratch.
        """
        self.scratch_dir = scratch_dir.rstrip("/")
        self.environment = environment
        self.version = version
        self.output_dir = output_dir.rstrip("/") if output_dir else None
        self.config: BCSDConfig | None = None

    @classmethod
    def from_config(cls, config: BCSDConfig, options: PipelineOptions) -> ArtifactCache:
        """
        Create ArtifactCache instance from BCSDConfig and PipelineOptions.

        Parameters
        ----------
        config : BCSDConfig
            Run-identity configuration.
        options : PipelineOptions
            Operational settings containing storage paths.

        Returns
        -------
        ArtifactCache
            Initialized cache manager.
        """
        cache = cls(
            scratch_dir=options.scratch_dir,
            environment=options.environment,
            version=options.version,
            output_dir=options.output_dir,
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
            Spatial bounds (lat_min, lat_max, lon_min, lon_max).

        Returns
        -------
        str
            Human-readable subset identifier.
        """
        if subset_bounds is None:
            return "global"
        lat_min, lat_max, lon_min, lon_max = subset_bounds
        return f"lat{lat_min}to{lat_max}_lon{lon_min}to{lon_max}"

    @staticmethod
    def _get_varconfig_id(variable_config: VariableConfig, mapping_type: MappingType) -> str:
        """8-character hash of VariableConfig fields + mapping_type. See ``VariableConfig.to_hash``."""
        return variable_config.to_hash(mapping_type)

    def _require_config(self) -> BCSDConfig:
        if self.config is None:
            raise RuntimeError(
                "No config bound to this cache. Use ArtifactCache.from_config(config) "
                "or pass config explicitly to the path methods."
            )
        return self.config

    # ── store-path helpers ────────────────────────────────────────────────────

    @property
    def _scratch_store(self) -> str:
        config = self._require_config()
        subset_id = self._get_subset_id(config.subset_bounds)
        return (
            f"{self.scratch_dir}/{self.environment}"
            f"/{self.version}"
            f"/{config.gcm}-{config.obs_dataset}-{subset_id}.icechunk"
        )

    @property
    def _output_store(self) -> str:
        config = self._require_config()
        subset_id = self._get_subset_id(config.subset_bounds)
        base = self.output_dir if self.output_dir else self.scratch_dir
        return (
            f"{base}/{self.environment}"
            f"/{self.version}"
            f"/{config.gcm}-{config.obs_dataset}-{subset_id}.icechunk"
        )

    # ── artifact location properties ─────────────────────────────────────────

    @property
    def obs_loc(self) -> StoreLocation:
        """StoreLocation for the obs-regridding artifact."""
        config = self._require_config()
        return StoreLocation(self._scratch_store, f"obs/{config.variable}")

    def historical_loc(self, hist_member: str) -> StoreLocation:
        """StoreLocation for the historical bias-correction artifact.

        Parameters
        ----------
        hist_member : str
            Resolved historical ensemble member ID.
        """
        config = self._require_config()
        return StoreLocation(
            self._scratch_store,
            f"historical/{config.variable}/{hist_member}",
        )

    @property
    def scenario_loc(self) -> StoreLocation:
        """StoreLocation for the scenario downscaling output."""
        from srm.config import SCENARIO_TO_GROUP

        config = self._require_config()
        scenario_group = SCENARIO_TO_GROUP[config.scenario]
        return StoreLocation(
            self._output_store,
            f"{scenario_group}/{config.variable}/{config.ensemble_member}",
        )

    def debiased_historical_loc(self, hist_member: str) -> StoreLocation:
        """StoreLocation for intermediate debiased-historical artifact."""
        config = self._require_config()
        return StoreLocation(
            self._scratch_store,
            f"debiased_historical/{config.variable}/{hist_member}",
        )

    def detrended_scenario_loc(self) -> StoreLocation:
        """StoreLocation for intermediate detrended-scenario artifact."""
        from srm.config import SCENARIO_TO_GROUP

        config = self._require_config()
        scenario_group = SCENARIO_TO_GROUP[config.scenario]
        return StoreLocation(
            self._scratch_store,
            f"detrended_scenario/{scenario_group}/{config.variable}/{config.ensemble_member}",
        )

    def trend_scenario_loc(self) -> StoreLocation:
        """StoreLocation for intermediate trend-scenario artifact."""
        from srm.config import SCENARIO_TO_GROUP

        config = self._require_config()
        scenario_group = SCENARIO_TO_GROUP[config.scenario]
        return StoreLocation(
            self._scratch_store,
            f"trend_scenario/{scenario_group}/{config.variable}/{config.ensemble_member}",
        )

    def debiased_scenario_loc(self) -> StoreLocation:
        """StoreLocation for intermediate debiased-scenario artifact."""
        from srm.config import SCENARIO_TO_GROUP

        config = self._require_config()
        scenario_group = SCENARIO_TO_GROUP[config.scenario]
        return StoreLocation(
            self._scratch_store,
            f"debiased_scenario/{scenario_group}/{config.variable}/{config.ensemble_member}",
        )

    def debiased_retrended_scenario_loc(self) -> StoreLocation:
        """StoreLocation for intermediate debiased-retrended-scenario artifact."""
        from srm.config import SCENARIO_TO_GROUP

        config = self._require_config()
        scenario_group = SCENARIO_TO_GROUP[config.scenario]
        return StoreLocation(
            self._scratch_store,
            f"debiased_retrended_scenario/{scenario_group}/{config.variable}/{config.ensemble_member}",
        )

    # ── existence check ───────────────────────────────────────────────────────

    def exists(self, loc: StoreLocation) -> bool:
        """
        Check if an artifact exists by scanning the icechunk commit ancestry.

        A commit is present iff a snapshot with message equal to ``loc.group``
        exists on the main branch. This is atomic — partial writes leave no
        matching commit.

        Parameters
        ----------
        loc : StoreLocation
            Location to check.

        Returns
        -------
        bool
            True if a commit with message ``loc.group`` is in the ancestry.
        """
        import icechunk

        try:
            if loc.store_path.startswith("s3://"):
                path_no_scheme = loc.store_path[len("s3://") :]
                bucket, _, prefix = path_no_scheme.partition("/")
                storage = icechunk.s3_storage(bucket=bucket, prefix=prefix)
            else:
                storage = icechunk.local_filesystem_storage(path=loc.store_path)

            repo = icechunk.Repository.open(storage)
            result = any(snapshot.message == loc.group for snapshot in repo.ancestry(branch="main"))
            if result:
                logger.debug("Cache hit: %s / %s", loc.store_path, loc.group)
            else:
                logger.debug(
                    "Cache miss (group not in ancestry): %s / %s", loc.store_path, loc.group
                )
            return result
        except Exception:
            logger.debug("Cache miss (store does not exist): %s", loc.store_path)
            return False

    # ── dependency helpers ────────────────────────────────────────────────────

    def check_dependencies(
        self, stage: str, config: BCSDConfig, hist_member: str | None = None
    ) -> dict[str, tuple[bool, StoreLocation]]:
        """
        Check if all dependencies for a stage exist.

        Parameters
        ----------
        stage : str
            Pipeline stage: 'prepare_observations', 'fit_historical', or 'transform_scenario'.
        config : BCSDConfig
            Configuration for the run.
        hist_member : str, optional
            Resolved historical ensemble member (required for transform_scenario).

        Returns
        -------
        dict[str, tuple[bool, StoreLocation]]
            Mapping of dependency name to (exists, StoreLocation) tuple.
        """
        if stage == "prepare_observations":
            return {}

        elif stage == "fit_historical":
            loc = self.obs_loc
            return {"obs_regridded": (self.exists(loc), loc)}

        elif stage == "transform_scenario":
            obs_loc = self.obs_loc
            hist_loc = self.historical_loc(hist_member or config.ensemble_member)
            return {
                "obs_regridded": (self.exists(obs_loc), obs_loc),
                "historical": (self.exists(hist_loc), hist_loc),
            }

        else:
            raise ValueError(f"Unknown stage: {stage}")

    def validate_dependencies(
        self, stage: str, config: BCSDConfig, hist_member: str | None = None
    ) -> None:
        """
        Validate that all dependencies exist, raising error if missing.

        Parameters
        ----------
        stage : str
            Pipeline stage.
        config : BCSDConfig
            Configuration for the run.
        hist_member : str, optional
            Resolved historical ensemble member.

        Raises
        ------
        ValueError
            If any required dependencies are missing.
        """
        deps = self.check_dependencies(stage, config, hist_member=hist_member)
        missing = {name: loc for name, (exists, loc) in deps.items() if not exists}

        if missing:
            dep_list = "\n  ".join(
                [f"{name}: {loc.store_path} / {loc.group}" for name, loc in missing.items()]
            )
            raise ValueError(
                f"Missing dependencies for stage '{stage}':\n  {dep_list}\n"
                f"Run the required upstream stages first."
            )

    def get_output_path(
        self, stage: str, config: BCSDConfig, hist_member: str | None = None
    ) -> str:
        """
        Get output store path for a given stage (returns store_path only, not group).

        Parameters
        ----------
        stage : str
            Pipeline stage.
        config : BCSDConfig
            Configuration for the run.
        hist_member : str, optional
            Resolved historical ensemble member.

        Returns
        -------
        str
            Full path to the icechunk store.
        """
        if stage == "prepare_observations":
            return self.obs_loc.store_path

        elif stage == "fit_historical":
            return self.historical_loc(hist_member or config.ensemble_member).store_path

        elif stage == "transform_scenario":
            if config.scenario is None:
                raise ValueError("scenario must be specified for transform_scenario stage")
            return self.scenario_loc.store_path

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
            Clear only specific stage ('obs', 'historical', 'scenarios').
        gcm : str, optional
            Clear only specific GCM.
        variable : str, optional
            Clear only specific variable.

        Returns
        -------
        int
            Number of artifacts deleted.
        """
        deleted_count = 0

        if stage:
            search_base = f"{self.scratch_dir}/{self.environment}/{self.version}/{stage}/"
        else:
            search_base = f"{self.scratch_dir}/{self.environment}/{self.version}/"

        try:
            import fsspec

            if self.scratch_dir.startswith("s3://"):
                fs = fsspec.filesystem("s3")
                search_base_no_scheme = search_base.replace("s3://", "")
                all_paths = fs.glob(f"{search_base_no_scheme}**/*.icechunk")
                all_paths = [f"s3://{p}" for p in all_paths]
            else:
                fs = fsspec.filesystem("local")
                all_paths = list(Path(search_base).rglob("*.icechunk"))
                all_paths = [str(p) for p in all_paths]

            for path in all_paths:
                if gcm and f"/{gcm}" not in path:
                    continue
                if variable and f"/{variable}/" not in path:
                    continue

                if self.scratch_dir.startswith("s3://"):
                    path_no_scheme = path.replace("s3://", "")
                    fs.rm(path_no_scheme, recursive=True)
                else:
                    import shutil

                    shutil.rmtree(path)

                deleted_count += 1
                logger.info("Deleted cache: %s", path)

        except Exception as e:
            logger.error("Error clearing cache: %s", e)

        return deleted_count

    def list_artifacts(
        self,
        stage: str | None = None,
        gcm: str | None = None,
        variable: str | None = None,
    ) -> list[str]:
        """
        List cached artifact store paths matching filters.

        Parameters
        ----------
        stage : str, optional
            List only specific stage.
        gcm : str, optional
            List only specific GCM.
        variable : str, optional
            List only specific variable.

        Returns
        -------
        list[str]
            List of icechunk store paths.
        """
        artifacts: set[str] = set()

        stages = [stage] if stage else ["obs", "historical", "scenarios"]

        try:
            import fsspec

            for stage_name in stages:
                if stage_name == "scenarios" and self.output_dir:
                    search_base = f"{self.output_dir}/{self.environment}/{self.version}/"
                else:
                    search_base = (
                        f"{self.scratch_dir}/{self.environment}/{self.version}/{stage_name}/"
                    )

                if self.scratch_dir.startswith("s3://"):
                    fs = fsspec.filesystem("s3")
                    search_base_no_scheme = search_base.replace("s3://", "")
                    try:
                        all_files = fs.glob(f"{search_base_no_scheme}**/*.icechunk")
                    except Exception:
                        continue

                    for path in all_files:
                        full_path = f"s3://{path}"
                        if gcm and f"/{gcm}" not in full_path:
                            continue
                        if variable and f"/{variable}/" not in full_path:
                            continue
                        artifacts.add(full_path)
                else:
                    search_path = Path(search_base)
                    if not search_path.exists():
                        continue

                    for path in search_path.rglob("*.icechunk"):
                        path_str = str(path)
                        if gcm and f"/{gcm}" not in path_str:
                            continue
                        if variable and f"/{variable}/" not in path_str:
                            continue
                        artifacts.add(path_str)

        except Exception as e:
            logger.error("Error listing artifacts: %s", e)

        return sorted(list(artifacts))
