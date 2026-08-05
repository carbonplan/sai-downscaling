"""
Artifact caching system for BCSD pipeline.

Manages icechunk-based cache storage with ancestry-based existence checks.
Supports three pipeline stages: obs_regridded, historical, and scenario,
plus intermediate artifacts when save_intermediate is enabled.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import icechunk
import zarr

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.config import _ROOT_MESSAGES, SCENARIO_TO_GROUP, _icechunk_storage_for_path

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
    config_variable : str or None, optional
        Variable whose ``VariableConfig`` governs this artifact's contents, or
        ``None`` when the artifact does not depend on ``VariableConfig`` (regridded
        observations) or when the governing config is not knowable in this process
        (an explicit sibling-variable lookup). Store paths do not encode
        ``VariableConfig``, so :meth:`ArtifactCache.exists` uses this to detect a
        cache hit that was computed under different bias-correction settings.
    """

    store_path: str
    group: str
    config_variable: str | None = None


class CacheCheckError(Exception):
    """Raised when an infrastructure error prevents verifying cache existence.

    Distinguishes real errors (S3 auth failure, permission denied, network error)
    from a simple cache miss. Callers can catch this to decide fallback behavior.
    """


class CacheConfigMismatchError(Exception):
    """Raised when a cached artifact was computed under a different VariableConfig.

    Store paths key on ``(gcm, obs_dataset, subset)`` and group paths on
    ``(stage, variable, member)``; neither encodes ``VariableConfig``. Without this
    check, a run using ``variable_overrides`` would silently treat an artifact built
    with different bias-correction settings as its own cache hit, and every
    downstream stage would consume it.
    """


class ArtifactCache:
    """
    Icechunk-based cache manager with ancestry-based existence checks.

    Cache locations are deterministic based on configuration parameters.
    Each artifact is identified by a ``StoreLocation`` that encodes both
    the icechunk repository path and the zarr group within it.
    """

    INTERMEDIATE_PREFIXES: tuple[str, ...] = (
        "detrended_scenario/",
        "trend_scenario/",
        "debiased_scenario/",
    )

    def __init__(
        self,
        scratch_dir: str = "s3://carbonplan-scratch/srm/cache/",
        environment: str = "qa",
        branch: str = "main",
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
        branch : str
            icechunk branch for output writes (e.g. ``"v2"``). Scratch intermediate
            artifacts always write to ``"main"`` regardless of this value.
        output_dir : str, optional
            Directory for final scenario outputs. If None, scenarios go to scratch.
        """
        self.scratch_dir = scratch_dir.rstrip("/")
        self.environment = environment
        self.branch = branch
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
            branch=options.branch,
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
            f"/{config.gcm}-{config.obs_dataset}-{subset_id}.icechunk"
        )

    @property
    def _output_store(self) -> str:
        config = self._require_config()
        subset_id = self._get_subset_id(config.subset_bounds)
        base = self.output_dir if self.output_dir else self.scratch_dir
        return f"{base}/{self.environment}/{config.gcm}-{config.obs_dataset}-{subset_id}.icechunk"

    # ── artifact location properties ─────────────────────────────────────────

    @property
    def obs_loc(self) -> StoreLocation:
        """StoreLocation for the obs-regridding artifact."""
        config = self._require_config()
        return StoreLocation(self._scratch_store, f"obs/{config.variable}")

    def historical_loc(self, hist_member: str, variable: str | None = None) -> StoreLocation:
        """StoreLocation for the fully downscaled historical artifact.

        Written to the output store — the fine-resolution downscaled historical
        is a deliverable, the analog of the fine scenario output.

        Parameters
        ----------
        hist_member : str
            Resolved historical ensemble member ID.
        variable : str, optional
            Override the variable from config. Used by tasmin to read/rewrite the
            sibling fine tasmax output during the tasmax<tasmin swap (issue #331).
        """
        config = self._require_config()
        var = variable or config.variable
        return StoreLocation(
            self._output_store,
            f"historical/{var}/{hist_member}",
            config_variable=None if variable is not None else config.variable,
        )

    def _scenario_group(self) -> str:
        """Return the icechunk group prefix for the bound config's scenario."""
        return SCENARIO_TO_GROUP[self._require_config().scenario]

    def scenario_output_loc(self, variable: str | None = None) -> StoreLocation:
        """StoreLocation for the fine scenario downscaling output.

        Parameters
        ----------
        variable : str, optional
            Override the variable from config. Used by tasmin to read/rewrite the
            sibling fine tasmax output during the tasmax<tasmin swap (issue #331).
        """
        config = self._require_config()
        var = variable or config.variable
        return StoreLocation(
            self._output_store,
            f"{self._scenario_group()}/{var}/{config.ensemble_member}",
            config_variable=None if variable is not None else config.variable,
        )

    @property
    def scenario_loc(self) -> StoreLocation:
        """StoreLocation for the scenario downscaling output."""
        return self.scenario_output_loc()

    def debiased_coarse_historical_loc(
        self, hist_member: str, variable: str | None = None
    ) -> StoreLocation:
        """StoreLocation for the debiased coarse historical artifact in the output store.

        Parameters
        ----------
        hist_member : str
            Resolved historical ensemble member ID.
        variable : str, optional
            Override the variable from config. Used by tasmin to read dtr/tasmax outputs.
        """
        config = self._require_config()
        var = variable or config.variable
        return StoreLocation(
            self._output_store,
            f"debiased_coarse/historical/{var}/{hist_member}",
            config_variable=None if variable is not None else config.variable,
        )

    def debiased_coarse_scenario_loc(self, variable: str | None = None) -> StoreLocation:
        """StoreLocation for the debiased coarse scenario artifact in the output store.

        Parameters
        ----------
        variable : str, optional
            Override the variable from config. Used by tasmin to read dtr/tasmax outputs.
        """
        config = self._require_config()
        var = variable or config.variable
        return StoreLocation(
            self._output_store,
            f"debiased_coarse/{self._scenario_group()}/{var}/{config.ensemble_member}",
            config_variable=None if variable is not None else config.variable,
        )

    def detrended_scenario_loc(self) -> StoreLocation:
        """StoreLocation for intermediate detrended-scenario artifact."""
        config = self._require_config()
        return StoreLocation(
            self._scratch_store,
            f"detrended_scenario/{self._scenario_group()}/{config.variable}/{config.ensemble_member}",
            config_variable=config.variable,
        )

    def trend_scenario_loc(self) -> StoreLocation:
        """StoreLocation for intermediate trend-scenario artifact."""
        config = self._require_config()
        return StoreLocation(
            self._scratch_store,
            f"trend_scenario/{self._scenario_group()}/{config.variable}/{config.ensemble_member}",
            config_variable=config.variable,
        )

    def debiased_scenario_loc(self) -> StoreLocation:
        """StoreLocation for intermediate debiased-scenario artifact."""
        config = self._require_config()
        return StoreLocation(
            self._scratch_store,
            f"debiased_scenario/{self._scenario_group()}/{config.variable}/{config.ensemble_member}",
            config_variable=config.variable,
        )

    # ── branch helpers ────────────────────────────────────────────────────────

    def _branch_for(self) -> str:
        """Return the icechunk branch for reads/writes.

        All artifacts (scratch and output) use ``self.branch``. ``main`` is kept
        as an empty anchor by ``_ensure_root_group``; every versioned branch
        forks from that clean snapshot so different branches never share ancestry.
        """
        return self.branch

    # ── existence check ───────────────────────────────────────────────────────

    def exists(self, loc: StoreLocation, *, max_attempts: int = 4) -> bool:
        """
        Check if an artifact exists by scanning the icechunk commit ancestry.

        A commit is present iff a snapshot with message equal to ``loc.group``
        exists on ``self.branch`` of the store. This is atomic — partial writes
        leave no matching commit.

        A store or branch that does not exist yet is a genuine cache miss and
        returns ``False``. Transient infrastructure errors — e.g. reading a
        branch ref while other workers concurrently commit to it — are retried
        with exponential backoff. A persistent failure raises
        :class:`CacheCheckError` rather than being silently reported as a miss,
        so a momentarily unreadable store never causes valid, already-committed
        outputs to be discarded and recomputed (or reported as failed).

        Parameters
        ----------
        loc : StoreLocation
            Location to check.
        max_attempts : int, optional
            Number of attempts before giving up on a transient error (default 4).

        Returns
        -------
        bool
            True if a commit with message ``loc.group`` is in the ancestry;
            False if the store or branch does not exist.

        Raises
        ------
        CacheCheckError
            If an existing store cannot be read after ``max_attempts`` attempts.
        CacheConfigMismatchError
            If the artifact exists but was computed under a different
            ``VariableConfig``. See :meth:`_verify_variable_config`.
        """
        branch = self._branch_for()
        storage = _icechunk_storage_for_path(loc.store_path)

        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                if not icechunk.Repository.exists(storage):
                    logger.debug("Cache miss (store does not exist): %s", loc.store_path)
                    return False
                repo = icechunk.Repository.open(storage)
                if branch not in repo.list_branches():
                    logger.debug("Cache miss (branch %r not created): %s", branch, loc.store_path)
                    return False
                hit = any(s.message == loc.group for s in repo.ancestry(branch=branch))
                logger.debug(
                    "Cache %s: %s / %s", "hit" if hit else "miss", loc.store_path, loc.group
                )
                if hit:
                    self._verify_variable_config(repo, branch, loc)
                return hit
            except CacheConfigMismatchError:
                # A real, decided answer about the artifact, not an infra failure.
                # Retrying would only repeat it and then mask it as a CacheCheckError.
                raise
            except Exception as err:  # transient infra error against an existing store
                last_error = err
                if attempt < max_attempts - 1:
                    logger.debug(
                        "Cache check error on %s / %s (attempt %d/%d), retrying: %s",
                        loc.store_path,
                        loc.group,
                        attempt + 1,
                        max_attempts,
                        err,
                    )
                    time.sleep(0.5 * 2**attempt)

        raise CacheCheckError(
            f"Could not verify {loc.store_path} / {loc.group} on branch {branch!r} "
            f"after {max_attempts} attempts"
        ) from last_error

    def _verify_variable_config(
        self, repo: icechunk.Repository, branch: str, loc: StoreLocation
    ) -> None:
        """Confirm a cache hit was computed under this run's ``VariableConfig``.

        Store and group paths encode ``(gcm, obs_dataset, subset)`` and
        ``(stage, variable, member)`` respectively, so two runs differing only in a
        ``variable_overrides`` entry resolve to the same location. Left unchecked,
        the second run reports a hit, skips the stage, and every downstream stage
        silently consumes artifacts built with different bias-correction settings.

        The comparison reads the ``srm_downscaling:config_json`` provenance attribute
        written by ``BCSDPipeline._build_attrs`` and compares only its nested
        ``variable_config``. Other config differences are out of scope here: they
        either already appear in the path or are legitimate (a wider predict period
        reusing a cached historical fit, for instance).

        Skipped, each deliberately, when:

        - no config is bound to this cache;
        - ``loc.config_variable`` is ``None``, meaning the artifact does not depend on
          ``VariableConfig`` (regridded obs) or is an explicit sibling-variable lookup
          whose intended config this process cannot know;
        - the attribute is absent, i.e. the artifact predates config provenance, or the
          group holds no readable zarr metadata. Unverifiable is not the same as
          mismatched, so this logs and allows the hit.

        Parameters
        ----------
        repo : icechunk.Repository
            Already-open repository for ``loc.store_path``.
        branch : str
            Branch the hit was found on.
        loc : StoreLocation
            Location of the artifact that matched.

        Raises
        ------
        CacheConfigMismatchError
            If the stored ``variable_config`` differs from this run's.
        """
        config = self.config
        if config is None or loc.config_variable is None:
            return
        if loc.config_variable != config.variable:
            return

        try:
            session = repo.readonly_session(branch=branch)
            attrs = dict(zarr.open_group(session.store, path=loc.group, mode="r").attrs)
            raw = attrs.get("srm_downscaling:config_json")
            stored = json.loads(raw)["variable_config"] if raw else None
        except CacheConfigMismatchError:
            raise
        except Exception as err:
            logger.debug("Could not read config provenance from %s: %s", loc.group, err)
            return

        if stored is None:
            logger.debug(
                "No config provenance on %s; cannot verify VariableConfig, allowing hit", loc.group
            )
            return

        current = config.variable_config.model_dump()
        differing = {k: (stored.get(k), v) for k, v in current.items() if stored.get(k) != v}
        if not differing:
            return

        detail = "; ".join(
            f"{k}: cached={was!r} current={now!r}" for k, (was, now) in differing.items()
        )
        raise CacheConfigMismatchError(
            f"{loc.store_path} / {loc.group} on branch {branch!r} was computed with a "
            f"different VariableConfig ({detail}). Store paths do not encode "
            "VariableConfig, so reusing this artifact would mix bias-correction "
            "settings. Run on a separate branch (--branch / BCSD_BRANCH), or force a "
            "recompute to overwrite it."
        )

    def list_groups_on_branch(self, store_path: str) -> list[str]:
        """Return all zarr group paths committed on the current branch of a store.

        Walks the icechunk ancestry and collects every snapshot message that
        looks like a zarr group path (contains a ``/``). Returns an empty list
        if the store does not exist or the branch has no user commits.

        Parameters
        ----------
        store_path : str
            S3 or local path to the icechunk repository.

        Returns
        -------
        list[str]
            Sorted list of group paths, e.g. ``["obs/tas", "obs/pr"]``.
        """
        branch = self._branch_for()
        try:
            storage = _icechunk_storage_for_path(store_path)
            repo = icechunk.Repository.open(storage)
            return sorted(
                s.message
                for s in repo.ancestry(branch=branch)
                if s.message not in _ROOT_MESSAGES and "/" in s.message
            )
        except Exception:
            return []

    def list_intermediate_groups(self) -> dict[str, list[str]]:
        """Return intermediate artifact groups on the current branch, keyed by store path.

        Scans both scratch and output stores and returns only groups whose
        paths start with one of ``INTERMEDIATE_PREFIXES``. Empty stores are
        omitted from the result.

        Returns
        -------
        dict[str, list[str]]
            Mapping of store path → sorted list of intermediate group paths.
        """
        result: dict[str, list[str]] = {}
        for store_path in {self._scratch_store, self._output_store}:
            groups = [
                g
                for g in self.list_groups_on_branch(store_path)
                if any(g.startswith(p) for p in self.INTERMEDIATE_PREFIXES)
            ]
            if groups:
                result[store_path] = groups
        return result

    def release(self, tag: str) -> None:
        """Freeze both scratch and output stores as a production release tag.

        Creates an icechunk tag on both stores using the same name so that
        ``readonly_session(tag=tag)`` on either store returns the exact state
        that produced this release.

        Parameters
        ----------
        tag : str
            Tag name (e.g. ``"v2"``). Must not already exist on either store.
        """
        import icechunk

        def _tag_store(store_path: str, branch: str) -> None:
            storage = _icechunk_storage_for_path(store_path)
            repo = icechunk.Repository.open(storage)
            snapshot_id = repo.lookup_branch(branch)
            repo.create_tag(tag, snapshot_id)
            logger.info("Tagged %s@%s as %r", store_path, branch, tag)

        _tag_store(self._output_store, self.branch)
        if self._scratch_store != self._output_store:
            _tag_store(self._scratch_store, self.branch)

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
            deps = {"obs_regridded": (self.exists(loc), loc)}
            if config.variable == "tasmin":
                member = hist_member or config.ensemble_member
                for name, var in (
                    ("debiased_coarse_dtr", "dtr"),
                    ("debiased_coarse_tasmax", "tasmax"),
                ):
                    dep_loc = self.debiased_coarse_historical_loc(member, variable=var)
                    deps[name] = (self.exists(dep_loc), dep_loc)
                # the tasmax<tasmin swap reads/rewrites the fine tasmax output
                fine_tasmax = self.historical_loc(member, variable="tasmax")
                deps["fine_tasmax"] = (self.exists(fine_tasmax), fine_tasmax)
            return deps

        elif stage == "transform_scenario":
            obs_loc = self.obs_loc
            hist_loc = self.historical_loc(hist_member or config.ensemble_member)
            deps = {
                "obs_regridded": (self.exists(obs_loc), obs_loc),
                "historical": (self.exists(hist_loc), hist_loc),
            }
            if config.variable == "tasmin":
                for name, var in (
                    ("debiased_coarse_dtr", "dtr"),
                    ("debiased_coarse_tasmax", "tasmax"),
                ):
                    dep_loc = self.debiased_coarse_scenario_loc(variable=var)
                    deps[name] = (self.exists(dep_loc), dep_loc)
                # the tasmax<tasmin swap reads/rewrites the fine tasmax output
                fine_tasmax = self.scenario_output_loc(variable="tasmax")
                deps["fine_tasmax"] = (self.exists(fine_tasmax), fine_tasmax)
            return deps

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
            search_base = f"{self.scratch_dir}/{self.environment}/{stage}/"
        else:
            search_base = f"{self.scratch_dir}/{self.environment}/"

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
                    search_base = f"{self.output_dir}/{self.environment}/"
                else:
                    search_base = f"{self.scratch_dir}/{self.environment}/{stage_name}/"

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
