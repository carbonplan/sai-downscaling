"""
Orchestration layer for batch BCSD execution with automatic task deduplication.

Manages efficient batch execution of BCSD runs across multiple configurations using
Coiled's batch API. Automatically detects cached artifacts, deduplicates shared stages
across ensemble members and scenarios, and submits only the necessary tasks.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Literal

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.cache import ArtifactCache, StoreLocation
from srm.pipeline import BCSDPipeline

logger = logging.getLogger(__name__)


class BCSDOrchestrator:
    """
    Manages batch submission of BCSD tasks with dependency awareness.

    This class handles intelligent task scheduling by:
    - Detecting cached artifacts and skipping unnecessary computations
    - Deduplicating tasks across stages (e.g., obs regridding only once per GCM/var)
    - Submitting tasks to Coiled using the batch API (coiled.batch.run)
    - Managing stage-by-stage execution of the pipeline

    The batch API submits Python commands that run on isolated VMs, with each
    task receiving its configuration via environment variables.

    Example
    -------
    >>> configs = [
    ...     BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member=0, scenario="ssp245", ...),
    ...     BCSDConfig(gcm="CESM2-WACCM", variable="tas", ensemble_member=1, scenario="ssp245", ...),
    ... ]
    >>> orchestrator = BCSDOrchestrator()
    >>> output_paths = orchestrator.run_full_workflow(configs, use_coiled=True)
    """

    # VM sizes per pipeline stage for a global run. fit_historical is memory-intensive
    # (QM fitting) and transform_scenario carries the spatial disaggregation rechunk, so
    # both run on larger instances than the obs regrid.
    _STAGE_VM_TYPES: dict[str, list[str]] = {
        "prepare_observations": ["r8g.4xlarge"],
        "fit_historical": ["r8g.12xlarge"],
        "transform_scenario": ["r8g.24xlarge"],
    }

    # VM sizes for a spatially subset run. Every stage applies subset_space before any
    # heavy compute, so the box size, not the source grid, sets the working set: the
    # South Africa QA box is roughly 150x fewer fine cells than global, under 1 GB per
    # variable-member. Sizing those runs off the global table left transform_scenario
    # holding 768 GiB of RAM for well under a gigabyte of data.
    _REGIONAL_STAGE_VM_TYPES: dict[str, list[str]] = {
        "prepare_observations": ["r8g.2xlarge"],
        "fit_historical": ["r8g.2xlarge"],
        "transform_scenario": ["r8g.4xlarge"],
    }

    _DEFAULT_VM_TYPE: list[str] = ["c8g.12xlarge"]

    @staticmethod
    def _is_regional(configs: list[BCSDConfig]) -> bool:
        """
        Report whether every config in a batch is spatially subset.

        A batch that mixes global and subset configs is treated as global, because one
        instance type covers the whole batch and it has to hold the largest task in it.

        Parameters
        ----------
        configs : list[BCSDConfig]
            Configurations making up a single batch submission.

        Returns
        -------
        bool
            True when the batch is non-empty and every config sets ``subset_bounds``.
        """
        return bool(configs) and all(config.subset_bounds is not None for config in configs)

    @classmethod
    def _vm_types_for(cls, stage: str, configs: list[BCSDConfig]) -> list[str]:
        """
        Select the instance types for a stage, scaled to the batch's spatial extent.

        Parameters
        ----------
        stage : str
            Pipeline stage name.
        configs : list[BCSDConfig]
            Configurations making up a single batch submission.

        Returns
        -------
        list[str]
            Candidate instance types to pass to ``coiled.batch.run``.
        """
        table = cls._REGIONAL_STAGE_VM_TYPES if cls._is_regional(configs) else cls._STAGE_VM_TYPES
        return table.get(stage, cls._DEFAULT_VM_TYPE)

    # AWS Batch takes resource requirements rather than instance types and picks the
    # instance itself. Each entry is sized to fill one instance of the class the Coiled
    # path requested, leaving ECS headroom, because every task uses all of its VM's cores
    # through threaded Dask and must not share a host.
    _STAGE_RESOURCES: dict[str, dict[str, int]] = {
        "prepare_observations": {"vcpu": 16, "memory_mib": 122880},  # r8g.4xlarge
        "fit_historical": {"vcpu": 48, "memory_mib": 368640},  # r8g.12xlarge
        "transform_scenario": {"vcpu": 96, "memory_mib": 737280},  # r8g.24xlarge
    }

    _REGIONAL_STAGE_RESOURCES: dict[str, dict[str, int]] = {
        "prepare_observations": {"vcpu": 8, "memory_mib": 61440},  # r8g.2xlarge
        "fit_historical": {"vcpu": 8, "memory_mib": 61440},  # r8g.2xlarge
        "transform_scenario": {"vcpu": 16, "memory_mib": 122880},  # r8g.4xlarge
    }

    @classmethod
    def _resources_for(cls, stage: str, configs: list[BCSDConfig]) -> dict[str, int]:
        """
        Select vCPU and memory for a stage, scaled to the batch's spatial extent.

        Parameters
        ----------
        stage : str
            Pipeline stage name.
        configs : list[BCSDConfig]
            Configurations making up a single batch submission.

        Returns
        -------
        dict[str, int]
            Mapping with ``vcpu`` and ``memory_mib`` keys.
        """
        table = cls._REGIONAL_STAGE_RESOURCES if cls._is_regional(configs) else cls._STAGE_RESOURCES
        return table[stage]

    def _config_payload_json(self, config: BCSDConfig) -> str:
        """Serialize one task's ``CONFIG_JSON`` payload.

        Computed fields (``run_id``, ``config_hash``, ``is_sai_scenario``) are excluded
        because ``BCSDConfig`` does not accept them as constructor inputs.
        """
        computed_fields = set(BCSDConfig.model_computed_fields.keys())
        return json.dumps(
            {
                **config.model_dump(exclude=computed_fields),
                "options": self.options.model_dump(),
            }
        )

    def _job_name(self, stage: str, configs: list[BCSDConfig]) -> str:
        """Build the deterministic job name shared by both remote executors."""
        gcms = "-".join(sorted({c.gcm for c in configs}))
        variables = "-".join(sorted({c.variable for c in configs}))
        batch_hash = hashlib.sha256(
            "".join(sorted(c.config_hash for c in configs)).encode()
        ).hexdigest()[:8]
        return f"bcsd-{stage}-{gcms}-{variables}-{batch_hash}"

    def _submit_batch_job(
        self, stage: str, configs: list[BCSDConfig], manifest_uri: str | None
    ) -> str:
        """
        Submit one AWS Batch job covering ``configs`` and return its job ID.

        A wave of two or more tasks becomes an array job whose children index into
        ``manifest_uri``. A wave of exactly one task becomes a plain job carrying
        ``CONFIG_JSON`` directly, because ``arrayProperties.size`` must be at least 2.

        Parameters
        ----------
        stage : str
            Pipeline stage name.
        configs : list[BCSDConfig]
            Configurations to run, in array-index order.
        manifest_uri : str or None
            Manifest location. Required when ``configs`` holds more than one entry.

        Returns
        -------
        str
            AWS Batch job ID.
        """
        import boto3

        client = boto3.client("batch", region_name=self.options.batch_region)
        resources = self._resources_for(stage, configs)
        environment = [{"name": "SRM_STAGE", "value": stage}]

        if len(configs) == 1:
            environment.append(
                {"name": "CONFIG_JSON", "value": self._config_payload_json(configs[0])}
            )
            array_kwargs: dict = {}
        else:
            if manifest_uri is None:
                raise ValueError("manifest_uri is required for a multi-task AWS Batch submission")
            environment.append({"name": "CONFIG_MANIFEST_URI", "value": manifest_uri})
            array_kwargs = {"arrayProperties": {"size": len(configs)}}

        response = client.submit_job(
            jobName=self._job_name(stage, configs),
            jobQueue=self.options.batch_job_queue,
            jobDefinition=self.options.batch_job_definition,
            containerOverrides={
                "command": ["uv", "run", "--no-sync", "python", "-m", "srm.batch_runner", stage],
                "environment": environment,
                "resourceRequirements": [
                    {"type": "VCPU", "value": str(resources["vcpu"])},
                    {"type": "MEMORY", "value": str(resources["memory_mib"])},
                ],
            },
            retryStrategy={"attempts": 3},
            tags={"Project": "SRM"},
            propagateTags=True,
            **array_kwargs,
        )
        logger.info(
            f"→ {stage}: submitted AWS Batch job {response['jobId']} "
            f"({len(configs)} task(s), {resources['vcpu']} vCPU, "
            f"extent={'regional' if self._is_regional(configs) else 'global'})"
        )
        return response["jobId"]

    # Every batch runs on-demand regardless of spatial extent. Regional runs were briefly
    # placed on spot to cut cost, but reclamation was frequent enough that jobs failed
    # outright rather than being absorbed by the retry loop in `_submit_to_coiled`, and a
    # regional batch is cheap enough that the savings do not pay for the lost runs.
    _SPOT_POLICY: str = "on-demand"

    def __init__(self, options: PipelineOptions):
        """
        Initialize orchestrator.

        Parameters
        ----------
        options : PipelineOptions
            Operational settings containing storage paths and runtime flags.
        """
        self.options = options
        self._cache: ArtifactCache | None = None

    def _get_cache(self) -> ArtifactCache:
        """Get or create cache instance from options."""
        if self._cache is None:
            self._cache = ArtifactCache(
                scratch_dir=self.options.scratch_dir,
                environment=self.options.environment,
                branch=self.options.branch,
                output_dir=self.options.output_dir,
            )
        return self._cache

    def _stage_loc(
        self,
        cache: ArtifactCache,
        stage: str,
        config: BCSDConfig,
        hist_member: str | None = None,
    ) -> StoreLocation:
        """Return the StoreLocation for a stage/config, binding config to cache.

        The stage-to-artifact mapping itself lives in
        :meth:`~srm.cache.ArtifactCache.stage_loc`, shared with the pipeline's write path
        and the downstream dependency gate. Coarse-only variables (issue #461) therefore
        skip, retry, and report against their ``debiased_coarse`` group here without this
        method needing to know about them.
        """
        cache.config = config
        return cache.stage_loc(stage, config, hist_member=hist_member)

    # Stages where tasmin reconstructs itself from its sibling tasmax/dtr stores
    # and therefore must run after them. Other stages (obs regridding) have no
    # cross-variable dependency and are never wave-split.
    _DERIVED_VARIABLE_STAGES: frozenset[str] = frozenset({"fit_historical", "transform_scenario"})

    def _dependency_waves(self, stage: str, configs: list[BCSDConfig]) -> list[list[int]]:
        """Split config indices into ordered execution waves within a stage.

        ``tasmin`` is never bias-corrected directly; it is reconstructed from its
        sibling ``tasmax``/``dtr`` stores (issue #363). Those siblings are written
        by separate tasks on the same icechunk branch, so in the stages that read
        them (``fit_historical``, ``transform_scenario``) ``tasmin`` must not start
        until they are committed and final — otherwise it fossilises a mid-flight
        (still-NaN) ``dtr``/``tasmax``. Stages without that coupling run as one wave.

        Parameters
        ----------
        stage : str
            Pipeline stage being submitted.
        configs : list[BCSDConfig]
            Configs to be executed in this stage.

        Returns
        -------
        list[list[int]]
            Ordered list of waves, each a list of indices into ``configs``. Empty
            waves are omitted.
        """
        if stage not in self._DERIVED_VARIABLE_STAGES:
            return [list(range(len(configs)))] if configs else []
        first = [i for i, c in enumerate(configs) if c.variable != "tasmin"]
        later = [i for i, c in enumerate(configs) if c.variable == "tasmin"]
        return [wave for wave in (first, later) if wave]

    def _run_in_dependency_waves(
        self,
        executor: Callable[[str, list[BCSDConfig]], list[str]],
        stage: str,
        configs: list[BCSDConfig],
    ) -> list[str]:
        """Run ``configs`` through ``executor`` one dependency wave at a time.

        Waves are executed sequentially (each blocks to completion before the next
        starts), while output paths are returned in the original ``configs`` order.
        """
        completed: list[str | None] = [None] * len(configs)
        for wave in self._dependency_waves(stage, configs):
            wave_configs = [configs[i] for i in wave]
            wave_paths = executor(stage, wave_configs)
            for i, path in zip(wave, wave_paths):
                completed[i] = path
        return completed  # type: ignore[return-value]

    def submit_stage(
        self,
        stage: Literal["prepare_observations", "fit_historical", "transform_scenario"],
        configs: list[BCSDConfig],
        force: bool = False,
        use_coiled: bool = True,
    ) -> list[str]:
        """
        Submit tasks for a specific stage, skipping cached outputs.

        Parameters
        ----------
        stage : {'prepare_observations', 'fit_historical', 'transform_scenario'}
            Pipeline stage to execute
        configs : list[BCSDConfig]
            List of configurations to process
        force : bool, optional
            Force recomputation even if cached
        use_coiled : bool, optional
            Use Coiled for distributed execution

        Returns
        -------
        list[str]
            List of output paths (either from cache or newly computed)
        """
        if not configs:
            return []

        cache = self._get_cache()

        # Filter out configs that are already cached
        configs_to_run = []
        output_paths = []

        for config in configs:
            hist_member = self._resolve_hist_member(config) if stage == "fit_historical" else None
            loc = self._stage_loc(cache, stage, config, hist_member=hist_member)

            if cache.exists(loc) and not force:
                branch = cache._branch_for()
                logger.info(
                    f"⊙ Skipping {config.run_id} - {loc.group} already cached (branch: {branch})"
                )
                output_paths.append(f"{loc.store_path}::{loc.group}")
            else:
                configs_to_run.append(config)
                output_paths.append(None)  # Placeholder

        if not configs_to_run:
            logger.info(f"✓ All {len(configs)} {stage} tasks already cached!")
            return output_paths

        logger.info(
            f"→ Submitting {len(configs_to_run)}/{len(configs)} {stage} tasks "
            f"({'Coiled' if use_coiled else 'local'})"
        )

        # Submit to Coiled or run locally, respecting intra-stage dependency
        # ordering (tasmin must run after its debiased-coarse tasmax/dtr inputs).
        executor = self._submit_to_coiled if use_coiled else self._run_local
        completed_paths = self._run_in_dependency_waves(executor, stage, configs_to_run)

        # Fill in the output_paths list
        completed_idx = 0
        for i, path in enumerate(output_paths):
            if path is None:
                output_paths[i] = completed_paths[completed_idx]
                completed_idx += 1

        logger.info(f"✓ Completed {len(configs_to_run)} {stage} tasks")

        return output_paths

    def _submit_to_coiled(
        self, stage: str, configs: list[BCSDConfig], max_retries: int = 3
    ) -> list[str]:
        """
        Submit tasks to Coiled using batch API.

        Uses coiled.batch.run() to submit shell commands that execute the
        srm.batch_runner module. Each task receives its configuration as a
        JSON-serialized environment variable (CONFIG_JSON).

        The batch job consists of N tasks (one per config) running in parallel. Instance
        type comes from ``_vm_types_for``, which sizes the batch by stage and by whether it
        is global or spatially subset; every batch is purchased on-demand. Tasks write
        their outputs to the cache, which are then verified and collected.

        Parameters
        ----------
        stage : str
            Pipeline stage ('prepare_observations', 'fit_historical', 'transform_scenario')
        configs : list[BCSDConfig]
            Configurations to process

        Returns
        -------
        list[str]
            Output paths from completed tasks

        Raises
        ------
        RuntimeError
            If batch job fails or outputs not found in cache
        """
        try:
            import coiled
        except ImportError:
            raise ImportError("Coiled is not installed. Install with: uv pip install coiled")

        cache = self._get_cache()
        command = ["python", "-m", "srm.batch_runner", stage]

        # Sized from the full config list rather than `remaining` so a retry batch that
        # happens to be all-regional cannot silently shrink a global run mid-flight.
        vm_type = self._vm_types_for(stage, configs)
        logger.info(
            f"→ {stage}: {len(configs)} tasks on {vm_type[0]} ({self._SPOT_POLICY}), "
            f"extent={'regional' if self._is_regional(configs) else 'global'}"
        )

        remaining = list(configs)
        attempt = 0

        while remaining and attempt < max_retries:
            attempt += 1
            if attempt > 1:
                logger.warning(
                    f"→ Retry {attempt}/{max_retries} for {len(remaining)} failed {stage} tasks"
                )

            task_var_dicts = [
                {"CONFIG_JSON": self._config_payload_json(config)} for config in remaining
            ]
            job_name = self._job_name(stage, remaining)

            job_result = coiled.batch.run(
                command=command,
                name=job_name,
                vm_type=vm_type,
                scheduler_vm_type=vm_type,
                region="us-west-2",
                map_over_task_var_dicts=task_var_dicts,
                forward_aws_credentials=False,
                spot_policy=self._SPOT_POLICY,
                logger=logger,
                tag={"Project": "SRM"},
                disk_size="100GB",
            )

            job_id = job_result["job_id"]
            logger.info(
                f"→ Submitted Coiled batch job {job_id} with {len(remaining)} tasks "
                f"(attempt {attempt}/{max_retries})"
            )

            final_state = coiled.batch.wait_for_job_done(job_id)
            logger.info(f"Batch job {job_id} finished with state: {final_state}")

            # Partition into succeeded / still-failed based on cache presence
            still_failed = [
                config
                for config in remaining
                if not cache.exists(
                    self._stage_loc(
                        cache,
                        stage,
                        config,
                        hist_member=self._resolve_hist_member(config)
                        if stage == "fit_historical"
                        else None,
                    )
                )
            ]

            if not still_failed:
                logger.info(f"✓ Batch job {job_id} completed successfully")
                remaining = []
                break

            logger.warning(
                f"✗ {len(still_failed)}/{len(remaining)} tasks still missing after job {job_id} "
                f"(state={final_state!r}). "
                f"Failed run_ids: {[c.run_id for c in still_failed]}"
            )
            remaining = still_failed

        if remaining:
            raise RuntimeError(
                f"Stage '{stage}' failed after {max_retries} attempt(s). "
                f"{len(remaining)} task(s) did not produce output: "
                f"{[c.run_id for c in remaining]}"
            )

        logger.info(f"✓ All {len(configs)} {stage} tasks completed")

        # Collect and return all output paths (now guaranteed to exist)
        result = []
        for config in configs:
            loc = self._stage_loc(
                cache,
                stage,
                config,
                hist_member=self._resolve_hist_member(config)
                if stage == "fit_historical"
                else None,
            )
            result.append(f"{loc.store_path}::{loc.group}")
        return result

    def _run_local(self, stage: str, configs: list[BCSDConfig]) -> list[str]:
        """
        Run tasks locally (sequential execution).

        Parameters
        ----------
        stage : str
            Pipeline stage
        configs : list[BCSDConfig]
            Configurations to process

        Returns
        -------
        list[str]
            Output paths from completed tasks
        """
        completed_paths = []
        cache = self._get_cache()

        for config in configs:
            pipeline = BCSDPipeline(config, self.options)

            if stage == "prepare_observations":
                pipeline.prepare_observations()
            elif stage == "fit_historical":
                pipeline.fit_historical()
            elif stage == "transform_scenario":
                pipeline.transform_scenario()
            else:
                raise ValueError(f"Unknown stage: {stage}")

            hist_member = self._resolve_hist_member(config) if stage == "fit_historical" else None
            loc = self._stage_loc(cache, stage, config, hist_member=hist_member)
            completed_paths.append(f"{loc.store_path}::{loc.group}")

        return completed_paths

    def run_full_workflow(
        self,
        configs: list[BCSDConfig],
        force: bool = False,
        use_coiled: bool = True,
    ) -> dict[str, list[str]]:
        """
        Run all three stages in sequence with automatic dependency management.

        This method intelligently deduplicates tasks across stages:
        - Stage 1 (obs): Only unique (GCM, variable) combinations
        - Stage 2 (historical): Only unique (GCM, variable, ensemble) combinations
        - Stage 3 (scenario): All configs

        Parameters
        ----------
        configs : list[BCSDConfig]
            List of configurations to process
        force : bool, optional
            Force recomputation of all stages
        use_coiled : bool, optional
            Use Coiled for distributed execution

        Returns
        -------
        dict[str, list[str]]
            Mapping of stage name to output paths:
            ``{'prepare_observations': [...], 'fit_historical': [...], 'transform_scenario': [...]}``
        """
        logger.info(f"╔═══ Starting BCSD workflow for {len(configs)} configurations")

        # Stage 1: Unique obs regridding tasks
        obs_configs = self._deduplicate_obs_configs(configs)
        logger.info(f"║ Stage 1: prepare_observations ({len(obs_configs)} unique tasks)")
        obs_paths = self.submit_stage(
            "prepare_observations", obs_configs, force=force, use_coiled=use_coiled
        )

        # Stage 2: Unique historical tasks
        hist_configs = self._deduplicate_historical_configs(configs)
        logger.info(f"║ Stage 2: fit_historical ({len(hist_configs)} unique tasks)")
        hist_paths = self.submit_stage(
            "fit_historical", hist_configs, force=force, use_coiled=use_coiled
        )

        # Stage 3: All scenario tasks
        logger.info(f"║ Stage 3: transform_scenario ({len(configs)} tasks)")
        scenario_paths = self.submit_stage(
            "transform_scenario", configs, force=force, use_coiled=use_coiled
        )

        logger.info("╚═══ Workflow complete! ✓")

        return {
            "prepare_observations": obs_paths,
            "fit_historical": hist_paths,
            "transform_scenario": scenario_paths,
        }

    def _deduplicate_obs_configs(self, configs: list[BCSDConfig]) -> list[BCSDConfig]:
        """
        Extract unique (GCM, obs_dataset, variable) combinations for obs regridding.

        obs_dataset is part of the key because the obs artifact store path embeds
        it; two configs differing only in obs_dataset must each regrid.

        Parameters
        ----------
        configs : list[BCSDConfig]
            All configurations

        Returns
        -------
        list[BCSDConfig]
            Deduplicated configs for obs stage
        """
        seen = set()
        unique = []
        for config in configs:
            key = (config.gcm, config.obs_dataset, config.variable)
            if key not in seen:
                seen.add(key)
                unique.append(config)
        return unique

    def _deduplicate_historical_configs(self, configs: list[BCSDConfig]) -> list[BCSDConfig]:
        """
        Extract unique (GCM, variable, ensemble) combinations for historical downscaling.

        Deduplication uses the resolved historical ensemble member so that multiple
        scenario configs that share the same lineage parent are not submitted as
        separate historical tasks. obs_dataset is part of the key because the
        historical artifact store path embeds it and fit_historical bias-corrects
        against obs; different obs_datasets require separate historical fits.
        downscaling_method is part of the key because fit_historical writes its
        output under a leading method segment (``{method}/historical/...`` and
        ``{method}/debiased_coarse/historical/...``); two configs differing only in
        method write different artifacts, so each one has to run.

        Parameters
        ----------
        configs : list[BCSDConfig]
            All configurations

        Returns
        -------
        list[BCSDConfig]
            Deduplicated configs for historical stage
        """
        seen = set()
        unique = []
        for config in configs:
            key = (
                config.gcm,
                config.obs_dataset,
                config.variable,
                config.downscaling_method,
                self._resolve_hist_member(config),
            )
            if key not in seen:
                seen.add(key)
                unique.append(config)
        return unique

    @staticmethod
    def _resolve_hist_member(config: BCSDConfig) -> str:
        """
        Return the resolved historical ensemble member for a config.

        Mirrors the lineage resolution in ``BCSDPipeline.__init__``: for SAI/SSP245
        scenarios the raw ``ensemble_member`` may map to a different historical parent
        member.  Falls back to ``config.ensemble_member`` when no lineage entry exists.

        Parameters
        ----------
        config : BCSDConfig
            Run configuration

        Returns
        -------
        str
            Resolved historical member (e.g. ``"r1i1p1f1"`` for CESM2-WACCM ``"001"``)
        """
        if config.scenario is None:
            return config.ensemble_member
        try:
            from srm.lineage import resolve_member_lineage

            return resolve_member_lineage(
                config.gcm, config.scenario, config.ensemble_member, config.variable
            ).historical
        except KeyError:
            return config.ensemble_member

    def get_status(self, configs: list[BCSDConfig]) -> dict[str, dict]:
        """
        Get cache status for all configs.

        Parameters
        ----------
        configs : list[BCSDConfig]
            Configurations to check

        Returns
        -------
        dict
            Status dictionary with completion info per stage
        """
        status = {
            "prepare_observations": {"total": 0, "cached": 0, "missing": []},
            "fit_historical": {"total": 0, "cached": 0, "missing": []},
            "transform_scenario": {"total": 0, "cached": 0, "missing": []},
        }

        if not configs:
            return status

        cache = self._get_cache()

        # Check obs (deduplicated)
        obs_configs = self._deduplicate_obs_configs(configs)
        status["prepare_observations"]["total"] = len(obs_configs)
        for config in obs_configs:
            loc = self._stage_loc(cache, "prepare_observations", config)
            if cache.exists(loc):
                status["prepare_observations"]["cached"] += 1
            else:
                status["prepare_observations"]["missing"].append(config.run_id)

        # Check historical (deduplicated)
        hist_configs = self._deduplicate_historical_configs(configs)
        status["fit_historical"]["total"] = len(hist_configs)
        for config in hist_configs:
            loc = self._stage_loc(
                cache, "fit_historical", config, hist_member=self._resolve_hist_member(config)
            )
            if cache.exists(loc):
                status["fit_historical"]["cached"] += 1
            else:
                status["fit_historical"]["missing"].append(config.run_id)

        # Check scenarios (all configs)
        status["transform_scenario"]["total"] = len(configs)
        for config in configs:
            if config.scenario:
                loc = self._stage_loc(cache, "transform_scenario", config)
                if cache.exists(loc):
                    status["transform_scenario"]["cached"] += 1
                else:
                    status["transform_scenario"]["missing"].append(config.run_id)

        return status
