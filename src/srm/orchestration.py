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
from dataclasses import dataclass
from functools import partial
from typing import Literal

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.cache import ArtifactCache, StoreLocation
from srm.cost import memory_mib, vcpus
from srm.pipeline import BCSDPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StagePlan:
    """What one stage of a run will submit, before anything is dispatched."""

    stage: str
    to_run: int
    cached: int
    vm_type: str
    regional: bool


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
    >>> output_paths = orchestrator.run_full_workflow(configs, executor="aws-batch")
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
        # AWS Batch takes resource requirements rather than an instance type and picks the
        # instance itself, so this asks for enough to fill the instance the Coiled path
        # would have requested. Derived from that one table rather than a parallel one:
        # every task uses all of its VM's cores through threaded Dask and must not share a
        # host, so the two must agree by construction.
        #
        # Indexed rather than defaulted, unlike _vm_types_for: an unsized stage should fail
        # at submission rather than run a long job on a guessed instance.
        table = cls._REGIONAL_STAGE_VM_TYPES if cls._is_regional(configs) else cls._STAGE_VM_TYPES
        vm_type = table[stage][0]
        return {"vcpu": vcpus(vm_type), "memory_mib": memory_mib(vm_type)}

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

    #: AWS Batch rejects a jobName longer than this.
    _MAX_JOB_NAME = 128

    #: Page cap when listing job definition revisions; each page holds up to 100.
    _MAX_JOB_DEFINITION_PAGES = 100

    def _job_name(self, stage: str, configs: list[BCSDConfig]) -> str:
        """Build the deterministic job name shared by both remote executors.

        The descriptive middle grows with the batch's GCMs and variables, so a wide
        enough wave would overrun the AWS Batch limit. Trimming keeps the hash, which
        is what makes the name unique, and only production waves are ever near it.
        """
        gcms = "-".join(sorted({c.gcm for c in configs}))
        variables = "-".join(sorted({c.variable for c in configs}))
        batch_hash = hashlib.sha256(
            "".join(sorted(c.config_hash for c in configs)).encode()
        ).hexdigest()[:8]
        name = f"bcsd-{stage}-{gcms}-{variables}-{batch_hash}"
        if len(name) > self._MAX_JOB_NAME:
            head = name[: self._MAX_JOB_NAME - len(batch_hash) - 1].rstrip("-")
            name = f"{head}-{batch_hash}"
        return name

    def resolve_job_definition(self) -> dict[str, str]:
        """
        Resolve the configured job definition to the revision and image AWS will run.

        A bare name resolves to the highest ACTIVE revision, which is what AWS Batch itself
        picks at submit time. Reporting it lets a cost preview name the code a run will
        execute, rather than only the definition it was asked for.

        Returns
        -------
        dict[str, str]
            ``job_definition`` as ``name:revision`` and the container ``image``.

        Raises
        ------
        ValueError
            If the job definition has no ACTIVE revision.
        """
        if self._resolved_job_definition is not None:
            return self._resolved_job_definition

        name = self.options.batch_job_definition
        client = self._batch_client()
        if ":" in name or name.startswith("arn:"):
            definitions = client.describe_job_definitions(jobDefinitions=[name])["jobDefinitions"]
        else:
            definitions = []
            kwargs: dict = {"jobDefinitionName": name, "status": "ACTIVE"}
            for _ in range(self._MAX_JOB_DEFINITION_PAGES):
                response = client.describe_job_definitions(**kwargs)
                definitions.extend(response["jobDefinitions"])
                token = response.get("nextToken")
                if not token:
                    break
                kwargs["nextToken"] = token
            else:
                # Bounded rather than `while True`: the loop is driven by a token the
                # server supplies, and stopping quietly would pin whichever revision
                # happened to be newest in the pages read so far.
                raise RuntimeError(
                    f"Job definition {name!r} paged past "
                    f"{self._MAX_JOB_DEFINITION_PAGES * 100} revisions without ending"
                )

        if not definitions:
            raise ValueError(f"No ACTIVE job definition found for {name!r}")

        latest = max(definitions, key=lambda d: d["revision"])
        self._resolved_job_definition = {
            # AWS reports the bare name back, so take it from the response rather than
            # trimming the configured value: splitting an ARN on ':' yields "arn".
            "job_definition": f"{latest['jobDefinitionName']}:{latest['revision']}",
            "image": latest["containerProperties"]["image"],
        }
        return self._resolved_job_definition

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
        client = self._batch_client()
        resources = self._resources_for(stage, configs)

        # Submit against a pinned revision rather than the bare name. AWS resolves a bare
        # name at submit time, so a revision registered after the cost preview would run
        # code the operator never saw. Falling back leaves today's behavior intact when the
        # definition cannot be described.
        try:
            job_definition = self.resolve_job_definition()["job_definition"]
        except Exception as exc:  # noqa: BLE001 - reported, and the fallback is the old path
            job_definition = self.options.batch_job_definition
            logger.warning(
                f"Could not resolve job definition {job_definition!r} to a revision "
                f"({type(exc).__name__}); AWS will resolve it at submit time."
            )
        # dask.system.CPU_COUNT is wrong here and, worse, unstable: ECS on EC2 sets a
        # cgroup share rather than a quota, so a container allocated 16 vCPU measures 96
        # when packed onto an r8g.24xlarge and 16 when it gets a box to itself. That
        # sizes ibicus's process pool from queue depth. Send the allocation instead.
        environment: list[dict[str, str]] = [
            {"name": "SRM_NR_PROCESSES", "value": str(resources["vcpu"])}
        ]

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
            jobDefinition=job_definition,
            containerOverrides={
                # Batch overrides CMD only; containerProperties and containerOverrides
                # have no entryPoint field at all. The image's ENTRYPOINT is therefore
                # just `uv run --no-sync`, and every job names the command it wants, so
                # one image serves both the stages and the deploy workflow's validate
                # steps. Changing either side alone breaks every submission.
                "command": ["python", "-m", "srm.batch_runner", stage],
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
        self._batch_client_instance = None
        self._resolved_job_definition: dict[str, str] | None = None

    def _batch_client(self):
        """Get or create the AWS Batch client, shared by submission and polling."""
        if self._batch_client_instance is None:
            import boto3

            self._batch_client_instance = boto3.client(
                "batch", region_name=self.options.batch_region
            )
        return self._batch_client_instance

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

    def _stage_loc_for(self, cache: ArtifactCache, stage: str, config: BCSDConfig) -> StoreLocation:
        """Resolve a config's artifact location, supplying the historical member when due."""
        hist_member = self._resolve_hist_member(config) if stage == "fit_historical" else None
        return self._stage_loc(cache, stage, config, hist_member=hist_member)

    def _preexisting_artifacts(
        self, cache: ArtifactCache, stage: str, configs: list[BCSDConfig]
    ) -> list[str]:
        """Report which configs already have an artifact before this run submits anything.

        Both executors decide success by cache presence, which is proof only for artifacts
        the run itself created. Normally every one qualifies, because ``submit_stage`` only
        queues uncached configs; under ``force`` it queues everything, and then a post-run
        sweep would accept the previous run's output as evidence that a failed recompute
        succeeded. Whatever this returns needs the job's own status to corroborate it.

        Returns
        -------
        list[str]
            ``run_id`` of every config whose artifact predates the submission.
        """
        return [
            config.run_id
            for config in configs
            if cache.exists(self._stage_loc_for(cache, stage, config))
        ]

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

    def _partition_by_cache(
        self,
        cache: ArtifactCache,
        stage: str,
        configs: list[BCSDConfig],
        force: bool = False,
    ) -> tuple[list[BCSDConfig], list[str | None]]:
        """Split configs into those still needing a run and the paths of those cached.

        Shared by :meth:`submit_stage` and :meth:`plan` so a cost preview cannot disagree
        with what the run then submits.

        Returns
        -------
        tuple
            ``(configs_to_run, output_paths)``, where ``output_paths`` is positional over
            ``configs`` and holds ``None`` wherever the artifact still has to be produced.
        """
        configs_to_run: list[BCSDConfig] = []
        output_paths: list[str | None] = []
        for config in configs:
            loc = self._stage_loc_for(cache, stage, config)
            if cache.exists(loc) and not force:
                output_paths.append(f"{loc.store_path}::{loc.group}")
            else:
                configs_to_run.append(config)
                output_paths.append(None)
        return configs_to_run, output_paths

    def plan_stage(self, stage: str, configs: list[BCSDConfig], force: bool = False) -> StagePlan:
        """
        Report what one stage would submit for ``configs``, without submitting anything.

        Mirrors :meth:`submit_stage`, which takes the caller's list as given. That differs
        from :meth:`plan`, which deduplicates the obs and historical stages the way
        :meth:`run_full_workflow` does, so a single-stage estimate must not reuse it.

        Parameters
        ----------
        stage : str
            Pipeline stage name.
        configs : list[BCSDConfig]
            Configurations exactly as they would be handed to :meth:`submit_stage`.
        force : bool, optional
            Ignore cached artifacts.

        Returns
        -------
        StagePlan
            Counts and sizing for that one stage.
        """
        cache = self._get_cache()
        to_run, _ = self._partition_by_cache(cache, stage, configs, force)
        return StagePlan(
            stage=stage,
            to_run=len(to_run),
            cached=len(configs) - len(to_run),
            vm_type=self._vm_types_for(stage, to_run or configs)[0],
            regional=self._is_regional(to_run or configs),
        )

    def plan(self, configs: list[BCSDConfig], force: bool = False) -> list[StagePlan]:
        """
        Report what each stage would submit, without submitting anything.

        Deduplication and cache filtering run exactly as :meth:`run_full_workflow` would,
        so the counts are what the run will actually dispatch. Stage 3's artifacts are not
        written by stages 1 and 2, so surveying all three up front stays accurate.

        Parameters
        ----------
        configs : list[BCSDConfig]
            The full set of configurations for the run.
        force : bool, optional
            Ignore cached artifacts, matching the same flag on the run.

        Returns
        -------
        list[StagePlan]
            One entry per stage, in execution order.
        """
        return [
            self.plan_stage(stage, stage_configs, force)
            for stage, stage_configs in (
                ("prepare_observations", self._deduplicate_obs_configs(configs)),
                ("fit_historical", self._deduplicate_historical_configs(configs)),
                ("transform_scenario", configs),
            )
        ]

    def submit_stage(
        self,
        stage: Literal["prepare_observations", "fit_historical", "transform_scenario"],
        configs: list[BCSDConfig],
        force: bool = False,
        executor: str | None = None,
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
        executor : {'coiled', 'aws-batch', 'local'}, optional
            Where to run the tasks. Defaults to ``options.executor``.

        Returns
        -------
        list[str]
            List of output paths (either from cache or newly computed)
        """
        # Resolved before any short circuit below, so a mistyped executor cannot report
        # success on a stage that happened to have nothing to submit.
        executor_name = executor or self.options.executor
        executors = {
            "coiled": self._submit_to_coiled,
            "aws-batch": self._submit_to_aws_batch,
            "local": self._run_local,
        }
        if executor_name not in executors:
            raise ValueError(
                f"Unknown executor {executor_name!r}; expected one of {sorted(executors)}"
            )

        if not configs:
            return []

        cache = self._get_cache()
        configs_to_run, output_paths = self._partition_by_cache(cache, stage, configs, force)

        branch = cache._branch_for()
        for config, path in zip(configs, output_paths):
            if path is not None:
                group = path.split("::")[-1]
                logger.info(
                    f"⊙ Skipping {config.run_id} - {group} already cached (branch: {branch})"
                )

        if not configs_to_run:
            logger.info(f"✓ All {len(configs)} {stage} tasks already cached!")
            return output_paths

        logger.info(
            f"→ Submitting {len(configs_to_run)}/{len(configs)} {stage} tasks ({executor_name})"
        )

        # Dispatch to the selected executor, respecting intra-stage dependency
        # ordering (tasmin must run after its debiased-coarse tasmax/dtr inputs).
        runner = executors[executor_name]
        completed_paths = self._run_in_dependency_waves(runner, stage, configs_to_run)

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

        loc_for = partial(self._stage_loc_for, cache, stage)
        preexisting = self._preexisting_artifacts(cache, stage, configs)

        remaining = list(configs)
        attempt = 0
        final_state: str | None = None

        while remaining and attempt < max_retries:
            attempt += 1
            if attempt > 1:
                logger.warning(
                    f"→ Retry {attempt}/{max_retries} for {len(remaining)} failed {stage} tasks"
                )

            # The same number the batch path sends, from the same table, so the two
            # executors partition identically. vm_type is the candidate list coiled picks
            # from; its first entry is what _resources_for sizes the batch request against.
            task_var_dicts = [
                {
                    "CONFIG_JSON": self._config_payload_json(config),
                    "SRM_NR_PROCESSES": str(vcpus(vm_type[0])),
                }
                for config in remaining
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
            still_failed = [config for config in remaining if not cache.exists(loc_for(config))]

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

        # A clean Coiled job reports "done"; one that lost a task reports "done (errors)",
        # and a timeout returns None. Either way the pre-existing artifacts stay unproven.
        if preexisting and final_state != "done":
            raise RuntimeError(
                f"Stage '{stage}' Coiled batch job finished in state {final_state!r} and "
                f"{len(preexisting)} artifact(s) already existed before it ran, so their "
                f"presence now proves nothing: {preexisting}. Check the job's task logs "
                "before treating this run as complete."
            )

        logger.info(f"✓ All {len(configs)} {stage} tasks completed")

        # Collect and return all output paths (now guaranteed to exist)
        return [f"{loc.store_path}::{loc.group}" for loc in map(loc_for, configs)]

    #: AWS Batch job states that mean the job will not progress further.
    _BATCH_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED"})

    #: States a job occupies before any container starts. A job that never leaves them is
    #: waiting on the queue, not doing work.
    _BATCH_QUEUED_STATES = frozenset({"SUBMITTED", "PENDING", "RUNNABLE"})

    #: How long a job may sit queued before the wait gives up. Generous enough to absorb a
    #: compute environment scaling up from zero, short enough that a queue with no instance
    #: large enough to place the task fails with a diagnostic rather than hanging until the
    #: surrounding CI job times out hours later with nothing to read.
    _MAX_QUEUED_SECONDS = 3600

    def _terminate_batch_job(self, client, job_id: str, reason: str) -> None:
        """Cancel a job, best effort. A failure here must not replace why we gave up."""
        try:
            client.terminate_job(jobId=job_id, reason=reason)
        except Exception as exc:  # noqa: BLE001 - reported, never raised over the real cause
            logger.warning(f"Could not terminate AWS Batch job {job_id}: {exc}")

    def _await_batch_job(
        self,
        client,
        job_id: str,
        poll_seconds: int = 30,
        max_queued_seconds: float | None = None,
    ) -> str:
        """
        Block until an AWS Batch job reaches a terminal state.

        There is deliberately no cap on a job that is running: a global
        ``transform_scenario`` has been measured at 17 hours, and a wall-clock limit that
        tolerated that would never fire in time to be useful. The cap is on the queue
        instead, which is where a job hangs when nothing is wrong with the code.

        Parameters
        ----------
        client : botocore.client.BaseClient
            AWS Batch client.
        job_id : str
            Job to poll.
        poll_seconds : int, optional
            Delay between ``describe_jobs`` calls.
        max_queued_seconds : float, optional
            How long the job may stay in :data:`_BATCH_QUEUED_STATES` before it is
            terminated. Defaults to :data:`_MAX_QUEUED_SECONDS`. Once the job starts, this
            no longer applies.

        Returns
        -------
        str
            ``SUCCEEDED``, ``FAILED``, or ``UNKNOWN`` when the job record has aged out
            of ``describe_jobs``.

        Raises
        ------
        RuntimeError
            If the job never starts within ``max_queued_seconds``.
        """
        import time

        if max_queued_seconds is None:
            max_queued_seconds = self._MAX_QUEUED_SECONDS
        queued_deadline = time.monotonic() + max_queued_seconds
        started = False

        while True:
            jobs = client.describe_jobs(jobs=[job_id])["jobs"]
            if not jobs:
                # AWS Batch drops terminated jobs from describe_jobs after about a day.
                # Distinct from FAILED: the record's absence says nothing about the
                # outcome, so the cache sweep decides. The one exception is a --force run,
                # where artifacts predating the job make their presence unprovable and
                # _submit_to_aws_batch raises on any status that is not SUCCEEDED.
                logger.warning(f"AWS Batch job {job_id} is no longer described; status unknown")
                return "UNKNOWN"
            job = jobs[0]
            status = job["status"]
            if status in self._BATCH_TERMINAL_STATES:
                summary = job.get("arrayProperties", {}).get("statusSummary")
                logger.info(
                    f"AWS Batch job {job_id} finished with state: {status}"
                    + (f", children: {summary}" if summary else "")
                )
                return status
            started = started or status not in self._BATCH_QUEUED_STATES
            if not started and time.monotonic() >= queued_deadline:
                # Terminated rather than abandoned: a job left queued can still start
                # hours later, spend money with nobody watching, and leave output that the
                # next run's cache check reads as this run's.
                self._terminate_batch_job(
                    client, job_id, f"never started within {max_queued_seconds:.0f}s"
                )
                raise RuntimeError(
                    f"AWS Batch job {job_id} sat in {status} for "
                    f"{max_queued_seconds / 60:.0f} minutes without starting and has been "
                    f"terminated. Check that queue {self.options.batch_job_queue!r} has a "
                    "compute environment with capacity for the requested vCPU and memory."
                )
            time.sleep(poll_seconds)

    def _submit_to_aws_batch(self, stage: str, configs: list[BCSDConfig]) -> list[str]:
        """
        Run a stage's tasks on AWS Batch.

        Retries are delegated to the service through ``retryStrategy``, so unlike
        ``_submit_to_coiled`` there is no resubmission loop here. Cache presence decides
        success, because a task can exit zero without producing output. That is proof only
        for artifacts this run created, so any that already existed beforehand, which
        ``force`` makes possible, additionally require the job to report ``SUCCEEDED``.

        Parameters
        ----------
        stage : str
            Pipeline stage name.
        configs : list[BCSDConfig]
            Configurations to process.

        Returns
        -------
        list[str]
            ``store::group`` paths, one per config, in input order.

        Raises
        ------
        RuntimeError
            If any config has no artifact in the cache after the job finishes, or if the
            job did not succeed and some artifacts predate it.
        """
        from srm.batch_manifest import write_manifest

        cache = self._get_cache()

        loc_for = partial(self._stage_loc_for, cache, stage)

        manifest_uri: str | None = None
        if len(configs) > 1:
            manifest_uri = (
                f"{self.options.scratch_dir.rstrip('/')}/manifests/"
                f"{self._job_name(stage, configs)}.json"
            )
            write_manifest(
                manifest_uri, stage, [json.loads(self._config_payload_json(c)) for c in configs]
            )

        preexisting = self._preexisting_artifacts(cache, stage, configs)

        job_id = self._submit_batch_job(stage, configs, manifest_uri)
        status = self._await_batch_job(self._batch_client(), job_id)

        failed = [config for config in configs if not cache.exists(loc_for(config))]
        if failed:
            raise RuntimeError(
                f"Stage '{stage}' on AWS Batch job {job_id}: "
                f"{len(failed)} task(s) did not produce output: {[c.run_id for c in failed]}"
            )

        if preexisting and status != "SUCCEEDED":
            raise RuntimeError(
                f"Stage '{stage}' AWS Batch job {job_id} reported {status} and "
                f"{len(preexisting)} artifact(s) already existed before it ran, so their "
                f"presence now proves nothing: {preexisting}. Check the job's children in "
                "CloudWatch before treating this run as complete."
            )

        logger.info(f"✓ All {len(configs)} {stage} tasks completed")
        return [f"{loc.store_path}::{loc.group}" for loc in map(loc_for, configs)]

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
        executor: str | None = None,
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
        executor : {'coiled', 'aws-batch', 'local'}, optional
            Where to run the tasks. Defaults to ``options.executor``.

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
            "prepare_observations", obs_configs, force=force, executor=executor
        )

        # Stage 2: Unique historical tasks
        hist_configs = self._deduplicate_historical_configs(configs)
        logger.info(f"║ Stage 2: fit_historical ({len(hist_configs)} unique tasks)")
        hist_paths = self.submit_stage(
            "fit_historical", hist_configs, force=force, executor=executor
        )

        # Stage 3: All scenario tasks
        logger.info(f"║ Stage 3: transform_scenario ({len(configs)} tasks)")
        scenario_paths = self.submit_stage(
            "transform_scenario", configs, force=force, executor=executor
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
