"""
Orchestration layer for batch BCSD execution with automatic caching and task deduplication.

This module provides the BCSDOrchestrator class which manages efficient batch
execution of BCSD runs across multiple configurations using Coiled's batch API.
It automatically detects cached artifacts and submits only necessary tasks.
"""

from __future__ import annotations

import logging
from typing import Literal

from srm.bcsd_config import BCSDConfig
from srm.cache import ArtifactCache
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

    # VM sizes per pipeline stage. fit_historical is memory-intensive (QM fitting),
    # so it runs on a larger instance; the other two stages are fine on the base size.
    _STAGE_VM_TYPES: dict[str, list[str]] = {
        "prepare_observations": ["r8g.4xlarge"],
        "fit_historical": ["r8g.12xlarge"],
        "transform_scenario": ["r8g.24xlarge"],
    }

    def __init__(self):
        """
        Initialize orchestrator.

        Note: The cache is created on-demand using cache_dir from the configs
        to ensure consistency between orchestrator and batch jobs.
        """
        self._cache_instances = {}  # Cache instances keyed by (cache_dir, environment)

    def _get_cache(self, config: BCSDConfig) -> ArtifactCache:
        """Get or create cache instance for config's cache_dir and output_dir."""
        cache_key = (config.cache_dir, config.output_dir, config.environment, config.version)
        if cache_key not in self._cache_instances:
            self._cache_instances[cache_key] = ArtifactCache(
                cache_dir=config.cache_dir,
                environment=config.environment,
                version=config.version,
                output_dir=config.output_dir,
            )
        return self._cache_instances[cache_key]

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

        # Get cache instance from first config (all configs should use same cache_dir)
        cache = self._get_cache(configs[0])

        # Filter out configs that are already cached
        configs_to_run = []
        output_paths = []

        for config in configs:
            output_path = cache.get_output_path(stage, config)

            if cache.exists(output_path) and not force:
                if config.verbose:
                    logger.info(f"⊙ Skipping {config.run_id} - output exists: {output_path}")
                output_paths.append(output_path)
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

        # Submit to Coiled or run locally
        if use_coiled:
            completed_paths = self._submit_to_coiled(stage, configs_to_run)
        else:
            completed_paths = self._run_local(stage, configs_to_run)

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

        The batch job consists of N tasks (one per config), all running in
        parallel on r8g.2xlarge VMs (64GB RAM, 8 vCPUs). Tasks write their
        outputs to the cache, which are then verified and collected.

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
            import json

            import coiled
        except ImportError:
            raise ImportError("Coiled is not installed. Install with: uv pip install coiled")

        # Exclude computed fields (run_id, config_hash, detrend_data, etc.) since they
        # are derived values and BCSDConfig does not accept them as constructor inputs.
        computed_fields = set(BCSDConfig.model_computed_fields.keys())
        cache = self._get_cache(configs[0])
        command = ["python", "-m", "srm.batch_runner", stage]

        remaining = list(configs)
        attempt = 0

        while remaining and attempt < max_retries:
            attempt += 1
            if attempt > 1:
                logger.warning(
                    f"→ Retry {attempt}/{max_retries} for {len(remaining)} failed {stage} tasks"
                )

            task_var_dicts = [
                {"CONFIG_JSON": json.dumps(config.model_dump(exclude=computed_fields))}
                for config in remaining
            ]

            vm_type = self._STAGE_VM_TYPES.get(stage, ["c8g.12xlarge"])
            job_result = coiled.batch.run(
                command=command,
                name=f"bcsd-{stage}-{remaining[0].gcm}",
                vm_type=vm_type,
                scheduler_vm_type=vm_type,
                region="us-west-2",
                map_over_task_var_dicts=task_var_dicts,
                forward_aws_credentials=False,
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
                if not cache.exists(cache.get_output_path(stage, config))
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
        return [cache.get_output_path(stage, config) for config in configs]

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

        for config in configs:
            pipeline = BCSDPipeline(config)

            if stage == "prepare_observations":
                path = pipeline.prepare_observations()
            elif stage == "fit_historical":
                path = pipeline.fit_historical()
            elif stage == "transform_scenario":
                path = pipeline.transform_scenario()
            else:
                raise ValueError(f"Unknown stage: {stage}")

            completed_paths.append(path)

        return completed_paths

    def run_full_workflow(
        self,
        configs: list[BCSDConfig],
        force: bool = False,
        use_coiled: bool = True,
    ) -> list[str]:
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
        list[str]
            Final scenario output paths for all configs
        """
        logger.info(f"╔═══ Starting BCSD workflow for {len(configs)} configurations")

        # Stage 1: Unique obs regridding tasks
        obs_configs = self._deduplicate_obs_configs(configs)
        logger.info(f"║ Stage 1: prepare_observations ({len(obs_configs)} unique tasks)")
        self.submit_stage("prepare_observations", obs_configs, force=force, use_coiled=use_coiled)

        # Stage 2: Unique historical tasks
        hist_configs = self._deduplicate_historical_configs(configs)
        logger.info(f"║ Stage 2: fit_historical ({len(hist_configs)} unique tasks)")
        self.submit_stage("fit_historical", hist_configs, force=force, use_coiled=use_coiled)

        # Stage 3: All scenario tasks
        logger.info(f"║ Stage 3: transform_scenario ({len(configs)} tasks)")
        output_paths = self.submit_stage(
            "transform_scenario", configs, force=force, use_coiled=use_coiled
        )

        logger.info("╚═══ Workflow complete! ✓")

        return output_paths

    def _deduplicate_obs_configs(self, configs: list[BCSDConfig]) -> list[BCSDConfig]:
        """
        Extract unique (GCM, variable) combinations for obs regridding.

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
            key = (config.gcm, config.variable)
            if key not in seen:
                seen.add(key)
                unique.append(config)
        return unique

    def _deduplicate_historical_configs(self, configs: list[BCSDConfig]) -> list[BCSDConfig]:
        """
        Extract unique (GCM, variable, ensemble) combinations for historical downscaling.

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
            key = (config.gcm, config.variable, config.ensemble_member)
            if key not in seen:
                seen.add(key)
                unique.append(config)
        return unique

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

        cache = self._get_cache(configs[0])

        # Check obs (deduplicated)
        obs_configs = self._deduplicate_obs_configs(configs)
        status["prepare_observations"]["total"] = len(obs_configs)
        for config in obs_configs:
            path = cache.get_obs_path(config)
            if cache.exists(path):
                status["prepare_observations"]["cached"] += 1
            else:
                status["prepare_observations"]["missing"].append(config.run_id)

        # Check historical (deduplicated)
        hist_configs = self._deduplicate_historical_configs(configs)
        status["fit_historical"]["total"] = len(hist_configs)
        for config in hist_configs:
            path = cache.get_historical_path(config)
            if cache.exists(path):
                status["fit_historical"]["cached"] += 1
            else:
                status["fit_historical"]["missing"].append(config.run_id)

        # Check scenarios (all configs)
        status["transform_scenario"]["total"] = len(configs)
        for config in configs:
            if config.scenario:
                path = cache.get_scenario_path(config)
                if cache.exists(path):
                    status["transform_scenario"]["cached"] += 1
                else:
                    status["transform_scenario"]["missing"].append(config.run_id)

        return status
