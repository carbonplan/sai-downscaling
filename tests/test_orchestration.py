"""
Unit tests for BCSDOrchestrator.

Tests focus on:
- Cache instance management (_get_cache)
- Task deduplication for obs and historical stages
- submit_stage cache-hit short-circuit and routing to _run_local / Coiled
- _run_local stage dispatch
- run_full_workflow deduplication and stage ordering
- get_status correctness

BCSDPipeline is always mocked so no real compute or S3 access is required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_icechunk_group as _make_icechunk_group

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.cache import ArtifactCache
from srm.orchestration import BCSDOrchestrator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# South Africa QA box, the standard regional extent used by the snapshot configs.
_SA_BOUNDS = (-38, -19, 13, 36)


@pytest.fixture
def pipeline_options(tmp_path) -> PipelineOptions:
    return PipelineOptions(
        scratch_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
    )


@pytest.fixture
def orchestrator(pipeline_options) -> BCSDOrchestrator:
    return BCSDOrchestrator(pipeline_options)


def _make_config(
    gcm="CESM2-WACCM",
    variable="tas",
    ensemble_member="r1i1p1f1",
    scenario="SSP245",
    subset_bounds=None,
    downscaling_method="BCSD",
) -> BCSDConfig:
    return BCSDConfig(
        gcm=gcm,
        downscaling_method=downscaling_method,
        variable=variable,
        ensemble_member=ensemble_member,
        scenario=scenario,
        predict_period_start=2015,
        predict_period_end=2100,
        subset_bounds=subset_bounds,
    )


@pytest.fixture
def config() -> BCSDConfig:
    return _make_config()


@pytest.fixture
def multi_configs() -> list[BCSDConfig]:
    """Three configs covering two GCMs and two variables for deduplication tests."""
    return [
        _make_config(gcm="CESM2-WACCM", variable="tas", ensemble_member="r1i1p1f1"),
        _make_config(gcm="CESM2-WACCM", variable="tas", ensemble_member="r2i1p1f1"),
        _make_config(gcm="CESM2-WACCM", variable="pr", ensemble_member="r1i1p1f1"),
        _make_config(gcm="MIROC-ES2H", variable="tas", ensemble_member="01"),
    ]


# ---------------------------------------------------------------------------
# _get_cache
# ---------------------------------------------------------------------------


class TestGetCache:
    def test_returns_artifact_cache_instance(self, orchestrator):
        cache = orchestrator._get_cache()
        assert isinstance(cache, ArtifactCache)

    def test_cache_uses_options_paths_and_env(self, orchestrator):
        cache = orchestrator._get_cache()
        opts = orchestrator.options
        assert opts.scratch_dir.rstrip("/") in cache.scratch_dir
        assert cache.environment == opts.environment
        assert cache.branch == opts.branch

    def test_same_call_returns_same_instance(self, orchestrator):
        cache_a = orchestrator._get_cache()
        cache_b = orchestrator._get_cache()
        assert cache_a is cache_b

    def test_different_branch_orchestrator_has_different_cache(self, tmp_path):
        opts_v2 = PipelineOptions(scratch_dir=str(tmp_path / "cache"), branch="v2")
        opts_v3 = PipelineOptions(scratch_dir=str(tmp_path / "cache"), branch="v3")
        orch_v2 = BCSDOrchestrator(opts_v2)
        orch_v3 = BCSDOrchestrator(opts_v3)
        assert orch_v2._get_cache().branch == "v2"
        assert orch_v3._get_cache().branch == "v3"


# ---------------------------------------------------------------------------
# _deduplicate_obs_configs
# ---------------------------------------------------------------------------


class TestDeduplicateObs:
    def test_same_gcm_variable_deduplicates(self, orchestrator, multi_configs):
        # multi_configs has CESM2-WACCM/tas ensemble 0 and 1 → should deduplicate to 1
        cesm_tas = [c for c in multi_configs if c.gcm == "CESM2-WACCM" and c.variable == "tas"]
        result = orchestrator._deduplicate_obs_configs(cesm_tas)
        assert len(result) == 1

    def test_preserves_first_config_of_duplicate_group(self, orchestrator, multi_configs):
        cesm_tas = [c for c in multi_configs if c.gcm == "CESM2-WACCM" and c.variable == "tas"]
        result = orchestrator._deduplicate_obs_configs(cesm_tas)
        assert result[0] is cesm_tas[0]

    def test_different_gcm_not_deduplicated(self, orchestrator, multi_configs):
        result = orchestrator._deduplicate_obs_configs(multi_configs)
        gcm_var_pairs = [(c.gcm, c.variable) for c in result]
        assert ("CESM2-WACCM", "tas") in gcm_var_pairs
        assert ("CESM2-WACCM", "pr") in gcm_var_pairs
        assert ("MIROC-ES2H", "tas") in gcm_var_pairs

    def test_four_configs_produce_three_unique_obs_tasks(self, orchestrator, multi_configs):
        result = orchestrator._deduplicate_obs_configs(multi_configs)
        assert len(result) == 3

    def test_empty_configs_returns_empty(self, orchestrator):
        assert orchestrator._deduplicate_obs_configs([]) == []

    def test_single_config_unchanged(self, orchestrator, config):
        result = orchestrator._deduplicate_obs_configs([config])
        assert len(result) == 1
        assert result[0] is config


# ---------------------------------------------------------------------------
# _deduplicate_historical_configs
# ---------------------------------------------------------------------------


class TestDeduplicateHistorical:
    def test_same_gcm_variable_ensemble_deduplicates(self, orchestrator, config):
        result = orchestrator._deduplicate_historical_configs([config, config])
        assert len(result) == 1

    def test_different_ensemble_not_deduplicated(self, orchestrator, multi_configs):
        result = orchestrator._deduplicate_historical_configs(multi_configs)
        # two CESM2-WACCM/tas (ens 0 and 1) + one CESM2-WACCM/pr + one MIROC-ES2H/tas = 4
        assert len(result) == 4

    def test_all_unique_combinations_preserved(self, orchestrator, multi_configs, subtests):
        result = orchestrator._deduplicate_historical_configs(multi_configs)
        seen_keys = {(c.gcm, c.variable, c.ensemble_member) for c in result}
        for cfg in multi_configs:
            with subtests.test(run_id=cfg.run_id):
                assert (cfg.gcm, cfg.variable, cfg.ensemble_member) in seen_keys

    def test_empty_configs_returns_empty(self, orchestrator):
        assert orchestrator._deduplicate_historical_configs([]) == []

    def test_different_downscaling_method_not_deduplicated(self, orchestrator):
        """fit_historical writes under a leading method segment, so each method must run.

        The two configs are identical apart from ``downscaling_method``. A method-blind
        key collapses them into one task, and the method that loses the race never gets
        its ``{method}/historical/...`` artifact written.
        """
        bcsd = _make_config(downscaling_method="BCSD")
        qdmsd = _make_config(downscaling_method="QDMSD")
        result = orchestrator._deduplicate_historical_configs([bcsd, qdmsd])
        assert [c.downscaling_method for c in result] == ["BCSD", "QDMSD"]

    def test_same_method_still_deduplicates(self, orchestrator):
        """Adding the method to the key must not stop same-method configs collapsing."""
        result = orchestrator._deduplicate_historical_configs(
            [_make_config(downscaling_method="QDMSD"), _make_config(downscaling_method="QDMSD")]
        )
        assert len(result) == 1


class TestDeduplicateObsIsMethodBlind:
    """obs/{variable} is shared across methods, so the obs key must stay method-blind."""

    def test_different_downscaling_method_deduplicates(self, orchestrator):
        result = orchestrator._deduplicate_obs_configs(
            [_make_config(downscaling_method="BCSD"), _make_config(downscaling_method="QDMSD")]
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# submit_stage
# ---------------------------------------------------------------------------


class TestSubmitStage:
    def test_returns_empty_for_empty_configs(self, orchestrator):
        result = orchestrator.submit_stage("prepare_observations", [], use_coiled=False)
        assert result == []

    def test_skips_all_when_all_cached(self, orchestrator, config):
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(cache, "prepare_observations", config)
        _make_icechunk_group(loc, branch=cache.branch)

        with patch.object(orchestrator, "_run_local") as mock_local:
            result = orchestrator.submit_stage("prepare_observations", [config], use_coiled=False)

        mock_local.assert_not_called()
        assert result == [f"{loc.store_path}::{loc.group}"]

    def test_calls_run_local_for_uncached_task(self, orchestrator, config):
        computed_path = "computed_obs_path"
        with patch.object(orchestrator, "_run_local", return_value=[computed_path]) as mock_local:
            result = orchestrator.submit_stage("prepare_observations", [config], use_coiled=False)

        mock_local.assert_called_once_with("prepare_observations", [config])
        assert result == [computed_path]

    def test_calls_submit_to_coiled_when_use_coiled(self, orchestrator, config):
        with patch.object(
            orchestrator, "_submit_to_coiled", return_value=["coiled_path"]
        ) as mock_coiled:
            result = orchestrator.submit_stage("prepare_observations", [config], use_coiled=True)

        mock_coiled.assert_called_once_with("prepare_observations", [config])
        assert result == ["coiled_path"]

    def test_force_runs_even_when_cached(self, orchestrator, config):
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(cache, "prepare_observations", config)
        _make_icechunk_group(loc, branch=cache.branch)

        with patch.object(orchestrator, "_run_local", return_value=[loc.store_path]) as mock_local:
            orchestrator.submit_stage(
                "prepare_observations", [config], force=True, use_coiled=False
            )

        mock_local.assert_called_once()

    def test_mixed_cached_and_uncached(self, orchestrator, multi_configs):
        """Cached tasks return paths directly; uncached tasks go to _run_local."""
        # Use configs with DIFFERENT variables so their obs paths are distinct.
        # multi_configs[0] = CESM2-WACCM/tas, multi_configs[2] = CESM2-WACCM/pr
        cfg_cached = multi_configs[0]  # tas
        cfg_uncached = multi_configs[2]  # pr

        cache = orchestrator._get_cache()
        loc_cached = orchestrator._stage_loc(cache, "prepare_observations", cfg_cached)
        _make_icechunk_group(loc_cached, branch=cache.branch)

        computed_path = "newly_computed"

        with patch.object(orchestrator, "_run_local", return_value=[computed_path]) as mock_local:
            result = orchestrator.submit_stage(
                "prepare_observations",
                [cfg_cached, cfg_uncached],
                use_coiled=False,
            )

        # cached config returns its qualified path; uncached goes through _run_local
        assert f"{loc_cached.store_path}::{loc_cached.group}" in result
        assert computed_path in result
        mock_local.assert_called_once_with("prepare_observations", [cfg_uncached])


# ---------------------------------------------------------------------------
# dependency-aware ordering: tasmin is derived from debiased-coarse tasmax and
# dtr, so it must run only after those sibling variables are committed and final
# (issue #363).
# ---------------------------------------------------------------------------


class TestTasminOrdering:
    def _run_local_recorder(self, calls):
        def fake(stage, cfgs):
            calls.append([c.variable for c in cfgs])
            return [f"path-{c.variable}-{i}" for i, c in enumerate(cfgs)]

        return fake

    def test_tasmin_runs_in_a_later_wave_than_tasmax_and_dtr(self, orchestrator):
        configs = [
            _make_config(variable="tasmax", ensemble_member="008"),
            _make_config(variable="dtr", ensemble_member="008"),
            _make_config(variable="tasmin", ensemble_member="008"),
        ]
        calls: list[list[str]] = []
        with patch.object(orchestrator, "_run_local", side_effect=self._run_local_recorder(calls)):
            orchestrator.submit_stage("transform_scenario", configs, use_coiled=False)

        assert len(calls) == 2, "tasmin should be submitted in a separate, later wave"
        assert "tasmin" not in calls[0]
        assert set(calls[0]) == {"tasmax", "dtr"}
        assert calls[1] == ["tasmin"]

    def test_no_extra_wave_when_no_tasmin(self, orchestrator):
        configs = [
            _make_config(variable="tasmax", ensemble_member="008"),
            _make_config(variable="dtr", ensemble_member="008"),
        ]
        calls: list[list[str]] = []
        with patch.object(orchestrator, "_run_local", side_effect=self._run_local_recorder(calls)):
            orchestrator.submit_stage("transform_scenario", configs, use_coiled=False)

        assert len(calls) == 1
        assert set(calls[0]) == {"tasmax", "dtr"}

    def test_prepare_observations_not_wave_split(self, orchestrator):
        # obs regridding has no cross-variable dependency; tasmin must not be
        # peeled into a second wave (that would spin up an extra job for nothing).
        configs = [
            _make_config(variable="tasmax", ensemble_member="008"),
            _make_config(variable="tasmin", ensemble_member="008"),
        ]
        calls: list[list[str]] = []
        with patch.object(orchestrator, "_run_local", side_effect=self._run_local_recorder(calls)):
            orchestrator.submit_stage("prepare_observations", configs, use_coiled=False)

        assert len(calls) == 1
        assert set(calls[0]) == {"tasmax", "tasmin"}

    def test_output_paths_preserve_input_order_across_waves(self, orchestrator):
        configs = [
            _make_config(variable="tasmin", ensemble_member="008"),
            _make_config(variable="tasmax", ensemble_member="008"),
        ]
        with patch.object(orchestrator, "_run_local", side_effect=self._run_local_recorder([])):
            result = orchestrator.submit_stage("transform_scenario", configs, use_coiled=False)

        # result[i] must correspond to configs[i] even though tasmin ran last
        assert "tasmin" in result[0]
        assert "tasmax" in result[1]


# ---------------------------------------------------------------------------
# coarse-only variables: dtr is bias corrected so tasmin can be derived from it, but
# is never disaggregated, so its debiased_coarse group is what marks a stage complete
# (issue #461).
# ---------------------------------------------------------------------------


class TestCoarseOnlyStageLoc:
    def test_fit_historical_points_at_the_coarse_group(self, orchestrator):
        config = _make_config(variable="dtr", ensemble_member="008")
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(cache, "fit_historical", config, hist_member="001")
        assert loc.group == "bcsd/debiased_coarse/historical/dtr/001"

    def test_transform_scenario_points_at_the_coarse_group(self, orchestrator):
        config = _make_config(variable="dtr", ensemble_member="008")
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(cache, "transform_scenario", config)
        assert loc.group == "bcsd/debiased_coarse/ssp245/dtr/008"

    def test_normal_variable_still_points_at_the_fine_group(self, orchestrator):
        config = _make_config(variable="tas", ensemble_member="003")
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(cache, "transform_scenario", config)
        assert loc.group == "bcsd/ssp245/tas/003"

    def test_submit_stage_skips_dtr_when_only_the_coarse_group_exists(self, orchestrator):
        # Without this, a finished dtr task looks uncached and re-runs every invocation.
        config = _make_config(variable="dtr", ensemble_member="008")
        cache = orchestrator._get_cache()
        cache.config = config
        coarse_loc = cache.debiased_coarse_scenario_loc()
        _make_icechunk_group(coarse_loc, branch=cache.branch)

        with patch.object(orchestrator, "_run_local") as mock_local:
            result = orchestrator.submit_stage("transform_scenario", [config], use_coiled=False)

        mock_local.assert_not_called()
        assert result == [f"{coarse_loc.store_path}::{coarse_loc.group}"]

    def test_submit_stage_does_not_skip_dtr_on_a_stale_fine_group(self, orchestrator):
        # A fine dtr group left behind by a pre-#461 run must not read as complete.
        config = _make_config(variable="dtr", ensemble_member="008")
        cache = orchestrator._get_cache()
        cache.config = config
        _make_icechunk_group(cache.scenario_loc, branch=cache.branch)

        with patch.object(orchestrator, "_run_local", return_value=["computed"]) as mock_local:
            orchestrator.submit_stage("transform_scenario", [config], use_coiled=False)

        mock_local.assert_called_once()


# ---------------------------------------------------------------------------
# _submit_to_coiled (retry logic)
# ---------------------------------------------------------------------------


class TestSubmitToCoiled:
    """Tests for the retry logic in _submit_to_coiled."""

    def _make_coiled_mock(self, job_states: list[str]):
        """Return a mock coiled module that cycles through the given job states."""
        mock_coiled = MagicMock()
        mock_coiled.batch.run.return_value = {"job_id": 1}
        mock_coiled.batch.wait_for_job_done.side_effect = job_states
        return mock_coiled

    def test_success_on_first_attempt(self, orchestrator, config):
        """All tasks succeed on the first attempt — no retry needed."""
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(cache, "prepare_observations", config)
        _make_icechunk_group(loc, branch=cache.branch)

        mock_coiled = self._make_coiled_mock(["done"])

        with patch.dict("sys.modules", {"coiled": mock_coiled}):
            result = orchestrator._submit_to_coiled("prepare_observations", [config])

        assert result == [f"{loc.store_path}::{loc.group}"]
        assert mock_coiled.batch.run.call_count == 1

    @pytest.mark.parametrize("subset_bounds", [None, _SA_BOUNDS], ids=["global", "regional"])
    def test_every_batch_is_purchased_on_demand(self, orchestrator, subset_bounds):
        """Regional batches ran on spot briefly, but reclamation failed jobs outright."""
        config = _make_config(subset_bounds=subset_bounds)
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(cache, "prepare_observations", config)
        _make_icechunk_group(loc, branch=cache.branch)

        mock_coiled = self._make_coiled_mock(["done"])

        with patch.dict("sys.modules", {"coiled": mock_coiled}):
            orchestrator._submit_to_coiled("prepare_observations", [config])

        assert mock_coiled.batch.run.call_args.kwargs["spot_policy"] == "on-demand"

    def test_retries_only_failed_tasks(self, orchestrator, multi_configs):
        """After a partial failure, only the still-missing configs are retried."""
        cfg_ok = multi_configs[0]  # will succeed on first attempt
        cfg_fail = multi_configs[2]  # will fail first, succeed on retry

        cache = orchestrator._get_cache()
        loc_ok = orchestrator._stage_loc(cache, "prepare_observations", cfg_ok)
        loc_fail = orchestrator._stage_loc(cache, "prepare_observations", cfg_fail)

        # First job: only cfg_ok writes its output
        _make_icechunk_group(loc_ok, branch=cache.branch)

        batch_run_calls = []

        def fake_run(**kwargs):
            task_dicts = kwargs["map_over_task_var_dicts"]
            batch_run_calls.append(task_dicts)
            # On the second call (retry), write the fail output so it looks cached
            if len(batch_run_calls) == 2:
                _make_icechunk_group(loc_fail, branch=cache.branch)
            return {"job_id": len(batch_run_calls)}

        mock_coiled = MagicMock()
        mock_coiled.batch.run.side_effect = fake_run
        mock_coiled.batch.wait_for_job_done.return_value = "done (errors)"

        with patch.dict("sys.modules", {"coiled": mock_coiled}):
            result = orchestrator._submit_to_coiled(
                "prepare_observations", [cfg_ok, cfg_fail], max_retries=3
            )

        # Two batch.run calls: initial + one retry
        assert mock_coiled.batch.run.call_count == 2
        # Second call should only have sent cfg_fail
        import json

        retry_configs = [json.loads(d["CONFIG_JSON"]) for d in batch_run_calls[1]]
        assert all(c["variable"] == cfg_fail.variable for c in retry_configs)
        assert f"{loc_ok.store_path}::{loc_ok.group}" in result
        assert f"{loc_fail.store_path}::{loc_fail.group}" in result

    def test_raises_after_max_retries_exhausted(self, orchestrator, config):
        """RuntimeError is raised when all retries are exhausted."""
        # Output is never written → perpetually failing
        mock_coiled = self._make_coiled_mock(["done (errors)", "done (errors)", "done (errors)"])

        with patch.dict("sys.modules", {"coiled": mock_coiled}):
            with pytest.raises(RuntimeError, match="failed after 3 attempt"):
                orchestrator._submit_to_coiled("prepare_observations", [config], max_retries=3)

        assert mock_coiled.batch.run.call_count == 3

    def test_error_message_includes_run_ids(self, orchestrator, config):
        """The RuntimeError message names the configs that failed."""
        mock_coiled = self._make_coiled_mock(["done (errors)"])

        with patch.dict("sys.modules", {"coiled": mock_coiled}):
            with pytest.raises(RuntimeError, match=config.run_id):
                orchestrator._submit_to_coiled("prepare_observations", [config], max_retries=1)

    def test_succeeds_on_second_attempt(self, orchestrator, config):
        """Task fails once then succeeds on retry."""
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(cache, "prepare_observations", config)

        call_count = 0

        def fake_run(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                _make_icechunk_group(loc, branch=cache.branch)
            return {"job_id": call_count}

        mock_coiled = MagicMock()
        mock_coiled.batch.run.side_effect = fake_run
        mock_coiled.batch.wait_for_job_done.return_value = "done (errors)"

        with patch.dict("sys.modules", {"coiled": mock_coiled}):
            result = orchestrator._submit_to_coiled("prepare_observations", [config], max_retries=3)

        assert mock_coiled.batch.run.call_count == 2
        assert result == [f"{loc.store_path}::{loc.group}"]


# ---------------------------------------------------------------------------
# _run_local
# ---------------------------------------------------------------------------


class TestRunLocal:
    def test_routes_prepare_observations(self, orchestrator, config):
        with patch("srm.orchestration.BCSDPipeline") as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance

            result = orchestrator._run_local("prepare_observations", [config])

        MockPipeline.assert_called_once_with(config, orchestrator.options)
        mock_instance.prepare_observations.assert_called_once()
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(cache, "prepare_observations", config)
        assert result == [f"{loc.store_path}::{loc.group}"]

    def test_routes_fit_historical(self, orchestrator, config):
        with patch("srm.orchestration.BCSDPipeline") as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance

            result = orchestrator._run_local("fit_historical", [config])

        mock_instance.fit_historical.assert_called_once()
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(
            cache, "fit_historical", config, hist_member=config.ensemble_member
        )
        assert result == [f"{loc.store_path}::{loc.group}"]

    def test_routes_transform_scenario(self, orchestrator, config):
        with patch("srm.orchestration.BCSDPipeline") as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance

            result = orchestrator._run_local("transform_scenario", [config])

        mock_instance.transform_scenario.assert_called_once()
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(cache, "transform_scenario", config)
        assert result == [f"{loc.store_path}::{loc.group}"]

    def test_unknown_stage_raises(self, orchestrator, config):
        with pytest.raises(ValueError, match="Unknown stage"):
            orchestrator._run_local("bad_stage", [config])

    def test_creates_pipeline_per_config(self, orchestrator, multi_configs):
        with patch("srm.orchestration.BCSDPipeline") as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance
            mock_instance.prepare_observations.return_value = "path"

            orchestrator._run_local("prepare_observations", multi_configs)

        assert MockPipeline.call_count == len(multi_configs)

    def test_returns_path_per_config(self, orchestrator, multi_configs):
        with patch("srm.orchestration.BCSDPipeline") as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance

            result = orchestrator._run_local("prepare_observations", multi_configs)

        assert len(result) == len(multi_configs)
        assert all("::" in r for r in result)

    def test_all_stage_names_route_correctly(self, orchestrator, config, subtests):
        stage_to_method = {
            "prepare_observations": "prepare_observations",
            "fit_historical": "fit_historical",
            "transform_scenario": "transform_scenario",
        }
        for stage, method_name in stage_to_method.items():
            with subtests.test(stage=stage):
                with patch("srm.orchestration.BCSDPipeline") as MockPipeline:
                    mock_instance = MagicMock()
                    MockPipeline.return_value = mock_instance
                    getattr(mock_instance, method_name).return_value = f"{stage}_path"

                    result = orchestrator._run_local(stage, [config])

                getattr(mock_instance, method_name).assert_called_once()
                assert len(result) == 1 and "::" in result[0]


# ---------------------------------------------------------------------------
# run_full_workflow
# ---------------------------------------------------------------------------


class TestRunFullWorkflow:
    def test_calls_all_three_stages(self, orchestrator, config):
        with patch.object(orchestrator, "submit_stage", return_value=["path"]) as mock_submit:
            orchestrator.run_full_workflow([config], use_coiled=False)

        assert mock_submit.call_count == 3

    def test_stages_called_in_correct_order(self, orchestrator, config):
        call_order = []

        def record_stage(stage, *_, **__):
            call_order.append(stage)
            return ["path"]

        with patch.object(orchestrator, "submit_stage", side_effect=record_stage):
            orchestrator.run_full_workflow([config], use_coiled=False)

        assert call_order == ["prepare_observations", "fit_historical", "transform_scenario"]

    def test_obs_stage_receives_deduplicated_configs(self, orchestrator, multi_configs):
        submitted = {}

        def capture(stage, configs, **kwargs):
            submitted[stage] = configs
            return ["path"] * len(configs)

        with patch.object(orchestrator, "submit_stage", side_effect=capture):
            orchestrator.run_full_workflow(multi_configs, use_coiled=False)

        # 4 configs → 3 unique (gcm, variable) pairs for obs
        assert len(submitted["prepare_observations"]) == 3

    def test_historical_stage_receives_deduplicated_configs(self, orchestrator, multi_configs):
        submitted = {}

        def capture(stage, configs, **kwargs):
            submitted[stage] = configs
            return ["path"] * len(configs)

        with patch.object(orchestrator, "submit_stage", side_effect=capture):
            orchestrator.run_full_workflow(multi_configs, use_coiled=False)

        # 4 configs → 4 unique (gcm, variable, ensemble) combinations
        assert len(submitted["fit_historical"]) == 4

    def test_scenario_stage_receives_all_configs(self, orchestrator, multi_configs):
        submitted = {}

        def capture(stage, configs, **kwargs):
            submitted[stage] = configs
            return ["path"] * len(configs)

        with patch.object(orchestrator, "submit_stage", side_effect=capture):
            orchestrator.run_full_workflow(multi_configs, use_coiled=False)

        assert len(submitted["transform_scenario"]) == len(multi_configs)

    def test_returns_all_stage_paths(self, orchestrator, config):
        def mock_submit(stage, configs, **kwargs):
            return [f"{stage}_path_{i}" for i in range(len(configs))]

        with patch.object(orchestrator, "submit_stage", side_effect=mock_submit):
            result = orchestrator.run_full_workflow([config], use_coiled=False)

        assert isinstance(result, dict)
        assert set(result) == {"prepare_observations", "fit_historical", "transform_scenario"}
        assert result["transform_scenario"] == ["transform_scenario_path_0"]

    def test_force_propagated_to_all_stages(self, orchestrator, config, subtests):
        recorded_forces = {}

        def capture(stage, configs, force=False, **kwargs):
            recorded_forces[stage] = force
            return ["path"] * len(configs)

        with patch.object(orchestrator, "submit_stage", side_effect=capture):
            orchestrator.run_full_workflow([config], force=True, use_coiled=False)

        for stage in ("prepare_observations", "fit_historical", "transform_scenario"):
            with subtests.test(stage=stage):
                assert recorded_forces[stage] is True


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_empty_configs_returns_zeros(self, orchestrator):
        status = orchestrator.get_status([])
        for stage_info in status.values():
            assert stage_info["total"] == 0
            assert stage_info["cached"] == 0
            assert stage_info["missing"] == []

    def test_all_stages_present_in_result(self, orchestrator, config):
        status = orchestrator.get_status([config])
        assert set(status.keys()) == {
            "prepare_observations",
            "fit_historical",
            "transform_scenario",
        }

    def test_nothing_cached_all_missing(self, orchestrator, config):
        status = orchestrator.get_status([config])
        assert status["prepare_observations"]["cached"] == 0
        assert config.run_id in status["prepare_observations"]["missing"]

    def test_obs_artifact_counted_as_cached(self, orchestrator, config):
        cache = orchestrator._get_cache()
        _make_icechunk_group(
            orchestrator._stage_loc(cache, "prepare_observations", config), branch=cache.branch
        )

        status = orchestrator.get_status([config])
        assert status["prepare_observations"]["cached"] == 1
        assert status["prepare_observations"]["missing"] == []

    def test_historical_artifact_counted_as_cached(self, orchestrator, config):
        cache = orchestrator._get_cache()
        _make_icechunk_group(
            orchestrator._stage_loc(cache, "fit_historical", config), branch=cache.branch
        )

        status = orchestrator.get_status([config])
        assert status["fit_historical"]["cached"] == 1
        assert status["fit_historical"]["missing"] == []

    def test_scenario_artifact_counted_as_cached(self, orchestrator, config):
        cache = orchestrator._get_cache()
        loc = orchestrator._stage_loc(cache, "transform_scenario", config)
        _make_icechunk_group(loc, branch=cache.branch)

        status = orchestrator.get_status([config])
        assert status["transform_scenario"]["cached"] == 1
        assert status["transform_scenario"]["missing"] == []

    def test_obs_deduplicated_in_totals(self, orchestrator, multi_configs):
        # 4 configs with 3 unique (gcm, variable) → obs total should be 3
        status = orchestrator.get_status(multi_configs)
        assert status["prepare_observations"]["total"] == 3

    def test_historical_totals_include_all_ensemble_combos(self, orchestrator, multi_configs):
        # 4 configs all with unique (gcm, variable, ensemble) → historical total 4
        status = orchestrator.get_status(multi_configs)
        assert status["fit_historical"]["total"] == 4

    def test_scenario_totals_equal_config_count(self, orchestrator, multi_configs):
        status = orchestrator.get_status(multi_configs)
        assert status["transform_scenario"]["total"] == len(multi_configs)

    def test_missing_run_ids_listed(self, orchestrator, multi_configs):
        status = orchestrator.get_status(multi_configs)
        missing = status["transform_scenario"]["missing"]
        for cfg in multi_configs:
            assert cfg.run_id in missing

    def test_cached_plus_missing_equals_total(self, orchestrator, multi_configs, subtests):
        # Cache one obs artifact and verify counts are consistent
        cache = orchestrator._get_cache()
        _make_icechunk_group(
            orchestrator._stage_loc(cache, "prepare_observations", multi_configs[0]),
            branch=cache.branch,
        )

        status = orchestrator.get_status(multi_configs)
        for stage_name, stage_info in status.items():
            with subtests.test(stage=stage_name):
                assert stage_info["cached"] + len(stage_info["missing"]) == stage_info["total"]


# ---------------------------------------------------------------------------
# VM sizing
# ---------------------------------------------------------------------------

_STAGES = ("prepare_observations", "fit_historical", "transform_scenario")


class TestVmSizing:
    """Instance type scales with a batch's spatial extent.

    Global batches must keep the original sizing, since they produce published output.
    Regional batches subset before any heavy compute, so they run on smaller instances.
    Purchase option does not vary with extent; see ``TestSubmitToCoiled``.
    """

    def test_global_batch_keeps_original_sizing(self, subtests):
        configs = [_make_config()]
        for stage in _STAGES:
            with subtests.test(stage=stage):
                assert (
                    BCSDOrchestrator._vm_types_for(stage, configs)
                    == (BCSDOrchestrator._STAGE_VM_TYPES[stage])
                )

    def test_regional_batch_is_smaller_than_global(self, subtests):
        configs = [_make_config(subset_bounds=_SA_BOUNDS)]
        for stage in _STAGES:
            with subtests.test(stage=stage):
                regional = BCSDOrchestrator._vm_types_for(stage, configs)
                assert regional == BCSDOrchestrator._REGIONAL_STAGE_VM_TYPES[stage]
                assert regional != BCSDOrchestrator._STAGE_VM_TYPES[stage]

    def test_mixed_batch_sized_as_global(self, subtests):
        """One instance type covers the batch, so it must fit the largest task in it."""
        configs = [_make_config(subset_bounds=_SA_BOUNDS), _make_config()]
        assert BCSDOrchestrator._is_regional(configs) is False
        for stage in _STAGES:
            with subtests.test(stage=stage):
                assert (
                    BCSDOrchestrator._vm_types_for(stage, configs)
                    == (BCSDOrchestrator._STAGE_VM_TYPES[stage])
                )

    def test_empty_batch_sized_as_global(self):
        assert BCSDOrchestrator._is_regional([]) is False
        assert (
            BCSDOrchestrator._vm_types_for("transform_scenario", [])
            == (BCSDOrchestrator._STAGE_VM_TYPES["transform_scenario"])
        )

    def test_unknown_stage_falls_back_to_default(self, subtests):
        for configs in ([_make_config()], [_make_config(subset_bounds=_SA_BOUNDS)]):
            with subtests.test(regional=BCSDOrchestrator._is_regional(configs)):
                assert BCSDOrchestrator._vm_types_for("no_such_stage", configs) == (
                    BCSDOrchestrator._DEFAULT_VM_TYPE
                )


# ---------------------------------------------------------------------------
# AWS Batch: resource sizing and job submission
# ---------------------------------------------------------------------------


class TestAwsBatchResources:
    def test_global_transform_scenario_fills_r8g_24xlarge(self, orchestrator, config):
        res = orchestrator._resources_for("transform_scenario", [config])
        assert res == {"vcpu": 96, "memory_mib": 737280}

    def test_regional_batch_uses_smaller_resources(self, orchestrator):
        regional = _make_config(subset_bounds=_SA_BOUNDS)
        res = orchestrator._resources_for("transform_scenario", [regional])
        assert res == {"vcpu": 16, "memory_mib": 122880}

    def test_mixed_batch_is_treated_as_global(self, orchestrator, config):
        regional = _make_config(subset_bounds=_SA_BOUNDS)
        res = orchestrator._resources_for("fit_historical", [config, regional])
        assert res == {"vcpu": 48, "memory_mib": 368640}


class TestSubmitBatchJob:
    def test_multi_task_wave_submits_an_array_job(self, orchestrator, multi_configs):
        client = MagicMock()
        client.submit_job.return_value = {"jobId": "abc-123"}
        with patch("boto3.client", return_value=client):
            job_id = orchestrator._submit_batch_job(
                "transform_scenario", multi_configs, "s3://bucket/manifest.json"
            )
        assert job_id == "abc-123"
        kwargs = client.submit_job.call_args.kwargs
        assert kwargs["arrayProperties"] == {"size": len(multi_configs)}
        env = {e["name"]: e["value"] for e in kwargs["containerOverrides"]["environment"]}
        assert env["CONFIG_MANIFEST_URI"] == "s3://bucket/manifest.json"
        assert "CONFIG_JSON" not in env

    def test_single_task_wave_submits_a_plain_job_with_config_json(self, orchestrator, config):
        client = MagicMock()
        client.submit_job.return_value = {"jobId": "solo-1"}
        with patch("boto3.client", return_value=client):
            orchestrator._submit_batch_job("fit_historical", [config], None)
        kwargs = client.submit_job.call_args.kwargs
        assert "arrayProperties" not in kwargs
        env = {e["name"]: e["value"] for e in kwargs["containerOverrides"]["environment"]}
        assert json.loads(env["CONFIG_JSON"])["variable"] == config.variable

    def test_submission_carries_retries_and_project_tag(self, orchestrator, multi_configs):
        client = MagicMock()
        client.submit_job.return_value = {"jobId": "abc-123"}
        with patch("boto3.client", return_value=client):
            orchestrator._submit_batch_job(
                "transform_scenario", multi_configs, "s3://bucket/manifest.json"
            )
        kwargs = client.submit_job.call_args.kwargs
        assert kwargs["retryStrategy"] == {"attempts": 3}
        assert kwargs["tags"] == {"Project": "SRM"}
        assert kwargs["propagateTags"] is True

    def test_command_override_carries_only_the_stage(self, orchestrator, multi_configs):
        # containerOverrides.command replaces CMD, not ENTRYPOINT. Repeating the
        # interpreter invocation here appends it to the image's ENTRYPOINT and every
        # task dies on argument parsing.
        client = MagicMock()
        client.submit_job.return_value = {"jobId": "abc-123"}
        with patch("boto3.client", return_value=client):
            orchestrator._submit_batch_job(
                "transform_scenario", multi_configs, "s3://bucket/manifest.json"
            )
        overrides = client.submit_job.call_args.kwargs["containerOverrides"]
        assert overrides["command"] == ["transform_scenario"]

    def test_submission_requests_stage_resources(self, orchestrator, multi_configs):
        client = MagicMock()
        client.submit_job.return_value = {"jobId": "abc-123"}
        with patch("boto3.client", return_value=client):
            orchestrator._submit_batch_job(
                "transform_scenario", multi_configs, "s3://bucket/manifest.json"
            )
        reqs = {
            r["type"]: r["value"]
            for r in client.submit_job.call_args.kwargs["containerOverrides"][
                "resourceRequirements"
            ]
        }
        assert reqs == {"VCPU": "96", "MEMORY": "737280"}


class TestAwaitBatchJob:
    def test_returns_terminal_status(self, orchestrator):
        client = MagicMock()
        client.describe_jobs.side_effect = [
            {"jobs": [{"status": "RUNNING"}]},
            {"jobs": [{"status": "SUCCEEDED"}]},
        ]
        with patch("time.sleep"):
            assert orchestrator._await_batch_job(client, "abc-123", poll_seconds=0) == "SUCCEEDED"
        assert client.describe_jobs.call_count == 2

    def test_returns_failed_without_raising(self, orchestrator):
        client = MagicMock()
        client.describe_jobs.return_value = {"jobs": [{"status": "FAILED"}]}
        with patch("time.sleep"):
            assert orchestrator._await_batch_job(client, "abc-123", poll_seconds=0) == "FAILED"


class TestSubmitToAwsBatch:
    def test_returns_paths_when_cache_confirms_every_task(self, orchestrator, multi_configs):
        client = MagicMock()
        client.submit_job.return_value = {"jobId": "abc-123"}
        client.describe_jobs.return_value = {"jobs": [{"status": "SUCCEEDED"}]}
        with (
            patch("boto3.client", return_value=client),
            patch("srm.batch_manifest.write_manifest", return_value="s3://b/m.json"),
            patch.object(ArtifactCache, "exists", return_value=True),
            patch("time.sleep"),
        ):
            paths = orchestrator._submit_to_aws_batch("transform_scenario", multi_configs)
        assert len(paths) == len(multi_configs)
        assert all("::" in p for p in paths)

    def test_raises_when_cache_is_missing_output(self, orchestrator, multi_configs):
        client = MagicMock()
        client.submit_job.return_value = {"jobId": "abc-123"}
        client.describe_jobs.return_value = {"jobs": [{"status": "SUCCEEDED"}]}
        with (
            patch("boto3.client", return_value=client),
            patch("srm.batch_manifest.write_manifest", return_value="s3://b/m.json"),
            patch.object(ArtifactCache, "exists", return_value=False),
            patch("time.sleep"),
        ):
            with pytest.raises(RuntimeError, match="did not produce output"):
                orchestrator._submit_to_aws_batch("transform_scenario", multi_configs)

    def test_single_config_skips_manifest_write(self, orchestrator, config):
        client = MagicMock()
        client.submit_job.return_value = {"jobId": "solo-1"}
        client.describe_jobs.return_value = {"jobs": [{"status": "SUCCEEDED"}]}
        with (
            patch("boto3.client", return_value=client),
            patch("srm.batch_manifest.write_manifest") as mock_write,
            patch.object(ArtifactCache, "exists", return_value=True),
            patch("time.sleep"),
        ):
            orchestrator._submit_to_aws_batch("fit_historical", [config])
        mock_write.assert_not_called()
