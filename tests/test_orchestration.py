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

from unittest.mock import MagicMock, patch

import pytest

from srm.bcsd_config import BCSDConfig
from srm.cache import ArtifactCache
from srm.orchestration import BCSDOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_icechunk_store(path: str) -> None:
    import icechunk
    import numpy as np
    import xarray as xr
    from icechunk.xarray import to_icechunk

    storage = icechunk.local_filesystem_storage(path=path)
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session("main")
    ds = xr.Dataset({"dummy": xr.DataArray(np.array([1.0]), dims=["x"])})
    to_icechunk(ds, session, mode="w")
    session.commit("write complete")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orchestrator() -> BCSDOrchestrator:
    return BCSDOrchestrator()


def _make_config(
    tmp_path, gcm="CESM2-WACCM", variable="tas", ensemble_member="r1i1p1f1", scenario="ssp245"
) -> BCSDConfig:
    return BCSDConfig(
        gcm=gcm,
        variable=variable,
        ensemble_member=ensemble_member,
        scenario=scenario,
        predict_period_start=2015,
        predict_period_end=2100,
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
    )


@pytest.fixture
def config(tmp_path) -> BCSDConfig:
    return _make_config(tmp_path)


@pytest.fixture
def multi_configs(tmp_path) -> list[BCSDConfig]:
    """Three configs covering two GCMs and two variables for deduplication tests."""
    return [
        _make_config(tmp_path, gcm="CESM2-WACCM", variable="tas", ensemble_member="r1i1p1f1"),
        _make_config(tmp_path, gcm="CESM2-WACCM", variable="tas", ensemble_member="r2i1p1f1"),
        _make_config(tmp_path, gcm="CESM2-WACCM", variable="pr", ensemble_member="r1i1p1f1"),
        _make_config(tmp_path, gcm="MIROC-ES2H", variable="tas", ensemble_member="01"),
    ]


# ---------------------------------------------------------------------------
# _get_cache
# ---------------------------------------------------------------------------


class TestGetCache:
    def test_returns_artifact_cache_instance(self, orchestrator, config):
        cache = orchestrator._get_cache(config)
        assert isinstance(cache, ArtifactCache)

    def test_cache_uses_config_paths_and_env(self, orchestrator, config):
        cache = orchestrator._get_cache(config)
        assert config.cache_dir.rstrip("/") in cache.cache_dir
        assert cache.environment == config.environment
        assert cache.version == config.version

    def test_same_key_returns_same_instance(self, orchestrator, config):
        cache_a = orchestrator._get_cache(config)
        cache_b = orchestrator._get_cache(config)
        assert cache_a is cache_b

    def test_different_version_creates_new_instance(self, orchestrator, config):
        cache_v1 = orchestrator._get_cache(config)
        config_v2 = config.model_copy(update={"version": "v2"})
        cache_v2 = orchestrator._get_cache(config_v2)
        assert cache_v1 is not cache_v2
        assert cache_v2.version == "v2"


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


# ---------------------------------------------------------------------------
# submit_stage
# ---------------------------------------------------------------------------


class TestSubmitStage:
    def test_returns_empty_for_empty_configs(self, orchestrator):
        result = orchestrator.submit_stage("prepare_observations", [], use_coiled=False)
        assert result == []

    def test_skips_all_when_all_cached(self, orchestrator, config):
        cache = orchestrator._get_cache(config)
        obs_path = cache.get_obs_path(config)
        _make_icechunk_store(obs_path)

        with patch.object(orchestrator, "_run_local") as mock_local:
            result = orchestrator.submit_stage("prepare_observations", [config], use_coiled=False)

        mock_local.assert_not_called()
        assert result == [obs_path]

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
        cache = orchestrator._get_cache(config)
        obs_path = cache.get_obs_path(config)
        _make_icechunk_store(obs_path)

        with patch.object(orchestrator, "_run_local", return_value=[obs_path]) as mock_local:
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

        cache = orchestrator._get_cache(cfg_cached)
        obs_cached_path = cache.get_obs_path(cfg_cached)
        _make_icechunk_store(obs_cached_path)

        computed_path = "newly_computed"

        with patch.object(orchestrator, "_run_local", return_value=[computed_path]) as mock_local:
            result = orchestrator.submit_stage(
                "prepare_observations",
                [cfg_cached, cfg_uncached],
                use_coiled=False,
            )

        # cached config returns its path; uncached goes through _run_local
        assert obs_cached_path in result
        assert computed_path in result
        mock_local.assert_called_once_with("prepare_observations", [cfg_uncached])


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
        cache = orchestrator._get_cache(config)
        output_path = cache.get_output_path("prepare_observations", config)
        _make_icechunk_store(output_path)

        mock_coiled = self._make_coiled_mock(["done"])

        with patch.dict("sys.modules", {"coiled": mock_coiled}):
            result = orchestrator._submit_to_coiled("prepare_observations", [config])

        assert result == [output_path]
        assert mock_coiled.batch.run.call_count == 1

    def test_retries_only_failed_tasks(self, orchestrator, multi_configs):
        """After a partial failure, only the still-missing configs are retried."""
        cfg_ok = multi_configs[0]  # will succeed on first attempt
        cfg_fail = multi_configs[2]  # will fail first, succeed on retry

        cache = orchestrator._get_cache(cfg_ok)
        ok_path = cache.get_output_path("prepare_observations", cfg_ok)
        fail_path = cache.get_output_path("prepare_observations", cfg_fail)

        # First job: only cfg_ok writes its output
        _make_icechunk_store(ok_path)

        batch_run_calls = []

        def fake_run(**kwargs):
            task_dicts = kwargs["map_over_task_var_dicts"]
            batch_run_calls.append(task_dicts)
            # On the second call (retry), write the fail output so it looks cached
            if len(batch_run_calls) == 2:
                _make_icechunk_store(fail_path)
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
        assert ok_path in result
        assert fail_path in result

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
        cache = orchestrator._get_cache(config)
        output_path = cache.get_output_path("prepare_observations", config)

        call_count = 0

        def fake_run(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                _make_icechunk_store(output_path)
            return {"job_id": call_count}

        mock_coiled = MagicMock()
        mock_coiled.batch.run.side_effect = fake_run
        mock_coiled.batch.wait_for_job_done.return_value = "done (errors)"

        with patch.dict("sys.modules", {"coiled": mock_coiled}):
            result = orchestrator._submit_to_coiled("prepare_observations", [config], max_retries=3)

        assert mock_coiled.batch.run.call_count == 2
        assert result == [output_path]


# ---------------------------------------------------------------------------
# _run_local
# ---------------------------------------------------------------------------


class TestRunLocal:
    def test_routes_prepare_observations(self, orchestrator, config):
        with patch("srm.orchestration.BCSDPipeline") as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance
            mock_instance.prepare_observations.return_value = "obs_path"

            result = orchestrator._run_local("prepare_observations", [config])

        MockPipeline.assert_called_once_with(config)
        mock_instance.prepare_observations.assert_called_once()
        assert result == ["obs_path"]

    def test_routes_fit_historical(self, orchestrator, config):
        with patch("srm.orchestration.BCSDPipeline") as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance
            mock_instance.fit_historical.return_value = "hist_path"

            result = orchestrator._run_local("fit_historical", [config])

        mock_instance.fit_historical.assert_called_once()
        assert result == ["hist_path"]

    def test_routes_transform_scenario(self, orchestrator, config):
        with patch("srm.orchestration.BCSDPipeline") as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance
            mock_instance.transform_scenario.return_value = "scenario_path"

            result = orchestrator._run_local("transform_scenario", [config])

        mock_instance.transform_scenario.assert_called_once()
        assert result == ["scenario_path"]

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
        paths = [f"path_{i}" for i in range(len(multi_configs))]
        with patch("srm.orchestration.BCSDPipeline") as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance
            mock_instance.prepare_observations.side_effect = paths

            result = orchestrator._run_local("prepare_observations", multi_configs)

        assert result == paths

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
                assert result == [f"{stage}_path"]


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
        cache = orchestrator._get_cache(config)
        _make_icechunk_store(cache.get_obs_path(config))

        status = orchestrator.get_status([config])
        assert status["prepare_observations"]["cached"] == 1
        assert status["prepare_observations"]["missing"] == []

    def test_historical_artifact_counted_as_cached(self, orchestrator, config):
        cache = orchestrator._get_cache(config)
        _make_icechunk_store(cache.get_historical_path(config))

        status = orchestrator.get_status([config])
        assert status["fit_historical"]["cached"] == 1
        assert status["fit_historical"]["missing"] == []

    def test_scenario_artifact_counted_as_cached(self, orchestrator, config):
        cache = orchestrator._get_cache(config)
        _make_icechunk_store(cache.get_scenario_path(config))

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
        cache = orchestrator._get_cache(multi_configs[0])
        _make_icechunk_store(cache.get_obs_path(multi_configs[0]))

        status = orchestrator.get_status(multi_configs)
        for stage_name, stage_info in status.items():
            with subtests.test(stage=stage_name):
                assert stage_info["cached"] + len(stage_info["missing"]) == stage_info["total"]
