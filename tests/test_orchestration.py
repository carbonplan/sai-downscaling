"""Unit tests for DownscalingOrchestrator with DownscalingPipeline and AWS clients mocked."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_icechunk_group as _make_icechunk_group

from saidownscale.cache import ArtifactCache
from saidownscale.cost import memory_mib, vcpus
from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions
from saidownscale.orchestration import DownscalingOrchestrator

_SA_BOUNDS = (-38, -19, 13, 36)
_STAGES = ("prepare_observations", "fit_historical", "transform_scenario")


@pytest.fixture
def pipeline_options(tmp_path) -> PipelineOptions:
    return PipelineOptions(
        scratch_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
    )


@pytest.fixture
def orchestrator(pipeline_options) -> DownscalingOrchestrator:
    return DownscalingOrchestrator(pipeline_options)


def _make_config(
    gcm="CESM2-WACCM6",
    variable="tas",
    ensemble_member="r1i1p1f1",
    scenario="SSP245",
    subset_bounds=None,
    downscaling_method="BCSD",
) -> DownscalingConfig:
    return DownscalingConfig(
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
def config() -> DownscalingConfig:
    return _make_config()


@pytest.fixture
def multi_configs() -> list[DownscalingConfig]:
    return [
        _make_config(gcm="CESM2-WACCM6", variable="tas", ensemble_member="r1i1p1f1"),
        _make_config(gcm="CESM2-WACCM6", variable="tas", ensemble_member="r2i1p1f1"),
        _make_config(gcm="CESM2-WACCM6", variable="pr", ensemble_member="r1i1p1f1"),
        _make_config(gcm="UKESM1-1-LL", variable="tas", ensemble_member="01"),
    ]


def _cache_stage(orchestrator, stage, config):
    cache = orchestrator._get_cache()
    loc = orchestrator._stage_loc(cache, stage, config)
    _make_icechunk_group(loc, branch=cache.branch)
    return loc


def _qualified(loc) -> str:
    return f"{loc.store_path}::{loc.group}"


def _job_definition(revision: int, image: str) -> dict:
    return {
        "jobDefinitionName": "srm-downscaling",
        "revision": revision,
        "containerProperties": {"image": image},
    }


def test_get_cache_is_built_from_options_and_memoized(orchestrator):
    cache = orchestrator._get_cache()
    opts = orchestrator.options
    assert isinstance(cache, ArtifactCache)
    assert opts.scratch_dir.rstrip("/") in cache.scratch_dir
    assert (cache.environment, cache.branch) == (opts.environment, opts.branch)
    assert orchestrator._get_cache() is cache


def test_deduplicate_obs_keys_on_gcm_and_variable_only(orchestrator, multi_configs):
    result = orchestrator._deduplicate_obs_configs(multi_configs)
    assert [(c.gcm, c.variable) for c in result] == [
        ("CESM2-WACCM6", "tas"),
        ("CESM2-WACCM6", "pr"),
        ("UKESM1-1-LL", "tas"),
    ]
    assert result[0] is multi_configs[0]
    methods = [_make_config(downscaling_method="BCSD"), _make_config(downscaling_method="QDMSD")]
    assert len(orchestrator._deduplicate_obs_configs(methods)) == 1
    assert orchestrator._deduplicate_obs_configs([]) == []


def test_deduplicate_historical_keys_on_member_and_method(orchestrator, config, multi_configs):
    result = orchestrator._deduplicate_historical_configs(multi_configs)
    assert {(c.gcm, c.variable, c.ensemble_member) for c in result} == {
        (c.gcm, c.variable, c.ensemble_member) for c in multi_configs
    }
    assert len(result) == 4
    assert len(orchestrator._deduplicate_historical_configs([config, config])) == 1
    qdmsd = _make_config(downscaling_method="QDMSD")
    assert len(orchestrator._deduplicate_historical_configs([qdmsd, qdmsd])) == 1
    both = [_make_config(downscaling_method="BCSD"), qdmsd]
    assert [c.downscaling_method for c in orchestrator._deduplicate_historical_configs(both)] == [
        "BCSD",
        "QDMSD",
    ]
    assert orchestrator._deduplicate_historical_configs([]) == []


def test_submit_stage_skips_cached_tasks_unless_forced(orchestrator, config):
    loc = _cache_stage(orchestrator, "prepare_observations", config)

    with patch.object(orchestrator, "_run_local") as mock_local:
        result = orchestrator.submit_stage("prepare_observations", [config], executor="local")
    mock_local.assert_not_called()
    assert result == [_qualified(loc)]

    with patch.object(orchestrator, "_run_local", return_value=[loc.store_path]) as mock_local:
        orchestrator.submit_stage("prepare_observations", [config], force=True, executor="local")
    mock_local.assert_called_once()


def test_submit_stage_runs_only_uncached_tasks(orchestrator, multi_configs):
    cfg_cached, cfg_uncached = multi_configs[0], multi_configs[2]
    loc_cached = _cache_stage(orchestrator, "prepare_observations", cfg_cached)

    with patch.object(orchestrator, "_run_local", return_value=["newly_computed"]) as mock_local:
        result = orchestrator.submit_stage(
            "prepare_observations", [cfg_cached, cfg_uncached], executor="local"
        )

    assert _qualified(loc_cached) in result
    assert "newly_computed" in result
    mock_local.assert_called_once_with("prepare_observations", [cfg_uncached])


def test_submit_stage_routes_to_the_selected_executor(pipeline_options, config, subtests):
    cases = [
        ("aws-batch", None, "_submit_to_aws_batch"),
        ("local", "coiled", "_submit_to_coiled"),
        ("coiled", "local", "_run_local"),
    ]
    for configured, explicit, method in cases:
        with subtests.test(configured=configured, explicit=explicit):
            pipeline_options.executor = configured
            orch = DownscalingOrchestrator(pipeline_options)
            kwargs = {"executor": explicit} if explicit else {}
            with patch.object(orch, method, return_value=["s3://x::g"]) as mock_exec:
                result = orch.submit_stage("prepare_observations", [config], **kwargs)
            mock_exec.assert_called_once_with("prepare_observations", [config])
            assert result == ["s3://x::g"]


def test_submit_stage_rejects_an_unknown_executor(orchestrator, config, subtests):
    with subtests.test("no configs"):
        with pytest.raises(ValueError, match="aws_batch"):
            orchestrator.submit_stage("prepare_observations", [], executor="aws_batch")
    with subtests.test("work pending"):
        with pytest.raises(ValueError, match="aws_batch"):
            orchestrator.submit_stage("prepare_observations", [config], executor="aws_batch")
    with subtests.test("everything cached"):
        _cache_stage(orchestrator, "prepare_observations", config)
        with pytest.raises(ValueError, match="aws_batch"):
            orchestrator.submit_stage("prepare_observations", [config], executor="aws_batch")
    with subtests.test("valid executor, no configs"):
        assert orchestrator.submit_stage("prepare_observations", [], executor="local") == []


def test_tasmin_runs_in_a_later_wave_than_tasmax_and_dtr(orchestrator, subtests):
    """Issue #363: tasmin is derived from committed debiased-coarse tasmax and dtr."""
    cases = [
        ("transform_scenario", ["tasmin", "tasmax", "dtr"], [{"tasmax", "dtr"}, {"tasmin"}]),
        ("transform_scenario", ["tasmax", "dtr"], [{"tasmax", "dtr"}]),
        ("prepare_observations", ["tasmax", "tasmin"], [{"tasmax", "tasmin"}]),
    ]
    for stage, variables, expected_waves in cases:
        with subtests.test(stage=stage, variables=variables):
            configs = [_make_config(variable=v, ensemble_member="008") for v in variables]
            calls: list[set[str]] = []

            def fake(stage, cfgs, calls=calls):
                calls.append({c.variable for c in cfgs})
                return [f"path-{c.variable}" for c in cfgs]

            with patch.object(orchestrator, "_run_local", side_effect=fake):
                result = orchestrator.submit_stage(stage, configs, executor="local")

            assert calls == expected_waves
            assert result == [f"path-{v}" for v in variables]


def test_coarse_only_dtr_stage_loc_points_at_debiased_coarse(orchestrator, subtests):
    """Issue #461: dtr is never disaggregated, so its coarse group marks a stage complete."""
    cache = orchestrator._get_cache()
    dtr = _make_config(variable="dtr", ensemble_member="008")
    cases = [
        ("fit_historical", dtr, "001", "bcsd/debiased_coarse/historical/dtr/001"),
        ("transform_scenario", dtr, None, "bcsd/debiased_coarse/ssp245/dtr/008"),
        ("transform_scenario", _make_config(ensemble_member="003"), None, "bcsd/ssp245/tas/003"),
    ]
    for stage, cfg, hist_member, expected in cases:
        with subtests.test(stage=stage, variable=cfg.variable):
            loc = orchestrator._stage_loc(cache, stage, cfg, hist_member=hist_member)
            assert loc.group == expected


def test_submit_stage_skips_dtr_when_only_the_coarse_group_exists(orchestrator):
    config = _make_config(variable="dtr", ensemble_member="008")
    cache = orchestrator._get_cache()
    cache.config = config
    coarse_loc = cache.debiased_coarse_scenario_loc()
    _make_icechunk_group(coarse_loc, branch=cache.branch)

    with patch.object(orchestrator, "_run_local") as mock_local:
        result = orchestrator.submit_stage("transform_scenario", [config], executor="local")

    mock_local.assert_not_called()
    assert result == [_qualified(coarse_loc)]


def test_submit_stage_does_not_skip_dtr_on_a_stale_fine_group(orchestrator):
    """A fine dtr group left by a pre-#461 run must not read as complete."""
    config = _make_config(variable="dtr", ensemble_member="008")
    cache = orchestrator._get_cache()
    cache.config = config
    _make_icechunk_group(cache.scenario_loc, branch=cache.branch)

    with patch.object(orchestrator, "_run_local", return_value=["computed"]) as mock_local:
        orchestrator.submit_stage("transform_scenario", [config], executor="local")

    mock_local.assert_called_once()


def _coiled_mock(job_states: list[str]) -> MagicMock:
    mock_coiled = MagicMock()
    mock_coiled.batch.run.return_value = {"job_id": 1}
    mock_coiled.batch.wait_for_job_done.side_effect = job_states
    return mock_coiled


def test_coiled_succeeds_on_demand_for_every_extent(orchestrator, subtests):
    for subset_bounds in (None, _SA_BOUNDS):
        with subtests.test(regional=subset_bounds is not None):
            config = _make_config(subset_bounds=subset_bounds)
            loc = _cache_stage(orchestrator, "prepare_observations", config)
            mock_coiled = _coiled_mock(["done"])

            with patch.dict("sys.modules", {"coiled": mock_coiled}):
                result = orchestrator._submit_to_coiled("prepare_observations", [config])

            assert result == [_qualified(loc)]
            assert mock_coiled.batch.run.call_count == 1
            assert mock_coiled.batch.run.call_args.kwargs["spot_policy"] == "on-demand"


def test_coiled_retries_only_failed_tasks(orchestrator, multi_configs):
    cfg_ok, cfg_fail = multi_configs[0], multi_configs[2]
    cache = orchestrator._get_cache()
    loc_ok = _cache_stage(orchestrator, "prepare_observations", cfg_ok)
    loc_fail = orchestrator._stage_loc(cache, "prepare_observations", cfg_fail)
    batch_run_calls = []

    def fake_run(**kwargs):
        batch_run_calls.append(kwargs["map_over_task_var_dicts"])
        if len(batch_run_calls) == 2:
            _make_icechunk_group(loc_fail, branch=cache.branch)
        return {"job_id": len(batch_run_calls)}

    mock_coiled = MagicMock()
    mock_coiled.batch.run.side_effect = fake_run
    mock_coiled.batch.wait_for_job_done.side_effect = ["done (errors)", "done"]

    with patch.dict("sys.modules", {"coiled": mock_coiled}):
        result = orchestrator._submit_to_coiled(
            "prepare_observations", [cfg_ok, cfg_fail], max_retries=3
        )

    assert mock_coiled.batch.run.call_count == 2
    retry_configs = [json.loads(d["CONFIG_JSON"]) for d in batch_run_calls[1]]
    assert all(c["variable"] == cfg_fail.variable for c in retry_configs)
    assert _qualified(loc_ok) in result
    assert _qualified(loc_fail) in result


def test_coiled_failed_job_does_not_pass_on_preexisting_artifacts(orchestrator, config):
    """With force, the cache may hold the previous run's output, which proves nothing."""
    _cache_stage(orchestrator, "prepare_observations", config)
    mock_coiled = _coiled_mock(["done (errors)"])

    with patch.dict("sys.modules", {"coiled": mock_coiled}):
        with pytest.raises(RuntimeError, match="already existed"):
            orchestrator._submit_to_coiled("prepare_observations", [config])


def test_coiled_raises_naming_run_ids_after_max_retries(orchestrator, config):
    mock_coiled = _coiled_mock(["done (errors)"] * 3)

    with patch.dict("sys.modules", {"coiled": mock_coiled}):
        with pytest.raises(RuntimeError, match="failed after 3 attempt") as exc_info:
            orchestrator._submit_to_coiled("prepare_observations", [config], max_retries=3)

    assert config.run_id in str(exc_info.value)
    assert mock_coiled.batch.run.call_count == 3


def test_run_local_routes_each_stage(orchestrator, config, subtests):
    cache = orchestrator._get_cache()
    for stage in _STAGES:
        with subtests.test(stage=stage):
            with patch("saidownscale.orchestration.DownscalingPipeline") as MockPipeline:
                result = orchestrator._run_local(stage, [config])

            MockPipeline.assert_called_once_with(config, orchestrator.options)
            getattr(MockPipeline.return_value, stage).assert_called_once()
            hist_member = config.ensemble_member if stage == "fit_historical" else None
            loc = orchestrator._stage_loc(cache, stage, config, hist_member=hist_member)
            assert result == [_qualified(loc)]
    with subtests.test(stage="bad_stage"):
        with pytest.raises(ValueError, match="Unknown stage"):
            orchestrator._run_local("bad_stage", [config])


def test_run_local_builds_one_pipeline_and_path_per_config(orchestrator, multi_configs):
    with patch("saidownscale.orchestration.DownscalingPipeline") as MockPipeline:
        result = orchestrator._run_local("prepare_observations", multi_configs)

    assert MockPipeline.call_count == len(multi_configs)
    assert len(result) == len(multi_configs)
    assert all("::" in r for r in result)


def test_run_full_workflow_runs_deduplicated_stages_in_order(orchestrator, multi_configs):
    calls = []

    def capture(stage, configs, force=False, **kwargs):
        calls.append((stage, len(configs), force))
        return [f"{stage}_path_{i}" for i in range(len(configs))]

    with patch.object(orchestrator, "submit_stage", side_effect=capture):
        result = orchestrator.run_full_workflow(multi_configs, force=True, executor="local")

    assert calls == [
        ("prepare_observations", 3, True),
        ("fit_historical", 4, True),
        ("transform_scenario", 4, True),
    ]
    assert set(result) == set(_STAGES)
    assert result["transform_scenario"] == [f"transform_scenario_path_{i}" for i in range(4)]


def test_get_status(orchestrator, config, multi_configs, subtests):
    with subtests.test("empty"):
        for stage_info in orchestrator.get_status([]).values():
            assert stage_info == {"total": 0, "cached": 0, "missing": []}

    with subtests.test("nothing cached"):
        status = orchestrator.get_status(multi_configs)
        assert {s: info["total"] for s, info in status.items()} == {
            "prepare_observations": 3,
            "fit_historical": 4,
            "transform_scenario": 4,
        }
        assert all(info["cached"] == 0 for info in status.values())
        assert all(c.run_id in status["transform_scenario"]["missing"] for c in multi_configs)

    for stage in _STAGES:
        with subtests.test("cached", stage=stage):
            _cache_stage(orchestrator, stage, config)
            status = orchestrator.get_status([config])
            assert status[stage]["cached"] == 1
            assert status[stage]["missing"] == []

    with subtests.test("cached plus missing equals total"):
        for info in orchestrator.get_status(multi_configs).values():
            assert info["cached"] + len(info["missing"]) == info["total"]


def test_vm_and_batch_resources_scale_with_extent(subtests):
    regional_table = DownscalingOrchestrator._REGIONAL_STAGE_VM_TYPES
    global_table = DownscalingOrchestrator._STAGE_VM_TYPES
    regional = _make_config(subset_bounds=_SA_BOUNDS)
    cases = {
        "global": ([_make_config()], global_table, {"vcpu": 96, "memory_mib": 737280}),
        "regional": ([regional], regional_table, {"vcpu": 16, "memory_mib": 122880}),
        "mixed": ([regional, _make_config()], global_table, {"vcpu": 96, "memory_mib": 737280}),
        "empty": ([], global_table, {"vcpu": 96, "memory_mib": 737280}),
    }
    for name, (configs, table, transform_resources) in cases.items():
        with subtests.test(name):
            assert DownscalingOrchestrator._is_regional(configs) is (table is regional_table)
            for stage in _STAGES:
                vm_types = DownscalingOrchestrator._vm_types_for(stage, configs)
                assert vm_types == table[stage]
                if table is regional_table:
                    assert vm_types != global_table[stage]
                assert DownscalingOrchestrator._resources_for(stage, configs) == {
                    "vcpu": vcpus(vm_types[0]),
                    "memory_mib": memory_mib(vm_types[0]),
                }
            resources = DownscalingOrchestrator._resources_for("transform_scenario", configs)
            assert resources == transform_resources


def test_unknown_stage_defaults_its_vm_but_refuses_batch_resources(config, subtests):
    for configs in ([config], [_make_config(subset_bounds=_SA_BOUNDS)]):
        with subtests.test(regional=DownscalingOrchestrator._is_regional(configs)):
            assert DownscalingOrchestrator._vm_types_for("no_such_stage", configs) == (
                DownscalingOrchestrator._DEFAULT_VM_TYPE
            )
            with pytest.raises(KeyError):
                DownscalingOrchestrator._resources_for("no_such_stage", configs)


def _batch_env(client) -> dict[str, str]:
    overrides = client.submit_job.call_args.kwargs["containerOverrides"]
    return {e["name"]: e["value"] for e in overrides["environment"]}


def test_submit_batch_job(pipeline_options, config, multi_configs, subtests):
    with subtests.test("multi-task wave is an array job"):
        client = MagicMock()
        client.submit_job.return_value = {"jobId": "abc-123"}
        with patch("boto3.client", return_value=client):
            job_id = DownscalingOrchestrator(pipeline_options)._submit_batch_job(
                "transform_scenario", multi_configs, "s3://bucket/manifest.json"
            )
        kwargs = client.submit_job.call_args.kwargs
        env = _batch_env(client)
        reqs = {r["type"]: r["value"] for r in kwargs["containerOverrides"]["resourceRequirements"]}
        assert job_id == "abc-123"
        assert kwargs["arrayProperties"] == {"size": len(multi_configs)}
        assert env["CONFIG_MANIFEST_URI"] == "s3://bucket/manifest.json"
        assert "CONFIG_JSON" not in env
        assert env["SRM_NR_PROCESSES"] == "96"
        assert reqs == {"VCPU": "96", "MEMORY": "737280"}
        assert kwargs["retryStrategy"] == {"attempts": 3}
        assert kwargs["tags"] == {"Project": "SRM"}
        assert kwargs["propagateTags"] is True
        assert kwargs["containerOverrides"]["command"] == [
            "python",
            "-m",
            "saidownscale.batch_runner",
            "transform_scenario",
        ]

    with subtests.test("single-task wave is a plain job carrying CONFIG_JSON"):
        client = MagicMock()
        client.submit_job.return_value = {"jobId": "solo-1"}
        with patch("boto3.client", return_value=client):
            DownscalingOrchestrator(pipeline_options)._submit_batch_job(
                "fit_historical", [config], None
            )
        assert "arrayProperties" not in client.submit_job.call_args.kwargs
        assert json.loads(_batch_env(client)["CONFIG_JSON"])["variable"] == config.variable


def test_batch_command_matches_the_image_entrypoint():
    """Batch overrides CMD only, so the job command must start where ENTRYPOINT ends."""
    dockerfile = (Path(__file__).parent.parent / "Dockerfile").read_text()
    entrypoint = next(line for line in dockerfile.splitlines() if line.startswith("ENTRYPOINT"))
    assert json.loads(entrypoint.removeprefix("ENTRYPOINT").strip()) == ["uv", "run", "--no-sync"]


def _array_parent(**counts) -> dict:
    states = ("STARTING", "RUNNING", "SUCCEEDED", "FAILED", "RUNNABLE", "SUBMITTED", "PENDING")
    summary = dict.fromkeys(states, 0) | counts
    return {"status": "PENDING", "arrayProperties": {"size": 20, "statusSummary": summary}}


def test_await_batch_job_returns_the_terminal_status(orchestrator, subtests):
    """Array parents stay PENDING, so child roll-ups are the start signal (v0.14.1 60-min kill)."""
    running, succeeded = {"status": "RUNNING"}, {"status": "SUCCEEDED"}
    cases = {
        "running job outlives the queue deadline": ([running, running, succeeded], 0, "SUCCEEDED"),
        "queued then started": ([{"status": "RUNNABLE"}, running, succeeded], 10_000, "SUCCEEDED"),
        "failed": ([{"status": "FAILED"}], None, "FAILED"),
        "dropped record": ([], None, "UNKNOWN"),
        "array with running children": ([_array_parent(RUNNING=21), succeeded], 0, "SUCCEEDED"),
        "array with finished children": (
            [_array_parent(SUCCEEDED=15, FAILED=5), succeeded],
            0,
            "SUCCEEDED",
        ),
    }
    for name, (jobs, max_queued, expected) in cases.items():
        with subtests.test(name):
            client = MagicMock()
            responses = [{"jobs": [job]} for job in jobs] or [{"jobs": []}]
            client.describe_jobs.side_effect = responses
            with patch("time.sleep"):
                status = orchestrator._await_batch_job(
                    client, "job-1", poll_seconds=0, max_queued_seconds=max_queued
                )
            assert status == expected
            assert client.describe_jobs.call_count == len(responses)
            client.terminate_job.assert_not_called()


def test_await_batch_job_terminates_a_job_that_never_starts(orchestrator, subtests):
    queue = orchestrator.options.batch_job_queue
    empty_array = {"status": "PENDING", "arrayProperties": {"size": 20, "statusSummary": {}}}
    cases = {
        "plain job": ({"status": "RUNNABLE"}, "RUNNABLE", None),
        "array whose children never place": (_array_parent(RUNNABLE=20), "PENDING", None),
        "array with no summary yet": (empty_array, "PENDING", None),
        "failed terminate keeps the real error": (
            {"status": "RUNNABLE"},
            "RUNNABLE",
            RuntimeError("AccessDenied"),
        ),
    }
    for name, (job, status, terminate_error) in cases.items():
        with subtests.test(name):
            client = MagicMock()
            client.describe_jobs.return_value = {"jobs": [job]}
            client.terminate_job.side_effect = terminate_error
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="without starting") as exc_info:
                    orchestrator._await_batch_job(
                        client, "stuck-1", poll_seconds=0, max_queued_seconds=0
                    )
            assert f"sat in {status}" in str(exc_info.value)
            assert queue in str(exc_info.value)
            assert client.terminate_job.call_args.kwargs["jobId"] == "stuck-1"


def test_submit_to_aws_batch_trusts_only_fresh_or_succeeded_artifacts(
    pipeline_options, multi_configs, subtests
):
    n = len(multi_configs)
    succeeded = [{"status": "SUCCEEDED"}]
    cases = {
        "succeeded, present": (succeeded, [True] * (3 * n), None),
        "succeeded, missing": (succeeded, [False] * (3 * n), "did not produce output"),
        "failed, stale": ([{"status": "FAILED"}], [True] * (3 * n), "FAILED"),
        "unknown, stale": ([], [True] * (3 * n), "UNKNOWN"),
        "unknown, fresh": ([], [False] * n + [True] * (2 * n), None),
    }
    for name, (jobs, exists, error) in cases.items():
        with subtests.test(name):
            orchestrator = DownscalingOrchestrator(pipeline_options)
            client = MagicMock()
            client.submit_job.return_value = {"jobId": "abc-123"}
            client.describe_jobs.return_value = {"jobs": jobs}
            with (
                patch("boto3.client", return_value=client),
                patch("saidownscale.batch_manifest.write_manifest", return_value="s3://b/m.json"),
                patch.object(ArtifactCache, "exists", side_effect=exists),
                patch("time.sleep"),
            ):
                if error:
                    with pytest.raises(RuntimeError, match=error):
                        orchestrator._submit_to_aws_batch("transform_scenario", multi_configs)
                else:
                    paths = orchestrator._submit_to_aws_batch("transform_scenario", multi_configs)
                    assert len(paths) == n
                    assert all("::" in p for p in paths)


def test_submit_to_aws_batch_single_config_skips_manifest_write(orchestrator, config):
    client = MagicMock()
    client.submit_job.return_value = {"jobId": "solo-1"}
    client.describe_jobs.return_value = {"jobs": [{"status": "SUCCEEDED"}]}
    with (
        patch("boto3.client", return_value=client),
        patch("saidownscale.batch_manifest.write_manifest") as mock_write,
        patch.object(ArtifactCache, "exists", return_value=True),
        patch("time.sleep"),
    ):
        orchestrator._submit_to_aws_batch("fit_historical", [config])
    mock_write.assert_not_called()


def test_job_name_fits_the_aws_batch_limit(orchestrator, config):
    configs = [
        SimpleNamespace(
            gcm=f"SOME-VERY-LONG-MODEL-NAME-{i:02d}",
            variable=f"variable_{i:02d}",
            config_hash=f"hash{i:04d}",
        )
        for i in range(20)
    ]
    batch_hash = hashlib.sha256(
        "".join(sorted(c.config_hash for c in configs)).encode()
    ).hexdigest()[:8]

    name = orchestrator._job_name("transform_scenario", configs)
    assert len(name) <= 128
    assert name.endswith(batch_hash)

    short = orchestrator._job_name("fit_historical", [config])
    assert short.startswith("saidownscale-fit_historical-CESM2-WACCM6-tas-")
    assert len(short) < 128


def test_plan_counts_deduplicated_stages_and_sizes(orchestrator, multi_configs):
    plans = {p.stage: p for p in orchestrator.plan(multi_configs)}
    assert set(plans) == set(_STAGES)
    assert plans["prepare_observations"].to_run == 3
    assert plans["fit_historical"].to_run == 4
    assert (plans["transform_scenario"].to_run, plans["transform_scenario"].cached) == (4, 0)
    assert plans["transform_scenario"].vm_type == "r8g.24xlarge"
    assert plans["transform_scenario"].regional is False

    regional = {p.stage: p for p in orchestrator.plan([_make_config(subset_bounds=_SA_BOUNDS)])}
    assert regional["transform_scenario"].vm_type == "r8g.4xlarge"
    assert regional["transform_scenario"].regional is True


def test_plan_excludes_cached_artifacts_unless_forced(orchestrator, config):
    _cache_stage(orchestrator, "transform_scenario", config)

    plan = {p.stage: p for p in orchestrator.plan([config])}["transform_scenario"]
    forced = {p.stage: p for p in orchestrator.plan([config], force=True)}["transform_scenario"]
    single = orchestrator.plan_stage("transform_scenario", [config])
    assert (plan.to_run, plan.cached) == (0, 1)
    assert (forced.to_run, forced.cached) == (1, 0)
    assert (single.to_run, single.cached) == (0, 1)


def test_plans_agree_with_what_submit_stage_runs(orchestrator, multi_configs, subtests):
    _cache_stage(orchestrator, "transform_scenario", multi_configs[0])
    planned = {
        "transform_scenario": {p.stage: p for p in orchestrator.plan(multi_configs)}[
            "transform_scenario"
        ],
        "prepare_observations": orchestrator.plan_stage("prepare_observations", multi_configs),
    }
    assert planned["prepare_observations"].to_run == len(multi_configs)
    for stage, plan in planned.items():
        with subtests.test(stage=stage):
            with patch.object(orchestrator, "_run_local", return_value=[]) as mock_local:
                orchestrator.submit_stage(stage, multi_configs, executor="local")
            submitted = sum(len(call.args[1]) for call in mock_local.call_args_list)
            assert plan.to_run == submitted


def test_resolve_job_definition(pipeline_options, subtests):
    arn = "arn:aws:batch:us-west-2:123456789012:job-definition/srm-downscaling:9"
    cases = {
        "pinned revision": (
            "srm-downscaling:7",
            [[_job_definition(7, "ecr/img:sha7")]],
            ("srm-downscaling:7", "ecr/img:sha7"),
            {"jobDefinitions": ["srm-downscaling:7"]},
        ),
        "bare name picks the highest active revision": (
            "srm-downscaling",
            [
                [
                    _job_definition(2, "ecr/img:old"),
                    _job_definition(5, "ecr/img:new"),
                    _job_definition(3, "ecr/img:mid"),
                ]
            ],
            ("srm-downscaling:5", "ecr/img:new"),
            {"status": "ACTIVE"},
        ),
        "arn resolves to the bare name": (
            arn,
            [[_job_definition(9, "ecr/img:sha9")]],
            ("srm-downscaling:9", "ecr/img:sha9"),
            {},
        ),
        "follows next token": (
            "srm-downscaling",
            [[_job_definition(1, "ecr/img:old")], [_job_definition(102, "ecr/img:new")]],
            ("srm-downscaling:102", "ecr/img:new"),
            {"nextToken": "page2"},
        ),
    }
    for name, (configured, pages, (job_def, image), last_kwargs) in cases.items():
        with subtests.test(name):
            pipeline_options.batch_job_definition = configured
            orch = DownscalingOrchestrator(pipeline_options)
            client = MagicMock()
            responses = [{"jobDefinitions": page} for page in pages]
            if len(responses) > 1:
                responses[0]["nextToken"] = "page2"
            client.describe_job_definitions.side_effect = responses
            with patch("boto3.client", return_value=client):
                resolved = orch.resolve_job_definition()
            assert resolved == {"job_definition": job_def, "image": image}
            assert client.describe_job_definitions.call_count == len(responses)
            call_kwargs = client.describe_job_definitions.call_args.kwargs
            assert {k: call_kwargs[k] for k in last_kwargs} == last_kwargs

    with subtests.test("missing definition raises"):
        pipeline_options.batch_job_definition = "srm-downscaling"
        orch = DownscalingOrchestrator(pipeline_options)
        client = MagicMock()
        client.describe_job_definitions.return_value = {"jobDefinitions": []}
        with patch("boto3.client", return_value=client):
            with pytest.raises(ValueError, match="srm-downscaling"):
                orch.resolve_job_definition()


def test_submission_pins_the_resolved_revision(pipeline_options, config, subtests):
    pipeline_options.batch_job_definition = "srm-downscaling"

    def make_client():
        client = MagicMock()
        client.submit_job.return_value = {"jobId": "abc-123"}
        client.describe_job_definitions.return_value = {
            "jobDefinitions": [_job_definition(4, "ecr/img:sha4")]
        }
        return client

    with subtests.test("resolved once and reused across a run"):
        orch = DownscalingOrchestrator(pipeline_options)
        client = make_client()
        with patch("boto3.client", return_value=client):
            orch.resolve_job_definition()
            orch._submit_batch_job("fit_historical", [config], None)
            orch._submit_batch_job("transform_scenario", [config], None)
        assert client.describe_job_definitions.call_count == 1
        assert client.submit_job.call_args.kwargs["jobDefinition"] == "srm-downscaling:4"

    with subtests.test("falls back to the configured name when resolution fails"):
        orch = DownscalingOrchestrator(pipeline_options)
        client = make_client()
        client.describe_job_definitions.side_effect = RuntimeError("denied")
        with patch("boto3.client", return_value=client):
            orch._submit_batch_job("fit_historical", [config], None)
        assert client.submit_job.call_args.kwargs["jobDefinition"] == "srm-downscaling"
