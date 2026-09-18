"""Unit tests for pre-dispatch cost estimation."""

from __future__ import annotations

import pytest

from saidownscale.cost import (
    COILED_FEE_PER_VCPU_HOUR,
    EC2_RATE_PER_VCPU_HOUR,
    MEMORY_MIB_PER_VCPU,
    STAGE_HOURS,
    burn_rate_per_hour,
    duration_range,
    estimate_wave,
    estimate_workflow,
    memory_mib,
    vcpus,
)


class TestRates:
    def test_vcpus_known_instance(self):
        assert vcpus("r8g.24xlarge") == 96
        assert vcpus("r8g.2xlarge") == 8

    def test_vcpus_rejects_unknown_instance(self):
        with pytest.raises(KeyError, match="r9z.99xlarge"):
            vcpus("r9z.99xlarge")

    def test_burn_rate_is_vcpus_times_tasks_times_rate(self):
        # 83 tasks x 96 vCPU x $0.0589
        assert burn_rate_per_hour("r8g.24xlarge", 83, "aws-batch") == pytest.approx(469.3, abs=0.5)

    def test_coiled_costs_more_than_aws_by_the_platform_fee(self):
        aws = burn_rate_per_hour("r8g.24xlarge", 10, "aws-batch")
        coiled = burn_rate_per_hour("r8g.24xlarge", 10, "coiled")
        assert coiled > aws
        ratio = (EC2_RATE_PER_VCPU_HOUR + COILED_FEE_PER_VCPU_HOUR) / EC2_RATE_PER_VCPU_HOUR
        assert coiled / aws == pytest.approx(ratio)

    def test_unknown_executor_raises_rather_than_guessing(self):
        # Falling back to the EC2 rate would price a coiled run ~46% under its real cost
        # in the very table meant to inform the confirmation.
        with pytest.raises(KeyError, match="slurm"):
            burn_rate_per_hour("r8g.24xlarge", 1, "slurm")

    def test_local_executor_is_free(self):
        assert burn_rate_per_hour("r8g.24xlarge", 10, "local") == 0.0


class TestDurations:
    def test_global_is_slower_than_regional(self):
        assert duration_range("transform_scenario", regional=False) > duration_range(
            "transform_scenario", regional=True
        )

    def test_range_is_ordered_low_to_high(self):
        for stage in STAGE_HOURS:
            for regional in (True, False):
                low, high = duration_range(stage, regional=regional)
                assert 0 < low <= high, (stage, regional)


class TestWaveEstimate:
    def test_cost_range_is_burn_rate_times_duration(self):
        est = estimate_wave("transform_scenario", 83, "r8g.24xlarge", False, "aws-batch")
        low_h, high_h = duration_range("transform_scenario", regional=False)
        assert est.low_cost == pytest.approx(est.burn_rate * low_h)
        assert est.high_cost == pytest.approx(est.burn_rate * high_h)

    def test_zero_tasks_costs_nothing(self):
        est = estimate_wave("transform_scenario", 0, "r8g.24xlarge", False, "aws-batch")
        assert est.burn_rate == 0.0 and est.high_cost == 0.0


class TestWorkflowEstimate:
    def _waves(self, executor="aws-batch"):
        return estimate_workflow(
            [
                ("prepare_observations", 7, "r8g.4xlarge", False),
                ("fit_historical", 11, "r8g.12xlarge", False),
                ("transform_scenario", 32, "r8g.24xlarge", False),
            ],
            executor,
        )

    def test_total_is_the_sum_over_stages(self):
        wf = self._waves()
        assert wf.low_cost == pytest.approx(sum(w.low_cost for w in wf.waves))
        assert wf.high_cost == pytest.approx(sum(w.high_cost for w in wf.waves))

    def test_peak_burn_is_the_max_not_the_sum(self):
        # Stages run one after another, so only one is ever burning at a time.
        wf = self._waves()
        assert wf.peak_burn_rate == pytest.approx(max(w.burn_rate for w in wf.waves))
        assert wf.peak_burn_rate < sum(w.burn_rate for w in wf.waves)

    def test_switching_to_coiled_raises_the_total(self):
        assert self._waves("coiled").high_cost > self._waves("aws-batch").high_cost

    def test_empty_plan_is_free(self):
        wf = estimate_workflow([], "aws-batch")
        assert wf.high_cost == 0.0 and wf.peak_burn_rate == 0.0


class TestInstanceMemory:
    """One instance table serves both the cost estimate and the Batch resource request."""

    def test_memory_is_derived_from_vcpus(self):
        assert memory_mib("r8g.24xlarge") == 96 * MEMORY_MIB_PER_VCPU == 737280
        assert memory_mib("r8g.2xlarge") == 8 * MEMORY_MIB_PER_VCPU == 61440

    def test_leaves_headroom_below_the_instance_total(self):
        # r8g carries 8 GiB per vCPU; requesting all of it would never place, because the
        # ECS agent and the OS are not free.
        assert MEMORY_MIB_PER_VCPU < 8 * 1024

    def test_unknown_instance_raises(self):
        with pytest.raises(KeyError, match="r9z.99xlarge"):
            memory_mib("r9z.99xlarge")
