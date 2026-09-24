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


def test_instance_vcpus_and_memory(subtests):
    assert MEMORY_MIB_PER_VCPU < 8 * 1024, "r8g has 8 GiB per vCPU; requesting all never places"
    for instance, n_vcpus, mib in (("r8g.24xlarge", 96, 737280), ("r8g.2xlarge", 8, 61440)):
        with subtests.test(instance=instance):
            assert vcpus(instance) == n_vcpus
            assert memory_mib(instance) == n_vcpus * MEMORY_MIB_PER_VCPU == mib
    for lookup in (vcpus, memory_mib):
        with subtests.test(unknown=lookup.__name__):
            with pytest.raises(KeyError, match="r9z.99xlarge"):
                lookup("r9z.99xlarge")


def test_burn_rate_per_executor():
    aws = burn_rate_per_hour("r8g.24xlarge", 83, "aws-batch")
    coiled = burn_rate_per_hour("r8g.24xlarge", 83, "coiled")
    assert aws == pytest.approx(469.3, abs=0.5)
    ratio = (EC2_RATE_PER_VCPU_HOUR + COILED_FEE_PER_VCPU_HOUR) / EC2_RATE_PER_VCPU_HOUR
    assert coiled / aws == pytest.approx(ratio)
    assert burn_rate_per_hour("r8g.24xlarge", 10, "local") == 0.0


def test_unknown_executor_raises_rather_than_guessing():
    """Falling back to the EC2 rate would underprice a coiled run by about 46%."""
    with pytest.raises(KeyError, match="slurm"):
        burn_rate_per_hour("r8g.24xlarge", 1, "slurm")


def test_duration_ranges_are_ordered_and_global_is_slower():
    for stage in STAGE_HOURS:
        for regional in (True, False):
            low, high = duration_range(stage, regional=regional)
            assert 0 < low <= high, (stage, regional)
    assert duration_range("transform_scenario", regional=False) > duration_range(
        "transform_scenario", regional=True
    )


def test_wave_cost_is_burn_rate_times_duration():
    est = estimate_wave("transform_scenario", 83, "r8g.24xlarge", False, "aws-batch")
    low_h, high_h = duration_range("transform_scenario", regional=False)
    assert est.low_cost == pytest.approx(est.burn_rate * low_h)
    assert est.high_cost == pytest.approx(est.burn_rate * high_h)
    empty = estimate_wave("transform_scenario", 0, "r8g.24xlarge", False, "aws-batch")
    assert empty.burn_rate == 0.0 and empty.high_cost == 0.0


def _workflow(executor):
    return estimate_workflow(
        [
            ("prepare_observations", 7, "r8g.4xlarge", False),
            ("fit_historical", 11, "r8g.12xlarge", False),
            ("transform_scenario", 32, "r8g.24xlarge", False),
        ],
        executor,
    )


def test_workflow_sums_cost_but_peaks_burn_rate():
    wf = _workflow("aws-batch")
    assert wf.low_cost == pytest.approx(sum(w.low_cost for w in wf.waves))
    assert wf.high_cost == pytest.approx(sum(w.high_cost for w in wf.waves))
    assert wf.peak_burn_rate == pytest.approx(max(w.burn_rate for w in wf.waves))
    assert wf.peak_burn_rate < sum(w.burn_rate for w in wf.waves)
    assert _workflow("coiled").high_cost > wf.high_cost


def test_empty_workflow_is_free():
    wf = estimate_workflow([], "aws-batch")
    assert wf.high_cost == 0.0 and wf.peak_burn_rate == 0.0
