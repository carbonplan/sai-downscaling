"""
Pre-dispatch cost estimation for a BCSD run.

A production wave commits real money before it produces anything, so the CLI shows what
a run will cost and asks before submitting. Two numbers carry very different confidence
and are labelled accordingly:

- **Burn rate** is exact. Task count, instance type, and the per-vCPU-hour price are all
  known at submission time.
- **Duration** is a guess, and it is the input this pipeline predicts worst. The ranges
  below come from measured history, but ``transform_scenario`` on a global run has been
  seen to take anywhere from 1.5 to 17 hours.

Every number below is hardcoded, and all of them go stale: AWS republishes prices, Coiled
changes its fee, and the pipeline's own runtime moves with each release. :data:`RATES_AS_OF`
records when they were last checked, and the CLI prints it alongside the estimate so an old
table announces its own age rather than quietly under-quoting a run.

Nothing here does I/O; the orchestrator supplies the task counts and instance types.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Date the prices and durations in this module were last checked, as ``YYYY-MM-DD``.
#: Bump it whenever a rate or a duration range below changes.
RATES_AS_OF = "2026-08-28"

#: vCPU count per instance type used by either executor.
INSTANCE_VCPU: dict[str, int] = {
    "r8g.2xlarge": 8,
    "r8g.4xlarge": 16,
    "r8g.8xlarge": 32,
    "r8g.12xlarge": 48,
    "r8g.16xlarge": 64,
    "r8g.24xlarge": 96,
    "c8g.12xlarge": 48,
}

#: Memory requested per vCPU, in MiB. The r8g family carries 8 GiB per vCPU; requesting all
#: of it would never place, because the ECS agent and the OS are not free. This is 15/16 of
#: the total, the headroom the hand-written resource table used to encode per instance.
MEMORY_MIB_PER_VCPU = 7680

#: On-demand EC2 price for the r8g family, per vCPU-hour.
EC2_RATE_PER_VCPU_HOUR = 0.0589

#: Coiled's platform fee, charged per CPU-hour on top of the EC2 price. Removing it is
#: the entire point of the AWS Batch executor (issue #648).
COILED_FEE_PER_VCPU_HOUR = 0.05

RATE_PER_VCPU_HOUR: dict[str, float] = {
    "aws-batch": EC2_RATE_PER_VCPU_HOUR,
    "coiled": EC2_RATE_PER_VCPU_HOUR + COILED_FEE_PER_VCPU_HOUR,
    "local": 0.0,
}

# Wall-clock hours a wave takes, as (p50, p90) of measured Coiled history: 644 timed
# batch_runner jobs as of 2026-08-28, split into global and regional by cost per task,
# which separates them by two orders of magnitude.
#
#   stage                 extent      n    p50     p90     max
#   prepare_observations  regional  157   0.083   0.201   0.371
#   fit_historical        regional  265   0.015   0.080   0.337
#   fit_historical        global      5   0.983   3.871   3.871
#   transform_scenario    regional  189   0.037   0.082   0.409
#   transform_scenario    global     25   1.503   4.316  17.277
#
# Regenerate with coiled.batch.list_jobs(limit=1000), taking completed - started per job.
# These are per wave, not per task: a wave's tasks run concurrently, so adding tasks
# widens the burn rate rather than the clock.
STAGE_HOURS: dict[str, dict[str, tuple[float, float]]] = {
    "prepare_observations": {
        "regional": (0.08, 0.20),
        # No global observation regrid has been measured. Estimated at roughly the
        # regional p90 scaled for the full grid; it is 1-2% of a run's cost, so the
        # imprecision barely moves the total.
        "global": (0.20, 1.00),
    },
    "fit_historical": {
        "regional": (0.02, 0.08),
        "global": (0.98, 3.87),
    },
    "transform_scenario": {
        "regional": (0.04, 0.08),
        "global": (1.50, 4.32),
    },
}


@dataclass(frozen=True, slots=True)
class WaveEstimate:
    """What one stage's submission is expected to cost."""

    stage: str
    n_tasks: int
    vm_type: str
    vcpu_total: int
    burn_rate: float
    low_hours: float
    high_hours: float
    low_cost: float
    high_cost: float


@dataclass(frozen=True, slots=True)
class WorkflowEstimate:
    """What a whole run is expected to cost, across its sequential stages."""

    waves: list[WaveEstimate]
    peak_burn_rate: float
    low_cost: float
    high_cost: float


def vcpus(vm_type: str) -> int:
    """
    Look up an instance type's vCPU count.

    Raises
    ------
    KeyError
        If the instance type is not in :data:`INSTANCE_VCPU`, which means a sizing table
        gained an entry this one did not.
    """
    try:
        return INSTANCE_VCPU[vm_type]
    except KeyError:
        raise KeyError(
            f"Unknown instance type {vm_type!r}; add it to srm.cost.INSTANCE_VCPU"
        ) from None


def memory_mib(vm_type: str) -> int:
    """Memory to request for a task that should fill one instance of ``vm_type``.

    Derived rather than tabulated so the AWS Batch resource request and the cost estimate
    cannot drift apart: both read one instance table.
    """
    return vcpus(vm_type) * MEMORY_MIB_PER_VCPU


def burn_rate_per_hour(vm_type: str, n_tasks: int, executor: str) -> float:
    """Dollars per hour while every task in a wave is running. Exact, not an estimate."""
    try:
        rate = RATE_PER_VCPU_HOUR[executor]
    except KeyError:
        # Defaulting to the bare EC2 rate would under-price a coiled run by a third in the
        # very table meant to inform the confirmation.
        raise KeyError(
            f"Unknown executor {executor!r}; add it to srm.cost.RATE_PER_VCPU_HOUR"
        ) from None
    return vcpus(vm_type) * n_tasks * rate


def duration_range(stage: str, regional: bool) -> tuple[float, float]:
    """Measured (p50, p90) wall-clock hours for one wave of ``stage``."""
    return STAGE_HOURS[stage]["regional" if regional else "global"]


def estimate_wave(
    stage: str, n_tasks: int, vm_type: str, regional: bool, executor: str
) -> WaveEstimate:
    """Estimate one stage's submission."""
    burn = burn_rate_per_hour(vm_type, n_tasks, executor) if n_tasks else 0.0
    low_h, high_h = duration_range(stage, regional)
    return WaveEstimate(
        stage=stage,
        n_tasks=n_tasks,
        vm_type=vm_type,
        vcpu_total=vcpus(vm_type) * n_tasks,
        burn_rate=burn,
        low_hours=low_h,
        high_hours=high_h,
        low_cost=burn * low_h,
        high_cost=burn * high_h,
    )


def estimate_workflow(stages: list[tuple[str, int, str, bool]], executor: str) -> WorkflowEstimate:
    """
    Estimate a whole run from its per-stage plan.

    Parameters
    ----------
    stages : list of (stage, n_tasks, vm_type, regional)
        One entry per stage that will actually submit work.
    executor : str
        Where the tasks run, which sets the per-vCPU-hour rate.

    Returns
    -------
    WorkflowEstimate
        Costs summed across stages, and the peak burn rate. Stages run one after another,
        so the peak is the largest single stage rather than their sum.
    """
    waves = [estimate_wave(s, n, vm, regional, executor) for s, n, vm, regional in stages]
    return WorkflowEstimate(
        waves=waves,
        peak_burn_rate=max((w.burn_rate for w in waves), default=0.0),
        low_cost=sum(w.low_cost for w in waves),
        high_cost=sum(w.high_cost for w in waves),
    )
