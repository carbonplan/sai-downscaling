"""
Quality checks for BCSD pipeline artifact DataArrays.

Provides run_artifact_checks() to inspect intermediate and output arrays
at any pipeline step — the same three checks (all_nan, out_of_range,
sporadic_nan_days) that a post-hoc ``bcsd qa`` run would apply to a cached
artifact, but callable on a live in-memory DataArray before it is written.

Intended for use in the pipeline-stage-debugger notebook; the inspect()
helper there calls run_artifact_checks() after every pipeline sub-step so
regressions are caught immediately rather than discovered downstream.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import xarray as xr

from srm.qaqc import VAR_SPATIAL_RANGES
from srm.validation import CheckStatus

_STATUS_SYMBOL: dict[CheckStatus, str] = {
    CheckStatus.PASS: "✓",
    CheckStatus.FAIL: "✗",
    CheckStatus.SKIP: "–",
    CheckStatus.UNKNOWN: "?",
    CheckStatus.XFAIL: "⚠",
    CheckStatus.XPASS: "⚠",
}


@dataclasses.dataclass
class ArtifactCheckResult:
    status: CheckStatus
    message: str = ""
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)
    nan_fraction: float = 0.0


def _check_all_nan(da: xr.DataArray) -> ArtifactCheckResult:
    nan_fraction = float(np.isnan(da.values).mean())
    if nan_fraction == 1.0:
        return ArtifactCheckResult(
            status=CheckStatus.FAIL,
            message="all values are NaN",
            detail={"nan_fraction": nan_fraction},
            nan_fraction=nan_fraction,
        )
    return ArtifactCheckResult(
        status=CheckStatus.PASS,
        detail={"nan_fraction": nan_fraction},
        nan_fraction=nan_fraction,
    )


def _check_out_of_range(da: xr.DataArray, variable: str) -> ArtifactCheckResult:
    if variable not in VAR_SPATIAL_RANGES:
        return ArtifactCheckResult(
            status=CheckStatus.SKIP,
            message=f"no range defined for variable {variable!r}",
        )
    ranges = VAR_SPATIAL_RANGES[variable]
    min_lo, _ = ranges["min"]
    _, max_hi = ranges["max"]

    arr = da.values
    valid_mask = ~np.isnan(arr)
    if not valid_mask.any():
        return ArtifactCheckResult(
            status=CheckStatus.SKIP,
            message="no non-NaN values to check",
        )

    valid_values = arr[valid_mask]
    oor_mask = (valid_values < min_lo) | (valid_values > max_hi)
    oor_fraction = float(oor_mask.mean())
    actual_min = float(valid_values.min())
    actual_max = float(valid_values.max())

    detail = {
        "out_of_range_fraction": oor_fraction,
        "actual_min": actual_min,
        "actual_max": actual_max,
        "expected_min": min_lo,
        "expected_max": max_hi,
    }
    if oor_fraction > 0:
        return ArtifactCheckResult(
            status=CheckStatus.FAIL,
            message=f"{oor_fraction:.2%} of values outside [{min_lo}, {max_hi}]",
            detail=detail,
        )
    return ArtifactCheckResult(status=CheckStatus.PASS, detail=detail)


def _check_sporadic_nan_days(da: xr.DataArray) -> ArtifactCheckResult:
    """Flag time steps that are partially (not fully) NaN."""
    if "time" not in da.dims:
        return ArtifactCheckResult(status=CheckStatus.SKIP, message="no time dimension")

    spatial_dims = [d for d in da.dims if d != "time"]
    time_nan_fracs = np.isnan(da.values).mean(axis=tuple(da.dims.index(d) for d in spatial_dims))
    sporadic_mask = (time_nan_fracs > 0) & (time_nan_fracs < 1.0)
    sporadic_count = int(sporadic_mask.sum())

    detail: dict[str, Any] = {"sporadic_nan_day_count": sporadic_count}
    if sporadic_count > 0:
        times = da.time.values[sporadic_mask]
        detail["sample_dates"] = [str(t)[:10] for t in times[:5]]
        return ArtifactCheckResult(
            status=CheckStatus.FAIL,
            message=f"{sporadic_count} time steps have partial NaN coverage",
            detail=detail,
        )
    return ArtifactCheckResult(status=CheckStatus.PASS, detail=detail)


def run_artifact_checks(
    da: xr.DataArray,
    *,
    gcm: str = "",
    variable: str = "",
    ensemble_member: str = "",
    scenario: str | None = None,
    stage: str = "",
) -> dict[str, ArtifactCheckResult]:
    """Run standard artifact quality checks on a computed DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Array to inspect. Must already be computed (not lazy dask).
    gcm, variable, ensemble_member, scenario, stage : str
        Run-identity fields used for context in log output.

    Returns
    -------
    dict[str, ArtifactCheckResult]
        Keys: ``"all_nan"``, ``"out_of_range"``, ``"sporadic_nan_days"``.
    """
    return {
        "all_nan": _check_all_nan(da),
        "out_of_range": _check_out_of_range(da, variable),
        "sporadic_nan_days": _check_sporadic_nan_days(da),
    }
