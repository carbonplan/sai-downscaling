"""
Quality checks for in-memory BCSD pipeline artifact arrays.

Provides :func:`run_artifact_checks` to inspect intermediate and output arrays at any
pipeline step, running NaN, out-of-range, and sporadic-NaN-days checks on a live
:class:`xarray.DataArray` before it is written to the store. Distinct from
:mod:`saidownscale.qaqc`, which operates on consolidated output datatrees.

Also provides :func:`assert_no_nans`, the hard NaN gate the pipeline itself calls at
every step where NaNs are known to cause silent corruption (issue #517). Unlike
:func:`run_artifact_checks`, which reports a status a caller may ignore, it raises
:class:`NaNCheckError` and aborts the run.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import xarray as xr

from saidownscale.qaqc import VAR_SPATIAL_RANGES
from saidownscale.validation import CheckStatus

# Leading-axis slab size for NaN scanning. A global scenario's fine-grid array is
# ~100 GB of float32 held in memory, and np.isnan() over the whole thing would
# allocate a 25 GB boolean temporary. Scanning in slabs bounds that temporary to a
# few hundred MB and yields the offending time indices as a side effect.
#
# This is why the scan is not simply `.isnull()`: the two run at the same speed, since
# both are bound by the same pass over memory, but the whole-array form asks for that
# 25 GB temporary and reports a bare count rather than which time steps to go look at.
_NAN_SCAN_BLOCK = 64

# Offending labels quoted in the error message before it switches to "(+N more)".
_MAX_REPORTED_LABELS = 5

_STATUS_SYMBOL: dict[CheckStatus, str] = {
    CheckStatus.PASS: "✓",
    CheckStatus.FAIL: "✗",
    CheckStatus.SKIP: "–",
    CheckStatus.UNKNOWN: "?",
    CheckStatus.XFAIL: "⚠",
    CheckStatus.XPASS: "⚠",
}


class NaNCheckError(RuntimeError):
    """Raised when a pipeline array holds NaNs where none are permitted.

    This is deliberately a hard error rather than a warning, because NaNs reaching the
    debiaser or the disaggregation residuals silently corrupt the published output. The
    run must stop rather than continue with degraded data (issue #517).
    """


def _format_label(label: Any) -> str:
    """Render a coordinate label compactly — dates as YYYY-MM-DD, everything else as-is."""
    if isinstance(label, np.datetime64):
        return str(label)[:10]
    return str(label)


def _align_mask_dims(where: xr.DataArray, da: xr.DataArray, *, name: str) -> xr.DataArray:
    """Transpose a mask onto ``da``'s trailing dimensions, rejecting any other layout.

    The mask is reduced to a bare numpy array before scanning, which discards dimension
    names. Shape alone cannot then tell a ``(lat, lon)`` mask from a ``(time, lat)`` one
    when the sizes happen to coincide, and numpy would broadcast the wrong axes together
    without complaint. Validating names here is what makes that silent mis-check
    impossible.
    """
    unknown = [d for d in where.dims if d not in da.dims]
    if unknown:
        raise ValueError(
            f"mask for {name!r} has dimensions {unknown} that are not in the array "
            f"being checked (dimensions {list(da.dims)})"
        )
    expected = tuple(d for d in da.dims if d in set(where.dims))
    if expected != da.dims[da.ndim - len(expected) :]:
        raise ValueError(
            f"mask for {name!r} covers {list(expected)}, which are not the trailing "
            f"dimensions of the array being checked (dimensions {list(da.dims)})"
        )
    return where.transpose(*expected) if tuple(where.dims) != expected else where


def _validate_mask_shape(mask: np.ndarray, values: np.ndarray, *, name: str) -> None:
    """Require the mask to be either the full shape or exactly trailing-aligned.

    Anything else — notably a same-ndim mask with a size-1 leading axis — would broadcast
    inconsistently across slab boundaries. Rejecting it up front turns a confusing
    mid-scan numpy error into a clear one.
    """
    if mask.shape == values.shape:
        return
    if mask.ndim < values.ndim and mask.shape == values.shape[values.ndim - mask.ndim :]:
        return
    raise ValueError(
        f"mask for {name!r} has shape {mask.shape}, which is neither the full shape "
        f"{values.shape} nor aligned with its trailing dimensions"
    )


def _scan_for_nans(
    values: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[int, int, list[int]]:
    """Count NaNs slab-by-slab along the leading axis.

    Parameters
    ----------
    values : np.ndarray
        Array to scan.
    mask : np.ndarray | None
        Boolean array selecting the cells subject to the check. A mask with fewer
        dimensions than ``values`` (the usual case: a 2-D lat/lon mask against a
        3-D time/lat/lon array) broadcasts across the leading axis.

    Returns
    -------
    tuple[int, int, list[int]]
        NaN count, number of cells checked, and the leading-axis indices holding
        at least one NaN.
    """
    if values.ndim == 0:
        is_nan = bool(np.isnan(values))
        return int(is_nan), 1, []

    n_nan = 0
    n_checked = 0
    offenders: list[int] = []
    # _validate_mask_shape has already reduced the mask to exactly one of these forms.
    full_shape_mask = mask is not None and mask.shape == values.shape

    for start in range(0, values.shape[0], _NAN_SCAN_BLOCK):
        stop = min(start + _NAN_SCAN_BLOCK, values.shape[0])
        bad = np.isnan(values[start:stop])
        if mask is None:
            n_checked += bad.size
        elif full_shape_mask:
            slab_mask = mask[start:stop]
            bad &= slab_mask
            n_checked += int(np.count_nonzero(slab_mask))
        else:
            # A lower-dimensional mask applies identically to every leading step.
            bad &= mask
            n_checked += int(np.count_nonzero(mask)) * (stop - start)
        # Reducing per step costs about three times what testing the slab does, and a
        # clean slab has nothing to report either way. Production arrays are clean end
        # to end, so skipping the reduction takes a 130 GB fine-grid scan from roughly
        # 11 s to 5 s, measured with the lat/lon mask regional runs pass for residuals_fine.
        # Global runs now scan that array unmasked (saidownscale.downscaling_utils.is_global_grid),
        # which does not change the shape of the tradeoff. n_checked is already accumulated
        # above, so skipped slabs still count toward the reported percentage.
        if not bad.any():
            continue
        per_step = bad.reshape(stop - start, -1).sum(axis=1)
        n_nan += int(per_step.sum())
        offenders.extend(start + int(i) for i in np.nonzero(per_step)[0])

    return n_nan, n_checked, offenders


def assert_no_nans(
    data: xr.DataArray | np.ndarray,
    *,
    name: str,
    where: xr.DataArray | np.ndarray | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Abort the run if ``data`` holds any NaN.

    Parameters
    ----------
    data : xr.DataArray | np.ndarray
        Array to check. Dask-backed arrays are computed; every array this is called
        on inside the pipeline is already numpy-backed, so the check is one pass
        over memory rather than a re-evaluation of a dask graph.
    name : str
        Name of the array, quoted in the error message.
    where : xr.DataArray | np.ndarray | None, optional
        Boolean mask selecting the cells subject to the check. Cells outside it are
        ignored. Used for fine-grid arrays on regional domains, where fine cells
        outside the coarse interpolation domain are legitimately NaN — see
        :func:`saidownscale.downscaling_utils.coarse_domain_mask`.
    context : dict[str, Any] | None, optional
        Run-identity fields (``gcm``, ``variable``, ``ensemble_member``, ``scenario``,
        ``stage``) appended to the error message. ``None`` values are dropped. Passed
        as a mapping rather than ``**kwargs`` so a stray key can never bind to
        ``where``.

    Raises
    ------
    NaNCheckError
        If any checked cell is NaN. There is no severity level and no way to
        downgrade this to a warning.
    """
    if isinstance(data, xr.DataArray):
        da = data.compute() if data.chunks is not None else data
        values = np.asarray(da.data)
        leading_dim = da.dims[0] if da.ndim else None
        labels = (
            np.asarray(da.coords[leading_dim].values)
            if leading_dim is not None and leading_dim in da.coords
            else None
        )
    else:
        da = None
        values = np.asarray(data)
        leading_dim = None
        labels = None

    if where is None:
        mask = None
    else:
        if isinstance(where, xr.DataArray) and da is not None:
            where = _align_mask_dims(where, da, name=name)
        mask = np.asarray(where.values if isinstance(where, xr.DataArray) else where, dtype=bool)
        _validate_mask_shape(mask, values, name=name)

    n_nan, n_checked, offenders = _scan_for_nans(values, mask)
    if n_nan == 0:
        return

    percent = 100.0 * n_nan / n_checked if n_checked else 100.0
    lines = [
        f"{name!r} contains {n_nan} NaN values "
        f"({percent:.2f}% of {n_checked} checked cells). "
        "NaNs are not permitted here — see issue #517."
    ]
    if offenders:
        shown = offenders[:_MAX_REPORTED_LABELS]
        rendered = [_format_label(labels[i]) if labels is not None else str(i) for i in shown]
        line = f"  affected {leading_dim if labels is not None else 'index'}: " + ", ".join(
            rendered
        )
        if len(offenders) > len(shown):
            line += f" (+{len(offenders) - len(shown)} more)"
        lines.append(line)
    described = {k: v for k, v in (context or {}).items() if v is not None}
    if described:
        lines.append("  context: " + " ".join(f"{k}={v}" for k, v in described.items()))

    raise NaNCheckError("\n".join(lines))


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
