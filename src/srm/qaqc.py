"""
QA/QC checks for consolidated BCSD output stores.

Runs spatial, temporal, and physical-constraint checks against merged output
datatrees, including NaN detection, range validation, and temperature monotonicity.
Distinct from :mod:`srm.qa_checks`, which operates on in-memory arrays during
pipeline execution.
"""

from __future__ import annotations

import contextlib
import io
import itertools
from collections.abc import Callable, Iterable
from pathlib import Path

import cf_xarray  # noqa: F401  # registers CF accessor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from srm import catalog
from srm.config import GROUP_TO_SCENARIO
from srm.lineage import resolve_member_lineage

# Spatial range bounds for unit-mismatch detection (e.g. Celsius instead of Kelvin).
# Ranges are wide intentionally — based on ERA5 observed range +/- large margins.
_N_TIME_SAMPLES = 15  # window size used by multi-step identity/spread checks


def _sample_time_window(n: int, k: int = _N_TIME_SAMPLES) -> slice:
    """Contiguous k-step window centered in [0, n), avoiding day 0.

    Using a contiguous window minimises zarr chunk access — at most two time
    chunks are fetched instead of up to k chunks from scattered random indices.
    """
    if n <= k:
        return slice(0, n)
    start = max(1, (n - k) // 2)
    return slice(start, start + k)


VAR_SPATIAL_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "tas": {"min": (100, 400), "max": (100, 400)},
    "tasmin": {"min": (100, 400), "max": (100, 400)},
    "tasmax": {"min": (100, 400), "max": (100, 400)},
    "pr": {"min": (0, 1e-7), "max": (0.0001, 0.03)},
    "rsds": {"min": (-1, 100), "max": (100, 1200)},
    "hurs": {"min": (0, 40), "max": (40, 900)},
    "dtr": {"min": (0, 10), "max": (10, 150)},
}

# --- Check-table highlighting -------------------------------------------------------------------
# Per-column definition of what counts as a problem in a QA check table, applied by `highlight`.
# Column names are shared across the QA notebooks because they name the same checks. A column in
# neither registry is left unstyled, so adding a column to a check without registering it here gets
# no color rather than the wrong one.

_STYLE_FAIL = "background-color: #f8d7da; color: #842029"
_STYLE_WARN = "background-color: #fff3cd; color: #664d03"
_STYLE_OK = "background-color: #d1e7dd; color: #0f5132"


def _is_false(col: pd.Series) -> pd.Series:
    return ~col.astype(bool)


def _is_true(col: pd.Series) -> pd.Series:
    return col.astype(bool)


def _is_nonzero(col: pd.Series) -> pd.Series:
    return col != 0


def _is_nonempty(col: pd.Series) -> pd.Series:
    return col.str.len() > 0


def _is_not_one(col: pd.Series) -> pd.Series:
    return col != 1


#: Check columns whose failure condition is a defect to act on.
CHECK_RED: dict[str, Callable[[pd.Series], pd.Series]] = {
    "pass": _is_false,
    "no_nans": _is_false,
    "no_interior_nans": _is_false,
    "no_duplicate_timesteps": _is_false,
    "no_all_zero_days": _is_false,
    "reasonable_range": _is_false,
    "within_input_time_bounds": _is_false,
    "regular_time_axis": _is_false,
    "bridged_pre_sai": _is_true,
    "stale_end_tail": _is_true,
    "missing": _is_nonempty,
    "unexpected": _is_nonempty,
    "unchecked": _is_nonempty,
    "duplicate_pairs": _is_nonempty,
    "all_zero_days": _is_nonempty,
    "n_distinct_grids": _is_not_one,
    "days_max<min": _is_nonzero,
    "neg_cell_days": _is_nonzero,
    "cells_with_neg": _is_nonzero,
    "frac_cells_neg": _is_nonzero,
    "high_cell_days": _is_nonzero,
    "cells_with_high": _is_nonzero,
}

#: Check columns reported for inspection rather than failed, so they color amber rather than red.
CHECK_AMBER: dict[str, Callable[[pd.Series], pd.Series]] = {
    "n_irregular_days": _is_nonzero,
}


def highlight(df: pd.DataFrame, *, demote: Iterable[str] = ()) -> pd.io.formats.style.Styler:
    """Color a QA check table so failing cells stand out.

    Every column in :data:`CHECK_RED` or :data:`CHECK_AMBER` is colored green where its condition
    holds and red (or amber) where it does not. Unregistered columns are left alone.

    Parameters
    ----------
    df : pandas.DataFrame
        A check table. Its index and columns must both be unique, which ``Styler.apply`` requires.
    demote : iterable of str, optional
        Registered red columns to color amber instead, for a notebook where that column's failure
        is structural rather than a defect. On a regional subset, for instance, every fine-grid leaf
        fails ``no_nans`` on the interpolation border left by downscaling.

    Returns
    -------
    pandas.io.formats.style.Styler
        Styler over ``df``, ready to ``display``.

    Raises
    ------
    KeyError
        If ``demote`` names a column absent from :data:`CHECK_RED`, so a typo cannot silently leave
        that column red.
    """
    demoted = set(demote)
    if unknown := demoted - set(CHECK_RED):
        raise KeyError(f"cannot demote unregistered columns: {sorted(unknown)}")
    red = {name: rule for name, rule in CHECK_RED.items() if name not in demoted}
    amber = CHECK_AMBER | {name: CHECK_RED[name] for name in demoted}

    def paint(col: pd.Series) -> list[str]:
        name = str(col.name)  # Series.name is Hashable; check-table columns are always strings
        for rules, style in ((red, _STYLE_FAIL), (amber, _STYLE_WARN)):
            if name in rules:
                return np.where(rules[name](col), style, _STYLE_OK).tolist()
        return [""] * len(col)

    return df.style.apply(paint)


class ValidationResult:
    def __init__(self, is_valid: bool, issues: list[str]):
        self.is_valid = is_valid
        self.issues = issues

    def __bool__(self):
        return self.is_valid

    def __repr__(self):
        status = "valid" if self.is_valid else "invalid"
        if self.issues:
            return f"{status}: {', '.join(self.issues)}"
        return status


def check_nans(ds):
    return ds.isnull().sum().values


def outlandishly_high_precip(ds):
    # highest ever recorded daily precipitation value was 1825 mm/day
    # on Reunion in 1966 according to https://www.weather.gov/owp/hdsc_world_record
    # so we'll say anything over 2000 is outlandishly high
    outlandishly_high_threshold = 2000
    return (ds["pr"] > outlandishly_high_threshold).sum().values


def outlandishly_high_temp(da):
    # highest ever recorded temperature value was 56.7 at Furnace Creek, CA in 1913
    # according to https://wmo.int/sites/default/files/2025-07/Table_Records_25Jul2025.pdf
    # so we'll say anything over 65 degC is outlandishly high
    outlandishly_high_threshold = 65 + 273.15  # convert to kelvin
    return (da > outlandishly_high_threshold).sum().values


def outlandishly_low_temp(da):
    # highest ever recorded temperature value was -89.2 at Vostok, Antarctica in 1983
    # according to https://wmo.int/sites/default/files/2025-07/Table_Records_25Jul2025.pdf
    # so we'll say anything under -100 is outlandishly low
    outlandishly_low_threshold = -100 + 273.15  # convert to kelvin
    return (da < outlandishly_low_threshold).sum().values


def negative_precip(ds):
    return (ds["pr"] < 0).sum().values


def check_temperature_monotonic(ds):
    min_exceeds_mean = (ds["tasmin"] > ds["tas"]).sum().values
    mean_exceeds_max = (ds["tas"] > ds["tasmax"]).sum().values
    min_exceeds_max = (ds["tasmin"] > ds["tasmax"]).sum().values
    return min_exceeds_mean, mean_exceeds_max, min_exceeds_max


def check_physical_constraints(ds):
    print(f"Number of negative precipitation values: {negative_precip(ds)}")
    print(f"Number of outlandishly high precipitation values: {outlandishly_high_precip(ds)}")
    print(f"Number of outlandishly high tas values: {outlandishly_high_temp(ds['tas'])}")
    print(f"Number of outlandishly high tasmax values: {outlandishly_high_temp(ds['tasmax'])}")
    print(f"Number of outlandishly high tasmin values: {outlandishly_high_temp(ds['tasmin'])}")
    print(f"Number of outlandishly low tas values: {outlandishly_low_temp(ds['tas'])}")
    print(f"Number of outlandishly low tasmax values: {outlandishly_low_temp(ds['tasmax'])}")
    print(f"Number of outlandishly low tasmin values: {outlandishly_low_temp(ds['tasmin'])}")
    min_exceeds_mean, mean_exceeds_max, min_exceeds_max = check_temperature_monotonic(ds)
    print(f"Number of times tasmin exceeds tas: {min_exceeds_mean}")
    print(f"Number of times tas exceeds tasmax: {mean_exceeds_max}")
    print(f"Number of times tasmin exceeds tasmax: {min_exceeds_max}")


def confirm_coords(ds, obs_dataset: str = "ERA5"):
    obs = catalog.get(obs_dataset).to_xarray()
    xr.testing.assert_equal(obs[["lat", "lon"]].coords, ds[["lat", "lon"]].coords)
    # TODO: add in the expected time coordinates
    return "Latitude and longitude match expectation"


class DatasetChecker:
    def __init__(self, ds_info):
        if isinstance(ds_info, xr.Dataset):
            self.ds_info = None
            self.ds = ds_info
        else:
            self.ds_info = ds_info
            self.ds = ds_info.to_xarray()

    def _validate_coord(
        self,
        cf_key: str,
        expected_name: str,
        expected_range: tuple[float, float],
        check_monotonic: bool,
    ) -> ValidationResult:
        issues = []
        try:
            coord = self.ds[cf_key]
            coord_name = coord.name
        except KeyError:
            return ValidationResult(False, [f"no {expected_name} coord '{cf_key}' found"])

        if coord_name != expected_name:
            issues.append(f"{expected_name} name is '{coord_name}', expected '{expected_name}'")

        coord_min = float(coord.min())
        coord_max = float(coord.max())

        if coord_min < expected_range[0] or coord_max > expected_range[1]:
            issues.append(
                f"{expected_name} range [{coord_min}, {coord_max}] outside {expected_range}"
            )

        if check_monotonic:
            if not self.ds.indexes[cf_key].is_monotonic_increasing:
                issues.append(f"{expected_name} is not monotonically increasing")

        return ValidationResult(len(issues) == 0, issues)

    def _resolve_ensemble_member(self, obj):
        """Return first ensemble_member slice where no variable/value is entirely null.

        Checks one member at a time and stops as soon as a non-null member is found,
        minimising S3 reads for the common case where member 0 has data.

        The null probe uses a single time step (``isel(time=0)``) rather than the full
        multi-step sample passed in ``obj``. Reading all time indices just to detect
        null members fetches O(N_TIME_SAMPLES × N_vars) chunks from S3 unnecessarily;
        a single time step is sufficient to identify an entirely-null member.
        """
        if "ensemble_member" not in getattr(obj, "dims", {}):
            return obj

        # Probe with one time step to avoid reading the full scattered sample from S3.
        probe = obj.isel(time=0) if "time" in getattr(obj, "dims", {}) else obj
        spatial_dims = [d for d in probe.dims if d != "ensemble_member"]

        for i in range(obj.sizes["ensemble_member"]):
            member_probe = probe.isel(ensemble_member=i)
            if isinstance(member_probe, xr.Dataset):
                if all(
                    not bool(member_probe[v].isnull().all(dim=spatial_dims).compute())
                    for v in member_probe.data_vars
                ):
                    return obj.isel(ensemble_member=i)
            else:
                if not bool(member_probe.isnull().all().compute()):
                    return obj.isel(ensemble_member=i)

        return None

    def validate_lon(
        self,
        expected_range: tuple[float, float] = (-180, 180),
        check_monotonic: bool = False,
    ) -> ValidationResult:
        return self._validate_coord("lon", "lon", expected_range, check_monotonic)

    def validate_lat(
        self,
        expected_range: tuple[float, float] = (-90, 90),
        check_monotonic: bool = False,
    ) -> ValidationResult:
        return self._validate_coord("lat", "lat", expected_range, check_monotonic)

    def validate_expected_variables(self) -> ValidationResult:
        if self.ds_info is None or not self.ds_info.expected_vars:
            return ValidationResult(True, [])
        actual_names = set(self.ds.data_vars)
        expected_names = {spec.name for spec in self.ds_info.expected_vars}
        missing_names = expected_names - actual_names
        if missing_names:
            return ValidationResult(
                False,
                [f"Missing variables: {sorted(missing_names)}. Found: {sorted(actual_names)}"],
            )
        return ValidationResult(True, [])

    def validate_units(self) -> ValidationResult:
        if self.ds_info is None or not self.ds_info.expected_vars:
            return ValidationResult(True, [])
        issues = []
        for spec in self.ds_info.expected_vars:
            if spec.name in self.ds:
                actual_unit = self.ds[spec.name].attrs.get("units")
                if actual_unit != spec.units:
                    issues.append(
                        f"Variable '{spec.name}' has units '{actual_unit}', expected '{spec.units}'"
                    )
        return ValidationResult(len(issues) == 0, issues)

    def validate_time_axis(self) -> ValidationResult:
        if "time" not in self.ds.dims:
            return ValidationResult(False, ["Dataset has no 'time' dimension"])

        time_vals = self.ds.time.values
        if len(time_vals) < 2:
            return ValidationResult(True, [])

        issues = []
        if not self.ds.indexes["time"].is_monotonic_increasing:
            issues.append("time axis is not monotonically increasing")
        diffs = np.diff(time_vals).astype("timedelta64[D]").astype(int)
        n_gaps = int((diffs > 1).sum())
        if n_gaps > 0:
            issues.append(f"{n_gaps} internal gap(s) > 1 day in time axis")
        return ValidationResult(len(issues) == 0, issues)

    def validate_calendar(self) -> ValidationResult:
        issues = []
        if "time" not in self.ds.dims:
            return ValidationResult(False, ["Dataset has no 'time' dimension"])

        calendar = self.ds.time.encoding.get("calendar")
        if calendar is None:
            issues.append("time coord missing 'calendar' in encoding")
        elif calendar != "proleptic_gregorian":
            issues.append(f"calendar is '{calendar}', expected 'proleptic_gregorian'")

        if not np.issubdtype(self.ds.time.dtype, np.datetime64):
            issues.append(f"time dtype is {self.ds.time.dtype}, expected datetime64")

        return ValidationResult(len(issues) == 0, issues)

    def validate_negative_precip(self, isel_kwargs: dict | None = None) -> ValidationResult:
        if "pr" not in self.ds:
            return ValidationResult(True, [])
        da = self.ds["pr"].isel(**isel_kwargs) if isel_kwargs else self.ds["pr"]
        count = int((da < 0).sum().compute())
        if count > 0:
            return ValidationResult(False, [f"Found {count} negative precipitation value(s)"])
        return ValidationResult(True, [])

    def validate_spatial_range(self, var: str, isel_kwargs: dict | None = None) -> ValidationResult:
        if var not in self.ds:
            return ValidationResult(True, [])
        if var not in VAR_SPATIAL_RANGES:
            return ValidationResult(True, [])

        da = self.ds[var].isel(**isel_kwargs) if isel_kwargs else self.ds[var]
        spatial_min = float(da.min().compute())
        spatial_max = float(da.max().compute())

        if np.isnan(spatial_min) or np.isnan(spatial_max):
            return ValidationResult(True, [])

        min_lo, min_hi = VAR_SPATIAL_RANGES[var]["min"]
        max_lo, max_hi = VAR_SPATIAL_RANGES[var]["max"]

        issues = []
        if not (min_lo <= spatial_min <= min_hi):
            issues.append(
                f"{var} spatial min {spatial_min:.4g} outside expected [{min_lo}, {min_hi}]"
            )
        if not (max_lo <= spatial_max <= max_hi):
            issues.append(
                f"{var} spatial max {spatial_max:.4g} outside expected [{max_lo}, {max_hi}]"
            )
        return ValidationResult(len(issues) == 0, issues)

    def validate_dtr_consistency(
        self, isel_kwargs: dict | None = None, atol: float = 0.5
    ) -> ValidationResult:
        required = {"dtr", "tasmax", "tasmin"}
        if not required.issubset(self.ds.data_vars):
            return ValidationResult(True, [])

        subset = (
            self.ds[list(required)].isel(**isel_kwargs) if isel_kwargs else self.ds[list(required)]
        )
        expected_dtr = (subset["tasmax"] - subset["tasmin"]).compute()
        actual_dtr = subset["dtr"].compute()
        max_diff = float(abs(actual_dtr - expected_dtr).max())

        if max_diff > atol:
            return ValidationResult(
                False,
                [f"dtr deviates from tasmax-tasmin by up to {max_diff:.4g} K (atol={atol})"],
            )
        return ValidationResult(True, [])

    def validate_temp_consistency(self, isel_kwargs: dict | None = None) -> ValidationResult:
        required = {"tas", "tasmin", "tasmax"}
        if not required.issubset(self.ds.data_vars):
            return ValidationResult(True, [])

        subset = (
            self.ds[list(required)].isel(**isel_kwargs) if isel_kwargs else self.ds[list(required)]
        )
        # Reuse module-level predicate.
        min_gt_mean, mean_gt_max, min_gt_max = check_temperature_monotonic(subset)

        issues = []
        if min_gt_mean > 0:
            issues.append(f"tas < tasmin at {min_gt_mean} grid point(s)")
        if mean_gt_max > 0:
            issues.append(f"tasmax < tas at {mean_gt_max} grid point(s)")
        if min_gt_max > 0:
            issues.append(f"tasmax < tasmin at {min_gt_max} grid point(s)")
        return ValidationResult(len(issues) == 0, issues)

    def validate_tasmax_ge_tasmin(self, isel_kwargs: dict | None = None) -> ValidationResult:
        """Blocking cross-variable gate: ``tasmax >= tasmin`` everywhere (issue #331).

        Unlike :meth:`validate_temp_consistency` this needs only ``tasmax`` and
        ``tasmin`` (not ``tas``), so it also covers the temperature-extremes-only
        outputs. NaN-safe (NaN comparisons are False). A nonzero count means the
        reconcile step did not land — the run must not ship.
        """
        required = {"tasmin", "tasmax"}
        if not required.issubset(self.ds.data_vars):
            return ValidationResult(True, [])
        subset = (
            self.ds[list(required)].isel(**isel_kwargs) if isel_kwargs else self.ds[list(required)]
        )
        n = int((subset["tasmax"] < subset["tasmin"]).sum().compute())
        if n > 0:
            return ValidationResult(False, [f"tasmax < tasmin at {n} grid point(s)"])
        return ValidationResult(True, [])

    def validate_no_identical_vars(self) -> ValidationResult:
        if "time" not in self.ds.dims or self.ds.sizes["time"] == 0:
            return ValidationResult(True, [])

        n = self.ds.sizes["time"]
        # Start at day 0.  CESM2-WACCM, #424 give an invalid first day (tasmax == tasmin == tas).
        window = slice(0, min(n, _N_TIME_SAMPLES))
        ds_sample = self.ds.isel(time=window)

        if "ensemble_member" in ds_sample.dims:
            ds_sample = self._resolve_ensemble_member(ds_sample)
            if ds_sample is None:
                return ValidationResult(True, [])

        # One compute call loads the contiguous window for all variables at once.
        ds_computed = ds_sample.compute()
        n_steps = ds_computed.sizes["time"]

        var_names = list(ds_computed.data_vars)
        issues = []
        for v1, v2 in itertools.combinations(var_names, 2):
            a = ds_computed[v1]
            b = ds_computed[v2]
            # Both all-NaN across the window → fill-data equality, not a real data bug.
            if a.isnull().all() and b.isnull().all():
                continue
            if a.equals(b):
                issues.append(
                    f"{v1} and {v2} are identical across {n_steps} consecutive time steps"
                )
        return ValidationResult(len(issues) == 0, issues)

    def validate_ensemble_member_dim(self) -> ValidationResult:
        if "ensemble_member" in self.ds.dims and self.ds.sizes["ensemble_member"] >= 1:
            return ValidationResult(True, [])
        return ValidationResult(False, ["ensemble_member dimension missing or empty"])

    def validate_ensemble_spread(self, var: str = "tas") -> ValidationResult:
        if var not in self.ds:
            return ValidationResult(True, [])
        if "ensemble_member" not in self.ds.dims:
            return ValidationResult(True, [])
        if self.ds.sizes["ensemble_member"] < 2:
            return ValidationResult(True, [])
        if "time" not in self.ds.dims or self.ds.sizes["time"] == 0:
            return ValidationResult(True, [])

        n = self.ds.sizes["time"]
        window = _sample_time_window(n)
        # One compute: global mean for all members × the contiguous time window.
        means = self.ds[var].isel(time=window).mean(dim=["lat", "lon"]).compute()
        n_steps = means.sizes["time"]

        # A member pair is flagged only when their global means are equal across every
        # step in the window (and neither series contains NaN).
        n_members = self.ds.sizes["ensemble_member"]
        issues = []
        for i, j in itertools.combinations(range(n_members), 2):
            mi = means.isel(ensemble_member=i).values
            mj = means.isel(ensemble_member=j).values
            if np.any(np.isnan(mi)) or np.any(np.isnan(mj)):
                continue
            if np.all(mi == mj):
                issues.append(
                    f"{var} global mean identical for members "
                    f"{means.ensemble_member.values[i]} and {means.ensemble_member.values[j]} "
                    f"across {n_steps} consecutive time steps"
                )
        return ValidationResult(len(issues) == 0, issues)


# ---------------------------------------------------------------------------
# Module-level helper functions (promoted from notebook helpers)
# ---------------------------------------------------------------------------


def point_missingness(ds: xr.Dataset, var: str) -> xr.DataArray:
    """Return a boolean DataArray marking NaN positions at a single grid point.

    Selects the grid point nearest to (lat=0, lon=0). Result has shape (time,) or
    (time, ensemble_member) depending on the dataset structure.
    """
    return ds[var].sel(lat=0, lon=0, method="nearest").isnull()


def summarize_time_coverage(ds: xr.Dataset, label: str) -> dict:
    """Return a summary dict of the time axis for DataFrame display.

    Keys: source, start, end, n_times, n_missing, n_duplicates, monotonic.
    n_missing is computed against an expected gap-free daily index from
    the first to last observed time step.
    """

    time = ds.indexes["time"]
    expected = pd.date_range(str(time[0])[:10], str(time[-1])[:10], freq="D")
    dt_index = pd.DatetimeIndex([str(t)[:10] for t in time])
    return {
        "source": label,
        "start": str(time[0])[:10],
        "end": str(time[-1])[:10],
        "n_times": len(time),
        "n_missing": max(0, len(expected) - len(time)),
        "n_duplicates": int(dt_index.duplicated().sum()),
        "monotonic": bool(time.is_monotonic_increasing),
    }


def summarize_grid(ds: xr.Dataset, label: str) -> dict:
    """Return a summary dict of the lat/lon grid for DataFrame display.

    Keys: source, lat_min, lat_max, lon_min, lon_max, n_lat, n_lon,
    lat_res, lon_res, lat_monotonic, lon_monotonic, lat_range_ok, lon_range_ok.
    """

    lat = ds["lat"].values
    lon = ds["lon"].values
    return {
        "source": label,
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lon_min": float(lon.min()),
        "lon_max": float(lon.max()),
        "n_lat": len(lat),
        "n_lon": len(lon),
        "lat_res": float(np.diff(lat).mean()) if len(lat) > 1 else float("nan"),
        "lon_res": float(np.diff(lon).mean()) if len(lon) > 1 else float("nan"),
        "lat_monotonic": bool(pd.Index(lat).is_monotonic_increasing),
        "lon_monotonic": bool(pd.Index(lon).is_monotonic_increasing),
        "lat_range_ok": bool(lat.min() >= -90 and lat.max() <= 90),
        "lon_range_ok": bool(lon.min() >= -180 and lon.max() <= 180),
    }


def check_units_and_range(
    ds: xr.Dataset, label: str, *, isel_kwargs: dict | None = None
) -> list[dict]:
    """Return per-variable unit and spatial-range check rows for DataFrame display.

    For each variable in VAR_SPATIAL_RANGES that is present in ds, records the declared
    units attribute, the actual min/max, and whether those values fall within the expected
    ranges. By default checks the full dataset; pass ``isel_kwargs`` (e.g.
    ``{"time": 1}``) to restrict to a subset for faster spot-checks.

    Keys per row: source, variable, units, min, max, min_ok, max_ok.
    """
    rows = []
    for var, ranges in VAR_SPATIAL_RANGES.items():
        if var not in ds:
            continue
        da = ds[var].isel(**isel_kwargs) if isel_kwargs else ds[var]
        da = da.compute()
        actual_min = float(da.min())
        actual_max = float(da.max())
        min_lo, min_hi = ranges["min"]
        max_lo, max_hi = ranges["max"]
        rows.append(
            {
                "source": label,
                "variable": var,
                "units": ds[var].attrs.get("units"),
                "min": actual_min,
                "max": actual_max,
                "min_ok": bool(min_lo <= actual_min <= min_hi),
                "max_ok": bool(max_lo <= actual_max <= max_hi),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Ensemble-spread check (#316)
# ---------------------------------------------------------------------------
#
# Stitched groups put the gap-filled block at the front of the time axis, so day
# 0 lands in it. MIROC-ES2H `ssp245` fills 2015-2019 from ESGF through a 10 -> 3
# member map, making r01/r04/r07/r10 identical there by construction.
#
# Mid-record sampling would dodge that silently and never check 2015-2019, which
# feeds the training period. So split on the gap-fill attrs instead: check the
# native window for distinctness, the bridge window for the duplication
# `gap_fill_member_map` predicts.

# Stores written before the ETL switched glyphs still carry "→"; matching one
# glyph alone would find no pairs and report a vacuous pass.
_GAP_FILL_ARROWS = ("→", "->")


def _parse_gap_fill_member_map(raw: str) -> dict[str, str]:
    """Parse a ``gap_fill_member_map`` attr into ``{member: source_member}``.

    Parameters
    ----------
    raw : str
        Comma-separated ``member->source`` pairs. Either arrow glyph is accepted.

    Returns
    -------
    dict of str to str

    Raises
    ------
    ValueError
        If an entry has no recognized arrow, or nothing parses. Raising keeps an
        unreadable map from passing as a validated one.
    """
    mapping: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        for arrow in _GAP_FILL_ARROWS:
            if arrow in entry:
                member, source = entry.split(arrow, 1)
                mapping[member.strip()] = source.strip()
                break
        else:
            raise ValueError(
                f"gap_fill_member_map entry {entry!r} has no recognized arrow "
                f"(expected one of {_GAP_FILL_ARROWS})"
            )
    if not mapping:
        raise ValueError(f"gap_fill_member_map {raw!r} parsed to no member pairs")
    return mapping


def _gap_fill_boundary(ds: xr.Dataset, gap_fill_period: str) -> int | None:
    """Return the first time index after the gap-filled block.

    ``gap_fill_period`` is the ETL's ``"<start>-<end>"`` year range (e.g.
    ``"2015-2019"``). None when nothing follows the bridge.
    """
    end_year = int(gap_fill_period.split("-")[-1])
    native = np.asarray(ds.time.dt.year.values) > end_year
    return int(np.argmax(native)) if native.any() else None


def _spread_row(
    ds: xr.Dataset,
    label: str,
    var: str,
    day_index: int,
    window: str,
    check: str,
    expected_map: dict[str, str] | None = None,
) -> dict:
    """Compute one ensemble-spread row from the per-member global mean on a day."""
    means_da = ds[var].isel(time=day_index).mean(dim=["lat", "lon"]).compute()
    members = [str(m) for m in means_da.ensemble_member.values]
    means = [float(v) for v in means_da.values]

    observed = _group_members_by_mean(members, means)
    if expected_map is None:
        ok = len(set(means)) == len(means)
        expected_groups = None
    else:
        # Compare the whole partition, not just the duplicate count: that is what
        # catches a mis-wired map.
        expected: dict[str, list[str]] = {}
        for member in members:
            expected.setdefault(expected_map.get(member, member), []).append(member)
        expected_groups = _format_groups(expected.values())
        ok = {frozenset(g) for g in expected.values()} == {frozenset(g) for g in observed.values()}

    return {
        "source": label,
        "variable": var,
        "window": window,
        "check": check,
        "day": str(ds.time.values[day_index])[:10],
        "n_members": len(members),
        "n_distinct": len(set(means)),
        "ok": ok,
        "members": members,
        "means": means,
        "groups": _format_groups(observed.values()),
        "expected_groups": expected_groups,
    }


def _group_members_by_mean(members: list[str], means: list[float]) -> dict[float, list[str]]:
    """Group member labels by their exact global-mean value."""
    groups: dict[float, list[str]] = {}
    for member, mean in zip(members, means, strict=True):
        groups.setdefault(mean, []).append(member)
    return groups


def _format_groups(groups) -> str:
    """Render member groupings as a stable string for table display."""
    return " | ".join("+".join(sorted(g)) for g in sorted(groups, key=lambda g: sorted(g)[0]))


def check_ensemble_spread(
    ds: xr.Dataset, label: str, var: str = "tas", day_index: int = 0
) -> list[dict]:
    """Return ensemble-spread summary rows for DataFrame/plot display (#316).

    Takes the global spatial mean of `var` per ensemble member on one day. Groups
    carrying the ETL's ``gap_fill_period``/``gap_fill_member_map`` attrs return two
    rows, ``bridge`` and ``native``; all others return a single ``full`` row.

    Parameters
    ----------
    ds : xarray.Dataset
        Group dataset, with `ensemble_member`, `time`, `lat` and `lon` dims.
    label : str
        Row label, e.g. ``"MIROC-ES2H ssp245"``.
    var : str, default "tas"
        Variable to take the global mean of.
    day_index : int, default 0
        Offset into each window, reported back in the ``day`` column.

    Returns
    -------
    list of dict
        Keys ``source``, ``variable``, ``window``, ``check``, ``day``,
        ``n_members``, ``n_distinct``, ``ok``, ``members``, ``means``, ``groups``,
        ``expected_groups``. ``ok`` means all members distinct on ``full``/``native``
        rows, and duplicates matching ``gap_fill_member_map`` on ``bridge`` rows.
        A single row with empty members/means when the dataset has no
        `ensemble_member` dim or lacks `var`.
    """
    if var not in ds or "ensemble_member" not in ds.dims:
        return [
            {
                "source": label,
                "variable": var,
                "window": "full",
                "check": "not applicable",
                "day": None,
                "n_members": 0,
                "n_distinct": 0,
                "ok": True,
                "members": [],
                "means": [],
                "groups": "",
                "expected_groups": None,
            }
        ]

    gap_fill_period = ds.attrs.get("gap_fill_period")
    boundary = _gap_fill_boundary(ds, gap_fill_period) if gap_fill_period else None
    if boundary is None:
        return [_spread_row(ds, label, var, day_index, "full", "all members distinct")]

    expected_map = _parse_gap_fill_member_map(ds.attrs["gap_fill_member_map"])
    return [
        _spread_row(
            ds,
            label,
            var,
            day_index,
            "bridge",
            f"duplicates match gap_fill_member_map ({gap_fill_period})",
            expected_map=expected_map,
        ),
        _spread_row(ds, label, var, boundary + day_index, "native", "all members distinct"),
    ]


def disagg_test_calculate_metrics(x, y, time_dim="time"):
    # Align on time so positional pairing can't silently drift
    x, y = xr.align(x, y, join="inner")

    # Only count cells/times where BOTH are finite, so every metric uses the same n
    good = x.notnull() & y.notnull()
    x = x.where(good)
    y = y.where(good)

    resid = y - x  # deviation from 1:1 line

    # n = good.sum(time_dim)
    bias = resid.mean(time_dim)
    mae = np.abs(resid).mean(time_dim)
    rmse = np.sqrt((resid**2).mean(time_dim))
    max_dev = np.abs(resid).max(time_dim)
    std_resid = resid.std(time_dim)
    # rmse_perp = rmse / np.sqrt(2)

    # R² vs 1:1 (Nash–Sutcliffe): 1 = perfect, can go negative
    # ss_tot uses x (the reference/observations), not y (the model) -- NSE measures
    # how much of the *true* variability the model explains, not the model's own variance.
    ss_res = (resid**2).sum(time_dim)
    ss_tot = ((x - x.mean(time_dim)) ** 2).sum(time_dim)
    r2_oneone = 1 - ss_res / ss_tot

    # Kling-Gupta Efficiency (Gupta et al. 2009): decomposes skill into
    # correlation (r), variability ratio (alpha), and bias ratio (beta), so
    # errors from timing/pattern, spread, and mean bias can be told apart
    # instead of collapsing into one NSE number. 1 = perfect.
    kge_r = xr.corr(x, y, dim=time_dim)
    kge_alpha = y.std(time_dim) / x.std(time_dim)
    kge_beta = y.mean(time_dim) / x.mean(time_dim)
    kge = 1 - np.sqrt((kge_r - 1) ** 2 + (kge_alpha - 1) ** 2 + (kge_beta - 1) ** 2)

    metrics = xr.Dataset(
        {
            "bias": bias,
            "mae": mae,
            "rmse": rmse,
            "max_dev": max_dev,
            "std_resid": std_resid,
            "r2_oneone": r2_oneone,
            "kge": kge,
            "kge_r": kge_r,
            "kge_alpha": kge_alpha,
            "kge_beta": kge_beta,
        }
    )

    return metrics


def disagg_test_plot_summary_stats(
    metrics,
    vmax_rmse=None,
    vmax_bias=None,
    vmax_std_resid=None,
    vmax_mae=None,
    vmax_max_dev=None,
    savefig_path=None,
):
    nrows = 2
    ncols = 3

    plt.figure(figsize=(20, 12))

    plt.subplot(nrows, ncols, 1)
    if vmax_rmse is None:
        metrics["rmse"].plot(vmin=0)
    else:
        metrics["rmse"].plot(vmin=0, vmax=vmax_rmse)
    plt.title("RMSE")

    plt.subplot(nrows, ncols, 2)
    if vmax_bias is None:
        metrics["bias"].plot()
    else:
        metrics["bias"].plot(vmax=vmax_bias, vmin=-vmax_bias, cmap=plt.cm.RdBu_r)
    plt.title("Bias relative to coarse debiased \n (goal: bias=0)")

    plt.subplot(nrows, ncols, 3)
    if vmax_std_resid is None:
        metrics["std_resid"].plot(vmin=0)
    else:
        metrics["std_resid"].plot(vmin=0, vmax=vmax_std_resid)
    plt.title("Std residual")

    plt.subplot(nrows, ncols, 4)
    if vmax_mae is None:
        metrics["mae"].plot(vmin=0)
    else:
        metrics["mae"].plot(vmin=0, vmax=vmax_mae)
    plt.title("MAE")

    plt.subplot(nrows, ncols, 5)
    if vmax_max_dev is None:
        metrics["max_dev"].plot(vmin=0)
    else:
        metrics["max_dev"].plot(vmin=0, vmax=vmax_max_dev)
    plt.title("Maximum deviation")

    plt.subplot(nrows, ncols, 6)
    metrics["r2_oneone"].plot(vmin=0.9, vmax=1, cmap=plt.cm.viridis_r)
    plt.title("R2 relative to 1:1 line")

    plt.tight_layout()
    if savefig_path is not None:
        plt.savefig(savefig_path)


def disagg_test_print_evaluation_for_metric(
    metric, metrics_to_evaluate, metric_max_thresh=None, metric_min_thresh=None
):
    print("---------------" + metric + "---------------")
    print("Max:")
    print(np.nanmax(metrics_to_evaluate[metric]))
    print("Min:")
    print(np.nanmin(metrics_to_evaluate[metric]))
    if metric_max_thresh is not None:
        print("Fraction above threshold:")
        print((metrics_to_evaluate[metric] > metric_max_thresh).mean(dim=["lat", "lon"]).values)
    if metric_min_thresh is not None:
        print("Fraction below threshold:")
        print((metrics_to_evaluate[metric] < metric_min_thresh).mean(dim=["lat", "lon"]).values)


def disagg_test_print_all_evaluation_metrics(
    metrics,
    variable,
    scenario,
    ensemble_member,
    timescale,
    disagg_eval_tresholds,
    is_regional_subset=True,
    log_path=None,
):
    thresholds = disagg_eval_tresholds[variable]
    if is_regional_subset:
        metrics_to_evaluate = metrics.isel(lat=slice(1, -1), lon=slice(1, -1))
    else:
        metrics_to_evaluate = metrics

    rows = []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for metric, limits in thresholds.items():
            disagg_test_print_evaluation_for_metric(
                metric=metric,
                metrics_to_evaluate=metrics_to_evaluate,
                metric_max_thresh=limits["eval_max"],
                metric_min_thresh=limits["eval_min"],
            )
            rows.append(
                {
                    "variable": variable,
                    "scenario": scenario,
                    "ensemble_member": ensemble_member,
                    "timescale": timescale,
                    "metric": metric,
                    "max": float(np.nanmax(metrics_to_evaluate[metric])),
                    "min": float(np.nanmin(metrics_to_evaluate[metric])),
                    "fraction_above": float(
                        (metrics_to_evaluate[metric] > limits["eval_max"])
                        .mean(dim=["lat", "lon"])
                        .values
                    ),
                    "fraction_below": float(
                        (metrics_to_evaluate[metric] < limits["eval_min"])
                        .mean(dim=["lat", "lon"])
                        .values
                    ),
                }
            )

    print(buf.getvalue(), end="")
    if log_path is not None:
        csv_path = Path(log_path).with_suffix(".csv")
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, mode="w", header=True, index=False)


def periodic_rolling(da, dim, window, agg="mean", **kwargs):
    pad = window // 2
    padded = da.pad({dim: pad}, mode="wrap")
    if agg == "max":
        out = padded.rolling({dim: window}, center=True, **kwargs).max()
    elif agg == "min":
        out = padded.rolling({dim: window}, center=True, **kwargs).min()
    elif agg == "mean":
        out = padded.rolling({dim: window}, center=True, **kwargs).mean()
    return out.isel({dim: slice(pad, -pad)})


def obs_doy_bounds(
    obs_fine_subset: xr.DataArray, window: int = 30
) -> tuple[xr.DataArray, xr.DataArray]:
    """Lazy per-day-of-year lower/upper envelope of the fine-grid observations.

    The bounds are the ``window``-day centered periodic-rolling minimum and
    maximum of the observed day-of-year extremes. This ingredient depends only on
    the variable, not on the scenario or member, so callers can compute it once
    per variable and reuse it across leaves in a single ``dask.compute``.

    Parameters
    ----------
    obs_fine_subset : xarray.DataArray
        Fine-grid observations with a ``time`` dimension.
    window : int, default 30
        Width, in days, of the centered periodic rolling window.

    Returns
    -------
    obs_min_rolling, obs_max_rolling : xarray.DataArray
        Lazy ``(dayofyear, lat, lon)`` arrays; no computation is triggered.
    """
    obs_fine_subset = obs_fine_subset.chunk({"time": 365, "lat": 180, "lon": 360})
    obs_max = obs_fine_subset.groupby("time.dayofyear").max()
    obs_min = obs_fine_subset.groupby("time.dayofyear").min()
    obs_max_rolling = periodic_rolling(da=obs_max, dim="dayofyear", window=window, agg="max")
    obs_min_rolling = periodic_rolling(da=obs_min, dim="dayofyear", window=window, agg="min")
    return obs_min_rolling, obs_max_rolling


def scenario_delta_doy(
    raw_scenario_subset: xr.DataArray,
    raw_historical_subset: xr.DataArray,
    window: int = 30,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Lazy coarse-grid day-of-year change signal relative to the historical mean.

    The upper (lower) delta is the ``window``-day centered periodic-rolling
    maximum (minimum) of the scenario's day-of-year extremes, minus the
    historical day-of-year mean climatology. When the historical input carries an
    ``ensemble_member`` dimension the mean is taken across members as well; a
    single pre-selected member is used as-is.

    Parameters
    ----------
    raw_scenario_subset : xarray.DataArray
        Coarse-grid scenario data with a ``time`` dimension.
    raw_historical_subset : xarray.DataArray
        Coarse-grid historical data with a ``time`` dimension and, optionally, an
        ``ensemble_member`` dimension.
    window : int, default 30
        Width, in days, of the centered periodic rolling window.

    Returns
    -------
    delta_doy_min, delta_doy_max : xarray.DataArray
        Lazy ``(dayofyear, lat, lon)`` coarse-grid arrays.
    """
    scenario_doy_max = raw_scenario_subset.groupby("time.dayofyear").max(dim="time")
    scenario_doy_min = raw_scenario_subset.groupby("time.dayofyear").min(dim="time")
    hist_doy_mean = raw_historical_subset.groupby("time.dayofyear").mean(dim="time")
    if "ensemble_member" in hist_doy_mean.dims:
        hist_doy_mean = hist_doy_mean.mean(dim="ensemble_member")

    scenario_doy_max_rolling = periodic_rolling(
        da=scenario_doy_max, dim="dayofyear", window=window, agg="max"
    )
    scenario_doy_min_rolling = periodic_rolling(
        da=scenario_doy_min, dim="dayofyear", window=window, agg="min"
    )
    hist_doy_mean_rolling = periodic_rolling(
        da=hist_doy_mean, dim="dayofyear", window=window, agg="mean"
    )

    delta_doy_max = scenario_doy_max_rolling - hist_doy_mean_rolling
    delta_doy_min = scenario_doy_min_rolling - hist_doy_mean_rolling
    return delta_doy_min, delta_doy_max


def calculate_reasonable_bounds_doy(
    raw_scenario_subset: xr.DataArray,
    raw_historical_subset: xr.DataArray,
    obs_fine_subset: xr.DataArray,
    window: int = 30,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Lazy per-day-of-year plausible lower/upper bounds on the fine grid.

    Combines the fine-grid observed envelope (:func:`obs_doy_bounds`) with the
    coarse-grid scenario change signal (:func:`scenario_delta_doy`) interpolated
    to the observation grid: ``bound = obs_envelope + regridded_change``. The
    result is lazy; call ``.compute()`` to materialize it.

    The returned arrays satisfy ``high_bound >= low_bound`` wherever the
    observations are defined, because both the observed envelope and the scenario
    change contribute a non-negative max-minus-min span.

    Parameters
    ----------
    raw_scenario_subset : xarray.DataArray
        Coarse-grid scenario data with a ``time`` dimension.
    raw_historical_subset : xarray.DataArray
        Coarse-grid historical data (see :func:`scenario_delta_doy`).
    obs_fine_subset : xarray.DataArray
        Fine-grid observations with ``time``, ``lat``, and ``lon``.
    window : int, default 30
        Width, in days, of the centered periodic rolling window.

    Returns
    -------
    low_bound, high_bound : xarray.DataArray
        Lazy ``(dayofyear, lat, lon)`` fine-grid plausible bounds.
    """
    obs_min_rolling, obs_max_rolling = obs_doy_bounds(obs_fine_subset, window=window)
    delta_doy_min, delta_doy_max = scenario_delta_doy(
        raw_scenario_subset, raw_historical_subset, window=window
    )

    # Nearest-neighbor coarse -> fine regrid via reindex, not interp(method="nearest"). The two are
    # numerically identical for nearest selection (differing only in edge fill: reindex fills the
    # poles by nearest where interp leaves NaN). reindex is pure indexing and stays on a single dask
    # backend, whereas interp routes through apply_ufunc and mixes classic dask arrays with the
    # query-planning (dask_array) backend used on Coiled, raising "Mixing chunked array types".
    regrid_kwargs = {
        "lat": obs_fine_subset["lat"],
        "lon": obs_fine_subset["lon"],
        "method": "nearest",
    }
    delta_doy_max_finegrid = delta_doy_max.reindex(**regrid_kwargs)
    delta_doy_min_finegrid = delta_doy_min.reindex(**regrid_kwargs)

    high_bound = obs_max_rolling + delta_doy_max_finegrid
    low_bound = obs_min_rolling + delta_doy_min_finegrid
    return low_bound, high_bound


# ---------------------------------------------------------------------------
# Exceedance region finding
# ---------------------------------------------------------------------------
#
# The plausible-value check produces, per leaf, a pair of 2-D maps of how far the
# downscaled output strays outside its envelope. Those maps say a leaf is flagged
# but not *where*: reading a global map by eye conflates an isolated coastline or
# sea-ice fleck with a coherent patch worth investigating.
#
# find_exceedance_regions turns such a map into ranked connected components, so
# that distinction becomes a column. The two private helpers below exist because
# the maps are global: longitude is periodic, so a patch straddling the
# antimeridian must not be reported as two regions, and its centroid must not
# average to the opposite side of the planet.


_REGION_COLUMNS = [
    "region",
    "n_cells",
    "area_frac",
    "worst_value",
    "worst_lat",
    "worst_lon",
    "centroid_lat",
    "centroid_lon",
    "lat_min",
    "lat_max",
    "lon_min",
    "lon_max",
    "wraps_lon",
]


def _merge_labels_across_lon_seam(labels: np.ndarray) -> np.ndarray:
    """Join labelled components that touch across the periodic longitude seam.

    ``scipy.ndimage.label`` treats the array as a flat plane, so a patch
    straddling the antimeridian is reported as two regions. Longitude is
    periodic, so components flagged at both the first and last column of the
    same (or a diagonally adjacent, matching 8-connectivity) row are one region.
    Latitude is *not* periodic, so the poles get no equivalent treatment.
    """
    n_lat = labels.shape[0]
    pairs: list[tuple[int, int]] = []
    for offset in (-1, 0, 1):
        left_rows = np.arange(max(0, -offset), min(n_lat, n_lat - offset))
        if left_rows.size == 0:
            continue
        left = labels[left_rows, 0]
        right = labels[left_rows + offset, -1]
        touching = (left > 0) & (right > 0)
        pairs.extend(zip(left[touching].tolist(), right[touching].tolist(), strict=True))

    if not pairs:
        return labels

    parent = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in pairs:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    remap = np.arange(labels.max() + 1)
    for label in range(1, labels.max() + 1):
        if label in parent:
            remap[label] = find(label)
    return remap[labels]


def _circular_mean_lon(lon_values: np.ndarray) -> float:
    """Mean longitude that behaves correctly across the antimeridian.

    An arithmetic mean of a region spanning the seam averages, say, -179 and 179
    to 0 -- the wrong side of the planet. Averaging unit vectors instead keeps
    the centroid inside the region.
    """
    radians = np.deg2rad(lon_values)
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))


def find_exceedance_regions(
    exceedance: xr.DataArray,
    *,
    direction: str,
    min_cells: int = 1,
    top_n: int = 10,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Rank the coherent regions where a leaf leaves its plausible envelope.

    Turns a 2-D exceedance map into connected components, so the distinction
    between an isolated fleck (usually a coastline or sea-ice edge effect) and a
    coherent patch worth investigating becomes a column rather than something
    eyeballed off a global map. 8-connectivity is used, so diagonally touching
    cells join, and components spanning the longitude seam are merged.

    Parameters
    ----------
    exceedance : xarray.DataArray
        Signed 2-D ``(lat, lon)`` map: ``over`` (flagged above zero) or ``under``
        (flagged below zero), as produced by the plausible-value check.
    direction : {"high", "low"}
        Which sign counts as flagged. ``"high"`` flags ``> 0``, ``"low"`` flags
        ``< 0``.
    min_cells : int, default 1
        Drop components smaller than this, filtering isolated flecks.
    top_n : int, default 10
        Keep at most this many regions, ranked by exceedance magnitude.
    scale : float, default 1.0
        Multiplier applied to ``worst_value`` only, for reporting units (86400
        for ``pr`` in mm day-1). Area metrics stay dimensionless.

    Returns
    -------
    pandas.DataFrame
        One row per region, ranked by ``abs(worst_value)`` descending, with
        columns ``region``, ``n_cells``, ``area_frac``, ``worst_value``,
        ``worst_lat``, ``worst_lon``, ``centroid_lat``, ``centroid_lon``,
        ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``, ``wraps_lon``.
        Empty (but correctly columned) when nothing is flagged.

    Notes
    -----
    ``area_frac`` is an unweighted cell-count fraction, using the same convention
    as the check's own ``frac_area_too_high = (over > 0).mean(["lat", "lon"])``.
    That overweights high latitudes relative to true surface area; a
    cos(lat)-weighted figure would be more physical but would not share a
    convention with the summary table.

    Do not expect ``area_frac`` to sum to the check's flagged fraction. It only
    does so with ``min_cells=1`` and a ``top_n`` large enough to keep every
    component, and neither is the normal case: a flagged variable is typically
    thousands of small components, so realistic settings return the few worst
    ones and account for a small share of the flagged area. The sum over
    returned regions is a lower bound on the check's fraction, not a
    reconciliation of it.

    ``lon_min``/``lon_max`` are meaningless for a region flagged as
    ``wraps_lon``: such a region straddles the seam, so its bounding box is two
    boxes, not one. ``centroid_lon`` stays usable in that case because it is a
    circular mean, which is why callers should prefer it for centring a view.
    """
    from scipy import ndimage

    if direction not in {"high", "low"}:
        raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")

    values = np.asarray(exceedance.values, dtype="float64")
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D (lat, lon) map, got dims {exceedance.dims}")

    finite = np.isfinite(values)
    mask = (values > 0) & finite if direction == "high" else (values < 0) & finite
    if not mask.any():
        return pd.DataFrame(columns=_REGION_COLUMNS)

    labels, _ = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 2))
    labels = _merge_labels_across_lon_seam(labels)

    lat = np.asarray(exceedance["lat"].values, dtype="float64")
    lon = np.asarray(exceedance["lon"].values, dtype="float64")
    total_cells = values.size
    last_column = values.shape[1] - 1

    rows = []
    for label in np.unique(labels[labels > 0]):
        selected = labels == label
        n_cells = int(selected.sum())
        if n_cells < min_cells:
            continue

        rows_idx, cols_idx = np.nonzero(selected)
        magnitudes = np.abs(values[rows_idx, cols_idx])
        worst = int(np.argmax(magnitudes))
        region_lats = lat[rows_idx]
        region_lons = lon[cols_idx]
        wraps = bool(selected[:, 0].any() and selected[:, last_column].any())

        rows.append(
            {
                "n_cells": n_cells,
                "area_frac": n_cells / total_cells,
                "worst_value": float(values[rows_idx[worst], cols_idx[worst]]) * scale,
                "worst_lat": float(region_lats[worst]),
                "worst_lon": float(region_lons[worst]),
                "centroid_lat": float(region_lats.mean()),
                "centroid_lon": _circular_mean_lon(region_lons),
                "lat_min": float(region_lats.min()),
                "lat_max": float(region_lats.max()),
                "lon_min": float(region_lons.min()),
                "lon_max": float(region_lons.max()),
                "wraps_lon": wraps,
            }
        )

    if not rows:
        return pd.DataFrame(columns=_REGION_COLUMNS)

    regions = pd.DataFrame.from_records(rows)
    regions = regions.reindex(regions.worst_value.abs().sort_values(ascending=False).index)
    regions = regions.head(top_n).reset_index(drop=True)
    regions.insert(0, "region", np.arange(1, len(regions) + 1))
    return regions[_REGION_COLUMNS]


# ---------------------------------------------------------------------------
# Trend-distortion analysis
#
# "Distortion" is a difference of differences: how much a scenario-to-scenario delta changes once
# debiasing and downscaling are applied, relative to the same delta computed on raw GCM output.
# These are the pure pieces of that analysis; plotting and cache IO live in the notebook.
# ---------------------------------------------------------------------------

# The four pipeline stages a delta can be computed at. `coarsened_downscaled_debiased` is the
# fine-grid downscaled output recoarsened onto the raw grid, which is what makes it comparable to
# `raw` and `coarse_debiased` cell for cell.
DISTORTION_STAGES = (
    "raw",
    "coarse_debiased",
    "downscaled_debiased",
    "coarsened_downscaled_debiased",
)

# {name: (tested_stage, reference_stage)}. The third pair isolates what downscaling adds on top of
# debiasing; the first two measure debiasing alone and the two together.
DISTORTION_STAGE_PAIRS: dict[str, tuple[str, str]] = {
    "coarse_debiased_vs_raw": ("coarse_debiased", "raw"),
    "downscaled_debiased_vs_raw": ("coarsened_downscaled_debiased", "raw"),
    "downscaled_debiased_vs_coarse_debiased": (
        "coarsened_downscaled_debiased",
        "coarse_debiased",
    ),
}

# {family: (after_scenario_group, before_scenario_group)}. `g6_ssp` isolates the effect of SAI by
# differencing the intervention against its no-SAI counterfactual over the same window.
DISTORTION_FAMILIES: dict[str, tuple[str, str]] = {
    "ssp_hist": ("ssp245", "historical"),
    "g6_hist": ("g6_1p5k", "historical"),
    "g6_ssp": ("g6_1p5k", "ssp245"),
}

_COMPARISON_COLUMNS = [
    "comparison_id",
    "family",
    "variable",
    "after_scenario",
    "after_member",
    "before_scenario",
    "before_member",
]

_SKIPPED_COMPARISON_COLUMNS = [
    "family",
    "variable",
    "after_scenario",
    "after_member",
    "before_scenario",
    "reason",
]


def _baseline_member(
    gcm: str, after_group: str, after_member: str, variable: str, before_group: str
) -> str | None:
    """Member of ``before_group`` that a leaf of ``after_group`` should be differenced against.

    Resolved through :mod:`srm.lineage`, never guessed. Returns ``None`` when the lineage table
    registers no such parent for the combination.
    """
    label = GROUP_TO_SCENARIO[after_group]
    lineage = resolve_member_lineage(gcm, label, after_member, variable)
    if before_group == "historical":
        return lineage.historical
    if before_group == "ssp245":
        return lineage.ssp245_bridge
    raise ValueError(f"no lineage rule for a {before_group!r} baseline")


def enumerate_scenario_comparisons(
    leaves,
    *,
    gcm: str,
    families: dict[str, tuple[str, str]] | None = None,
    variables=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every scenario comparison the available leaves support, with its baseline member resolved.

    The baseline member comes from :func:`srm.lineage.resolve_member_lineage`, which is what makes
    this safe to run across variables. For the ``g6_ssp`` family the correct SSP245 bridge member
    varies by variable on CESM2-WACCM: ``tas``/``pr``/``rsds``/``hurs`` bridge member ``003`` while
    ``tasmax``/``tasmin``/``dtr`` bridge ``008`` (issue #448). A hand-written member therefore cannot
    be reused across variables, and pairing on one would compare different realizations.

    Parameters
    ----------
    leaves : iterable of tuple
        ``(scenario_group, variable, member)`` triples present in the store, in group naming
        (``historical``, ``ssp245``, ``g6_1p5k``).
    gcm : str
        GCM name as registered in :mod:`srm.lineage`.
    families : dict, optional
        ``{family: (after_group, before_group)}``. Defaults to :data:`DISTORTION_FAMILIES`.
    variables : container, optional
        Restrict to these variables. ``None`` keeps every variable found in ``leaves``.

    Returns
    -------
    comparisons : pandas.DataFrame
        Columns ``comparison_id``, ``family``, ``variable``, ``after_scenario``, ``after_member``,
        ``before_scenario``, ``before_member``.
    skipped : pandas.DataFrame
        Comparisons that could not be built, with a ``reason`` column. Skipping rather than
        fabricating a baseline is what keeps a lineage gap or a missing leaf visible.
    """
    available = {tuple(key) for key in leaves}
    families = DISTORTION_FAMILIES if families is None else families

    rows, skips = [], []
    for family, (after_group, before_group) in families.items():
        for scenario, variable, member in sorted(available):
            if scenario != after_group:
                continue
            if variables is not None and variable not in variables:
                continue
            skip = dict(
                family=family,
                variable=variable,
                after_scenario=scenario,
                after_member=member,
                before_scenario=before_group,
            )
            try:
                before_member = _baseline_member(gcm, scenario, member, variable, before_group)
            except KeyError as exc:
                skips.append({**skip, "reason": f"no registered lineage: {exc.args[0][:80]}"})
                continue
            if before_member is None:
                skips.append({**skip, "reason": f"lineage registers no {before_group} parent"})
                continue
            if (before_group, variable, before_member) not in available:
                missing = "/".join((before_group, variable, before_member))
                skips.append({**skip, "reason": f"baseline leaf {missing} absent from the store"})
                continue
            rows.append(
                {
                    "comparison_id": f"{family}_{variable}_{member}-vs-{before_member}",
                    "family": family,
                    "variable": variable,
                    "after_scenario": scenario,
                    "after_member": member,
                    "before_scenario": before_group,
                    "before_member": before_member,
                }
            )

    return (
        pd.DataFrame(rows, columns=_COMPARISON_COLUMNS),
        pd.DataFrame(skips, columns=_SKIPPED_COMPARISON_COLUMNS),
    )


def compute_deltas(
    after: dict[str, xr.DataArray],
    before: dict[str, xr.DataArray],
    stages=DISTORTION_STAGES,
) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
    """``after - before`` per pipeline stage, in absolute terms and as a percent of ``before``.

    Both arguments map a stage name to that stage's period-mean grid. Nothing is computed here, so
    the caller decides when to materialize.

    Cells where ``before`` is exactly zero give an undefined percent change and come back as NaN
    rather than infinity. That matters because an infinity would propagate into the percent extremes
    and, since a distortion flag requires both tolerances, silently flag the cell.
    """
    deltas = {stage: after[stage] - before[stage] for stage in stages}
    deltas_pct = {
        stage: deltas[stage] * 100 / before[stage].where(before[stage] != 0) for stage in stages
    }
    return deltas, deltas_pct


def distortion_fields(
    deltas_absolute: dict[str, xr.DataArray],
    deltas_pct: dict[str, xr.DataArray],
    stage_pair: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Absolute and percent distortion for one entry of :data:`DISTORTION_STAGE_PAIRS`."""
    tested, reference = DISTORTION_STAGE_PAIRS[stage_pair]
    return (
        deltas_absolute[tested] - deltas_absolute[reference],
        deltas_pct[tested] - deltas_pct[reference],
    )


def calculate_distortion_flags(
    distortion_absolute: xr.DataArray,
    distortion_pct: xr.DataArray,
    *,
    tolerance_absolute: float,
    tolerance_pct: float,
) -> xr.DataArray:
    """Flag cells where a distortion exceeds *both* tolerances.

    Requiring both is what keeps the flag off tiny absolute swings in already-dry cells (large
    percent, small absolute) and off modest percent swings in very wet ones (large absolute, small
    percent).

    ``tolerance_pct=0`` disables the percent condition rather than applying it as ``abs(pct) > 0``.
    The two differ: the literal reading is true almost everywhere, so it looks like a no-op, but it
    would drop any cell whose percent change is NaN. That is the intended setting for temperature in
    kelvin, where a 1 K change on a 300 K mean reads as 0.3 %.
    """
    is_distorted = abs(distortion_absolute) > tolerance_absolute
    if tolerance_pct == 0:
        return is_distorted
    return is_distorted & (abs(distortion_pct) > tolerance_pct)


def sign_flip_mask(
    delta_tested: xr.DataArray, delta_reference: xr.DataArray, *, threshold: float
) -> xr.DataArray:
    """Flag cells where two stages disagree on the *sign* of a delta by more than ``threshold``.

    Stricter and more literal than a tolerance on the size of the distortion: this is the case where
    "does the intervention increase or decrease this variable here" gets a different answer depending
    on which pipeline stage you read it from. Both stages must clear ``threshold`` in magnitude, so a
    pair straddling zero by a hair does not count.
    """
    return ((delta_reference > threshold) & (delta_tested < -threshold)) | (
        (delta_reference < -threshold) & (delta_tested > threshold)
    )


def area_weights(lat: xr.DataArray) -> xr.DataArray:
    """``cos(latitude)`` weights for area-weighted means on a rectilinear grid.

    A plain spatial mean gives a polar cell the same weight as an equatorial one. On the 192 x 288
    coarse grid a cell at 89 degrees covers roughly a sixtieth of the area of one at the equator, so
    an unweighted flagged fraction overstates whatever happens near the poles. Values are clipped at
    zero so floating-point noise at the poles cannot contribute a negative weight.
    """
    weights = xr.DataArray(
        np.cos(np.deg2rad(np.asarray(lat, dtype="float64"))), dims=lat.dims, coords=lat.coords
    )
    return weights.clip(min=0.0).rename("area_weights")


def weighted_fraction(
    mask: xr.DataArray,
    weights: xr.DataArray,
    *,
    valid: xr.DataArray | None = None,
    dims=("lat", "lon"),
) -> float:
    """Area-weighted fraction of ``mask``, over the cells where ``valid`` is true.

    The denominator is the valid area, not the whole grid. Conservative recoarsening can leave NaN
    cells at the grid edge, and counting those as "not distorted" would deflate every reported
    fraction by the size of that band rather than excluding it from the question.

    Returns NaN when no valid cell has weight, which is the honest answer for an empty domain.
    """
    if valid is None:
        valid = mask.notnull()
    flagged = xr.where(mask.fillna(False).astype(bool), 1.0, 0.0)
    weights = weights.broadcast_like(flagged).where(valid, 0.0)
    total = float(weights.sum(dims))
    if total == 0:
        return float("nan")
    return float((flagged * weights).sum(dims)) / total


def distortion_summary(
    distortion_absolute: xr.DataArray,
    distortion_pct: xr.DataArray,
    flag: xr.DataArray,
    *,
    weights: xr.DataArray,
    sign_flip: xr.DataArray | None = None,
) -> dict:
    """Scalar summary of one stage pair's distortion, for one row of the summary CSV.

    The distortion fields are expected to already be in reporting units, because the tolerances that
    produced ``flag`` are. Converting here instead would put the flag and the magnitude it reports on
    two different scales.
    """
    valid = distortion_absolute.notnull()
    summary = {
        "frac_area_distorted": weighted_fraction(flag, weights, valid=valid),
        "distortion_abs_max": float(distortion_absolute.max()),
        "distortion_abs_min": float(distortion_absolute.min()),
        "distortion_pct_max": float(distortion_pct.max()),
        "distortion_pct_min": float(distortion_pct.min()),
    }
    if sign_flip is not None:
        summary["frac_area_sign_flip"] = weighted_fraction(sign_flip, weights, valid=valid)
    return summary
