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
from pathlib import Path

import cf_xarray  # noqa: F401  # registers CF accessor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from srm import catalog

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


def check_ensemble_spread(ds: xr.Dataset, label: str, var: str = "tas", day_index: int = 0) -> dict:
    """Return an ensemble-spread summary dict for DataFrame/plot display.

    Computes the global spatial mean of `var` for each ensemble member on `day_index`.
    Returns spread_ok=True when all member means are distinct (no exact duplicates).
    Returns empty members/means lists when the dataset has no ensemble_member dimension
    or the variable is absent.

    Keys: source, variable, members, means, spread_ok.
    """
    if var not in ds or "ensemble_member" not in ds.dims:
        return {"source": label, "variable": var, "members": [], "means": [], "spread_ok": True}

    da = ds[var].isel(time=day_index)
    means_da = da.mean(dim=["lat", "lon"]).compute()
    members = [str(m) for m in means_da.ensemble_member.values]
    means = [float(v) for v in means_da.values]
    spread_ok = len(set(means)) == len(means)
    return {
        "source": label,
        "variable": var,
        "members": members,
        "means": means,
        "spread_ok": spread_ok,
    }


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
