from __future__ import annotations

import itertools

import cf_xarray  # noqa: F401  # registers CF accessor
import numpy as np
import pandas as pd
import xarray as xr

from srm import catalog

# Spatial range bounds for unit-mismatch detection (e.g. Celsius instead of Kelvin).
# Ranges are wide intentionally — based on ERA5 observed range +/- large margins.
_N_TIME_SAMPLES = 15  # number of pseudo-random time indices used by multi-step checks
_TIME_SAMPLE_SEED = 0  # fixed seed → deterministic draws across runs


def _sample_time_indices(
    n: int, k: int = _N_TIME_SAMPLES, seed: int = _TIME_SAMPLE_SEED
) -> list[int]:
    """k deterministic pseudo-random indices drawn from [1, n-1] (day 0 excluded)."""
    if n <= 1:
        return [0]
    pool = list(range(1, n))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(pool, size=min(k, len(pool)), replace=False))


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
        """
        if "ensemble_member" not in getattr(obj, "dims", {}):
            return obj

        other_dims = [d for d in obj.dims if d != "ensemble_member"]

        for i in range(obj.sizes["ensemble_member"]):
            member = obj.isel(ensemble_member=i)
            if isinstance(member, xr.Dataset):
                if all(
                    not bool(member[v].isnull().all(dim=other_dims).compute())
                    for v in member.data_vars
                ):
                    return member
            else:
                if not bool(member.isnull().all(dim=other_dims).compute()):
                    return member

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

    def validate_no_identical_vars(self) -> ValidationResult:
        if "time" not in self.ds.dims or self.ds.sizes["time"] == 0:
            return ValidationResult(True, [])

        time_indices = _sample_time_indices(self.ds.sizes["time"])
        ds_sample = self.ds.isel(time=time_indices)

        if "ensemble_member" in ds_sample.dims:
            ds_sample = self._resolve_ensemble_member(ds_sample)
            if ds_sample is None:
                return ValidationResult(True, [])

        # One compute call loads all sampled time steps for all variables at once.
        ds_computed = ds_sample.compute()

        var_names = list(ds_computed.data_vars)
        issues = []
        for v1, v2 in itertools.combinations(var_names, 2):
            a = ds_computed[v1]
            b = ds_computed[v2]
            # Both all-NaN across all samples → fill-data equality, not a real data bug.
            if a.isnull().all() and b.isnull().all():
                continue
            if a.equals(b):
                issues.append(
                    f"{v1} and {v2} are identical across {len(time_indices)} sampled time steps"
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

        time_indices = _sample_time_indices(self.ds.sizes["time"])
        # One compute: global mean for all members × all sampled time steps.
        means = self.ds[var].isel(time=time_indices).mean(dim=["lat", "lon"]).compute()

        # A member pair is flagged only when their global means are equal at ALL sampled
        # time steps (and neither series is NaN-only). A stitch artifact at one time step
        # is therefore outvoted by the remaining clean samples.
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
                    f"across {len(time_indices)} sampled time steps"
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
