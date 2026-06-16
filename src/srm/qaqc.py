from __future__ import annotations

import xarray as xr

from srm import catalog

# Spatial range bounds for a single day (isel(time=1)), computed across full spatial extent.
# Goal: catch obvious unit mismatches (e.g. Celsius instead of Kelvin, fraction instead of %).
# Ranges are wide intentionally — based on ERA5 observed range +/- large margins.
VAR_SPATIAL_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "tas": {"min": (100, 400), "max": (100, 400)},
    "tasmin": {"min": (100, 400), "max": (100, 400)},
    "tasmax": {"min": (100, 400), "max": (100, 400)},
    "pr": {"min": (0, 1e-7), "max": (0.0001, 0.03)},
    "rsds": {"min": (-1, 100), "max": (100, 1000)},
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
        import cf_xarray  # noqa: F401  # registers CF accessor
        import xarray as xr

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
        """Return first ensemble_member slice where no variable/value is entirely null."""
        import xarray as xr

        if "ensemble_member" not in getattr(obj, "dims", {}):
            return obj
        for i in range(obj.sizes["ensemble_member"]):
            candidate = obj.isel(ensemble_member=i)
            if isinstance(candidate, xr.Dataset):
                if all(
                    not bool(candidate[v].isnull().all().compute()) for v in candidate.data_vars
                ):
                    return candidate
            else:
                if not bool(candidate.isnull().all().compute()):
                    return candidate
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
        import numpy as np

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
        import numpy as np

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

    def validate_negative_precip(self, day_index: int = 1) -> ValidationResult:
        if "pr" not in self.ds:
            return ValidationResult(True, [])
        # Reuse module-level predicate; check single day for performance.
        count = int((self.ds["pr"].isel(time=day_index) < 0).sum().compute())
        if count > 0:
            return ValidationResult(False, [f"Found negative precipitation (day {day_index})"])
        return ValidationResult(True, [])

    def validate_spatial_range(self, var: str, day_index: int = 1) -> ValidationResult:
        if var not in self.ds:
            return ValidationResult(True, [])
        if var not in VAR_SPATIAL_RANGES:
            return ValidationResult(True, [])

        da = self.ds[var].isel(time=day_index)
        da = self._resolve_ensemble_member(da)
        if da is None:
            return ValidationResult(True, [])
        spatial_min = float(da.min().compute())
        spatial_max = float(da.max().compute())

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

    def validate_dtr_consistency(self, day_index: int = 1, atol: float = 0.5) -> ValidationResult:
        required = {"dtr", "tasmax", "tasmin"}
        if not required.issubset(self.ds.data_vars):
            return ValidationResult(True, [])

        day = self.ds[list(required)].isel(time=day_index)
        if "ensemble_member" in day.dims:
            day = day.isel(ensemble_member=0)

        expected_dtr = (day["tasmax"] - day["tasmin"]).compute()
        actual_dtr = day["dtr"].compute()
        max_diff = float(abs(actual_dtr - expected_dtr).max())

        if max_diff > atol:
            return ValidationResult(
                False,
                [f"dtr deviates from tasmax-tasmin by up to {max_diff:.4g} K (atol={atol})"],
            )
        return ValidationResult(True, [])

    def validate_temp_consistency(self, day_index: int = 1) -> ValidationResult:
        required = {"tas", "tasmin", "tasmax"}
        if not required.issubset(self.ds.data_vars):
            return ValidationResult(True, [])

        day = self.ds[list(required)].isel(time=day_index)
        if "ensemble_member" in day.dims:
            day = day.isel(ensemble_member=0)

        # Reuse module-level predicate.
        min_gt_mean, mean_gt_max, min_gt_max = check_temperature_monotonic(day)

        issues = []
        if min_gt_mean > 0:
            issues.append(f"tas < tasmin at {min_gt_mean} grid points (day {day_index})")
        if mean_gt_max > 0:
            issues.append(f"tasmax < tas at {mean_gt_max} grid points (day {day_index})")
        if min_gt_max > 0:
            issues.append(f"tasmax < tasmin at {min_gt_max} grid points (day {day_index})")
        return ValidationResult(len(issues) == 0, issues)

    def validate_no_identical_vars(
        self, member_index: int = 0, day_index: int = 1
    ) -> ValidationResult:
        import itertools

        da_slice = self.ds.isel(time=day_index)
        if "ensemble_member" in da_slice.dims:
            da_slice = self._resolve_ensemble_member(da_slice)
            if da_slice is None:
                return ValidationResult(True, [])

        var_names = list(da_slice.data_vars)
        issues = []
        for v1, v2 in itertools.combinations(var_names, 2):
            a = da_slice[v1].compute()
            b = da_slice[v2].compute()
            if a.equals(b):
                issues.append(
                    f"{v1} and {v2} are identical (member_index={member_index}, day={day_index})"
                )
        return ValidationResult(len(issues) == 0, issues)

    def validate_ensemble_spread(self, var: str = "tas", day_index: int = 0) -> ValidationResult:
        import itertools

        if var not in self.ds:
            return ValidationResult(True, [])
        if "ensemble_member" not in self.ds.dims:
            return ValidationResult(True, [])
        if self.ds.sizes["ensemble_member"] < 2:
            return ValidationResult(True, [])

        da = self.ds[var].isel(time=day_index)
        means = da.mean(dim=["lat", "lon"]).compute()

        issues = []
        for i, j in itertools.combinations(range(len(means)), 2):
            if means.values[i] == means.values[j]:
                issues.append(
                    f"{var} global mean identical for members "
                    f"{means.ensemble_member.values[i]} and {means.ensemble_member.values[j]} "
                    f"(day {day_index})"
                )
        return ValidationResult(len(issues) == 0, issues)
