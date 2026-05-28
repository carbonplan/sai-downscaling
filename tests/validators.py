from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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


class DatasetValidator:
    def __init__(self, ds_info):
        import cf_xarray  # noqa
        import xarray as xr

        if isinstance(ds_info, xr.Dataset):
            self.ds_info = None
            self.ds = ds_info
        else:
            self.ds_info = ds_info
            self.ds = ds_info.to_xarray(convert_calendar=False)

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
            if not (coord.diff(coord_name) > 0).all().item():
                issues.append(f"{expected_name} is not monotonically increasing")

        return ValidationResult(len(issues) == 0, issues)

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

    def validate_expected_chunking(self) -> ValidationResult:
        issues = []
        expected_chunks_dict = self.ds_info.expected_chunks
        expected_shards_dict = self.ds_info.expected_shards

        for var_spec in self.ds_info.expected_vars:
            var_name = var_spec.name
            var = self.ds[var_name]
            enc = var.encoding

            expected_chunks = tuple(expected_chunks_dict[dim] for dim in var.dims)
            expected_shards = tuple(expected_shards_dict[dim] for dim in var.dims)

            actual_chunks = enc.get("chunks")
            if actual_chunks != expected_chunks:
                issues.append(f"{var_name}: chunks {actual_chunks} != expected {expected_chunks}")

            actual_shards = enc.get("shards")
            if actual_shards != expected_shards:
                issues.append(f"{var_name}: shards {actual_shards} != expected {expected_shards}")

        return ValidationResult(len(issues) == 0, issues)

    def validate_expected_variables(self) -> ValidationResult:
        if not self.ds_info.expected_vars:
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
        if not self.ds_info.expected_vars:
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

    def validate_negative_precip(self) -> ValidationResult:
        if "pr" not in self.ds:
            return ValidationResult(True, [])

        has_negatives = (self.ds["pr"] < 0).any().compute()
        if has_negatives:
            return ValidationResult(False, ["Found negative precipitation data"])
        return ValidationResult(True, [])
