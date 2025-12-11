from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from srm.catalog import Dataset


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
    def __init__(self, ds_info: Dataset):
        self.ds_info = ds_info
        self.ds = ds_info.to_xarray().cf

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
            is_increasing = (coord.diff(coord.name) > 0).all().item()
            if not is_increasing:
                issues.append(f"{expected_name} is not monotonically increasing")

        return ValidationResult(len(issues) == 0, issues)

    def validate_lon(
        self, expected_range: tuple[float, float] = (-180, 180), check_monotonic: bool = False
    ) -> ValidationResult:
        return self._validate_coord("lon", "lon", expected_range, check_monotonic)

    def validate_lat(
        self, expected_range: tuple[float, float] = (-90, 90), check_monotonic: bool = False
    ) -> ValidationResult:
        return self._validate_coord("lat", "lat", expected_range, check_monotonic)

    def validate_expected_chunking(self) -> ValidationResult:
        actual_chunks = self.ds_info.get_chunking_dict()
        is_valid = actual_chunks == self.ds_info.expected_chunks
        issues = (
            []
            if is_valid
            else [f"chunks {actual_chunks} != expected {self.ds_info.expected_chunks}"]
        )
        return ValidationResult(is_valid, issues)
