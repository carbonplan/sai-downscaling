from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from srm.datasets import Dataset

from validators import DatasetValidator


class TestCatalogDatasets:
    """Validate input datasets in the catalog"""

    @pytest.fixture
    def validator(self, ds_info: Dataset) -> DatasetValidator:
        """Create validator from dataset class"""
        return DatasetValidator(ds_info)

    def test_longitude_valid(self, ds_info: Dataset, validator: DatasetValidator):
        result = validator.validate_lon(check_monotonic=True)
        assert result, f"{ds_info.name}: {result.issues}"

    def test_latitude_valid(self, ds_info: Dataset, validator: DatasetValidator):
        result = validator.validate_lat(check_monotonic=True)
        assert result, f"{ds_info.name}: {result.issues}"

    def test_expected_chunking(self, ds_info: Dataset, validator: DatasetValidator):
        if ds_info.expected_chunks is None:
            pytest.skip(f"{ds_info.name} has no chunking expectations")

        result = validator.validate_expected_chunking()
        assert result, f"{ds_info.name}: {result.issues}"

    def test_expected_variables(self, ds_info: Dataset, validator: DatasetValidator):
        """check existing data variables against known variables in catalog"""
        if not ds_info.expected_vars:
            pytest.skip(f"{ds_info.name} no var expections.")

        result = validator.validate_expected_variables()
        assert result, f"variable mismatch {ds_info.name}: {result.issues}"

    def test_variable_units(self, ds_info: Dataset, validator: DatasetValidator):
        """check variable units match"""
        if not ds_info.expected_vars:
            pytest.skip(f"{ds_info.name} has no variable expectations defined.")

        result = validator.validate_units()
        assert result, f"Unit mismatch for {ds_info.name}: {result.issues}"

    def test_calendar(self, ds_info: Dataset, validator: DatasetValidator):
        """check calendar is proleptic_gregorian and datetime64"""
        result = validator.validate_calendar()
        assert result, f"U {ds_info.name}: {result.issues}"

    @pytest.mark.slow
    def test_negative_precip(self, ds_info: Dataset, validator: DatasetValidator):
        """Only run on datasets that contain 'pr'"""
        if ds_info.expected_vars and not any(
            v.name == "pr" for v in ds_info.expected_vars
        ):
            pytest.skip(f"Dataset {ds_info.name} does not contain precipitation.")

        result = validator.validate_negative_precip()
        assert result, f"{ds_info.name}: {result.issues}"
