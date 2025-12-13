from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from srm.catalog import Dataset

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
