from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from srm.catalog import Dataset

from validators import DatasetValidator

from srm.datasets import VirtualDataset
from srm.validation import (
    GCM_OPTIONS,
    SCENARIO_OPTIONS,
    CheckStatus,
    DatasetValidator as SRMDatasetValidator,
)

pytestmark = pytest.mark.input_data


class TestCatalogDatasets:
    """Validate input datasets in the catalog"""

    def _skip_if_virtual(
        self, ds_info: Dataset, reason: str = "Not applicable to virtual datasets"
    ):
        if isinstance(ds_info, VirtualDataset):
            pytest.skip(reason)

    @pytest.fixture
    def validator(self, ds_info: Dataset) -> DatasetValidator:
        return DatasetValidator(ds_info)

    def test_longitude_valid(self, ds_info: Dataset, validator: DatasetValidator):
        self._skip_if_virtual(ds_info)
        result = validator.validate_lon(check_monotonic=True)
        assert result, f"{ds_info.name}: {result.issues}"

    def test_latitude_valid(self, ds_info: Dataset, validator: DatasetValidator):
        self._skip_if_virtual(ds_info)
        result = validator.validate_lat(check_monotonic=True)
        assert result, f"{ds_info.name}: {result.issues}"

    def test_expected_chunking(self, ds_info: Dataset, validator: DatasetValidator):
        self._skip_if_virtual(ds_info)
        if ds_info.expected_chunks is None:
            pytest.skip(f"{ds_info.name} has no chunking expectations")
        result = validator.validate_expected_chunking()
        assert result, f"{ds_info.name}: {result.issues}"

    def test_expected_variables(self, ds_info: Dataset, validator: DatasetValidator):
        """check existing data variables against known variables in catalog"""
        self._skip_if_virtual(ds_info)
        if not ds_info.expected_vars:
            pytest.skip(f"{ds_info.name} has no var expectations.")
        result = validator.validate_expected_variables()
        assert result, f"variable mismatch {ds_info.name}: {result.issues}"

    def test_variable_units(self, ds_info: Dataset, validator: DatasetValidator):
        """check variable units match"""
        self._skip_if_virtual(ds_info)
        if not ds_info.expected_vars:
            pytest.skip(f"{ds_info.name} has no variable expectations defined.")
        result = validator.validate_units()
        assert result, f"Unit mismatch for {ds_info.name}: {result.issues}"

    def test_calendar(self, ds_info: Dataset, validator: DatasetValidator):
        """check calendar is proleptic_gregorian and datetime64"""
        self._skip_if_virtual(ds_info)
        result = validator.validate_calendar()
        assert result, f"{ds_info.name}: {result.issues}"

    @pytest.mark.slow
    def test_negative_precip(self, ds_info: Dataset, validator: DatasetValidator):
        """Only run on datasets that contain 'pr'"""
        self._skip_if_virtual(ds_info)
        if ds_info.expected_vars and not any(v.name == "pr" for v in ds_info.expected_vars):
            pytest.skip(f"Dataset {ds_info.name} does not contain precipitation.")
        result = validator.validate_negative_precip()
        assert result, f"{ds_info.name}: {result.issues}"


class TestCrossScenarioConsistency:
    """D: Cross-scenario ensemble member consistency checks."""

    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_ssp245_hist_member_pairing(self, gcm):
        """D1: every SSP245 member has a match in the historical store."""
        result = SRMDatasetValidator(gcm=gcm, scenario="SSP245").check_ssp245_hist_member_pairing()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"

    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_g6_ssp245_member_pairing(self, gcm):
        """D2: every G6 member has a match in the SSP245 store."""
        result = SRMDatasetValidator(gcm=gcm, scenario="G6-1.5K").check_g6_ssp245_member_pairing()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"


class TestDataIntegrity:
    """E: Data integrity checks."""

    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_g6_not_identical_to_ssp245(self, gcm):
        """E1: G6-1.5K data must differ from SSP245 for the same ensemble member."""
        result = SRMDatasetValidator(gcm=gcm, scenario="G6-1.5K").check_g6_not_identical_to_ssp245()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"

    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    @pytest.mark.parametrize("scenario", list(SCENARIO_OPTIONS))
    def test_temporal_coverage(self, gcm, scenario):
        """E2: time axis must be gapless with correct first and last dates."""
        result = SRMDatasetValidator(gcm=gcm, scenario=scenario).check_temporal_coverage()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"
