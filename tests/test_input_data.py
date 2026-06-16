from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from srm.catalog import Dataset

from validators import VAR_SPATIAL_RANGES, DatasetValidator

from srm.datasets import Datatree, VirtualDataset, catalog
from srm.validation import (
    _SCENARIO_TO_GROUP,
    GCM_OPTIONS,
    SCENARIO_OPTIONS,
    CheckStatus,
    DatasetValidator as SRMDatasetValidator,
)

pytestmark = pytest.mark.input_data

# Datasets too large or non-GCM for the expensive spatial range check.
# ERA5 and GDEX still get lighter single-day consistency checks below.
_SKIP_SPATIAL_RANGE = frozenset(
    {
        "ERA5",
        "NASA-NEX-SSP245",
        "NASA-NEX-historical",
        "GDEX-GMF",
        "ocean-mask",
    }
)

# Non-climate or non-data datasets — skip all physics checks.
_SKIP_ALL_PHYSICS = frozenset({"ocean-mask"})

# ERA5 tasmin/tasmax vars are forecast (minimum/maximum_2m_temperature_since_previous_post_processing)
# while ERA5 tas derives is analysis: analysis instantaneous 2m_temperature.
# Comparsing these, we get small tasmax < tas and tasmin < tas check failures
# For ex: on day 1, 0.09% of grid points have tas < tasmin
# and 0.19% have tasmax < tas

_SKIP_TEMP_CONSISTENCY = frozenset({"ERA5"})


class TestCatalogDatasets:
    """Validate input datasets in the catalog"""

    @pytest.fixture
    def validator(self, ds_info: Dataset) -> DatasetValidator:
        return DatasetValidator(ds_info)

    # spatial-checks: coordinate_names, coordinate_ranges
    def test_longitude_valid(self, ds_info: Dataset, validator: DatasetValidator):
        result = validator.validate_lon(check_monotonic=True)
        assert result, f"{ds_info.name}: {result.issues}"

    # spatial-checks: coordinate_names, coordinate_ranges
    def test_latitude_valid(self, ds_info: Dataset, validator: DatasetValidator):
        result = validator.validate_lat(check_monotonic=True)
        assert result, f"{ds_info.name}: {result.issues}"

    # variable-checks: variable_presence
    def test_expected_variables(self, ds_info: Dataset, validator: DatasetValidator):
        """check existing data variables against known variables in catalog"""
        if not ds_info.expected_vars:
            pytest.skip(f"{ds_info.name} has no var expectations.")
        result = validator.validate_expected_variables()
        assert result, f"variable mismatch {ds_info.name}: {result.issues}"

    # variable-checks: units
    def test_variable_units(self, ds_info: Dataset, validator: DatasetValidator):
        """check variable units match"""
        if not ds_info.expected_vars:
            pytest.skip(f"{ds_info.name} has no variable expectations defined.")
        result = validator.validate_units()
        assert result, f"Unit mismatch for {ds_info.name}: {result.issues}"

    # temporal-checks: monotonic, no_duplicate_timestamps, no_internal_gaps
    def test_time_axis(self, ds_info: Dataset, validator: DatasetValidator):
        result = validator.validate_time_axis()
        assert result, f"{ds_info.name}: {result.issues}"

    # temporal-checks: calendar
    def test_calendar(self, ds_info: Dataset, validator: DatasetValidator):
        """check calendar is proleptic_gregorian and datetime64"""
        result = validator.validate_calendar()
        assert result, f"{ds_info.name}: {result.issues}"

    # variable-checks: reasonable_ranges (pr >= 0)
    def test_negative_precip(self, ds_info: Dataset, validator: DatasetValidator):
        """Only run on datasets that contain 'pr'"""
        if ds_info.expected_vars and not any(v.name == "pr" for v in ds_info.expected_vars):
            pytest.skip(f"Dataset {ds_info.name} does not contain precipitation.")
        result = validator.validate_negative_precip()
        assert result, f"{ds_info.name}: {result.issues}"


class TestCrossScenarioConsistency:
    """D: Cross-scenario ensemble member consistency checks."""

    # ensemble-checks: cross_scenario_member_pairing
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_ssp245_hist_member_pairing(self, gcm):
        """D1: every SSP245 member has a match in the historical store."""
        result = SRMDatasetValidator(gcm=gcm, scenario="SSP245").check_lineage_member_availability()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"

    # ensemble-checks: cross_scenario_member_pairing
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_g6_ssp245_member_pairing(self, gcm):
        """D2: every G6 member has a match in the SSP245 store."""
        result = SRMDatasetValidator(
            gcm=gcm, scenario="G6-1.5K"
        ).check_lineage_member_availability()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"


class TestVariablePhysics:
    """Variable range, temperature consistency, and identity checks."""

    def _skip_if_not_applicable(self, ds_info):
        if ds_info.name in _SKIP_ALL_PHYSICS:
            pytest.skip(f"{ds_info.name} is not a climate dataset")

    @pytest.fixture
    def validator(self, ds_info) -> DatasetValidator:
        return DatasetValidator(ds_info)

    # variable-checks: reasonable_ranges (spatial min/max on single day; catches unit mismatches)
    @pytest.mark.parametrize("var", list(VAR_SPATIAL_RANGES))
    def test_spatial_range(self, ds_info, validator, var):
        self._skip_if_not_applicable(ds_info)
        if ds_info.name in _SKIP_SPATIAL_RANGE:
            pytest.skip(f"{ds_info.name} excluded from spatial range checks")
        if var not in validator.ds.data_vars:
            pytest.skip(f"{var} not in {ds_info.name}")
        result = validator.validate_spatial_range(var)
        assert result, f"{ds_info.name}: {result.issues}"

    # variable-checks: temperature_consistency (tasmax > tas > tasmin; single day only — not all time steps)
    def test_temperature_consistency(self, ds_info, validator):
        self._skip_if_not_applicable(ds_info)
        if ds_info.name in _SKIP_TEMP_CONSISTENCY:
            pytest.skip(f"{ds_info.name} excluded from temp consistency check")
        result = validator.validate_temp_consistency()
        assert result, f"{ds_info.name}: {result.issues}"

    # variable-checks: no_identical_vars
    def test_no_identical_vars(self, ds_info, validator):
        self._skip_if_not_applicable(ds_info)
        result = validator.validate_no_identical_vars()
        assert result, f"{ds_info.name}: {result.issues}"


class TestSpatialConsistency:
    """All datasets from the same GCM must share the same lat/lon grid."""

    # spatial-checks: grid_consistency
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_same_gcm_grid(self, gcm):
        import numpy as np

        gcm_datasets = [
            entry
            for name, entry in catalog.datasets.items()
            if gcm in name and not isinstance(entry, VirtualDataset)
        ]

        def _to_ds(entry):
            if isinstance(entry, Datatree):
                return entry.to_xarray()["historical"].ds
            return entry.to_xarray()

        reference_ds = _to_ds(gcm_datasets[0])
        ref_lat = reference_ds["lat"].values
        ref_lon = reference_ds["lon"].values

        issues = []
        for entry in gcm_datasets[1:]:
            ds = _to_ds(entry)
            try:
                np.testing.assert_array_equal(ref_lat, ds["lat"].values)
                np.testing.assert_array_equal(ref_lon, ds["lon"].values)
            except AssertionError as exc:
                issues.append(f"{entry.name}: {exc}")

        assert not issues, "\n".join(issues)


class TestEnsembleSpread:
    """Ensemble spread must be nonzero — members should differ."""

    # ensemble-checks: spread (global mean of tas differs across members)
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    @pytest.mark.parametrize("scenario", list(SCENARIO_OPTIONS))
    def test_ensemble_spread_nonzero(self, gcm, scenario):
        key = f"{gcm}-unified-icechunk"
        try:
            entry = catalog.get(key)
        except KeyError:
            pytest.skip(f"{key} not in catalog")

        group = _SCENARIO_TO_GROUP[scenario]
        ds = entry.to_xarray()[group].ds
        validator = DatasetValidator(ds)
        result = validator.validate_ensemble_spread()
        assert result, f"{key}/{group}: {result.issues}"


class TestDataIntegrity:
    """E: Data integrity checks."""

    # variable-checks: no_identical_vars (cross-scenario)
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_g6_not_identical_to_ssp245(self, gcm):
        """E1: G6-1.5K data must differ from SSP245 for the same ensemble member."""
        result = SRMDatasetValidator(gcm=gcm, scenario="G6-1.5K").check_g6_not_identical_to_ssp245()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"

    # temporal-checks: coverage
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    @pytest.mark.parametrize("scenario", list(SCENARIO_OPTIONS))
    def test_temporal_coverage(self, gcm, scenario):
        """E2: time axis must be gapless with correct first and last dates."""
        result = SRMDatasetValidator(gcm=gcm, scenario=scenario).check_temporal_coverage()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"
