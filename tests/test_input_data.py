from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from srm.catalog import Dataset

from srm.config import SCENARIO_TO_GROUP
from srm.datasets import Datatree, VirtualDataset, catalog
from srm.qaqc import VAR_SPATIAL_RANGES, DatasetChecker as DatasetValidator
from srm.validation import (
    GCM_OPTIONS,
    SCENARIO_OPTIONS,
    CheckStatus,
    DatasetValidator as SRMDatasetValidator,
)

pytestmark = pytest.mark.input_data

# Spot-check slice: negative precip and spatial range are systematic errors (unit/sign
# problems) that manifest in any small sample. Reading the full array is unnecessary.
_SAMPLE_KWARGS = {"time": slice(0, 5)}

# Datasets too large or non-GCM for the expensive spatial range check.
# ERA5 and GDEX still get other consistency checks (time axis, calendar, units, etc.).
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


# ERA5 tasmin/tasmax are forecast fields (minimum/maximum_2m_temperature_since_previous_post_processing)
# while ERA5 tas is an analysis field (instantaneous 2m_temperature). The product mismatch
# causes systematic tasmax < tas and tasmin < tas violations across grid points.
_SKIP_TEMP_CONSISTENCY = frozenset({"ERA5"})

# Known data issues where identical-variable failures are expected. The test is marked
# xfail (not skipped) so that an unexpected pass signals the upstream issue was resolved.
# Maps ds_info.name → human-readable reason.
_XFAIL_IDENTICAL_VARS: dict[str, str] = {
    "CESM2-WACCM/ssp245": (
        "tasmax, tasmin, and tas are identical — known upstream data issue in the unified SSP245 store"
    ),
}


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

    # spatial-checks: coordinate_names, coordinate_ranges
    def test_longitude_valid(self, ds_info: Dataset, validator: DatasetValidator):
        self._skip_if_virtual(ds_info)
        result = validator.validate_lon(check_monotonic=True)
        assert result, f"{ds_info.name}: {result.issues}"

    # spatial-checks: coordinate_names, coordinate_ranges
    def test_latitude_valid(self, ds_info: Dataset, validator: DatasetValidator):
        self._skip_if_virtual(ds_info)
        result = validator.validate_lat(check_monotonic=True)
        assert result, f"{ds_info.name}: {result.issues}"

    # variable-checks: variable_presence
    def test_expected_variables(self, ds_info: Dataset, validator: DatasetValidator):
        """check existing data variables against known variables in catalog"""
        self._skip_if_virtual(ds_info)
        if not ds_info.expected_vars:
            pytest.skip(f"{ds_info.name} has no var expectations.")
        result = validator.validate_expected_variables()
        assert result, f"variable mismatch {ds_info.name}: {result.issues}"

    # variable-checks: units
    def test_variable_units(self, ds_info: Dataset, validator: DatasetValidator):
        """check variable units match"""
        self._skip_if_virtual(ds_info)
        if not ds_info.expected_vars:
            pytest.skip(f"{ds_info.name} has no variable expectations defined.")
        result = validator.validate_units()
        assert result, f"Unit mismatch for {ds_info.name}: {result.issues}"

    # temporal-checks: monotonic, no_duplicate_timestamps, no_internal_gaps
    def test_time_axis(self, ds_info: Dataset, validator: DatasetValidator):
        self._skip_if_virtual(ds_info)
        result = validator.validate_time_axis()
        assert result, f"{ds_info.name}: {result.issues}"

    # temporal-checks: calendar
    def test_calendar(self, ds_info: Dataset, validator: DatasetValidator):
        """check calendar is proleptic_gregorian and datetime64"""
        self._skip_if_virtual(ds_info)
        result = validator.validate_calendar()
        assert result, f"{ds_info.name}: {result.issues}"

    # variable-checks: reasonable_ranges (pr >= 0)
    def test_negative_precip(self, ds_info: Dataset, validator: DatasetValidator):
        """Spot-check first 5 time steps — negative pr is a systematic sign/unit error."""
        self._skip_if_virtual(ds_info)
        if ds_info.expected_vars and not any(v.name == "pr" for v in ds_info.expected_vars):
            pytest.skip(f"Dataset {ds_info.name} does not contain precipitation.")
        result = validator.validate_negative_precip(isel_kwargs=_SAMPLE_KWARGS)
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
        if isinstance(ds_info, VirtualDataset):
            pytest.skip("Not applicable to virtual datasets")
        if ds_info.name in _SKIP_ALL_PHYSICS:
            pytest.skip(f"{ds_info.name} is not a climate dataset")

    @pytest.fixture
    def validator(self, ds_info) -> DatasetValidator:
        return DatasetValidator(ds_info)

    # variable-checks: reasonable_ranges (spatial min/max; catches unit mismatches)
    @pytest.mark.parametrize("var", list(VAR_SPATIAL_RANGES))
    def test_spatial_range(self, ds_info, validator, var):
        self._skip_if_not_applicable(ds_info)
        if ds_info.name in _SKIP_SPATIAL_RANGE:
            pytest.skip(f"{ds_info.name} excluded from spatial range checks")
        if var not in validator.ds.data_vars:
            pytest.skip(f"{var} not in {ds_info.name}")
        result = validator.validate_spatial_range(var, isel_kwargs=_SAMPLE_KWARGS)
        assert result, f"{ds_info.name}: {result.issues}"

    # variable-checks: dtr_consistency (dtr ≈ tasmax − tasmin; catches unit mismatch in derived variable)
    def test_dtr_consistency(self, ds_info, validator):
        self._skip_if_not_applicable(ds_info)
        result = validator.validate_dtr_consistency(isel_kwargs=_SAMPLE_KWARGS)
        assert result, f"{ds_info.name}: {result.issues}"

    # variable-checks: temperature_consistency (tasmax > tas > tasmin; full dataset)
    def test_temperature_consistency(self, ds_info, validator):
        self._skip_if_not_applicable(ds_info)
        if ds_info.name in _SKIP_TEMP_CONSISTENCY:
            pytest.skip(f"{ds_info.name} excluded from temp consistency check")
        result = validator.validate_temp_consistency(isel_kwargs=_SAMPLE_KWARGS)
        assert result, f"{ds_info.name}: {result.issues}"

    # variable-checks: no_identical_vars
    def test_no_identical_vars(self, ds_info, validator):
        self._skip_if_not_applicable(ds_info)
        result = validator.validate_no_identical_vars()
        if not result and ds_info.name in _XFAIL_IDENTICAL_VARS:
            pytest.xfail(f"{ds_info.name}: {_XFAIL_IDENTICAL_VARS[ds_info.name]}")
        assert result, f"{ds_info.name}: {result.issues}"


class TestSpatialConsistency:
    """All scenario groups in a unified GCM datatree must share the same lat/lon grid."""

    # spatial-checks: grid_consistency
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_same_gcm_grid(self, gcm):
        import numpy as np

        entry = catalog.get(gcm)
        if entry is None or not isinstance(entry, Datatree):
            pytest.skip(f"No unified datatree found for {gcm}")

        dt = entry.to_xarray()
        available_groups = [g for g in SCENARIO_TO_GROUP.values() if g in dt.children]
        if len(available_groups) < 2:
            pytest.skip(f"Fewer than 2 scenario groups found for {gcm}")

        ref_ds = dt[available_groups[0]].to_dataset()
        ref_lat = ref_ds["lat"].values
        ref_lon = ref_ds["lon"].values

        issues = []
        for group in available_groups[1:]:
            ds = dt[group].to_dataset()
            try:
                np.testing.assert_array_equal(ref_lat, ds["lat"].values)
                np.testing.assert_array_equal(ref_lon, ds["lon"].values)
            except AssertionError as exc:
                issues.append(f"{gcm}/{group}: {exc}")

        assert not issues, "\n".join(issues)


class TestEnsembleSpread:
    """Ensemble spread must be nonzero — members should differ."""

    # ensemble-checks: spread (global mean of tas differs across members)
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    @pytest.mark.parametrize("scenario", list(SCENARIO_OPTIONS))
    def test_ensemble_spread_nonzero(self, gcm, scenario):
        gcm_entry = catalog.get(gcm)
        if gcm_entry is None or not isinstance(gcm_entry, Datatree):
            pytest.skip(f"No unified datatree found for {gcm}")

        group = SCENARIO_TO_GROUP.get(scenario)
        if group is None:
            pytest.skip(f"No group mapping for scenario {scenario}")

        dt = gcm_entry.to_xarray()
        if group not in dt.children:
            pytest.skip(f"Group '{group}' not present in datatree for {gcm}")

        ds = dt[group].to_dataset()
        validator = DatasetValidator(ds)
        result = validator.validate_ensemble_spread()
        assert result, f"{gcm}/{scenario}: {result.issues}"


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

    # variable-checks: no instantaneous initialisation record (issues #424, #521)
    def test_cesm2_waccm_first_record_is_a_daily_mean(self):
        """CAM prefixes each history stream with a zero-width initial-state record.

        The ETL drops it via ``decode_time_from_bounds``; this asserts none survived. An
        instantaneous shortwave field is dark over exactly the night hemisphere, so a first-day
        dark fraction anywhere near 0.5 means a snapshot is still being served as a daily mean.
        A genuine daily mean is dark only where the sun never rises, which in early January is
        the polar-night cap at roughly 13%.
        """
        import numpy as np

        entry = catalog.get("CESM2-WACCM")
        if entry is None or not isinstance(entry, Datatree):
            pytest.skip("No unified datatree found for CESM2-WACCM")

        dt = entry.to_xarray()
        issues = []
        for group in dt.children:
            ds = dt[group].to_dataset()
            if "rsds" not in ds.data_vars:
                continue
            for member in [str(m) for m in ds.ensemble_member.values]:
                first = ds["rsds"].sel(ensemble_member=member).isel(time=0).compute()
                if bool(np.isnan(first).all()):
                    continue  # member is NaN-padded before its own coverage starts
                dark = float((first == 0).mean())
                if dark > 0.4:
                    issues.append(
                        f"{group}/{member} {str(ds.time.values[0])[:10]}: first record is "
                        f"{dark:.0%} dark, i.e. an instantaneous snapshot rather than a daily mean"
                    )
        assert not issues, "\n".join(issues)

    # temporal-checks: coverage
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    @pytest.mark.parametrize("scenario", list(SCENARIO_OPTIONS))
    def test_temporal_coverage(self, gcm, scenario):
        """E2: time axis must be gapless with correct first and last dates."""
        result = SRMDatasetValidator(gcm=gcm, scenario=scenario).check_temporal_coverage()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"
