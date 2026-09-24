from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from saidownscale.catalog import Dataset

from saidownscale.config import SCENARIO_TO_GROUP
from saidownscale.datasets import Datatree, VirtualDataset, catalog
from saidownscale.qaqc import VAR_SPATIAL_RANGES, DatasetChecker as DatasetValidator
from saidownscale.validation import (
    GCM_OPTIONS,
    SCENARIO_OPTIONS,
    CheckStatus,
    DatasetValidator as SRMDatasetValidator,
)

pytestmark = pytest.mark.input_data

_SAMPLE_KWARGS = {"time": slice(0, 5)}

_SKIP_SPATIAL_RANGE = frozenset(
    {
        "ERA5",
        "NASA-NEX-SSP245",
        "NASA-NEX-historical",
        "GDEX-GMF",
        "ocean-mask",
    }
)

_SKIP_ALL_PHYSICS = frozenset({"ocean-mask"})

_SKIP_TEMP_CONSISTENCY = frozenset({"ERA5"})

_XFAIL_IDENTICAL_VARS: dict[str, str] = {
    "CESM2-WACCM6/ssp245": (
        "tasmax, tasmin, and tas are identical: known upstream data issue in the unified SSP245 store"
    ),
}


class TestCatalogDatasets:
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

    def test_expected_variables(self, ds_info: Dataset, validator: DatasetValidator):
        self._skip_if_virtual(ds_info)
        if not ds_info.expected_vars:
            pytest.skip(f"{ds_info.name} has no var expectations.")
        result = validator.validate_expected_variables()
        assert result, f"variable mismatch {ds_info.name}: {result.issues}"

    def test_variable_units(self, ds_info: Dataset, validator: DatasetValidator):
        self._skip_if_virtual(ds_info)
        if not ds_info.expected_vars:
            pytest.skip(f"{ds_info.name} has no variable expectations defined.")
        result = validator.validate_units()
        assert result, f"Unit mismatch for {ds_info.name}: {result.issues}"

    def test_time_axis(self, ds_info: Dataset, validator: DatasetValidator):
        self._skip_if_virtual(ds_info)
        result = validator.validate_time_axis()
        assert result, f"{ds_info.name}: {result.issues}"

    def test_calendar(self, ds_info: Dataset, validator: DatasetValidator):
        self._skip_if_virtual(ds_info)
        result = validator.validate_calendar()
        assert result, f"{ds_info.name}: {result.issues}"

    def test_negative_precip(self, ds_info: Dataset, validator: DatasetValidator):
        self._skip_if_virtual(ds_info)
        if ds_info.expected_vars and not any(v.name == "pr" for v in ds_info.expected_vars):
            pytest.skip(f"Dataset {ds_info.name} does not contain precipitation.")
        result = validator.validate_negative_precip(isel_kwargs=_SAMPLE_KWARGS)
        assert result, f"{ds_info.name}: {result.issues}"


class TestCrossScenarioConsistency:
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_ssp245_hist_member_pairing(self, gcm):
        result = SRMDatasetValidator(gcm=gcm, scenario="SSP245").check_lineage_member_availability()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"

    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_g6_ssp245_member_pairing(self, gcm):
        result = SRMDatasetValidator(
            gcm=gcm, scenario="G6-1.5K"
        ).check_lineage_member_availability()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"


class TestVariablePhysics:
    def _skip_if_not_applicable(self, ds_info):
        if isinstance(ds_info, VirtualDataset):
            pytest.skip("Not applicable to virtual datasets")
        if ds_info.name in _SKIP_ALL_PHYSICS:
            pytest.skip(f"{ds_info.name} is not a climate dataset")

    @pytest.fixture
    def validator(self, ds_info) -> DatasetValidator:
        return DatasetValidator(ds_info)

    @pytest.mark.parametrize("var", list(VAR_SPATIAL_RANGES))
    def test_spatial_range(self, ds_info, validator, var):
        self._skip_if_not_applicable(ds_info)
        if ds_info.name in _SKIP_SPATIAL_RANGE:
            pytest.skip(f"{ds_info.name} excluded from spatial range checks")
        if var not in validator.ds.data_vars:
            pytest.skip(f"{var} not in {ds_info.name}")
        result = validator.validate_spatial_range(var, isel_kwargs=_SAMPLE_KWARGS)
        assert result, f"{ds_info.name}: {result.issues}"

    def test_dtr_consistency(self, ds_info, validator):
        self._skip_if_not_applicable(ds_info)
        result = validator.validate_dtr_consistency(isel_kwargs=_SAMPLE_KWARGS)
        assert result, f"{ds_info.name}: {result.issues}"

    def test_temperature_consistency(self, ds_info, validator):
        self._skip_if_not_applicable(ds_info)
        if ds_info.name in _SKIP_TEMP_CONSISTENCY:
            pytest.skip(f"{ds_info.name}: tasmax/tasmin are forecast fields, tas is analysis")
        result = validator.validate_temp_consistency(isel_kwargs=_SAMPLE_KWARGS)
        assert result, f"{ds_info.name}: {result.issues}"

    def test_no_identical_vars(self, ds_info, validator):
        self._skip_if_not_applicable(ds_info)
        result = validator.validate_no_identical_vars()
        if not result and ds_info.name in _XFAIL_IDENTICAL_VARS:
            pytest.xfail(f"{ds_info.name}: {_XFAIL_IDENTICAL_VARS[ds_info.name]}")
        assert result, f"{ds_info.name}: {result.issues}"


class TestSpatialConsistency:
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_same_gcm_grid(self, gcm):
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
    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    def test_g6_not_identical_to_ssp245(self, gcm):
        result = SRMDatasetValidator(gcm=gcm, scenario="G6-1.5K").check_g6_not_identical_to_ssp245()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"

    def test_cesm2_waccm_first_record_is_a_daily_mean(self):
        """#424, #521: an instantaneous rsds snapshot is ~50% dark; a daily mean is ~13%."""
        entry = catalog.get("CESM2-WACCM6")
        if entry is None or not isinstance(entry, Datatree):
            pytest.skip("No unified datatree found for CESM2-WACCM6")

        dt = entry.to_xarray()
        issues = []
        for group in dt.children:
            ds = dt[group].to_dataset()
            if "rsds" not in ds.data_vars:
                continue
            for member in [str(m) for m in ds.ensemble_member.values]:
                first = ds["rsds"].sel(ensemble_member=member).isel(time=0).compute()
                if bool(np.isnan(first).all()):
                    continue
                dark = float((first == 0).mean())
                if dark > 0.4:
                    issues.append(
                        f"{group}/{member} {str(ds.time.values[0])[:10]}: first record is "
                        f"{dark:.0%} dark, i.e. an instantaneous snapshot rather than a daily mean"
                    )
        assert not issues, "\n".join(issues)

    @pytest.mark.parametrize("gcm", list(GCM_OPTIONS))
    @pytest.mark.parametrize("scenario", list(SCENARIO_OPTIONS))
    def test_temporal_coverage(self, gcm, scenario):
        result = SRMDatasetValidator(gcm=gcm, scenario=scenario).check_temporal_coverage()
        if result.status == CheckStatus.SKIP:
            pytest.skip(result.message)
        assert result.status == CheckStatus.PASS, f"{result.message} | {result.detail}"
