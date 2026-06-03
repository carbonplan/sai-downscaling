"""Unit tests for srm.validation.DatasetValidator.

All tests mock catalog.datasets so no S3 access is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pydantic
import pytest
import xarray as xr

from srm.validation import CheckStatus, DatasetValidator

# ── helpers ──────────────────────────────────────────────────────────────────


def _catalog_entry(ds: xr.Dataset) -> MagicMock:
    """Return a mock catalog entry whose to_xarray() returns ds (any kwargs accepted)."""
    entry = MagicMock()
    entry.to_xarray.return_value = ds
    return entry


def _failing_entry(msg: str = "S3 error") -> MagicMock:
    """Return a mock catalog entry whose to_xarray() raises RuntimeError."""
    entry = MagicMock()
    entry.to_xarray.side_effect = RuntimeError(msg)
    return entry


def _ds_with_members(*members: str) -> xr.Dataset:
    """Minimal xr.Dataset with an ensemble_member dimension."""
    n = len(members)
    return xr.Dataset(
        {"tas": (["ensemble_member", "time"], np.zeros((n, 3)))},
        coords={"ensemble_member": list(members), "time": range(3)},
    )


def _ds_no_members() -> xr.Dataset:
    """Minimal xr.Dataset WITHOUT an ensemble_member dimension."""
    return xr.Dataset(
        {"tas": (["time"], np.zeros(3))},
        coords={"time": range(3)},
    )


def _ds_with_time(start: str, end: str, freq: str = "D", calendar: str = "standard") -> xr.Dataset:
    """Minimal xr.Dataset with a cftime time axis."""
    times = xr.date_range(start=start, end=end, freq=freq, calendar=calendar, use_cftime=True)
    return xr.Dataset(
        {"tas": (["time"], np.zeros(len(times)))},
        coords={"time": times},
    )


@pytest.fixture()
def mock_datasets(monkeypatch):
    """
    Replace catalog.datasets with a plain dict.  Tests populate it as needed.
    """
    datasets: dict = {}
    monkeypatch.setattr("srm.validation.catalog.datasets", datasets)
    return datasets


# ── construction ─────────────────────────────────────────────────────────────


class TestDatasetValidatorConstruction:
    def test_valid_inputs(self):
        v = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245")
        assert v.gcm == "CESM2-WACCM"
        assert v.scenario == "SSP245"

    def test_invalid_gcm_raises(self):
        with pytest.raises(pydantic.ValidationError, match="GCM"):
            DatasetValidator(gcm="INVALID-GCM", scenario="SSP245")

    def test_invalid_scenario_raises(self):
        with pytest.raises(pydantic.ValidationError, match="scenario"):
            DatasetValidator(gcm="CESM2-WACCM", scenario="RCP85")

    def test_frozen(self):
        v = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245")
        with pytest.raises(pydantic.ValidationError):
            v.gcm = "MIROC-ES2H"  # type: ignore[misc]


# ── check_ensemble_member_dim ─────────────────────────────────────────────────


class TestCheckEnsembleMemberDim:
    def test_pass(self, mock_datasets):
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(_ds_with_members("r1i1p1f1"))
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_ensemble_member_dim()
        assert result.status == CheckStatus.PASS

    def test_fail_missing_dim(self, mock_datasets):
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(_ds_no_members())
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_ensemble_member_dim()
        assert result.status == CheckStatus.FAIL
        assert "missing" in result.message

    def test_fail_dataset_not_in_catalog(self, mock_datasets):
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_ensemble_member_dim()
        assert result.status == CheckStatus.FAIL
        assert "not found" in result.message

    def test_fail_load_error(self, mock_datasets):
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _failing_entry()
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_ensemble_member_dim()
        assert result.status == CheckStatus.FAIL
        assert "traceback" in result.detail


# ── check_lineage_member_availability ────────────────────────────────────────

# CESM2-WACCM G6-1.5K requires SSP245 bridge members: 001-003, 007-009
_CESM2_G6_SSP245_MEMBERS = ("001", "002", "003", "007", "008", "009")


class TestCheckLineageMemberAvailability:
    def test_skip_no_lineage_registered(self, mock_datasets):
        # UKESM has no lineage entries registered
        result = DatasetValidator(
            gcm="UKESM", scenario="SSP245"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.SKIP
        assert "No lineage registered" in result.message

    def test_skip_historical_scenario(self, mock_datasets):
        # No lineage entries for any historical scenario
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="historical"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.SKIP

    def test_fail_historical_store_missing(self, mock_datasets):
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="SSP245"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.FAIL
        assert "not found" in result.message

    def test_pass_ssp245_all_hist_present(self, mock_datasets):
        mock_datasets["CESM2-WACCM-historical-icechunk"] = _catalog_entry(_ds_with_members("001"))
        mock_datasets["pangeo-CESM2-WACCM-historical-icechunk"] = _catalog_entry(
            _ds_with_members("r1i1p1f1", "r2i1p1f1", "r3i1p1f1")
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="SSP245"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.PASS
        assert "historical" in result.message

    def test_fail_ssp245_hist_member_missing(self, mock_datasets):
        # Missing "001" from standard store — needed for tasmax/tasmin members 007-010
        mock_datasets["CESM2-WACCM-historical-icechunk"] = _catalog_entry(_ds_with_members())
        mock_datasets["pangeo-CESM2-WACCM-historical-icechunk"] = _catalog_entry(
            _ds_with_members("r1i1p1f1", "r2i1p1f1", "r3i1p1f1")
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="SSP245"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.FAIL
        assert "001" in result.detail["missing_historical"]

    def test_pass_g6_all_members_present(self, mock_datasets):
        mock_datasets["CESM2-WACCM-historical-icechunk"] = _catalog_entry(_ds_with_members("001"))
        mock_datasets["pangeo-CESM2-WACCM-historical-icechunk"] = _catalog_entry(
            _ds_with_members("r1i1p1f1", "r2i1p1f1", "r3i1p1f1")
        )
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(
            _ds_with_members(*_CESM2_G6_SSP245_MEMBERS)
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.PASS
        assert "SSP245" in result.message

    def test_fail_g6_ssp245_bridge_member_missing(self, mock_datasets):
        # Missing SSP245 "009" — bridge for G6 member 001 tasmax/tasmin
        mock_datasets["CESM2-WACCM-historical-icechunk"] = _catalog_entry(_ds_with_members("001"))
        mock_datasets["pangeo-CESM2-WACCM-historical-icechunk"] = _catalog_entry(
            _ds_with_members("r1i1p1f1", "r2i1p1f1", "r3i1p1f1")
        )
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(
            _ds_with_members("001", "002", "003", "007", "008")  # 009 missing
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.FAIL
        assert "009" in result.detail["missing_ssp245"]

    def test_fail_g6_ssp245_store_missing(self, mock_datasets):
        mock_datasets["CESM2-WACCM-historical-icechunk"] = _catalog_entry(_ds_with_members("001"))
        mock_datasets["pangeo-CESM2-WACCM-historical-icechunk"] = _catalog_entry(
            _ds_with_members("r1i1p1f1", "r2i1p1f1", "r3i1p1f1")
        )
        # SSP245 store absent
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.FAIL
        assert "not found" in result.message


# ── check_g6_not_identical_to_ssp245 ─────────────────────────────────────────


def _ds_with_data(value: float, member: str = "r1i1p1f1") -> xr.Dataset:
    """Dataset with one variable, one member, deterministic values."""
    data = np.full((1, 5, 10, 10), value)
    return xr.Dataset(
        {"tas": (["ensemble_member", "time", "lat", "lon"], data)},
        coords={
            "ensemble_member": [member],
            "time": range(5),
            "lat": range(10),
            "lon": range(10),
        },
    )


class TestCheckG6NotIdenticalToSsp245:
    def test_skip_wrong_scenario(self, mock_datasets):
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="SSP245"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.SKIP

    def test_skip_g6_missing(self, mock_datasets):
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.SKIP

    def test_skip_ssp245_missing(self, mock_datasets):
        mock_datasets["CESM2-WACCM-G6-1.5K-icechunk"] = _catalog_entry(_ds_with_data(1.0))
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.SKIP

    def test_fail_load_g6(self, mock_datasets):
        mock_datasets["CESM2-WACCM-G6-1.5K-icechunk"] = _failing_entry()
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(_ds_with_data(1.0))
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.FAIL
        assert "traceback" in result.detail

    def test_pass_data_differs(self, mock_datasets):
        mock_datasets["CESM2-WACCM-G6-1.5K-icechunk"] = _catalog_entry(_ds_with_data(1.0))
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(_ds_with_data(2.0))
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.PASS

    def test_fail_data_identical(self, mock_datasets):
        mock_datasets["CESM2-WACCM-G6-1.5K-icechunk"] = _catalog_entry(_ds_with_data(1.0))
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(_ds_with_data(1.0))
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.FAIL
        assert "identical" in result.message.lower()

    def test_skip_no_shared_members(self, mock_datasets):
        mock_datasets["CESM2-WACCM-G6-1.5K-icechunk"] = _catalog_entry(
            _ds_with_data(1.0, "r1i1p1f1")
        )
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(
            _ds_with_data(1.0, "r2i1p1f1")
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.SKIP
        assert "No shared ensemble members" in result.message

    def test_skip_no_common_vars(self, mock_datasets):
        g6_ds = xr.Dataset({"tas": (["time"], np.zeros(3))}, coords={"time": range(3)})
        ssp245_ds = xr.Dataset({"pr": (["time"], np.zeros(3))}, coords={"time": range(3)})
        mock_datasets["CESM2-WACCM-G6-1.5K-icechunk"] = _catalog_entry(g6_ds)
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(ssp245_ds)
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.SKIP
        assert "No common variables" in result.message


# ── check_temporal_coverage ───────────────────────────────────────────────────


class TestCheckTemporalCoverage:
    def test_skip_missing_dataset(self, mock_datasets):
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.SKIP

    def test_skip_no_time_dim(self, mock_datasets):
        ds = xr.Dataset({"tas": (["lat"], np.zeros(3))}, coords={"lat": range(3)})
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(ds)
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.SKIP

    def test_fail_load_error(self, mock_datasets):
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _failing_entry()
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.FAIL

    def test_pass_correct_ssp245_coverage(self, mock_datasets):
        ds = _ds_with_time("2015-01-01", "2101-01-01")
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(ds)
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.PASS
        assert result.detail["actual_start"] == "2015-01-01"
        assert result.detail["actual_end"] == "2101-01-01"

    def test_fail_wrong_start_date(self, mock_datasets):
        ds = _ds_with_time("2016-01-01", "2100-12-31")
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(ds)
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.FAIL
        assert "start date" in result.message

    def test_fail_wrong_end_date(self, mock_datasets):
        ds = _ds_with_time("2015-01-01", "2099-12-31")
        mock_datasets["CESM2-WACCM-SSP245-icechunk"] = _catalog_entry(ds)
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.FAIL
        assert "end date" in result.message

    def test_pass_correct_g6_coverage(self, mock_datasets):
        ds = _ds_with_time("2035-01-01", "2085-01-01")
        mock_datasets["CESM2-WACCM-G6-1.5K-icechunk"] = _catalog_entry(ds)
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="G6-1.5K").check_temporal_coverage()
        assert result.status == CheckStatus.PASS

    def test_pass_correct_historical_coverage(self, mock_datasets):
        ds = _ds_with_time("1978-01-01", "2015-01-16")
        mock_datasets["CESM2-WACCM-historical-icechunk"] = _catalog_entry(ds)
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="historical"
        ).check_temporal_coverage()
        assert result.status == CheckStatus.PASS

    def test_pass_historical_with_pangeo_coverage(self, mock_datasets):
        mock_datasets["CESM2-WACCM-historical-icechunk"] = _catalog_entry(
            _ds_with_time("1978-01-01", "2015-01-16")
        )
        mock_datasets["pangeo-CESM2-WACCM-historical-icechunk"] = _catalog_entry(
            _ds_with_time("1850-01-01", "2015-01-01")
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="historical"
        ).check_temporal_coverage()
        assert result.status == CheckStatus.PASS
        assert "pangeo_actual_start" in result.detail
        assert result.detail["pangeo_actual_start"] == "1850-01-01"
        assert result.detail["pangeo_actual_end"] == "2015-01-01"

    def test_fail_historical_pangeo_wrong_end(self, mock_datasets):
        mock_datasets["CESM2-WACCM-historical-icechunk"] = _catalog_entry(
            _ds_with_time("1978-01-01", "2015-01-16")
        )
        mock_datasets["pangeo-CESM2-WACCM-historical-icechunk"] = _catalog_entry(
            _ds_with_time("1850-01-01", "2014-12-31")  # wrong end for CESM2-WACCM
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="historical"
        ).check_temporal_coverage()
        assert result.status == CheckStatus.FAIL
        assert "pangeo-historical" in result.message


# ── validate() ───────────────────────────────────────────────────────────────


class TestValidate:
    def test_returns_list_of_results(self, mock_datasets):
        results = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").run_checks()
        assert len(results) == len(DatasetValidator._CHECKS)

    def test_check_ids_stamped(self, mock_datasets):
        results = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").run_checks()
        check_ids = [r.check_id for r in results]
        assert "ensemble_member_dim" in check_ids
        assert "temporal_coverage" in check_ids
        assert "lineage_member_availability" in check_ids
        assert "g6_not_identical_to_ssp245" in check_ids
        assert "ssp245_hist_member_pairing" not in check_ids
        assert "g6_ssp245_member_pairing" not in check_ids

    def test_all_results_have_correct_gcm_scenario(self, mock_datasets):
        results = DatasetValidator(gcm="MIROC-ES2H", scenario="historical").run_checks()
        for r in results:
            assert r.gcm == "MIROC-ES2H"
            assert r.scenario == "historical"
