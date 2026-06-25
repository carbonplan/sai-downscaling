"""Unit tests for srm.validation.DatasetValidator.

All tests mock catalog.datasets so no S3 access is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pydantic
import pytest
import xarray as xr

from srm.validation import (
    _DS_CHECKER_CHECKS,
    OUTPUT_CHECKS,
    CheckStatus,
    DatasetValidator,
    validate_output_store,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_datatree(**groups: xr.Dataset) -> xr.DataTree:
    """Build a real xr.DataTree from keyword group_name=dataset pairs."""
    return xr.DataTree.from_dict({f"/{name}": ds for name, ds in groups.items()})


def _datatree_entry(**groups: xr.Dataset) -> MagicMock:
    """Return a mock catalog entry whose to_xarray() returns an xr.DataTree."""
    entry = MagicMock()
    entry.to_xarray.return_value = _make_datatree(**groups)
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
    """Replace catalog.datasets with a plain dict.  Tests populate it as needed."""
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
        mock_datasets["CESM2-WACCM"] = _datatree_entry(ssp245=_ds_with_members("r1i1p1f1"))
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_ensemble_member_dim()
        assert result.status == CheckStatus.PASS

    def test_fail_missing_dim(self, mock_datasets):
        mock_datasets["CESM2-WACCM"] = _datatree_entry(ssp245=_ds_no_members())
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_ensemble_member_dim()
        assert result.status == CheckStatus.FAIL
        assert "missing" in result.message

    def test_fail_dataset_not_in_catalog(self, mock_datasets):
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_ensemble_member_dim()
        assert result.status == CheckStatus.FAIL
        assert "not found" in result.message

    def test_fail_load_error(self, mock_datasets):
        mock_datasets["CESM2-WACCM"] = _failing_entry()
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_ensemble_member_dim()
        assert result.status == CheckStatus.FAIL
        assert "traceback" in result.detail


# ── check_lineage_member_availability ────────────────────────────────────────

# CESM2-WACCM G6-1.5K requires SSP245 bridge members: 001-003, 007-009
_CESM2_G6_SSP245_MEMBERS = ("001", "002", "003", "007", "008", "009")

# Historical members needed for CESM2-WACCM (merged from standard + pangeo stores)
_CESM2_HIST_MEMBERS = ("001", "r1i1p1f1", "r2i1p1f1", "r3i1p1f1")


class TestCheckLineageMemberAvailability:
    def test_fail_ukesm_datatree_missing(self, mock_datasets):
        result = DatasetValidator(
            gcm="UKESM", scenario="SSP245"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.FAIL
        assert "not found" in result.message

    def test_pass_ukesm_ssp245_all_hist_present(self, mock_datasets):
        mock_datasets["UKESM"] = _datatree_entry(
            historical=_ds_with_members("r2i1p1f2", "r3i1p1f2", "r12i1p1f2")
        )
        result = DatasetValidator(
            gcm="UKESM", scenario="SSP245"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.PASS

    def test_skip_historical_scenario(self, mock_datasets):
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="historical"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.SKIP

    def test_fail_datatree_missing(self, mock_datasets):
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="SSP245"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.FAIL
        assert "not found" in result.message

    def test_pass_ssp245_all_hist_present(self, mock_datasets):
        mock_datasets["CESM2-WACCM"] = _datatree_entry(
            historical=_ds_with_members(*_CESM2_HIST_MEMBERS)
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="SSP245"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.PASS
        assert "historical" in result.message

    def test_fail_ssp245_hist_member_missing(self, mock_datasets):
        # Missing "001" from historical — needed for tasmax/tasmin members 007-010
        mock_datasets["CESM2-WACCM"] = _datatree_entry(
            historical=_ds_with_members("r1i1p1f1", "r2i1p1f1", "r3i1p1f1")  # "001" absent
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="SSP245"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.FAIL
        assert "001" in result.detail["missing_historical"]

    def test_pass_g6_all_members_present(self, mock_datasets):
        mock_datasets["CESM2-WACCM"] = _datatree_entry(
            historical=_ds_with_members(*_CESM2_HIST_MEMBERS),
            ssp245=_ds_with_members(*_CESM2_G6_SSP245_MEMBERS),
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.PASS
        assert "SSP245" in result.message

    def test_fail_g6_ssp245_bridge_member_missing(self, mock_datasets):
        # Missing SSP245 "009" — bridge for G6 member 001 tasmax/tasmin
        mock_datasets["CESM2-WACCM"] = _datatree_entry(
            historical=_ds_with_members(*_CESM2_HIST_MEMBERS),
            ssp245=_ds_with_members("001", "002", "003", "007", "008"),  # 009 missing
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.FAIL
        assert "009" in result.detail["missing_ssp245"]

    def test_fail_g6_ssp245_group_missing(self, mock_datasets):
        # Datatree has historical but no ssp245 group
        mock_datasets["CESM2-WACCM"] = _datatree_entry(
            historical=_ds_with_members(*_CESM2_HIST_MEMBERS)
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_lineage_member_availability()
        assert result.status == CheckStatus.FAIL
        assert "not present" in result.message


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

    def test_skip_g6_group_missing(self, mock_datasets):
        # Datatree exists but no g6_1p5k group
        mock_datasets["CESM2-WACCM"] = _datatree_entry(ssp245=_ds_with_data(1.0))
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.SKIP

    def test_fail_datatree_not_in_catalog(self, mock_datasets):
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.FAIL

    def test_skip_ssp245_group_missing(self, mock_datasets):
        mock_datasets["CESM2-WACCM"] = _datatree_entry(g6_1p5k=_ds_with_data(1.0))
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.SKIP

    def test_fail_load_datatree(self, mock_datasets):
        mock_datasets["CESM2-WACCM"] = _failing_entry()
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.FAIL
        assert "traceback" in result.detail

    def test_pass_data_differs(self, mock_datasets):
        mock_datasets["CESM2-WACCM"] = _datatree_entry(
            g6_1p5k=_ds_with_data(1.0),
            ssp245=_ds_with_data(2.0),
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.PASS

    def test_fail_data_identical(self, mock_datasets):
        mock_datasets["CESM2-WACCM"] = _datatree_entry(
            g6_1p5k=_ds_with_data(1.0),
            ssp245=_ds_with_data(1.0),
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.FAIL
        assert "identical" in result.message.lower()

    def test_skip_no_shared_members(self, mock_datasets):
        mock_datasets["CESM2-WACCM"] = _datatree_entry(
            g6_1p5k=_ds_with_data(1.0, "r1i1p1f1"),
            ssp245=_ds_with_data(1.0, "r2i1p1f1"),
        )
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.SKIP
        assert "No shared ensemble members" in result.message

    def test_skip_no_common_vars(self, mock_datasets):
        g6_ds = xr.Dataset({"tas": (["time"], np.zeros(3))}, coords={"time": range(3)})
        ssp245_ds = xr.Dataset({"pr": (["time"], np.zeros(3))}, coords={"time": range(3)})
        mock_datasets["CESM2-WACCM"] = _datatree_entry(g6_1p5k=g6_ds, ssp245=ssp245_ds)
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="G6-1.5K"
        ).check_g6_not_identical_to_ssp245()
        assert result.status == CheckStatus.SKIP
        assert "No common variables" in result.message


# ── check_temporal_coverage ───────────────────────────────────────────────────


class TestCheckTemporalCoverage:
    def test_fail_datatree_missing(self, mock_datasets):
        # GCM datatree absent → FAIL (not SKIP; can't determine temporal coverage at all)
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.FAIL

    def test_skip_scenario_group_missing(self, mock_datasets):
        # Datatree exists but SSP245 group absent → SKIP
        mock_datasets["CESM2-WACCM"] = _datatree_entry()
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.SKIP

    def test_skip_no_time_dim(self, mock_datasets):
        ds = xr.Dataset({"tas": (["lat"], np.zeros(3))}, coords={"lat": range(3)})
        mock_datasets["CESM2-WACCM"] = _datatree_entry(ssp245=ds)
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.SKIP

    def test_fail_load_error(self, mock_datasets):
        mock_datasets["CESM2-WACCM"] = _failing_entry()
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.FAIL

    def test_pass_correct_ssp245_coverage(self, mock_datasets):
        ds = _ds_with_time("2015-01-01", "2099-12-31")
        mock_datasets["CESM2-WACCM"] = _datatree_entry(ssp245=ds)
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.PASS
        assert result.detail["actual_start"] == "2015-01-01"
        assert result.detail["actual_end"] == "2099-12-31"

    def test_fail_wrong_start_date(self, mock_datasets):
        ds = _ds_with_time("2016-01-01", "2100-12-31")
        mock_datasets["CESM2-WACCM"] = _datatree_entry(ssp245=ds)
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.FAIL
        assert "start date" in result.message

    def test_fail_wrong_end_date(self, mock_datasets):
        ds = _ds_with_time("2015-01-01", "2098-12-31")
        mock_datasets["CESM2-WACCM"] = _datatree_entry(ssp245=ds)
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").check_temporal_coverage()
        assert result.status == CheckStatus.FAIL
        assert "end date" in result.message

    def test_pass_correct_g6_coverage(self, mock_datasets):
        ds = _ds_with_time("2035-01-01", "2085-01-01")
        mock_datasets["CESM2-WACCM"] = _datatree_entry(g6_1p5k=ds)
        result = DatasetValidator(gcm="CESM2-WACCM", scenario="G6-1.5K").check_temporal_coverage()
        assert result.status == CheckStatus.PASS

    def test_pass_correct_historical_coverage(self, mock_datasets):
        ds = _ds_with_time("1850-01-01", "2015-01-16")
        mock_datasets["CESM2-WACCM"] = _datatree_entry(historical=ds)
        result = DatasetValidator(
            gcm="CESM2-WACCM", scenario="historical"
        ).check_temporal_coverage()
        assert result.status == CheckStatus.PASS


# ── run_checks ────────────────────────────────────────────────────────────────


class TestValidate:
    def test_returns_list_of_results(self, mock_datasets):
        # Need cftime time axis so check_temporal_coverage doesn't error on ds.time.dt.calendar
        mock_datasets["CESM2-WACCM"] = _datatree_entry(
            ssp245=_ds_with_time("2020-01-01", "2020-01-10")
        )
        results = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").run_checks()
        assert len(results) == len(_DS_CHECKER_CHECKS) + 3

    def test_returns_single_error_when_open_fails(self, mock_datasets):
        # No datatree in catalog → _open_scenario_ds fails → run_checks returns [err]
        results = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").run_checks()
        assert len(results) == 1
        assert results[0].status == CheckStatus.FAIL

    def test_check_ids_stamped(self, mock_datasets):
        mock_datasets["CESM2-WACCM"] = _datatree_entry(
            ssp245=_ds_with_time("2020-01-01", "2020-01-10")
        )
        results = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245").run_checks()
        check_ids = [r.check_id for r in results]
        for check_id, _, _ in _DS_CHECKER_CHECKS:
            assert check_id in check_ids
        assert "temporal_coverage" in check_ids
        assert "lineage_member_availability" in check_ids
        assert "g6_not_identical_to_ssp245" in check_ids

    def test_all_results_have_correct_gcm_scenario(self, mock_datasets):
        results = DatasetValidator(gcm="MIROC-ES2H", scenario="historical").run_checks()
        for r in results:
            assert r.gcm == "MIROC-ES2H"
            assert r.scenario == "historical"


# ── output store validation ──────────────────────────────────────────────────


# In-range fill values per variable so validate_spatial_range passes on the dummy leaf.
_VAR_FILL = {"tas": 280.0, "tasmax": 290.0, "tasmin": 270.0, "pr": 0.001, "rsds": 200.0}


def _ds_single_var(var: str) -> xr.Dataset:
    """Minimal single-variable dataset matching a /scenario/variable/member leaf."""
    time = xr.date_range("2015-01-01", periods=3, freq="D", calendar="proleptic_gregorian")
    ds = xr.Dataset(
        {var: (["time", "lat", "lon"], np.full((3, 2, 2), _VAR_FILL.get(var, 1.0)))},
        coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    ds.time.encoding["calendar"] = "proleptic_gregorian"
    return ds


def _make_output_datatree(
    scenarios: list[str], variables: list[str], members: list[str]
) -> xr.DataTree:
    """Build a DataTree mirroring the real /scenario/variable/member output store structure.

    Intermediate nodes (/scenario, /scenario/variable) have no data_vars — only member
    leaves do.
    """
    return xr.DataTree.from_dict(
        {
            f"/{scenario}/{var}/{member}": _ds_single_var(var)
            for scenario in scenarios
            for var in variables
            for member in members
        }
    )


def test_validate_output_store():
    tree = _make_output_datatree(
        scenarios=["ssp245"],
        variables=["tasmax"],
        members=["006", "007"],
    )
    results = validate_output_store(tree)
    assert {r.scenario for r in results} == {"/ssp245/tasmax/006", "/ssp245/tasmax/007"}
    applicable = [c for c in OUTPUT_CHECKS if c[2].get("var") in (None, "tasmax")]
    assert len(results) == len(applicable) * 2
    assert all(r.gcm == "output" for r in results)
    # Single-variable leaves with valid coords/data must produce no failures.
    assert [r for r in results if r.status == CheckStatus.FAIL] == []


def test_validate_output_store_filters():
    tree = _make_output_datatree(
        scenarios=["ssp245", "historical"],
        variables=["tas", "pr"],
        members=["006", "007"],
    )

    def leaves(results):
        return sorted({r.scenario for r in results})

    assert leaves(validate_output_store(tree, scenarios=["ssp245"])) == [
        "/ssp245/pr/006",
        "/ssp245/pr/007",
        "/ssp245/tas/006",
        "/ssp245/tas/007",
    ]
    assert leaves(validate_output_store(tree, variables=["tas"])) == [
        "/historical/tas/006",
        "/historical/tas/007",
        "/ssp245/tas/006",
        "/ssp245/tas/007",
    ]
    assert leaves(validate_output_store(tree, scenarios=["ssp245"], variables=["pr"])) == [
        "/ssp245/pr/006",
        "/ssp245/pr/007",
    ]
    # Unknown filter values match nothing rather than erroring.
    assert validate_output_store(tree, scenarios=["nope"]) == []


def test_parse_variable():
    from srm.validation import parse_variable

    assert parse_variable("tas") == "tas"
    with pytest.raises(ValueError, match="Unknown variable"):
        parse_variable("TAS")


def test_open_output_datatree_rejects_non_s3():
    from srm.validation import _open_output_datatree

    with pytest.raises(ValueError, match="must be an s3:// URI"):
        _open_output_datatree("gs://bucket/key")
