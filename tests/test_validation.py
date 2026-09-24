"""Unit tests for saidownscale.validation; catalog.datasets is mocked, so no S3 access."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pydantic
import pytest
import xarray as xr

from saidownscale.downscaling_config import DownscalingConfig
from saidownscale.validation import (
    _DS_CHECKER_CHECKS,
    BLOCKING_CHECKS,
    OUTPUT_CHECKS,
    CheckStatus,
    DatasetValidator,
    _open_output_datatree,
    check_config_time_domain,
    parse_variable,
    validate_output_store,
)

PASS, FAIL, SKIP = CheckStatus.PASS, CheckStatus.FAIL, CheckStatus.SKIP
CESM, UKESM = "CESM2-WACCM6", "UKESM1-1-LL"


def _entry(**groups: xr.Dataset) -> MagicMock:
    entry = MagicMock()
    entry.to_xarray.return_value = xr.DataTree.from_dict(
        {f"/{name}": ds for name, ds in groups.items()}
    )
    return entry


def _failing_entry() -> MagicMock:
    entry = MagicMock()
    entry.to_xarray.side_effect = RuntimeError("S3 error")
    return entry


def _ds_with_members(*members: str) -> xr.Dataset:
    return xr.Dataset(
        {"tas": (["ensemble_member", "time"], np.zeros((len(members), 3)))},
        coords={"ensemble_member": list(members), "time": range(3)},
    )


def _ds_1d(var: str = "tas", dim: str = "time") -> xr.Dataset:
    return xr.Dataset({var: ([dim], np.zeros(3))}, coords={dim: range(3)})


def _ds_with_time(start: str, end: str) -> xr.Dataset:
    times = xr.date_range(start=start, end=end, freq="D", calendar="standard", use_cftime=True)
    return xr.Dataset({"tas": (["time"], np.zeros(len(times)))}, coords={"time": times})


def _ds_with_data(value: float, member: str = "r1i1p1f1") -> xr.Dataset:
    return xr.Dataset(
        {"tas": (["ensemble_member", "time", "lat", "lon"], np.full((1, 5, 10, 10), value))},
        coords={"ensemble_member": [member], "time": range(5), "lat": range(10), "lon": range(10)},
    )


@pytest.fixture()
def mock_datasets(monkeypatch):
    datasets: dict = {}
    monkeypatch.setattr("saidownscale.validation.catalog.datasets", datasets)
    return datasets


def _run_cases(subtests, mock_datasets, check: str, cases):
    """Case: (id, gcm, scenario, catalog entry or None, status, message substr, detail dict)."""
    for case_id, gcm, scenario, entry, status, in_message, detail in cases:
        with subtests.test(case=case_id):
            mock_datasets.clear()
            if entry is not None:
                mock_datasets[gcm] = entry
            result = getattr(DatasetValidator(gcm=gcm, scenario=scenario), check)()
            assert result.status == status
            if in_message:
                assert in_message in result.message
            for key, value in (detail or {}).items():
                assert value in (result.detail[key] if key else result.detail)


def test_dataset_validator_construction(subtests):
    with subtests.test(case="valid"):
        v = DatasetValidator(gcm=CESM, scenario="SSP245")
        assert (v.gcm, v.scenario) == (CESM, "SSP245")
    with subtests.test(case="frozen"), pytest.raises(pydantic.ValidationError):
        v.gcm = UKESM  # type: ignore[misc]
    for field, kwargs in [
        ("GCM", {"gcm": "INVALID-GCM", "scenario": "SSP245"}),
        ("scenario", {"gcm": CESM, "scenario": "RCP85"}),
    ]:
        with subtests.test(case=f"invalid_{field}"):
            with pytest.raises(pydantic.ValidationError, match=field):
                DatasetValidator(**kwargs)


def test_check_ensemble_member_dim(subtests, mock_datasets):
    _run_cases(
        subtests,
        mock_datasets,
        "check_ensemble_member_dim",
        [
            ("pass", CESM, "SSP245", _entry(ssp245=_ds_with_members("r1i1p1f1")), PASS, "", None),
            ("missing_dim", CESM, "SSP245", _entry(ssp245=_ds_1d()), FAIL, "missing", None),
            ("not_in_catalog", CESM, "SSP245", None, FAIL, "not found", None),
            ("load_error", CESM, "SSP245", _failing_entry(), FAIL, "", {None: "traceback"}),
        ],
    )


def test_check_lineage_member_availability(subtests, mock_datasets):
    hist = _ds_with_members("001", "r1i1p1f1", "r2i1p1f1", "r3i1p1f1")
    g6_bridge = _ds_with_members("001", "002", "003", "007", "008", "009")
    _run_cases(
        subtests,
        mock_datasets,
        "check_lineage_member_availability",
        [
            ("ukesm_missing", UKESM, "SSP245", None, FAIL, "not found", None),
            (
                "ukesm_pass",
                UKESM,
                "SSP245",
                _entry(historical=_ds_with_members("u-by791")),
                PASS,
                "",
                None,
            ),
            ("historical_skips", CESM, "historical", None, SKIP, "", None),
            ("cesm_missing", CESM, "SSP245", None, FAIL, "not found", None),
            ("ssp245_pass", CESM, "SSP245", _entry(historical=hist), PASS, "historical", None),
            (
                "ssp245_hist_001_missing",
                CESM,
                "SSP245",
                _entry(historical=_ds_with_members("r1i1p1f1", "r2i1p1f1", "r3i1p1f1")),
                FAIL,
                "",
                {"missing_historical": "001"},
            ),
            (
                "g6_pass",
                CESM,
                "G6-1.5K",
                _entry(historical=hist, ssp245=g6_bridge),
                PASS,
                "SSP245",
                None,
            ),
            (
                "g6_bridge_009_missing",
                CESM,
                "G6-1.5K",
                _entry(historical=hist, ssp245=g6_bridge.isel(ensemble_member=slice(0, 5))),
                FAIL,
                "",
                {"missing_ssp245": "009"},
            ),
            (
                "g6_ssp245_group_missing",
                CESM,
                "G6-1.5K",
                _entry(historical=hist),
                FAIL,
                "not present",
                None,
            ),
        ],
    )


def test_check_g6_not_identical_to_ssp245(subtests, mock_datasets):
    one, two = _ds_with_data(1.0), _ds_with_data(2.0)
    g6 = "G6-1.5K"
    _run_cases(
        subtests,
        mock_datasets,
        "check_g6_not_identical_to_ssp245",
        [
            ("wrong_scenario", CESM, "SSP245", None, SKIP, "", None),
            ("g6_group_missing", CESM, g6, _entry(ssp245=one), SKIP, "", None),
            ("not_in_catalog", CESM, g6, None, FAIL, "", None),
            ("ssp245_group_missing", CESM, g6, _entry(g6_1p5k=one), SKIP, "", None),
            ("load_error", CESM, g6, _failing_entry(), FAIL, "", {None: "traceback"}),
            ("differs", CESM, g6, _entry(g6_1p5k=one, ssp245=two), PASS, "", None),
            ("identical", CESM, g6, _entry(g6_1p5k=one, ssp245=one), FAIL, "identical", None),
            (
                "no_shared_members",
                CESM,
                g6,
                _entry(g6_1p5k=one, ssp245=_ds_with_data(1.0, "r2i1p1f1")),
                SKIP,
                "No shared ensemble members",
                None,
            ),
            (
                "no_common_vars",
                CESM,
                g6,
                _entry(g6_1p5k=_ds_1d("tas"), ssp245=_ds_1d("pr")),
                SKIP,
                "No common variables",
                None,
            ),
        ],
    )


def test_check_temporal_coverage(subtests, mock_datasets):
    def ssp245(start, end):
        return _entry(ssp245=_ds_with_time(start, end))

    _run_cases(
        subtests,
        mock_datasets,
        "check_temporal_coverage",
        [
            ("datatree_missing_fails", CESM, "SSP245", None, FAIL, "", None),
            ("scenario_group_missing_skips", CESM, "SSP245", _entry(), SKIP, "", None),
            ("no_time_dim", CESM, "SSP245", _entry(ssp245=_ds_1d(dim="lat")), SKIP, "", None),
            ("load_error", CESM, "SSP245", _failing_entry(), FAIL, "", None),
            (
                "ssp245_pass",
                CESM,
                "SSP245",
                ssp245("2015-01-01", "2099-12-31"),
                PASS,
                "",
                {"actual_start": "2015-01-01", "actual_end": "2099-12-31"},
            ),
            (
                "wrong_start",
                CESM,
                "SSP245",
                ssp245("2016-01-01", "2100-12-31"),
                FAIL,
                "start date",
                None,
            ),
            (
                "wrong_end",
                CESM,
                "SSP245",
                ssp245("2015-01-01", "2098-12-31"),
                FAIL,
                "end date",
                None,
            ),
            (
                "g6_pass",
                CESM,
                "G6-1.5K",
                _entry(g6_1p5k=_ds_with_time("2035-01-01", "2084-12-31")),
                PASS,
                "",
                None,
            ),
            (
                "historical_pass",
                CESM,
                "historical",
                _entry(historical=_ds_with_time("1850-01-01", "2014-12-31")),
                PASS,
                "",
                None,
            ),
        ],
    )


def test_run_checks(subtests, mock_datasets):
    with subtests.test(case="open_fails_returns_single_error"):
        results = DatasetValidator(gcm=CESM, scenario="SSP245").run_checks()
        assert [r.status for r in results] == [FAIL]

    with subtests.test(case="gcm_scenario_stamped"):
        results = DatasetValidator(gcm=UKESM, scenario="historical").run_checks()
        assert {(r.gcm, r.scenario) for r in results} == {(UKESM, "historical")}

    with subtests.test(case="all_checks_run"):
        mock_datasets[CESM] = _entry(ssp245=_ds_with_time("2020-01-01", "2020-01-10"))
        results = DatasetValidator(gcm=CESM, scenario="SSP245").run_checks()
        assert len(results) == len(_DS_CHECKER_CHECKS) + 3
        assert {r.check_id for r in results} == {c[0] for c in _DS_CHECKER_CHECKS} | {
            "temporal_coverage",
            "lineage_member_availability",
            "g6_not_identical_to_ssp245",
        }


_VAR_FILL = {"tas": 280.0, "tasmax": 290.0, "tasmin": 270.0, "rsds": 200.0}


def _leaf_values(var: str) -> np.ndarray:
    if var == "pr":
        values = np.zeros((3, 2, 2))
        values[0, 0, 0] = 1e-3
        return values
    return np.full((3, 2, 2), _VAR_FILL.get(var, 1.0))


def _ds_single_var(var: str, values: np.ndarray | None = None) -> xr.Dataset:
    time = xr.date_range("2015-01-01", periods=3, freq="D", calendar="proleptic_gregorian")
    ds = xr.Dataset(
        {var: (["time", "lat", "lon"], _leaf_values(var) if values is None else values)},
        coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    ds.time.encoding["calendar"] = "proleptic_gregorian"
    return ds


def _output_tree(scenarios, variables, members, method=None) -> xr.DataTree:
    prefix = f"/{method}" if method else ""
    return xr.DataTree.from_dict(
        {
            f"{prefix}/{scenario}/{var}/{member}": _ds_single_var(var)
            for scenario in scenarios
            for var in variables
            for member in members
        }
    )


def test_validate_output_store(subtests):
    with subtests.test(case="valid_leaves_pass"):
        results = validate_output_store(_output_tree(["ssp245"], ["tasmax"], ["006", "007"]))
        assert {r.scenario for r in results} == {"/ssp245/tasmax/006", "/ssp245/tasmax/007"}
        applicable = [c for c in OUTPUT_CHECKS if c[2].get("var") in (None, "tasmax")]
        assert len(results) == len(applicable) * 2
        assert all(r.gcm == "output" for r in results)
        assert [r for r in results if r.status == FAIL] == []

    tree = _output_tree(["ssp245", "historical"], ["tas", "pr"], ["006", "007"])
    for filters, expected in [
        ({"scenarios": ["ssp245"]}, ["ssp245/pr", "ssp245/tas"]),
        ({"variables": ["tas"]}, ["historical/tas", "ssp245/tas"]),
        ({"scenarios": ["ssp245"], "variables": ["pr"]}, ["ssp245/pr"]),
        ({"scenarios": ["nope"]}, []),
    ]:
        with subtests.test(case=f"filter_{filters}"):
            leaves = sorted({r.scenario for r in validate_output_store(tree, **filters)})
            assert leaves == [f"/{p}/{m}" for p in expected for m in ("006", "007")]


@pytest.mark.parametrize("method", [None, "bcsd", "qdmsd"])
def test_validate_output_store_layouts(method, subtests):
    """Legacy and method-namespaced trees must both run var-scoped checks, not silently skip."""
    prefix = f"/{method}" if method else ""

    with subtests.test(case="var_scoped_checks_run"):
        results = validate_output_store(_output_tree(["ssp245"], ["pr"], ["006"], method))
        var_scoped = {c[0] for c in OUTPUT_CHECKS if c[2].get("var") is not None}
        ran = {r.check_id for r in results}
        assert ran & var_scoped == {"negative_precip", "spatial_range_pr"}
        assert [r for r in results if r.status == FAIL] == []

    with subtests.test(case="bad_data_fails"):
        bad = _leaf_values("pr")
        bad[1, 1, 1] = -1.0
        tree = xr.DataTree.from_dict({f"{prefix}/ssp245/pr/006": _ds_single_var("pr", bad)})
        failed = [r for r in validate_output_store(tree) if r.check_id == "negative_precip"]
        assert [r.status for r in failed] == [FAIL]

    with subtests.test(case="tasmax_ge_tasmin_fires"):
        tree = _output_tree(["ssp245"], ["tasmax", "tasmin"], ["006", "007"], method)
        gate = [r for r in validate_output_store(tree) if r.check_id == "tasmax_ge_tasmin"]
        assert sorted(r.scenario for r in gate) == [
            "ssp245/tasmax_ge_tasmin/006",
            "ssp245/tasmax_ge_tasmin/007",
        ]
        assert all(r.status == PASS for r in gate)

    with subtests.test(case="scenario_filter"):
        tree = _output_tree(["ssp245", "historical"], ["tas"], ["006"], method)
        results = validate_output_store(tree, scenarios=["ssp245"])
        assert results
        assert {r.scenario for r in results} == {f"{prefix}/ssp245/tas/006"}


def test_validate_output_store_mixed_top_level_segments():
    """debiased_coarse sits beside scenarios, so each top-level child is classified alone."""
    tree = xr.DataTree.from_dict(
        {
            "/bcsd/ssp245/pr/006": _ds_single_var("pr"),
            "/bcsd/debiased_coarse/ssp245/pr/006": _ds_single_var("pr"),
            "/ssp245/pr/007": _ds_single_var("pr"),
        }
    )
    checked = {r.scenario for r in validate_output_store(tree) if r.check_id == "negative_precip"}
    assert checked == {"/bcsd/ssp245/pr/006", "/ssp245/pr/007"}


def test_parse_variable_and_open_output_datatree_reject_bad_input():
    assert parse_variable("tas") == "tas"
    with pytest.raises(ValueError, match="Unknown variable"):
        parse_variable("TAS")
    with pytest.raises(ValueError, match="must be an s3:// URI"):
        _open_output_datatree("gs://bucket/key")


def test_check_config_time_domain(subtests):
    """Members 006-010 end 2069 (#521); G6 must start at 2035, not in SSP245 bridge years (#448)."""
    ssp, g6, end = "SSP245", "G6-1.5K", "G6-1.5K-END"
    cases = [
        ("007", 2015, 2100, ssp, FAIL, ["2069"]),
        ("007", 2015, 2069, ssp, PASS, []),
        ("001", 2015, 2099, ssp, PASS, []),
        ("003", 2015, 2100, ssp, FAIL, ["2099"]),
        *[(m, 2015, 2070, ssp, FAIL, []) for m in ("006", "007", "008", "009", "010")],
        *[(m, 2015, 2069, ssp, PASS, []) for m in ("006", "008", "009", "010")],
        ("001", 2015, 2084, g6, FAIL, ["2035", "bridge"]),
        ("001", 2035, 2084, g6, PASS, []),
        ("001", 2035, 2086, g6, FAIL, ["2084"]),
        ("002", 2085, 2100, end, PASS, []),
        *[("002", start, 2100, end, FAIL, ["2085"]) for start in (2015, 2035, 2084)],
    ]
    for member, start, stop, scenario, status, in_message in cases:
        with subtests.test(member=member, scenario=scenario, period=f"{start}-{stop}"):
            config = DownscalingConfig(
                gcm=CESM,
                downscaling_method="BCSD",
                variable="tas",
                ensemble_member=member,
                scenario=scenario,
                predict_period_start=start,
                predict_period_end=stop,
            )
            r = check_config_time_domain(config)
            assert r.status == status
            assert r.check_id == "config_time_domain"
            assert r.ensemble_member == member
            for text in in_message:
                assert text in r.message

    with subtests.test(case="historical_only_skips"):
        config = DownscalingConfig(
            gcm=UKESM,
            variable="tas",
            ensemble_member="r2i1p1f2",
            scenario=None,
            downscaling_method="BCSD",
        )
        assert check_config_time_domain(config).status == SKIP


def _temp_leaf(var: str, data: np.ndarray) -> xr.Dataset:
    time = np.arange("2035-01-01", "2035-01-04", dtype="datetime64[D]").astype("datetime64[ns]")
    return xr.Dataset(
        {var: (("time", "lat", "lon"), data.astype("float32"))},
        coords={"time": time, "lat": [0.0, 1.0], "lon": [10.0, 11.0]},
    )


def test_tasmax_ge_tasmin_output_gate(subtests):
    """#331: blocking cross-variable gate; NaN cells must not read as inversions."""
    assert "tasmax_ge_tasmin" in BLOCKING_CHECKS

    def gate(tmax, tmin):
        tree = xr.DataTree.from_dict(
            {
                "/g6_1p5k/tasmax/003": _temp_leaf("tasmax", tmax),
                "/g6_1p5k/tasmin/003": _temp_leaf("tasmin", tmin),
            }
        )
        return [r for r in validate_output_store(tree) if r.check_id == "tasmax_ge_tasmin"]

    with subtests.test(case="inversion_fails"):
        tmax, tmin = np.full((3, 2, 2), 300.0), np.full((3, 2, 2), 290.0)
        tmin[0, 0, 0] = 305.0
        (result,) = gate(tmax, tmin)
        assert result.status == FAIL
        assert "tasmax < tasmin at 1" in result.message

    with subtests.test(case="nan_passes"):
        tmax, tmin = np.full((3, 2, 2), 300.0), np.full((3, 2, 2), 290.0)
        tmax[1, 1, 1] = np.nan
        (result,) = gate(tmax, tmin)
        assert result.status == PASS
