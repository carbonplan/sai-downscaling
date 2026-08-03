"""Unit tests for BCSDPipeline._load_ssp245_bridge gap-fill logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import xarray as xr

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.pipeline import BCSDPipeline


def _make_config(**overrides) -> BCSDConfig:
    defaults = dict(
        gcm="MIROC-ES2H",
        variable="tas",
        ensemble_member="r01",
        scenario="G6-1.5K",
        predict_period_start=2015,
        predict_period_end=2100,
        subset_bounds=(-35.0, -22.0, 16.0, 33.0),
        debias_approach="nonparametric_hybrid",
    )
    defaults.update(overrides)
    return BCSDConfig(**defaults)


def _make_options(tmp_path) -> PipelineOptions:
    return PipelineOptions(
        scratch_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
        rechunk_workflow=False,
    )


def _make_annual_ds(start_year: int, end_year: int, member: str, var: str = "tas") -> xr.Dataset:
    """Annual year-start timestamps (numpy datetime64) with an ensemble_member dimension."""
    times = xr.date_range(f"{start_year}", f"{end_year}", freq="YS", use_cftime=False)
    data = np.zeros((1, len(times), 2, 2))
    da = xr.DataArray(
        data,
        dims=["ensemble_member", "time", "lat", "lon"],
        coords={
            "ensemble_member": [member],
            "time": times,
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
        },
    )
    return xr.Dataset({var: da})


def _mock_catalog_get(
    key: str, *, geomip_start: int, esgf_start: int, primary_member: str = "r01"
) -> MagicMock:
    """Return a mock catalog entry whose to_xarray() returns the right dataset."""
    mock_entry = MagicMock()
    if "esgf" in key:
        mock_entry.to_xarray.return_value = _make_annual_ds(esgf_start, 2084, "r1i1p4f2")
    else:
        mock_entry.to_xarray.return_value = _make_annual_ds(geomip_start, 2084, primary_member)
    return mock_entry


def test_bridge_returns_primary_when_esgf_member_none(tmp_path):
    """When _ssp245_esgf_member is None, _load_ssp245_bridge returns only primary (no gap check)."""
    config = _make_config(gcm="CESM2-WACCM", ensemble_member="001")
    pipeline = BCSDPipeline(config, _make_options(tmp_path))
    assert pipeline._ssp245_esgf_member is None

    geomip_start = 2015
    with patch("srm.pipeline._catalog") as mock_cat:
        mock_cat.get.side_effect = lambda k: _mock_catalog_get(
            k, geomip_start=geomip_start, esgf_start=2015, primary_member="001"
        )
        result = pipeline._load_ssp245_bridge()

    calls = [c[0][0] for c in mock_cat.get.call_args_list]
    assert all("esgf" not in k for k in calls)
    assert int(result.time.dt.year.min()) == geomip_start


def test_bridge_returns_primary_when_no_gap(tmp_path):
    """When primary starts at or before train_period_end+1 (2015), no ESGF prepend."""
    config = _make_config()  # MIROC G6-1.5K r01
    pipeline = BCSDPipeline(config, _make_options(tmp_path))
    assert pipeline._ssp245_esgf_member == "r1i1p4f2"

    with patch("srm.pipeline._catalog") as mock_cat:
        mock_cat.get.side_effect = lambda k: _mock_catalog_get(
            k, geomip_start=2015, esgf_start=2015
        )
        result = pipeline._load_ssp245_bridge()

    calls = [c[0][0] for c in mock_cat.get.call_args_list]
    assert all("esgf" not in k for k in calls), "ESGF catalog should not be accessed when no gap"
    assert int(result.time.dt.year.min()) == 2015


def test_bridge_prepends_esgf_when_gap_detected(tmp_path):
    """When GeoMIP SSP245 starts at 2020, ESGF data fills 2015–2019."""
    config = _make_config()  # MIROC G6-1.5K r01
    pipeline = BCSDPipeline(config, _make_options(tmp_path))
    assert pipeline._ssp245_member == "r01"
    assert pipeline._ssp245_esgf_member == "r1i1p4f2"

    # Primary (GeoMIP) SSP245 starts at 2020; ESGF fills the 2015-2019 gap.
    # get_experiment now uses downscaling_utils.catalog (separate from _catalog),
    # so patch it directly to control the primary start year.
    geomip_da = _make_annual_ds(2020, 2084, "r01")["tas"]
    esgf_mock_entry = MagicMock()
    esgf_mock_entry.to_xarray.return_value = _make_annual_ds(2015, 2084, "r1i1p4f2")

    with (
        patch("srm.pipeline.get_experiment", return_value=geomip_da),
        patch("srm.pipeline._catalog") as mock_cat,
    ):
        mock_cat.get.return_value = esgf_mock_entry
        result = pipeline._load_ssp245_bridge()

    years = sorted(int(y) for y in np.unique(result.time.dt.year.values))
    assert years[0] == 2015, f"Bridge must start at 2015, got {years[0]}"
    assert 2019 in years, "ESGF gap years (2015-2019) must be present"
    assert 2020 in years, "GeoMIP data (from 2020) must be present"
    assert years[-1] == 2084
    assert len(years) == len(set(years)), "No duplicate years"

    # ESGF data is now fetched via catalog.get(gcm).to_xarray(group="esgf_ssp245")
    mock_cat.get.assert_called_with(config.gcm)
    esgf_mock_entry.to_xarray.assert_called_with(group="esgf_ssp245")

    # Provenance attrs
    assert result.attrs["bridge_type"] == "esgf_geomip_stitch"
    assert result.attrs["bridge_esgf_member"] == "r1i1p4f2"
    assert result.attrs["bridge_esgf_years"] == "2015-2019"
    assert result.attrs["bridge_geomip_member"] == "r01"
    assert result.attrs["bridge_geomip_years"] == "2020-2084"
    assert result.attrs["bridge_gcm"] == "MIROC-ES2H"
    assert result.attrs["bridge_variable"] == "tas"


def test_bridge_esgf_uses_correct_member(tmp_path):
    """ESGF bridge selects ssp245_esgf_member (r1i1p4f2), not the GeoMIP member (r01)."""
    config = _make_config()
    pipeline = BCSDPipeline(config, _make_options(tmp_path))

    esgf_ds = _make_annual_ds(2015, 2084, "r1i1p4f2")  # r1i1p4f2 only
    geomip_ds = _make_annual_ds(2020, 2084, "r01")

    def _side_effect(key):
        m = MagicMock()
        if "esgf" in key:
            m.to_xarray.return_value = esgf_ds
        else:
            m.to_xarray.return_value = geomip_ds
        return m

    with patch("srm.pipeline._catalog") as mock_cat:
        mock_cat.get.side_effect = _side_effect
        result = pipeline._load_ssp245_bridge()

    # If selection used "r01" on the ESGF ds (which only has r1i1p4f2), it would raise.
    assert result is not None
    assert int(result.time.dt.year.min()) == 2015


def test_bridge_calendar_aligned_to_primary(tmp_path):
    """ESGF data with a different calendar is converted to match the primary before concat."""
    config = _make_config()
    pipeline = BCSDPipeline(config, _make_options(tmp_path))

    # Use two different cftime calendars (noleap for GeoMIP, 360_day for ESGF).
    # to_proleptic_gregorian converts both to numpy datetime64 (proleptic_gregorian),
    # ensuring no calendar mismatch at concat time.
    def _make_cftime_ds(start: str, end: str, calendar: str, member: str) -> xr.Dataset:
        times = xr.date_range(start, end, freq="YS", use_cftime=True, calendar=calendar)
        data = np.zeros((1, len(times), 2, 2))
        da = xr.DataArray(
            data,
            dims=["ensemble_member", "time", "lat", "lon"],
            coords={
                "ensemble_member": [member],
                "time": times,
                "lat": [0.0, 1.0],
                "lon": [0.0, 1.0],
            },
        )
        return xr.Dataset({"tas": da})

    geomip_ds = _make_annual_ds(2020, 2084, "r01")  # numpy datetime64, already proleptic_gregorian
    esgf_ds = _make_cftime_ds("2015", "2084", "360_day", "r1i1p4f2")

    def _side_effect(key):
        m = MagicMock()
        m.to_xarray.return_value = esgf_ds if "esgf" in key else geomip_ds
        return m

    with patch("srm.pipeline._catalog") as mock_cat:
        mock_cat.get.side_effect = _side_effect
        result = pipeline._load_ssp245_bridge()

    # to_proleptic_gregorian converts both to numpy datetime64 — no cftime objects remain.
    assert isinstance(result.time.values[0], np.datetime64), (
        f"Expected numpy datetime64, got {type(result.time.values[0])}"
    )
    years = sorted(int(y) for y in np.unique(result.time.dt.year.values))
    assert years[0] == 2015
    assert 2019 in years
    assert 2020 in years


def test_bridge_empty_esgf_gap_returns_primary(tmp_path):
    """When the ESGF dataset has no data before primary_start_year, return primary with a warning."""
    config = _make_config()
    pipeline = BCSDPipeline(config, _make_options(tmp_path))

    # GeoMIP SSP245 starts at 2020; ESGF also starts at 2020 → no gap data → return primary.
    geomip_da = _make_annual_ds(2020, 2084, "r01")["tas"]
    esgf_mock_entry = MagicMock()
    esgf_mock_entry.to_xarray.return_value = _make_annual_ds(2020, 2084, "r1i1p4f2")

    with (
        patch("srm.pipeline.get_experiment", return_value=geomip_da),
        patch("srm.pipeline._catalog") as mock_cat,
    ):
        mock_cat.get.return_value = esgf_mock_entry
        result = pipeline._load_ssp245_bridge()

    assert int(result.time.dt.year.min()) == 2020


# ---------------------------------------------------------------------------
# SAI parent segment (G6-1.5K-END termination run)
# ---------------------------------------------------------------------------


def _g6_end_config() -> BCSDConfig:
    return _make_config(
        gcm="CESM2-WACCM",
        ensemble_member="002",
        scenario="G6-1.5K-END",
        predict_period_start=2085,
        predict_period_end=2100,
    )


def _experiment_by_scenario(ssp245: xr.DataArray, g6: xr.DataArray):
    """Dispatch a patched get_experiment on scenario, positional or keyword."""

    def _side_effect(*args, **kwargs):
        scenario = kwargs.get("scenario", args[1] if len(args) > 1 else None)
        return ssp245 if scenario == "SSP245" else g6

    return _side_effect


def test_bridge_appends_sai_parent_for_termination_run(tmp_path):
    """G6-1.5K-END starts in 2085, so the bridge must cover 2015-2084.

    SSP245 alone would supply those years from the no-SAI run; the parent G6-1.5K
    segment has to take over from 2035.
    """
    pipeline = BCSDPipeline(_g6_end_config(), _make_options(tmp_path))
    assert pipeline._sai_parent == ("G6-1.5K", "002")

    ssp245_da = _make_annual_ds(2015, 2099, "002")["tas"]
    g6_da = _make_annual_ds(2035, 2084, "002")["tas"]

    with patch(
        "srm.pipeline.get_experiment", side_effect=_experiment_by_scenario(ssp245_da, g6_da)
    ):
        result = pipeline._load_ssp245_bridge()

    years = sorted(int(y) for y in np.unique(result.time.dt.year.values))
    assert years[0] == 2015, "bridge must start where historical ends"
    assert years[-1] == 2084, "bridge must stop where the termination run begins"
    assert len(years) == len(set(years)), "no duplicate years"
    gaps = [(y1, y2) for y1, y2 in zip(years, years[1:]) if y2 - y1 > 1]
    assert gaps == [], f"unexpected gaps: {gaps}"

    assert result.attrs["bridge_sai_parent_scenario"] == "G6-1.5K"
    assert result.attrs["bridge_sai_parent_member"] == "002"
    assert result.attrs["bridge_sai_parent_years"] == "2035-2084"


def test_bridge_sai_parent_takes_precedence_over_ssp245(tmp_path):
    """Where the two overlap (2035-2084), the SAI parent wins.

    SSP245 002 runs to 2099, so without the truncation the no-SAI run would cover
    years the model actually spent under SAI. That failure is silent for the standard
    variables, since the stitch stays continuous either way.
    """
    pipeline = BCSDPipeline(_g6_end_config(), _make_options(tmp_path))

    ssp245_da = _make_annual_ds(2015, 2099, "002")["tas"] + 1.0
    g6_da = _make_annual_ds(2035, 2084, "002")["tas"]  # zeros

    with patch(
        "srm.pipeline.get_experiment", side_effect=_experiment_by_scenario(ssp245_da, g6_da)
    ):
        result = pipeline._load_ssp245_bridge()

    pre_2035 = result.sel(time=result["time.year"] < 2035)
    post_2035 = result.sel(time=result["time.year"] >= 2035)
    assert float(pre_2035.mean()) == 1.0, "2015-2034 must come from SSP245"
    assert float(post_2035.mean()) == 0.0, "2035-2084 must come from the parent SAI run"


def test_bridge_sai_parent_tmax_uses_truncated_ssp245_without_gap(tmp_path):
    """tmx bridges through SSP245 007, which stops at 2069.

    The parent G6 segment starts in 2035 and covers the rest, so the 2070-2084
    hole that an SSP245-only bridge would leave never appears. Unlike the standard
    variables, this case fails loudly rather than silently when it regresses.
    """
    config = _make_config(
        gcm="CESM2-WACCM",
        variable="tasmax",
        ensemble_member="002",
        scenario="G6-1.5K-END",
        predict_period_start=2085,
        predict_period_end=2100,
    )
    pipeline = BCSDPipeline(config, _make_options(tmp_path))
    assert pipeline._ssp245_member == "007"

    ssp245_da = _make_annual_ds(2015, 2069, "007", var="tasmax")["tasmax"]
    g6_da = _make_annual_ds(2035, 2084, "002", var="tasmax")["tasmax"]

    with patch(
        "srm.pipeline.get_experiment", side_effect=_experiment_by_scenario(ssp245_da, g6_da)
    ):
        result = pipeline._load_ssp245_bridge()

    years = sorted(int(y) for y in np.unique(result.time.dt.year.values))
    gaps = [(y1, y2) for y1, y2 in zip(years, years[1:]) if y2 - y1 > 1]
    assert gaps == [], f"unexpected gaps: {gaps}"
    assert years[-1] == 2084


def test_bridge_unchanged_when_no_sai_parent(tmp_path):
    """Plain G6-1.5K has no SAI parent, so the bridge stays SSP245-only."""
    config = _make_config(gcm="CESM2-WACCM", ensemble_member="002", scenario="G6-1.5K")
    pipeline = BCSDPipeline(config, _make_options(tmp_path))
    assert pipeline._sai_parent is None

    ssp245_da = _make_annual_ds(2015, 2099, "002")["tas"]
    g6_da = _make_annual_ds(2035, 2084, "002")["tas"]

    with patch(
        "srm.pipeline.get_experiment", side_effect=_experiment_by_scenario(ssp245_da, g6_da)
    ) as mock_get:
        result = pipeline._load_ssp245_bridge()

    scenarios = [
        c.kwargs.get("scenario", c.args[1] if len(c.args) > 1 else None)
        for c in mock_get.call_args_list
    ]
    assert scenarios == ["SSP245"], "the G6 store must not be opened"
    assert int(result.time.dt.year.max()) == 2099
    assert "bridge_sai_parent_member" not in result.attrs


def test_termination_stitch_is_continuous_through_2100():
    """End to end shape: historical + bridge + termination run leaves no year gap."""
    from srm.pipeline import stitch_historical_scenario

    model_hist = _make_da_with_member(1950, 2014, "r2i1p1f1")
    bridge = _make_da_with_member(2015, 2084, "002")
    model_scenario = _make_da_with_member(2085, 2100, "002")

    result = stitch_historical_scenario(
        model_hist=model_hist,
        model_scenario=model_scenario,
        train_period_end=2014,
        predict_period_start=2085,
        ssp_timeseries=bridge,
    )

    years = sorted(int(y) for y in np.unique(result["time.year"].values))
    assert years[0] == 1950
    assert years[-1] == 2100
    gaps = [(y1, y2) for y1, y2 in zip(years, years[1:]) if y2 - y1 > 1]
    assert gaps == [], f"unexpected gaps: {gaps}"


def _make_da_with_member(start_year: int, end_year: int, member: str) -> xr.DataArray:
    """Annual DataArray with a scalar ensemble_member coordinate (post-sel shape)."""
    times = xr.date_range(f"{start_year}", f"{end_year}", freq="YS", use_cftime=False)
    data = np.zeros((len(times), 2, 2))
    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": times,
            "lat": [0.0, 1.0],
            "lon": [0.0, 1.0],
            "ensemble_member": member,
        },
    )


def test_stitch_historical_scenario_mismatched_ensemble_member():
    """stitch_historical_scenario must not raise MergeError when hist/bridge/scenario
    carry different ensemble_member scalar coords (the MIROC G6-1.5K case)."""
    from srm.pipeline import stitch_historical_scenario

    model_hist = _make_da_with_member(1950, 2014, "r1i1p4f2")
    ssp_bridge = _make_da_with_member(2015, 2034, "r01")
    model_scenario = _make_da_with_member(2035, 2060, "r01")

    result = stitch_historical_scenario(
        model_hist=model_hist,
        model_scenario=model_scenario,
        train_period_end=2014,
        predict_period_start=2015,
        ssp_timeseries=ssp_bridge,
    )

    years = sorted(int(y) for y in np.unique(result["time.year"].values))
    assert years[0] == 1950
    assert years[-1] == 2060
    # No gap anywhere
    gaps = [(y1, y2) for y1, y2 in zip(years, years[1:]) if y2 - y1 > 1]
    assert gaps == [], f"Unexpected gaps: {gaps}"
    # ensemble_member must be a scalar coord from model_scenario, not a dim
    assert "ensemble_member" not in result.dims
    assert result.coords["ensemble_member"].values.item() == "r01"
