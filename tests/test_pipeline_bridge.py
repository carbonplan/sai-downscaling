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
        mapping_type="nonparametric_hybrid",
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

    with patch("srm.pipeline._catalog") as mock_cat:
        mock_cat.get.side_effect = lambda k: _mock_catalog_get(
            k, geomip_start=2020, esgf_start=2015
        )
        result = pipeline._load_ssp245_bridge()

    years = sorted(int(y) for y in np.unique(result.time.dt.year.values))
    assert years[0] == 2015, f"Bridge must start at 2015, got {years[0]}"
    assert 2019 in years, "ESGF gap years (2015-2019) must be present"
    assert 2020 in years, "GeoMIP data (from 2020) must be present"
    assert years[-1] == 2084
    assert len(years) == len(set(years)), "No duplicate years"

    calls = [c[0][0] for c in mock_cat.get.call_args_list]
    assert any("esgf" in k for k in calls), "ESGF catalog key must be accessed"
    assert any("esgf" not in k for k in calls), "Primary (GeoMIP) catalog key must be accessed"


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
