"""Local integration tests for the stitch + detrend computation path.

These tests run real computation (no mocking of stitch/detrend) on tiny synthetic
daily datasets. They require no S3 access and no Coiled, making them the
first-pass check when debugging detrending regressions.

Scenarios covered:
- MIROC-ES2H G6-1.5K: hist=r1i1p4f2, bridge=r01 (GeoMIP SSP245), scenario=r01
  This is the case that triggered the MergeError on conflicting ensemble_member coords.
- CESM2-WACCM G6-1.5K: hist=r1i1p1f1, bridge=001, scenario=001
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from saidownscale.downscaling_utils import calculate_baseline_climatology, detrend
from saidownscale.pipeline import _assert_stitched_continuity, stitch_historical_scenario


def _daily_da(start_year: int, end_year: int, member: str | None = None) -> xr.DataArray:
    """Daily DataArray with optional scalar ensemble_member coordinate.

    Uses a short date range (year-start to year-end, daily) with a 2×2 spatial grid.
    Values are random normal around 280 K so detrend has real numbers to work with.
    """
    times = xr.date_range(
        f"{start_year}-01-01",
        f"{end_year}-12-31",
        freq="D",
        use_cftime=True,
        calendar="proleptic_gregorian",
    )
    rng = np.random.default_rng(0)
    data = rng.normal(280.0, 5.0, size=(len(times), 2, 2)).astype("float32")
    coords: dict = {
        "time": times,
        "lat": [0.0, 1.0],
        "lon": [0.0, 1.0],
    }
    if member is not None:
        coords["ensemble_member"] = member
    return xr.DataArray(data, dims=["time", "lat", "lon"], coords=coords)


# ---------------------------------------------------------------------------
# stitch_historical_scenario
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hist_member,bridge_member,scenario_member",
    [
        ("r1i1p4f2", "r01", "r01"),  # MIROC G6-1.5K: the previously-broken case
        ("r1i1p1f1", "001", "001"),  # CESM2-WACCM G6-1.5K
        (None, None, None),  # no ensemble_member coord at all
    ],
    ids=["miroc_g6", "cesm_g6", "no_member_coord"],
)
def test_stitch_no_gap_and_no_merge_error(hist_member, bridge_member, scenario_member):
    """Stitched timeseries is continuous regardless of ensemble_member coord values."""
    model_hist = _daily_da(1980, 2014, hist_member)
    ssp_bridge = _daily_da(2015, 2034, bridge_member)
    model_scenario = _daily_da(2035, 2040, scenario_member)

    result = stitch_historical_scenario(
        model_hist=model_hist,
        model_scenario=model_scenario,
        train_period_end=2014,
        predict_period_start=2015,
        ssp_timeseries=ssp_bridge,
    )

    _assert_stitched_continuity(result)  # no gap, no duplicates

    years = np.unique(result["time.year"].values)
    assert int(years[0]) == 1980
    assert int(years[-1]) == 2040
    assert "ensemble_member" not in result.dims, "ensemble_member must not become a dimension"
    # Coord is re-attached from model_scenario when present
    if scenario_member is not None:
        assert result.coords["ensemble_member"].values.item() == scenario_member


# ---------------------------------------------------------------------------
# stitch + detrend: the full path exercised on Coiled for SAI scenarios
# ---------------------------------------------------------------------------


def test_detrend_miroc_g6_local():
    """Full stitch+detrend for a MIROC G6-1.5K-style config with mismatched members.

    This is the real computation path that was failing on Coiled with
    'Stitched timeseries has year-level gap(s): [(2014, 2020)]'.
    """
    # Historical: ends 2014, member r1i1p4f2
    model_hist = _daily_da(1990, 2014, "r1i1p4f2")

    # SSP245 bridge covers 2015-2034, member r01 (GeoMIP-style)
    ssp_bridge = _daily_da(2015, 2034, "r01")

    # G6-1.5K scenario: starts 2035, member r01
    model_scenario = _daily_da(2035, 2040, "r01")

    stitched = stitch_historical_scenario(
        model_hist=model_hist,
        model_scenario=model_scenario,
        train_period_end=2014,
        predict_period_start=2015,
        ssp_timeseries=ssp_bridge,
    )
    _assert_stitched_continuity(stitched)

    baseline_clim = calculate_baseline_climatology(
        da_baseline=model_hist,
        baseline_period_start=1990,
        baseline_period_end=2014,
    )

    detrended, trend = detrend(
        da=stitched, da_baseline_clim=baseline_clim, detrend_method="additive"
    )

    # Slice to the predict period (2035-2040)
    predict_slice = slice("2035", "2040")
    detrended = detrended.sel(time=predict_slice)
    trend = trend.sel(time=predict_slice)

    assert detrended.time.size > 0, "Detrended output must not be empty"
    assert trend.time.size > 0, "Trend output must not be empty"
    assert int(detrended["time.year"].min()) == 2035
    assert int(detrended["time.year"].max()) == 2040
    assert not np.isnan(detrended.values).all(), "Detrended output must not be all-NaN"
