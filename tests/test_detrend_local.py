"""Local stitch + detrend on tiny synthetic data; the first check for detrending regressions."""

from __future__ import annotations

import numpy as np
import xarray as xr

from saidownscale.downscaling_utils import calculate_baseline_climatology, detrend
from saidownscale.pipeline import _assert_stitched_continuity, stitch_historical_scenario


def _daily_da(start_year: int, end_year: int, member: str | None = None) -> xr.DataArray:
    times = xr.date_range(
        f"{start_year}-01-01",
        f"{end_year}-12-31",
        freq="D",
        use_cftime=True,
        calendar="proleptic_gregorian",
    )
    data = np.random.default_rng(0).normal(280.0, 5.0, size=(len(times), 2, 2)).astype("float32")
    coords: dict = {"time": times, "lat": [0.0, 1.0], "lon": [0.0, 1.0]}
    if member is not None:
        coords["ensemble_member"] = member
    return xr.DataArray(data, dims=["time", "lat", "lon"], coords=coords)


def test_stitch_and_detrend_with_mismatched_members(subtests):
    """Differing hist/bridge/scenario members once caused a MergeError and a year-level gap."""
    stitched = {}
    for case, (hist_member, bridge_member, scenario_member) in {
        "cesm_g6_tasmax": ("001", "009", "001"),
        "cesm_g6": ("r1i1p1f1", "001", "001"),
        "no_member_coord": (None, None, None),
    }.items():
        with subtests.test(case=case):
            model_hist = _daily_da(2011, 2014, hist_member)
            result = stitch_historical_scenario(
                model_hist=model_hist,
                model_scenario=_daily_da(2017, 2019, scenario_member),
                train_period_end=2014,
                predict_period_start=2015,
                ssp_timeseries=_daily_da(2015, 2016, bridge_member),
            )
            _assert_stitched_continuity(result)
            years = result["time.year"].values
            assert (int(years.min()), int(years.max())) == (2011, 2019)
            assert "ensemble_member" not in result.dims
            if scenario_member is not None:
                assert result.coords["ensemble_member"].values.item() == scenario_member
            stitched[case] = (model_hist, result)

    with subtests.test(case="detrend_stitched_cesm_g6_tasmax"):
        model_hist, result = stitched["cesm_g6_tasmax"]
        baseline_clim = calculate_baseline_climatology(
            da_baseline=model_hist, baseline_period_start=2011, baseline_period_end=2014
        )
        detrended, trend = detrend(
            da=result, da_baseline_clim=baseline_clim, detrend_method="additive"
        )
        detrended = detrended.sel(time=slice("2017", "2019"))
        assert trend.sel(time=slice("2017", "2019")).time.size > 0
        assert (int(detrended["time.year"].min()), int(detrended["time.year"].max())) == (
            2017,
            2019,
        )
        assert not np.isnan(detrended.values).all()
