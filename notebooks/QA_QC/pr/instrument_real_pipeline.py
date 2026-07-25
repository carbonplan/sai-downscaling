"""
Call the REAL, unmodified production code path (BCSDPipeline._detrend_scenario and
._apply_bias_correction_scenario from src/srm/pipeline.py) locally, feeding it the
CSV-derived point series reshaped into a fake 1x1 "grid" -- no Coiled, no S3 writes,
just numpy/xarray on a tiny array.

Purpose: point_debias_diagnostic.py's debias_scenario() is a *hand-written
reimplementation* of the real pipeline's debiasing call. It's been carefully checked
against the real code by reading, but reading isn't proof. This script runs the real
class methods directly so we can determine, by direct comparison:

  (a) real-code-on-CSV-data vs. ground truth (coarse_debiased_g6 from the CSV) --
      if this matches to float precision, the bug is in debias_scenario()'s
      reimplementation, not in production.
  (b) real-code-on-CSV-data vs. debias_scenario()'s own output -- if these two are
      bit-identical, the reimplementation is faithful and the remaining gap versus
      ground truth must come from something about the *data* real production ingested
      that isn't captured by this point extraction.

Run with the project venv:

    .venv/bin/python notebooks/QA_QC/pr/instrument_real_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import point_debias_diagnostic as pdd  # noqa: E402

from srm.bcsd_config import BCSDConfig, PipelineOptions  # noqa: E402
from srm.pipeline import BCSDPipeline  # noqa: E402

CSV_PATH = Path(__file__).parent / "bug_drilldown_timeseries.csv"
LAT, LON = 1.5, -82.5


def col_to_grid(df, col, lat=LAT, lon=LON):
    """1D (time,) CSV column -> 3D (time, lat=1, lon=1) DataArray, NaN-dropped."""
    s = df[["time", col]].dropna()
    da = xr.DataArray(
        s[col].values.reshape(-1, 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": s["time"].values, "lat": [lat], "lon": [lon]},
        name="pr",
    )
    return da


def main():
    df = pd.read_csv(CSV_PATH, parse_dates=["time"])

    era5_obs = col_to_grid(df, "era5_obs")
    raw_historical = col_to_grid(df, "raw_historical")
    raw_ssp = col_to_grid(df, "raw_ssp")
    raw_g6 = col_to_grid(df, "raw_g6")
    coarse_debiased_g6 = col_to_grid(df, "coarse_debiased_g6")

    config = BCSDConfig(
        gcm="CESM2-WACCM",
        variable="pr",
        ensemble_member="003",
        scenario="G6-1.5K",
        train_period_start=1978,
        train_period_end=2014,
        predict_period_start=2015,
        predict_period_end=2100,
    )
    options = PipelineOptions()
    pipeline = BCSDPipeline(config, options)

    print(
        f"resolved hist_member={pipeline._hist_member!r}, ssp245_member={pipeline._ssp245_member!r}"
    )

    # Real _detrend_scenario: stitches historical + ssp245 bridge + g6, slices to predict window.
    scenario_detrended, scenario_trend = pipeline._detrend_scenario(
        model_hist=raw_historical,
        model_scenario=raw_g6,
        ssp_timeseries=raw_ssp,
    )
    print(
        f"scenario_detrended time range: {scenario_detrended.time.values.min()} .. {scenario_detrended.time.values.max()}, n={scenario_detrended.time.size}"
    )
    assert scenario_trend is None, "pr should not be detrended"

    # Real _apply_bias_correction_scenario: the actual nonparametric_hybrid_2sided dispatch.
    real_debiased = pipeline._apply_bias_correction_scenario(
        obs_coarse=era5_obs,
        model_hist=raw_historical,
        scenario_detrended=scenario_detrended,
    )
    real_debiased_pt = real_debiased.isel(lat=0, lon=0)

    # My own hand-written reimplementation, for comparison, on the SAME (unstitched) g6 array
    # sliced the same way scenario_detrended would be for dates >= g6's native start.
    my_debiased, my_oor_low, my_oor_high = pdd.debias_scenario(
        era5_obs.isel(lat=0, lon=0),
        raw_historical.isel(lat=0, lon=0),
        raw_g6.isel(lat=0, lon=0),
    )

    gt = coarse_debiased_g6.isel(lat=0, lon=0)

    print()
    print("=" * 70)
    print("(a) real production code (fed CSV data) vs. ground truth")
    print("=" * 70)
    a_mine, a_prod = xr.align(real_debiased_pt, gt, join="inner")
    a_diff = a_mine - a_prod
    print(
        f"n={a_diff.time.size}, mean(diff)={float(a_diff.mean()):.6e}, max|diff|={float(np.abs(a_diff).max()):.6e}"
    )
    a_monthly = a_diff.groupby("time.month").mean()
    for m in range(1, 13):
        v = float(a_monthly.sel(month=m)) if m in a_monthly["month"].values else np.nan
        print(f"  month {m:2d}: {v:+.6e}")

    print()
    print("=" * 70)
    print("(b) real production code (fed CSV data) vs. debias_scenario() reimplementation")
    print("=" * 70)
    b_real, b_mine = xr.align(real_debiased_pt, my_debiased, join="inner")
    b_diff = b_real - b_mine
    print(
        f"n={b_diff.time.size}, mean(diff)={float(b_diff.mean()):.6e}, max|diff|={float(np.abs(b_diff).max()):.6e}"
    )
    nz = np.abs(b_diff.values) > 1e-9
    print(f"n nonzero (tol 1e-9): {int(nz.sum())}")
    if nz.sum() > 0:
        idx = np.where(nz)[0][:20]
        for i in idx:
            t = b_diff.time.values[i]
            print(
                f"  {pd.Timestamp(t).date()}  real={float(b_real.values[i]):.6f}  mine={float(b_mine.values[i]):.6f}  diff={float(b_diff.values[i]):+.6f}"
            )

    print()
    print("=" * 70)
    print(
        "(c) debias_scenario() reimplementation vs. ground truth (sanity check, matches earlier findings)"
    )
    print("=" * 70)
    c_mine, c_prod = xr.align(my_debiased, gt, join="inner")
    c_diff = c_mine - c_prod
    print(
        f"n={c_diff.time.size}, mean(diff)={float(c_diff.mean()):.6e}, max|diff|={float(np.abs(c_diff).max()):.6e}"
    )

    return real_debiased_pt, my_debiased, gt


if __name__ == "__main__":
    main()
