"""
Reproduction of the point_debias_diagnostic bug, using ``bug_drilldown_timeseries.csv``
plus one small supplemental live read of ``raw_historical`` for 2015-01-01..2015-01-16.

That supplemental read is unavoidable, not optional: the CSV's raw_historical column was
built by ``get_data_for_bug_drilldown.ipynb`` using a hardcoded
``train_slice = slice("1978-01-01", "2014-12-31")`` (see that notebook), which throws away
exactly the ~16 days of January 2015 that turned out to be the actual root cause of the
Jan/Dec-only debiased_g6 mismatch -- see model_hist_for_predict_start() in
point_debias_diagnostic.py for the full mechanism. Reproducing the bug (and its fix)
faithfully requires the FULL native-extent historical series, which the CSV as captured
does not contain. Everything else (obs, raw_ssp, raw_g6, and the ground-truth
coarse_debiased_* columns) still comes only from the CSV.

Run with the project venv:

    .venv/bin/python notebooks/QA_QC/pr/reproduce_from_csv.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import point_debias_diagnostic as pdd  # noqa: E402

from srm import catalog  # noqa: E402

CSV_PATH = Path(__file__).parent / "bug_drilldown_timeseries.csv"
LAT, LON = 1.5, -82.5
HIST_ENS = "r3i1p1f1"


def load_csv_series():
    """Load the CSV and return each column as a clean (NaN-free) xr.DataArray,
    sliced to its own valid time range, with a proper time coordinate.

    raw_historical is the one exception: it's supplemented with a small live read
    (2015-01-01..2015-01-16, ~16 days) that the CSV doesn't contain -- see module
    docstring. This intentionally keeps whatever NaN padding is natively present in
    that supplemental range (2015-01-02..16 are NaN in the live catalog) rather than
    dropping it, since production's real code sees those NaNs too
    (model_hist_for_predict_start's g6 slice includes them) and still produces the
    ground-truth output being compared against here.
    """
    df = pd.read_csv(CSV_PATH, parse_dates=["time"])

    def col_to_da(col):
        s = df[["time", col]].dropna()
        da = xr.DataArray(s[col].values, dims="time", coords={"time": s["time"].values}, name="pr")
        return da

    era5_obs = col_to_da("era5_obs")
    raw_ssp = col_to_da("raw_ssp")
    raw_g6 = col_to_da("raw_g6")
    coarse_debiased_ssp = col_to_da("coarse_debiased_ssp")
    coarse_debiased_g6 = col_to_da("coarse_debiased_g6")

    raw_historical_csv = col_to_da("raw_historical")  # 1978-01-01..2014-12-31 only
    raw_data = catalog.get("CESM2-WACCM").to_xarray()
    hist_tail_live = raw_data["historical"]["pr"].sel(ensemble_member=HIST_ENS)
    hist_tail_live = hist_tail_live.sel(lat=LAT, lon=LON, method="nearest")
    hist_tail_live = hist_tail_live.sel(time=slice("2015-01-01", "2015-01-16")).compute() * 86400
    raw_historical = xr.concat([raw_historical_csv, hist_tail_live], dim="time")

    return {
        "era5_obs": era5_obs,
        "raw_historical": raw_historical,
        "raw_ssp": raw_ssp,
        "raw_g6": raw_g6,
        "coarse_debiased_ssp": coarse_debiased_ssp,
        "coarse_debiased_g6": coarse_debiased_g6,
    }


def compare_to_production(
    name, my_debiased, production_debiased, out_of_range_low=None, out_of_range_high=None
):
    """Same diagnostic as bug_drilldown.ipynb cell 9: compares my_debiased
    (this script's output) against production_debiased (ground truth from the
    CSV), aligned on time, with a branch (nonparametric/parametric) and
    monthly breakdown.
    """
    mine, prod = xr.align(my_debiased, production_debiased, join="inner")
    diff = mine - prod
    n = diff.time.size

    print(f"[{name}] {n} overlapping days")
    print(f"  mean(mine):        {float(mine.mean()):.6e}")
    print(f"  mean(production):  {float(prod.mean()):.6e}")
    print(f"  mean(mine - prod): {float(diff.mean()):.6e}")
    print(f"  mine > production: {100 * float((diff > 0).mean()):.1f}% of days")
    print(f"  max |diff|:        {float(np.abs(diff).max()):.6e}")
    print(f"  corr(mine, prod):  {float(xr.corr(mine, prod)):.6f}")

    if out_of_range_low is not None and out_of_range_high is not None:
        low_a, high_a, diff_a = xr.align(out_of_range_low, out_of_range_high, diff, join="inner")
        is_low = low_a & ~high_a
        is_high = high_a
        is_nonparam = ~low_a & ~high_a
        for label, mask in [
            ("nonparametric", is_nonparam),
            ("Weibull low tail", is_low),
            ("Gumbel high tail", is_high),
        ]:
            n_branch = int(mask.sum())
            if n_branch == 0:
                print(f"  {label:18s}: 0 days")
                continue
            branch_mean_diff = float(diff_a.where(mask).mean())
            print(f"  {label:18s}: {n_branch:6d} days, mean(mine - prod) = {branch_mean_diff:+.6e}")

    print("  by month:")
    diff_monthly = diff.groupby("time.month").mean()
    for month in range(1, 13):
        val = (
            float(diff_monthly.sel(month=month))
            if month in diff_monthly["month"].values
            else np.nan
        )
        print(f"    month {month:2d}: mean(mine - prod) = {val:+.6e}")

    return diff


def main():
    data = load_csv_series()

    print("=" * 70)
    print("Running point_debias_diagnostic.run_analysis from CSV-reconstructed inputs")
    print("=" * 70)
    results = pdd.run_analysis(
        data["era5_obs"],
        data["raw_historical"],
        data["raw_ssp"],
        data["raw_g6"],
        do_diagnostics=False,
    )

    print()
    print("=" * 70)
    print("Comparison vs. ground truth (production's cached coarse_debiased_*)")
    print("=" * 70)
    diff_ssp = compare_to_production(
        "ssp245",
        results["debiased_ssp"],
        data["coarse_debiased_ssp"],
        results["out_of_range_low_ssp"],
        results["out_of_range_high_ssp"],
    )
    print()
    diff_g6 = compare_to_production(
        "g6-1.5k",
        results["debiased_g6"],
        data["coarse_debiased_g6"],
        results["out_of_range_low_g6"],
        results["out_of_range_high_g6"],
    )

    return diff_ssp, diff_g6


if __name__ == "__main__":
    main()
