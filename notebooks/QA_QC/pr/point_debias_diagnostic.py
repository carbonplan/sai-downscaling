"""
Point-level bias-correction diagnostic at lat=1.5, lon=-82.5 (precip).

Reuses the actual production building blocks from ``srm.pipeline`` --
``stitch_historical_scenario``, ``calculate_out_of_range_mask``, ``_make_debiaser``,
``_weibull_min_zero_bounded`` -- so this replicates exactly what the real BCSD
pipeline does for ``debias_approach="nonparametric_hybrid_2sided"`` (the default,
see src/srm/bcsd_config.py / src/srm/cli.py). It is NOT a simplified
reimplementation of the quantile-mapping logic.

Assumes ``var="pr"`` (everything in the investigation that led to this script was
precip-specific: the Weibull/Gumbel tail distributions in ``_make_debiaser`` only
apply to ["pr", "rsds", "hurs", "dtr"]). Change VAR below if you want a different
variable, but note the "why" section (part 3) is written specifically for the
Gumbel high-tail cdf_threshold saturation issue and won't be meaningful for tas/tasmax/tasmin.

Answers:
  1. How often is the parametric (Weibull low-tail / Gumbel high-tail) debiaser
     used vs. the nonparametric one, for ssp245 and g6-1.5k?
  2. Does bias correction distort the g6-minus-ssp245 mean difference over
     2040-2069, relative to the same difference computed on raw (undebiased) GCM
     output?
  3. Why -- decomposed into (a) asymmetric per-scenario debiasing shift,
     (b) asymmetric out-of-range frequency between the two scenarios, and
     (c) the cdf_threshold saturation mechanism (see Bug_explorations.ipynb),
     checked day-of-year-window by day-of-year-window for every flagged point
     in the 2040-2069 window rather than just the one day-of-year that was
     manually spot-checked before.

This is a single point, so no distributed cluster is spun up -- the data volume
is trivial (one grid cell's worth of daily values) and plain numpy/xarray is fast
enough. Add a frisky/dask client (see any of the other QA_QC notebooks for the
pattern) only if S3 reads turn out to be slow on your connection.
"""

import numpy as np
import scipy.stats
import xarray as xr
from ibicus.utils import threshold_cdf_vals

from srm import catalog
from srm.bcsd_config import VariableConfig
from srm.pipeline import (
    _make_debiaser,
    _weibull_min_zero_bounded,
    calculate_out_of_range_mask,
    stitch_historical_scenario,
)
from srm.utils import open_icechunk

# --------------------------------------------------------------------- config
VAR = "pr"
LAT, LON = 1.5, -82.5

BRANCH = "v0.11.1"
SSP_ENS, HIST_ENS, G6_ENS = "008", "r3i1p1f1", "003"

# Matches BCSDConfig defaults (src/srm/bcsd_config.py)
TRAIN_PERIOD_START, TRAIN_PERIOD_END = 1978, 2014
PREDICT_PERIOD_START, PREDICT_PERIOD_END = 2015, 2100  # generous; data naturally truncates

# The window the user wants the g6-vs-ssp245 mean-difference question evaluated over
DELTA_WINDOW = slice("2040-01-01", "2069-12-31")

CDF_THRESHOLD = 1e-10  # ibicus/QuantileMapping default, matches production

VAR_CFG = VariableConfig.for_variable(VAR)
RUNNING_WINDOW_LENGTH = VAR_CFG.running_window_length  # 31 for pr


# ------------------------------------------------------- scenario detrending
def build_scenario_input(model_hist, model_scenario, is_sai, ssp_bridge=None):
    """Replicates BCSDPipeline._detrend_scenario for pr (detrend_data=False).

    ssp245 (not SAI) needs no bridging -- it's already continuous from
    historical through the future. g6-1.5k (SAI) starts partway through the
    predict period, so it gets bridged with ssp245 for the gap, exactly as
    production does (issue #363).
    """
    if not is_sai:
        return model_scenario.sel(
            time=slice(f"{PREDICT_PERIOD_START}-01-01", f"{PREDICT_PERIOD_END}-12-31")
        )

    bridged = stitch_historical_scenario(
        model_hist=model_hist,
        model_scenario=model_scenario,
        train_period_end=TRAIN_PERIOD_END,
        predict_period_start=PREDICT_PERIOD_START,
        ssp_timeseries=ssp_bridge,
    )
    return bridged.sel(time=slice(f"{PREDICT_PERIOD_START}-01-01", f"{PREDICT_PERIOD_END}-12-31"))


# ------------------------------------------------------------------ debiasing
def debias_scenario(obs_coarse, model_hist, scenario_detrended):
    """Replicates BCSDPipeline._apply_bias_correction_scenario's
    debias_approach="nonparametric_hybrid_2sided" branch exactly, at one point.

    ibicus's Debiaser.apply() requires strictly 3D (time, x, y) arrays, so a
    single point's 1D timeseries gets reshaped to (T, 1, 1) and squeezed back
    afterward -- this is the only shape difference from what production does
    on the full grid; the quantile-mapping logic itself is untouched.

    obs_coarse/model_hist/scenario_detrended are forced to eager (computed)
    arrays here regardless of whether the caller passed lazy dask-backed ones
    in -- calculate_out_of_range_mask's groupby/rolling output stays lazy if
    its inputs are lazy, and xarray refuses `.where(mask, drop=True)` on an
    uncomputed dask boolean mask (used later in diagnose_cdf_saturation).
    """
    obs_coarse = obs_coarse.compute()
    model_hist = model_hist.compute()
    scenario_detrended = scenario_detrended.compute()

    obs_np = obs_coarse.values.reshape(-1, 1, 1)
    cm_hist_np = model_hist.values.reshape(-1, 1, 1)
    cm_future_np = scenario_detrended.values.reshape(-1, 1, 1)

    common_kwargs = dict(
        variable=VAR,
        detrending="no_detrending",
        running_window_mode=VAR_CFG.do_windowing,
        running_window_length=RUNNING_WINDOW_LENGTH,
        running_window_step_length=1,
        running_window_mode_over_years_of_cm_future=False,
    )
    apply_kwargs = dict(
        obs=obs_np,
        cm_hist=cm_hist_np,
        cm_future=cm_future_np,
        time_obs=obs_coarse["time"].values,
        time_cm_hist=model_hist["time"].values,
        time_cm_future=scenario_detrended["time"].values,
        parallel=False,  # single point -- multiprocessing would only add overhead
        progressbar=False,
        failsafe=True,
    )

    low_dist = _weibull_min_zero_bounded
    high_dist = scipy.stats.gumbel_r

    parametric_low_np = _make_debiaser(
        distribution=low_dist, mapping_type="parametric", **common_kwargs
    ).apply(**apply_kwargs)
    parametric_high_np = _make_debiaser(
        distribution=high_dist, mapping_type="parametric", **common_kwargs
    ).apply(**apply_kwargs)
    nonparametric_np = _make_debiaser(mapping_type="nonparametric", **common_kwargs).apply(
        **apply_kwargs
    )

    _, out_of_range_low, out_of_range_high = calculate_out_of_range_mask(
        model_hist=model_hist,
        scenario_detrended=scenario_detrended,
        center_window=RUNNING_WINDOW_LENGTH,
    )

    debiased_np = np.where(out_of_range_low.values, parametric_low_np[:, 0, 0], nonparametric_np[:, 0, 0])
    debiased_np = np.where(out_of_range_high.values, parametric_high_np[:, 0, 0], debiased_np)
    debiased_np = np.clip(debiased_np, a_min=0.0, a_max=None)  # production clip_bounds for pr

    debiased = xr.DataArray(
        debiased_np, dims="time", coords={"time": scenario_detrended["time"]}, name=VAR
    )
    return debiased, out_of_range_low, out_of_range_high


# --------------------------------------------------- (1) parametric usage
def report_parametric_usage(name, out_of_range_low, out_of_range_high):
    n = out_of_range_low.size
    n_low = int(out_of_range_low.values.sum())
    n_high = int(out_of_range_high.values.sum())
    n_param = n_low + n_high
    print(f"[{name}] {n} days total")
    print(f"  parametric (Weibull low tail):  {n_low:6d} days ({100 * n_low / n:.2f}%)")
    print(f"  parametric (Gumbel high tail):  {n_high:6d} days ({100 * n_high / n:.2f}%)")
    print(f"  parametric (either tail):       {n_param:6d} days ({100 * n_param / n:.2f}%)")
    print(f"  nonparametric:                  {n - n_param:6d} days ({100 * (n - n_param) / n:.2f}%)")


# --------------------------------------- (2) mean-difference distortion check
def evaluate_mean_distortion(raw_ssp, raw_g6, debiased_ssp, debiased_g6, window):
    raw_ssp_mean = float(raw_ssp.sel(time=window).mean())
    raw_g6_mean = float(raw_g6.sel(time=window).mean())
    debiased_ssp_mean = float(debiased_ssp.sel(time=window).mean())
    debiased_g6_mean = float(debiased_g6.sel(time=window).mean())

    raw_delta = raw_g6_mean - raw_ssp_mean
    debiased_delta = debiased_g6_mean - debiased_ssp_mean
    distortion = debiased_delta - raw_delta

    print(f"Raw       g6 - ssp245 mean (2040-2069):  {raw_delta:.6e}")
    print(f"Debiased  g6 - ssp245 mean (2040-2069):  {debiased_delta:.6e}")
    print(f"Distortion (debiased - raw):              {distortion:.6e}", end="")
    if raw_delta != 0:
        print(f"  ({100 * distortion / abs(raw_delta):.1f}% of the raw delta)")
    else:
        print()
    print()
    print(f"  ssp245: debiasing shifted the mean by {debiased_ssp_mean - raw_ssp_mean:.6e}")
    print(f"  g6:     debiasing shifted the mean by {debiased_g6_mean - raw_g6_mean:.6e}")
    print(
        "  -> if these two shifts are similar, debiasing is roughly *consistent* "
        "across scenarios and the g6-ssp245 signal is preserved; if they differ "
        "substantially, that asymmetry IS the distortion in the delta."
    )


# ------------------------------- (3) why: cdf_threshold saturation check
def diagnose_cdf_saturation(
    name, model_hist, scenario_detrended, out_of_range_high, window, cdf_threshold=CDF_THRESHOLD
):
    """For every day-of-year with a Gumbel high-tail out-of-range point inside
    `window`, refit Gumbel to model_hist's matching +/- running-window-length/2
    day-of-year neighborhood (replicating calculate_out_of_range_mask's own
    windowing) and check whether that point's cdf saturates against
    cdf_threshold -- the exact mechanism identified in Bug_explorations.ipynb,
    generalized here to every flagged day in the window instead of one
    hand-picked day-of-year.
    """
    half = RUNNING_WINDOW_LENGTH // 2
    sub_scenario = scenario_detrended.sel(time=window)
    sub_mask = out_of_range_high.sel(time=window)
    flagged = sub_scenario.where(sub_mask, drop=True)

    if flagged.time.size == 0:
        print(f"[{name}] no Gumbel high-tail out-of-range days in {window} -- nothing to check.")
        return

    doys = np.unique(flagged["time.dayofyear"].values)
    n_checked = 0
    n_saturated = 0
    for doy in doys:
        lo, hi = doy - half, doy + half
        # wrap around year boundary, same as calculate_out_of_range_mask's padding
        hist_doy = model_hist["time.dayofyear"]
        if lo < 1:
            in_window = (hist_doy >= lo + 365) | (hist_doy <= hi)
        elif hi > 365:
            in_window = (hist_doy >= lo) | (hist_doy <= hi - 365)
        else:
            in_window = (hist_doy >= lo) & (hist_doy <= hi)
        hist_window_vals = model_hist.where(in_window, drop=True).values

        fit_cm_hist = scipy.stats.gumbel_r.fit(hist_window_vals)
        pts = flagged.where(flagged["time.dayofyear"] == doy, drop=True).values
        raw_cdf = scipy.stats.gumbel_r.cdf(pts, *fit_cm_hist)
        clipped = threshold_cdf_vals(raw_cdf, cdf_threshold)
        saturated = clipped >= (1 - cdf_threshold)
        n_saturated += int(saturated.sum())
        n_checked += pts.size

    pct = 100 * n_saturated / n_checked if n_checked else 0.0
    print(
        f"[{name}] {n_checked} Gumbel high-tail out-of-range days in {window}, "
        f"{n_saturated} ({pct:.1f}%) hit the cdf_threshold ceiling "
        f"(cdf clipped to 1-{cdf_threshold:g}). Every saturated point on the same "
        f"day-of-year collapses to the identical corrected value, regardless of "
        f"how far out it actually is -- see cdf_threshold discussion for the mechanism."
    )


def run_analysis(obs_coarse, model_hist, model_ssp, model_g6):
    """Run the full diagnostic given already-loaded point timeseries.

    Call this directly from a notebook when you already have the four point
    series loaded -- e.g. as era5_pt, raw_historical_pt, raw_ssp_pt, raw_g6_pt:

        import sys
        sys.path.insert(0, "/path/to/notebooks/QA_QC/pr")
        import point_debias_diagnostic as pdd

        results = pdd.run_analysis(era5_pt, raw_historical_pt, raw_ssp_pt, raw_g6_pt)

    Expects:
      obs_coarse, model_hist : sliced to the training period
                                (TRAIN_PERIOD_START..TRAIN_PERIOD_END, i.e. 1978-2014)
      model_ssp, model_g6    : the raw future timeseries, each at native/full length.
                                model_ssp also doubles as the SSP245 bridge used to
                                stitch g6's gap before it starts (~2035), so don't
                                pre-truncate it to only where g6 has data.

    Returns a dict with the debiased series, the scenario inputs actually used
    (post-bridging), and the out-of-range masks, in case you want to inspect them
    further after the printed report.
    """
    ssp_input = build_scenario_input(model_hist, model_ssp, is_sai=False)
    g6_input = build_scenario_input(model_hist, model_g6, is_sai=True, ssp_bridge=model_ssp)

    print("Running nonparametric_hybrid_2sided debiasing (ssp245)...")
    debiased_ssp, oor_low_ssp, oor_high_ssp = debias_scenario(obs_coarse, model_hist, ssp_input)

    print("Running nonparametric_hybrid_2sided debiasing (g6-1.5k)...")
    debiased_g6, oor_low_g6, oor_high_g6 = debias_scenario(obs_coarse, model_hist, g6_input)

    print()
    print("=" * 70)
    print("(1) Parametric vs nonparametric usage")
    print("=" * 70)
    report_parametric_usage("ssp245", oor_low_ssp, oor_high_ssp)
    print()
    report_parametric_usage("g6-1.5k", oor_low_g6, oor_high_g6)

    print()
    print("=" * 70)
    print("(2) Does debiasing distort the g6 - ssp245 mean difference, 2040-2069?")
    print("=" * 70)
    evaluate_mean_distortion(ssp_input, g6_input, debiased_ssp, debiased_g6, DELTA_WINDOW)

    print()
    print("=" * 70)
    print("(3) Why -- cdf_threshold saturation check on the Gumbel high tail")
    print("=" * 70)
    diagnose_cdf_saturation("ssp245", model_hist, ssp_input, oor_high_ssp, DELTA_WINDOW)
    diagnose_cdf_saturation("g6-1.5k", model_hist, g6_input, oor_high_g6, DELTA_WINDOW)

    return {
        "ssp_input": ssp_input,
        "g6_input": g6_input,
        "debiased_ssp": debiased_ssp,
        "debiased_g6": debiased_g6,
        "out_of_range_low_ssp": oor_low_ssp,
        "out_of_range_high_ssp": oor_high_ssp,
        "out_of_range_low_g6": oor_low_g6,
        "out_of_range_high_g6": oor_high_g6,
    }