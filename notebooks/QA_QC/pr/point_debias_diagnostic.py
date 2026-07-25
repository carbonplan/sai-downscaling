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

from srm.bcsd_config import VariableConfig
from srm.pipeline import (
    _make_debiaser,
    _weibull_min_zero_bounded,
    calculate_out_of_range_mask,
    stitch_historical_scenario,
)

# --------------------------------------------------------------------- config
VAR = "pr"
LAT, LON = 1.5, -82.5

BRANCH = "v0.11.1"
SSP_ENS, HIST_ENS, G6_ENS = "008", "r3i1p1f1", "003"

# Matches BCSDConfig defaults (src/srm/bcsd_config.py)
TRAIN_PERIOD_START, TRAIN_PERIOD_END = 1978, 2014

# Matches the REAL production configs exactly (configs/production/cesm2-waccm/), not a
# shared guess. This matters: predict_period_start is used by
# BCSDPipeline._load_scenario_data (src/srm/pipeline.py ~line 1200) to slice model_hist
# itself -- via `time=slice(train_period_start, predict_period_start - 1)` -- which is
# NOT the same as slicing to the training period. See model_hist_for_predict_start()
# below for why using one shared predict_period_start/end for both scenarios (as this
# script used to) silently corrupts g6's model_hist and was the actual root cause of the
# Jan/Dec-only mismatch chased through this file's history.
SSP_PREDICT_PERIOD_START, SSP_PREDICT_PERIOD_END = (
    2015,
    2069,
)  # cesm2-waccm-ssp245-std-trunc.yaml (member 008)
G6_PREDICT_PERIOD_START, G6_PREDICT_PERIOD_END = 2035, 2084  # cesm2-waccm-g6.yaml (member 003)

# The window the user wants the g6-vs-ssp245 mean-difference question evaluated over
DELTA_WINDOW = slice("2040-01-01", "2069-12-31")

CDF_THRESHOLD = 1e-10  # ibicus/QuantileMapping default, matches production

VAR_CFG = VariableConfig.for_variable(VAR)
RUNNING_WINDOW_LENGTH = VAR_CFG.running_window_length  # 31 for pr


# ------------------------------------------------- scenario-specific model_hist
def model_hist_for_predict_start(model_hist_native, predict_period_start):
    """Replicates BCSDPipeline._load_scenario_data's model_hist slice exactly
    (src/srm/pipeline.py ~line 1200): ``time=slice(train_period_start, predict_period_start - 1)``.

    This is scenario-specific and is NOT the same as slicing model_hist to the
    training period (train_period_start..train_period_end), even though obs_coarse
    IS sliced that simple way in the same function. For ssp245
    (predict_period_start=2015) the two happen to coincide: predict_period_start - 1
    == train_period_end == 2014. For g6 (predict_period_start=2035) they do not:
    this pulls in everything through 2034, and because CESM2-WACCM's native
    historical series only actually extends to 2015-01-16 (confirmed against the
    live catalog), model_hist for g6 ends up spanning 1978-01-01..2015-01-16 --
    2014-12-31 plus one valid extra day (2015-01-01, pr=0.0095 mm) plus 15
    NaN-padded days (2015-01-02..16).

    Those ~16 extra days of January landing inside the +/-15-day window centered
    on day-of-year 1 (RUNNING_WINDOW_LENGTH // 2 == 15) are the actual root cause
    of the Jan/Dec-only debiased_g6 mismatch chased through this script's history
    -- confirmed by reproducing it exactly (mean diff -> 0.0, max diff -> 0.0
    across all 10904 compared days) once model_hist is built this way instead of
    truncated to train_period_end for both scenarios. It is NOT the stitch, NOT a
    calendar/leap-year issue, NOT a windowing bug in ibicus, and NOT bad input
    data -- every one of those was ruled out with direct evidence first.

    ``model_hist_native`` must be the FULL native-extent historical series (NOT
    pre-truncated to train_period_end) for this to reproduce production
    correctly -- callers must not hand this a copy already sliced to
    TRAIN_PERIOD_START..TRAIN_PERIOD_END, since that's exactly the shortcut that
    caused the bug.
    """
    return model_hist_native.sel(time=slice(f"{TRAIN_PERIOD_START}", f"{predict_period_start - 1}"))


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

    debiased_np = np.where(
        out_of_range_low.values, parametric_low_np[:, 0, 0], nonparametric_np[:, 0, 0]
    )
    debiased_np = np.where(out_of_range_high.values, parametric_high_np[:, 0, 0], debiased_np)
    debiased_np = np.clip(debiased_np, a_min=0.0, a_max=None)  # production clip_bounds for pr

    debiased = xr.DataArray(
        debiased_np, dims="time", coords={"time": scenario_detrended["time"]}, name=VAR
    )
    return debiased, out_of_range_low, out_of_range_high


# ------------------------ (4) is the correction quantile-dependent? why the delta shifts
def decompose_by_quantile_rank(name, raw, debiased, model_hist, window, n_bins=10):
    """Bins days by the quantile rank of their raw value within model_hist's
    empirical distribution -- the reference distribution quantile mapping
    actually maps against -- and reports the mean correction (debiased - raw)
    per bin.

    Quantile mapping's correction is generally a nonlinear function of a
    value's rank in cm_hist, not a uniform shift. If this comes back flat
    across bins, the correction is basically a constant offset and the
    scenario-to-scenario distortion in part (2) needs another explanation. If
    it's clearly not flat, that confirms the mechanism: two scenarios with
    different raw distributions occupy different parts of this curve and so
    get corrected by different average amounts, with neither being "wrong".
    """
    raw_w = raw.sel(time=window)
    debiased_w = debiased.sel(time=window)
    diff_w = debiased_w - raw_w

    hist_sorted = np.sort(model_hist.values)
    ranks = np.searchsorted(hist_sorted, raw_w.values) / hist_sorted.size
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(ranks, bins) - 1, 0, n_bins - 1)

    print(f"[{name}] correction (debiased - raw) by cm_hist quantile bin:")
    bin_means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        bin_mean_diff = float(diff_w.values[mask].mean())
        bin_means[b] = bin_mean_diff
        print(
            f"  quantile [{bins[b]:.1f}-{bins[b + 1]:.1f}): {n_b:5d} days "
            f"({100 * n_b / ranks.size:4.1f}%), mean correction = {bin_mean_diff:+.4f}"
        )
    return bin_means


def compare_quantile_occupancy(ssp_bins, g6_bins, n_bins=10):
    """Lines up the two scenarios' per-bin corrections (from
    decompose_by_quantile_rank) side by side, so it's visible where the
    differential correction that drives part (2)'s distortion actually
    accumulates across the distribution.
    """
    print(
        f"{'quantile bin':<16s}{'ssp245 correction':>20s}{'g6 correction':>16s}{'difference':>14s}"
    )
    for b in range(n_bins):
        s, g = ssp_bins[b], g6_bins[b]
        if np.isnan(s) or np.isnan(g):
            continue
        lo, hi = b / n_bins, (b + 1) / n_bins
        print(f"[{lo:.1f}-{hi:.1f})     {s:>+18.4f}  {g:>+14.4f}  {g - s:>+12.4f}")


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
    print(
        f"  nonparametric:                  {n - n_param:6d} days ({100 * (n - n_param) / n:.2f}%)"
    )


# --------------------------------------- (2) mean-difference distortion check
def evaluate_mean_distortion(raw_ssp, raw_g6, debiased_ssp, debiased_g6, window):
    raw_ssp_mean = float(raw_ssp.sel(time=window).mean())
    raw_g6_mean = float(raw_g6.sel(time=window).mean())
    debiased_ssp_mean = float(debiased_ssp.sel(time=window).mean())
    debiased_g6_mean = float(debiased_g6.sel(time=window).mean())

    raw_delta = raw_g6_mean - raw_ssp_mean
    debiased_delta = debiased_g6_mean - debiased_ssp_mean
    distortion = debiased_delta - raw_delta

    print(f"Raw       ssp245 mean (2040-2069):       {raw_ssp_mean:.6e}")
    print(f"Raw       g6 mean (2040-2069):           {raw_g6_mean:.6e}")
    print(f"Debiased  ssp245 mean (2040-2069):       {debiased_ssp_mean:.6e}")
    print(f"Debiased  g6 mean (2040-2069):           {debiased_g6_mean:.6e}")
    print()
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


def run_analysis(obs_coarse, model_hist_native, model_ssp, model_g6, do_diagnostics=False):
    """Run the full diagnostic given already-loaded point timeseries.

    Call this directly from a notebook when you already have the four point
    series loaded -- e.g. as era5_pt, raw_historical_pt, raw_ssp_pt, raw_g6_pt:

        import sys
        sys.path.insert(0, "/path/to/notebooks/QA_QC/pr")
        import point_debias_diagnostic as pdd

        results = pdd.run_analysis(era5_pt, raw_historical_pt, raw_ssp_pt, raw_g6_pt)

    Expects:
      obs_coarse         : sliced to the training period
                            (TRAIN_PERIOD_START..TRAIN_PERIOD_END, i.e. 1978-2014) --
                            matches BCSDPipeline._load_scenario_data's obs_coarse slice
                            exactly (src/srm/pipeline.py ~line 1198).
      model_hist_native   : the historical series at FULL NATIVE extent -- do NOT
                            pre-truncate this to the training period. Production slices
                            model_hist per-scenario using
                            ``time=slice(train_period_start, predict_period_start - 1)``
                            (src/srm/pipeline.py ~line 1200), which is scenario-specific
                            because predict_period_start differs between ssp245 (2015)
                            and g6 (2035). See model_hist_for_predict_start() for why
                            this matters -- pre-truncating to train_period_end silently
                            drops data production actually includes for g6 and was the
                            real root cause of the Jan/Dec-only debiased_g6 mismatch.
      model_ssp, model_g6 : the raw future timeseries, each at native/full length.
                            model_ssp also doubles as the SSP245 bridge used to
                            stitch g6's gap before it starts (~2035), so don't
                            pre-truncate it to only where g6 has data.

    Mirrors ``BCSDPipeline._detrend_scenario`` (src/srm/pipeline.py, ~lines
    1211-1244): ``pr`` is a non-detrended variable (VariableConfig.detrend_data
    is False), so no actual detrending happens for either scenario. But g6-1.5k
    *is* a SAI scenario (config.is_sai_scenario), and per issue #363 even
    non-detrended SAI variables still get stitched -- historical + ssp245 bridge
    + g6 -- and sliced to the predict window, so the debiased-coarse output spans
    predict_period_start..end instead of starting at g6's native start (~2035).
    ssp245 itself is not a SAI scenario, so it takes the early-return branch
    unchanged (``return model_scenario, None``) -- no stitching. Each scenario
    uses its own real production predict_period_start/end
    (SSP_PREDICT_PERIOD_START/END, G6_PREDICT_PERIOD_START/END) -- these were
    previously a single shared (and wrong for both) guess.

    Returns a dict with the debiased series, the scenario inputs actually used
    (post-bridging), the scenario-specific model_hist arrays actually used, and
    the out-of-range masks, in case you want to inspect them further after the
    printed report.
    """
    model_hist_ssp = model_hist_for_predict_start(model_hist_native, SSP_PREDICT_PERIOD_START)
    model_hist_g6 = model_hist_for_predict_start(model_hist_native, G6_PREDICT_PERIOD_START)

    ssp_predict_slice = slice(f"{SSP_PREDICT_PERIOD_START}", f"{SSP_PREDICT_PERIOD_END}")
    ssp_input = model_ssp.sel(time=ssp_predict_slice)

    g6_predict_slice = slice(f"{G6_PREDICT_PERIOD_START}", f"{G6_PREDICT_PERIOD_END}")
    g6_bridged = stitch_historical_scenario(
        model_hist=model_hist_g6,
        model_scenario=model_g6,
        train_period_end=TRAIN_PERIOD_END,
        predict_period_start=G6_PREDICT_PERIOD_START,
        ssp_timeseries=model_ssp,
    )
    g6_input = g6_bridged.sel(time=g6_predict_slice)

    print("Running nonparametric_hybrid_2sided debiasing (ssp245)...")
    debiased_ssp, oor_low_ssp, oor_high_ssp = debias_scenario(obs_coarse, model_hist_ssp, ssp_input)

    print("Running nonparametric_hybrid_2sided debiasing (g6-1.5k)...")
    debiased_g6, oor_low_g6, oor_high_g6 = debias_scenario(obs_coarse, model_hist_g6, g6_input)

    if do_diagnostics:
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
        diagnose_cdf_saturation("ssp245", model_hist_ssp, ssp_input, oor_high_ssp, DELTA_WINDOW)
        diagnose_cdf_saturation("g6-1.5k", model_hist_g6, g6_input, oor_high_g6, DELTA_WINDOW)

        print()
        print("=" * 70)
        print(
            "(4) Is the correction quantile-dependent? Where does the distortion in (2) come from?"
        )
        print("=" * 70)
        ssp_bins = decompose_by_quantile_rank(
            "ssp245", ssp_input, debiased_ssp, model_hist_ssp, DELTA_WINDOW
        )
        print()
        g6_bins = decompose_by_quantile_rank(
            "g6-1.5k", g6_input, debiased_g6, model_hist_g6, DELTA_WINDOW
        )
        print()
        compare_quantile_occupancy(ssp_bins, g6_bins)

    return {
        "ssp_input": ssp_input,
        "g6_input": g6_input,
        "model_hist_ssp": model_hist_ssp,
        "model_hist_g6": model_hist_g6,
        "debiased_ssp": debiased_ssp,
        "debiased_g6": debiased_g6,
        "out_of_range_low_ssp": oor_low_ssp,
        "out_of_range_high_ssp": oor_high_ssp,
        "out_of_range_low_g6": oor_low_g6,
        "out_of_range_high_g6": oor_high_g6,
    }
