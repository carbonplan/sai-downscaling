import io
import logging
import time

import boto3
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import icechunk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from icechunk.xarray import to_icechunk

from srm import catalog
from srm.config import _icechunk_storage_for_path
from srm.downscaling_utils import interpolate_coarse_to_fine_grid, interpolate_fine_to_coarse_grid
from srm.encoding import (
    CHUNK_LAT,
    CHUNK_LON,
    CHUNK_TIME,
    COMPRESSOR,
    SHARD_LAT,
    SHARD_LON,
    SHARD_TIME,
)
from srm.qaqc import VAR_SPATIAL_RANGES, calculate_distortion_flags, sign_flip_mask

logger = logging.getLogger(__name__)

# Per-dim view of the pipeline's *current* output chunk/shard shapes. srm.encoding.make_encoding
# hardcodes the 3-D (time, lat, lon) tuple order, which can't encode the 2-D time-invariant flag,
# so the flags index these by their own dims instead.
#
# These are only a fallback. An existing store was written with whatever these constants held at
# the time (they last changed in PR #631), so _flag_encoding prefers the layout already on disk.
FLAG_CHUNKS = {"time": CHUNK_TIME, "lat": CHUNK_LAT, "lon": CHUNK_LON}
FLAG_SHARDS = {"time": SHARD_TIME, "lat": SHARD_LAT, "lon": SHARD_LON}

INPUT_CHUNKS = {"time": SHARD_TIME, "lat": SHARD_LAT, "lon": SHARD_LON}

# directory where outputs from step 1 are saved for use in calculating flags in step 2
DIR_QA_FLAG_CONSTANT_INPUTS = "s3://carbonplan-srm/output/qa_flag_inputs/"

# Variable-specific tolerances for differences in scenario comparisons (i.e. trends) between the raw GCM and  debiased, downscaled output (re-coarsened to native GCM grid). Grid cells where the scenario comparison differs by more than the absolute tolerance (in that variable's units defined in this dictionary) AND the percent tolerance are flagged.
# the sign flip flag occurs when the GCM scenario comparison is above the sign_flip threshold and the downscaled scenario comparison is below the negative of that threshold (or vice versa)
# E.g. if the raw GCM G6-1.5K-SAI scenario is 0.3 degrees cooler than the SSP245 scenario, but the downscaled G6-1.5K-SAI scenario is 0.4 degrees warmer than the SSP245 scenario in a grid cell, that would trigger a sign flip flag for that grid cell because the GCM and downscaled scenario comparisons have opposite signs of change and the absolute magnitude of those changes are both greater than the sign_flip threshold of 0.25 degrees.
# We use a sign_flip threshold to avoid flagging grid cells where scenario comparisons are different signs but essentially zero. As an example, we wouldn't want to flag a grid cell where the raw GCM G6-1.5K-SAI -> SSP245 precipitation change is -0.00001 mm/day and the downscaled G6-1.5K-SAI -> SSP245 change is +0.00001 mm/day, even though they are different signs, because the absolute magnitude of those changes is extremely close to zero.

TREND_VARIABLE_SETTINGS = {
    "tas": {
        "units": "K",
        "scale": 1.0,
        "abs_tol": 0.25,
        "pct_tol": 0.0,
        "sign_flip": 0.1,
    },
    "tasmax": {"units": "K", "scale": 1.0, "abs_tol": 0.25, "pct_tol": 0.0, "sign_flip": 0.1},
    "tasmin": {"units": "K", "scale": 1.0, "abs_tol": 0.25, "pct_tol": 0.0, "sign_flip": 0.1},
    "pr": {
        "units": "mm/yr",
        "scale": 31536000.0,  # factor to convert from kg/m2/sec to mm/year
        "abs_tol": 10.0,
        "pct_tol": 2.0,
        "sign_flip": 5.0,
    },
    "rsds": {"units": "W m-2", "scale": 1.0, "abs_tol": 1.0, "pct_tol": 0.25, "sign_flip": 0.5},
    "hurs": {
        "units": "%",
        "scale": 1.0,
        "abs_tol": 5.0,
        "pct_tol": 1.0,
        "sign_flip": 0.5,
    },
}

SCENARIO_COMPARISONS = {
    # When comparing the SAI and counterfactual SSP scenario, compare the same time period in both scenarios, towards the end of the SAI run when the differences
    # between the two scenarios should be the largest
    # Note that several CESM ensemble members did not run beyond 2070 in the SSP245 scenario, so the CESM ensemble mean is drawing from a different number of
    # ensemble members for different years
    "g6_1p5k_ssp245": {
        "scenario1": "ssp245",
        "scenario1_time_slice": slice("2065-01-01", "2084-12-31"),
        "scenario2": "g6_1p5k",
        "scenario2_time_slice": slice("2065-01-01", "2084-12-31"),
    },
    # When comparing a future SSP scenario to the historical, compare the full historical baseline used in this dataset (1978-2014) to a future time slice for the SSP245 scenario
    # Here, we use the same time slice as the SAI scenario for consistency, but this could be changed to a different future time slice if desired, e.g. later in the SSP245 scenario (2080-2100)
    # or to an earlier time slice that all ensemble members completed (e.g. 2050-2070). The fact that several CESM ensemble members did not run beyond 2070 in the SSP245 scenario is a limitation
    # of the existing time slice comparison
    "ssp245_hist": {
        "scenario1": "historical",
        "scenario1_time_slice": slice("1978-01-01", "2014-12-31"),
        "scenario2": "ssp245",
        "scenario2_time_slice": slice("2065-01-01", "2084-12-31"),
    },
    # When comparing a future SAI scenario to the historical, compare the full historical baseline used in this dataset (1978-2014) to the end of the SAI simulation (last 20 years)
    "g6_1p5k_hist": {
        "scenario1": "historical",
        "scenario1_time_slice": slice("1978-01-01", "2014-12-31"),
        "scenario2": "g6_1p5k",
        "scenario2_time_slice": slice("2065-01-01", "2084-12-31"),
    },
    # When comparing a termination shock simulation to the preceding SAI simulation, compare the end of the SAI simulation (using 20 years here)
    # to the full termination shock scenario, run to 2100 (so 15 years after termination are available)
    "g6_1p5k_end_g6_1p5k": {
        "scenario1": "g6_1p5k",
        "scenario1_time_slice": slice("2065-01-01", "2084-12-31"),
        "scenario2": "g6_1p5k_end",
        "scenario2_time_slice": slice("2085-01-01", "2099-12-31"),
    },
}

FLAG_LIST_TIME_VARYING = [
    "annual_outlier_flag",
    "rsds_max_exceeded",
    "outside_global_plausible_range",
    "temperature_inconsistency",
]

FLAG_LIST_TIME_INVARIANT = [
    "flipped_sign_ssp245_g6_1p5k",
    "flipped_sign_historical_ssp245",
    "flipped_sign_historical_g6_1p5k",
    "trend_distortion_historical_ssp245",
    "trend_distortion_ssp245_g6_1p5k",
    "trend_distortion_historical_g6_1p5k",
    "flipped_sign_g6_1p5k_g6_1p5k_end",
    "trend_distortion_g6_1p5k_g6_1p5k_end",
]

ATTRS_TIME_INVARIANT = {
    "long_name": "Change distortion flag",
    "description": (
        "This flag identifies pixels where debiasing and/or downscaling meaningfully changes "
        "common scenario intercomparisons. For each variable and ensemble member, it evaluates "
        "future scenarios (G6-1.5k, G6-1.5k-end, and SSP2-4.5) and assesses whether the change "
        "signals either (1) among them or (2) between them and the historical scenario are "
        "distorted meaningfully in either magnitude or sign of change (using a 5% threshold "
        "window). A distortion in any scenario-intercomparison flags the entire ensemble. This "
        "flag is time-invariant. See repo for details about each distortion test."
    ),
    "possible_values": "This is a binary flag: 0=no known issue; 1=known issue",
    "short_name": "qa_flag_time_invariant",
}

ATTRS_TIME_VARYING = {
    "long_name": "Quality issues flag",
    "description": (
        "This quality flag flags pixel-days when a variable exceeds multiple quality checks "
        "including screening for outliers, exceedances from variable-specific plausible ranges, "
        "and physically impossible relationships (e.g. tasmax < tas). For variables with "
        "multiple quality checks, failing any check flags that pixel-day for the variable. This "
        "flag is time-variant. See repo for details about each quality test."
    ),
    "possible_values": "This is a binary flag: 0=no known issue; 1=known issue",
    "short_name": "qa_flag_time_varying",
}


def calculate_thresholds(
    obs_max: xr.DataArray,
    obs_min: xr.DataArray,
    obs_max_std: xr.DataArray,
    obs_min_std: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute per-pixel outlier bounds from observational climatology stats.

    outlier_thresh_high = obs_max + 5 * obs_max_std; outlier_thresh_low = obs_min - 5 * obs_min_std.
    Used by prep_annual_threshold_inputs to build the thresholds consumed by flag_outliers.
    """
    outlier_thresh_high = obs_max + (5 * obs_max_std)
    outlier_thresh_low = obs_min - (5 * obs_min_std)

    return outlier_thresh_low, outlier_thresh_high


def flag_outliers(
    da: xr.DataArray,
    outlier_thresh_low: xr.DataArray,
    outlier_thresh_high: xr.DataArray,
    timescale: str = "annual",
) -> xr.DataArray:
    """
    Flag values outside the observational-record thresholds.

    Parameters
    ----------
    da : xr.DataArray
        Values to check, e.g. one variable's daily or annual data.
    outlier_thresh_low, outlier_thresh_high : xr.DataArray
        Per-pixel lower/upper bounds from calculate_thresholds: annual values
        broadcastable against `da` directly, or per-dayofyear values
        selected against `da`'s calendar day when ``timescale="dayofyear"``.
    timescale : {"annual", "dayofyear"}
        Whether the thresholds vary by day of year or are single annual
        values.

    Returns
    -------
    xr.DataArray
        Boolean flag, True where `da` is above `outlier_thresh_high` or
        below `outlier_thresh_low`.

    Raises
    ------
    ValueError
        If `timescale` is not one of ``{"annual", "dayofyear"}``.
    """
    accepted_timescales = {"dayofyear", "annual"}
    if timescale not in accepted_timescales:
        raise ValueError(
            f"unsupported timescale value: {timescale}. Valid values include: {accepted_timescales}"
        )
    if timescale == "dayofyear":
        doy = da["time"].dt.dayofyear
        high_outlier = da > outlier_thresh_high.sel(dayofyear=doy)
        low_outlier = da < outlier_thresh_low.sel(dayofyear=doy)
    elif timescale == "annual":
        high_outlier = da > outlier_thresh_high
        low_outlier = da < outlier_thresh_low

    any_outlier = (low_outlier | high_outlier) > 0

    return any_outlier


def flag_rsds_above_max(da: xr.DataArray, zonal_doy_max_rsds: xr.DataArray) -> xr.DataArray:
    """
    Flag days when rsds exceeds the expected max for its latitude and day of year.

    Parameters
    ----------
    da : xr.DataArray
        rsds values to check.
    zonal_doy_max_rsds : xr.DataArray
        Per-latitude, per-dayofyear maximum plausible rsds, as loaded by
        load_rsds_lims.

    Returns
    -------
    xr.DataArray
        Boolean flag, True where `da` exceeds the aligned max.
    """
    aligned_max = zonal_doy_max_rsds.sel(dayofyear=da.time.dt.dayofyear)
    exceeds_max = da > aligned_max

    return exceeds_max


def flag_global_exceedances(
    da: xr.DataArray, var: str, var_ranges: dict = VAR_SPATIAL_RANGES
) -> xr.DataArray:
    """
    Flag values outside the globally-defined plausible range for `var`.

    Parameters
    ----------
    da : xr.DataArray
        Values to check.
    var : str
        Key into `var_ranges`.
    var_ranges : dict
        Mapping of variable name to ``{"min": (lo, hi), "max": (lo, hi)}``;
        the plausible range applied here is ``[min[0], max[1]]`` -- see
        VAR_SPATIAL_RANGES.

    Returns
    -------
    xr.DataArray
        Boolean flag, True where `da` is outside the plausible range.
    """

    var_range = var_ranges[var]
    var_min = var_range["min"][0]
    var_max = var_range["max"][1]

    too_high = da > var_max
    too_low = da < var_min

    outside_range = (too_low | too_high) > 0

    return outside_range


def flag_tasmax_tas_inconsistency(tas: xr.DataArray, tasmax: xr.DataArray) -> xr.DataArray:
    """
    Flag days where tasmax < tas, which is physically inconsistent.

    Parameters
    ----------
    tas, tasmax : xr.DataArray

    Returns
    -------
    xr.DataArray
        Boolean flag, True where `tasmax` is below `tas`.
    """
    inconsistent_days = tasmax < tas
    return inconsistent_days


def flag_tasmin_tas_inconsistency(tas: xr.DataArray, tasmin: xr.DataArray) -> xr.DataArray:
    """
    Flag days where tasmin > tas, which is physically inconsistent.

    Parameters
    ----------
    tas, tasmin : xr.DataArray

    Returns
    -------
    xr.DataArray
        Boolean flag, True where `tasmin` is above `tas`.
    """
    inconsistent_days = tasmin > tas
    return inconsistent_days


def write_individual_flags(
    flag_data: xr.DataArray,
    flag_name: str,
    tag: str,
    bucket: str,
    prefix: str,
    write_mode: str = "a",
    branch: str = "main",
) -> None:
    """
    Write one intermediate QA flag to `tag`'s icechunk store in scratch.

    Intermediate flags are per-tag, per-check boolean arrays written here as
    uint8; they are later read back by get_intermediate_flags and combined
    into the two final flags by combine_intermediate_flags.

    Parameters
    ----------
    flag_data : xr.DataArray
        Boolean flag array; cast to uint8 before writing.
    flag_name : str
        Name to give the written variable, e.g. one entry of
        FLAG_LIST_TIME_VARYING or FLAG_LIST_TIME_INVARIANT.
    tag : str
        Tag identifying which icechunk store to write to -- one store per
        gcm/var/scenario/ens/method combination, as produced by
        discover_leaves.
    bucket, prefix : str
        S3 location of the intermediate-flags stores; the store path is
        ``s3://{bucket}/{prefix}/{tag}.icechunk``.
    write_mode : {"w", "a"}
        Passed through to `to_icechunk`. ``"a"`` appends `flag_name` to an
        existing store (creating the store first if needed); ``"w"`` always
        writes fresh variable-level encoding.
    branch : str
        icechunk branch to write to.

    Raises
    ------
    TypeError
        If `flag_data` is not boolean.
    """
    flag_data = flag_data.rename(flag_name)
    if flag_data.dtype != bool:
        raise TypeError(f"flag_data must be boolean before casting to uint8, got {flag_data.dtype}")
    flag_data = flag_data.astype(np.uint8)
    flag_data.attrs = {
        "long_name": "Quality flag",
        "description": "0=no known issue; 1=known issue",
        "short_name": flag_name,
    }
    dims, sizes = flag_data.dims, flag_data.sizes
    chunks = tuple(min(FLAG_CHUNKS[dim], sizes[dim]) for dim in dims)
    shards = tuple(
        chunk * max(1, min(FLAG_SHARDS[dim], sizes[dim]) // chunk)
        for dim, chunk in zip(dims, chunks, strict=True)
    )
    flag_data = flag_data.chunk(dict(zip(dims, shards, strict=True)))

    storage = icechunk.s3_storage(bucket=bucket, prefix=f"{prefix}/{tag}.icechunk", from_env=True)
    repo = icechunk.Repository.open_or_create(storage)  # one repo per gcm/var/scenario/ens tag
    session = repo.writable_session(branch)

    # encoding is only valid the first time flag_name is written to this store;
    # xarray errors if encoding is passed for a variable that already exists there
    variable_exists = False
    if write_mode != "w":
        try:
            variable_exists = flag_name in xr.open_zarr(session.store, consolidated=False).variables
        except Exception:
            variable_exists = False  # store doesn't exist yet

    encoding = (
        {}
        if variable_exists
        else {
            flag_name: {
                "_FillValue": None,
                "chunks": chunks,
                "shards": shards,
                "compressors": [COMPRESSOR],
            }
        }
    )

    to_icechunk(
        flag_data,
        session,
        mode=write_mode,
        align_chunks=True,
        encoding=encoding,
    )

    session.commit(
        f"write {flag_name} for {tag}",
        rebase_with=icechunk.ConflictDetector(),
    )


def save_figure_to_s3(fig, bucket: str, s3_key: str) -> None:
    """
    Save a matplotlib figure directly to S3, without writing a local file first.

    matplotlib's savefig() only writes to a local path or file-like object -- there's
    no direct-to-S3 write -- so this renders the figure into an in-memory PNG buffer
    and uploads that buffer's bytes with boto3. No local file is ever created.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save, e.g. from ``plt.gcf()``.
    bucket : str
        S3 bucket name.
    s3_key : str
        Full S3 object key (path within the bucket) to write to, e.g.
        ``"scratch/output/qa-intermediate-flags/_plots/annual_outlier_flag_{tag}.png"``.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    try:
        boto3.client("s3").put_object(Bucket=bucket, Key=s3_key, Body=buf.getvalue())
    except Exception as exc:
        logger.warning("Failed to upload plot to s3://%s/%s: %s", bucket, s3_key, exc)


def plot_flags(
    flags,
    time_varying: bool = True,
    separate_low_high: bool = True,
    vlims=None,
    bucket: str | None = None,
    s3_key: str | None = None,
) -> None:
    """
    Plot a map of where a flag is set, for visual QA review.

    Parameters
    ----------
    flags : xr.DataArray, or (low_flag, high_flag) tuple of xr.DataArray
        A single flag array when ``separate_low_high=False``; a two-element
        ``(low_flag, high_flag)`` pair plotted side by side when
        ``separate_low_high=True``.
    time_varying : bool
        If True, each flag is summed over the "time" dim first to get a
        cell-day count; if False, flags are already 2-D (lat, lon).
    separate_low_high : bool
        Which of the two `flags` shapes above is being passed.
    vlims : (float, float), optional
        Fixed (vmin, vmax) color limits; otherwise limits are auto-scaled.
    bucket, s3_key : str, optional
        If both are given (and a figure is actually drawn -- see below), also
        upload the figure to ``s3://{bucket}/{s3_key}`` via save_figure_to_s3.

    Draws nothing (or prints "No flags found.") when no cells are flagged --
    in that case, nothing is uploaded either even if bucket/s3_key are given.
    """
    if separate_low_high:
        low_flag = flags[0]
        high_flag = flags[1]

        if time_varying:
            count_high_flag = high_flag.sum(dim="time").load()
            count_low_flag = low_flag.sum(dim="time").load()
        else:
            count_high_flag = high_flag
            count_low_flag = low_flag

        contains_high_flags = np.nansum(count_high_flag)
        contains_low_flags = np.nansum(count_low_flag)

        contains_flags = (contains_high_flags + contains_low_flags) > 0

        if contains_flags:
            plt.figure(figsize=(10, 3))

            ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
            if contains_low_flags:
                if vlims is not None:
                    count_low_flag.where(count_low_flag > 0).plot(
                        ax=ax1, transform=ccrs.PlateCarree(), vmin=vlims[0], vmax=vlims[1]
                    )
                else:
                    count_low_flag.where(count_low_flag > 0).plot(
                        ax=ax1, transform=ccrs.PlateCarree()
                    )
                ax1.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.4")
            ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
            if contains_high_flags:
                if vlims is not None:
                    count_high_flag.where(count_high_flag > 0).plot(
                        ax=ax2, transform=ccrs.PlateCarree(), vmin=vlims[0], vmax=vlims[1]
                    )
                else:
                    count_high_flag.where(count_high_flag > 0).plot(
                        ax=ax2, transform=ccrs.PlateCarree()
                    )
                ax2.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.4")
            plt.tight_layout()
            if bucket is not None and s3_key is not None:
                save_figure_to_s3(plt.gcf(), bucket=bucket, s3_key=s3_key)
            plt.show()
    else:
        flag = flags
        if time_varying:
            count_flag = flag.sum(dim="time").load()
        else:
            count_flag = flag

        contains_flags = np.nansum(count_flag)

        # draw the map only when something is flagged
        if contains_flags > 0:
            plt.figure(figsize=(5, 3))
            ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            limits = {} if contains_flags else {"vmin": 0, "vmax": 1}
            if vlims is not None:
                count_flag.where(count_flag > 0).plot(
                    ax=ax1, transform=ccrs.PlateCarree(), vmin=vlims[0], vmax=vlims[1]
                )
            else:
                count_flag.where(count_flag > 0).plot(
                    ax=ax1, transform=ccrs.PlateCarree(), **limits
                )
            ax1.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.4")
            unit = "cell-days" if time_varying else "cells"
            ax1.set_title(f"{count_flag.name}: {float(contains_flags):,.0f} flagged {unit}")
            if bucket is not None and s3_key is not None:
                save_figure_to_s3(plt.gcf(), bucket=bucket, s3_key=s3_key)
        else:
            logger.info("No flags found.")


def parse_tag(tag: str) -> tuple[str, str, str, str, str]:
    """
    Split a ``"{gcm}_{var}_{scenario}_{ens}_{method}"`` tag into its fields.

    The inverse of the tag construction in discover_leaves. `scenario` may
    itself contain underscores (e.g. ``"g6_1p5k"``), so it is recovered as
    everything between the fixed-position leading ``gcm``/``var`` and
    trailing ``ens``/``method`` fields, not by a fixed split index; `gcm`,
    `var`, `ens`, and `method` must not themselves contain underscores for
    this to round-trip correctly.

    Returns
    -------
    tuple[str, str, str, str, str]
        ``(gcm, var, scenario, ens, method)``.
    """
    parts = tag.split("_")
    if len(parts) < 4:
        raise ValueError(f"Tag is expected to have 4 parts: gcm_var_scenario_ens_method: {tag}")
    gcm, var, ens, method = parts[0], parts[1], parts[-2], parts[-1]
    scenario = "_".join(parts[2:-2])
    return gcm, var, scenario, ens, method


def get_data(
    tag: str, trees: dict[str, xr.DataTree], is_downscaled: bool, var_to_analyze: str | None = None
) -> xr.DataArray:
    """
    Look up the DataArray for `tag` within the already-opened `trees`.

    Parameters
    ----------
    tag : str
        A tag as produced by discover_leaves / parsed by parse_tag.
    trees : dict[str, xr.DataTree]
        Mapping of gcm name -> opened DataTree, as returned by discover_leaves.
    is_downscaled : bool
        Whether the data is downscaled (False means debiased_coarse output).
    var_to_analyze : str, optional
        Variable to pull from the leaf group; defaults to the variable
        encoded in `tag`. Pass this to pull a sibling variable from the same
        group -- e.g. tasmax, while iterating over `tas` tags.

    Returns
    -------
    xr.DataArray
        The requested variable's data at that leaf.
    """
    [gcm, var, scenario, ens, method] = parse_tag(tag)

    # This is a bit of a hack to handle the fact that datasets produced before the addition
    # of QDMSD don't have a method specified, and are just gcm_var_scenario_ens. In those cases,
    # we should just use the scenario/var/ens as the group path.
    if method == "no-method-specified":
        if is_downscaled:
            group_path = f"{scenario}/{var}/{ens}"
        else:
            group_path = f"debiased_coarse/{scenario}/{var}/{ens}"
    else:
        if is_downscaled:
            group_path = f"{method}/{scenario}/{var}/{ens}"
        else:
            group_path = f"{method}/debiased_coarse/{scenario}/{var}/{ens}"
    comparison_ds = trees[gcm][group_path]
    if var_to_analyze is None:
        var_to_analyze = var
    da = comparison_ds[var_to_analyze]

    return da


def run_flag_loop(
    tags: list[str],
    trees: dict[str, xr.DataTree],
    flag_name: str,
    compute_flag,
    bucket: str,
    prefix: str,
    is_downscaled: bool,
    var_filter: list[str] | None = None,
    write_mode: str = "a",
    plot: bool = True,
    save_plots: bool = False,
) -> None:
    """Loop over tags, compute one flag per leaf, write it, optionally plot it.

    Parameters
    ----------
    tags : list[str]
        Tags to iterate, as produced by discover_leaves.
    trees : dict[str, xr.DataTree]
    flag_name : str
        Name to write the computed flag under (see write_individual_flags).
    compute_flag : callable
        ``compute_flag(da, var) -> flag DataArray``. Bind whatever extra
        fixed arguments a specific check needs (e.g. `zonal_doy_max_rsds`)
        with a lambda at the call site -- this loop doesn't need to know
        what they are.
    var_filter : list of str, optional
        If given, only process tags whose variable equals this.
    write_mode : {"w", "a"}
        Passed through to write_individual_flags.
    plot : bool
        If True, call plot_flags on each computed flag.
    save_plots : bool
        If True (and plot is also True), also upload each plotted figure to
        S3 under ``{bucket}/{prefix}/_plots/{flag_name}_{tag}.png``.
    bucket, prefix : str
        Passed through to write_individual_flags.
    is_downscaled : bool
        Passed through to get_data.
    """
    logger.info("%d tags to process", len(tags))
    for tag in tags:
        [gcm, var, scenario, ens, method] = parse_tag(tag)
        if var_filter is not None and var not in var_filter:
            continue
        logger.info(tag)

        da = get_data(tag=tag, trees=trees, is_downscaled=is_downscaled)
        flag_data = compute_flag(da, var)

        write_individual_flags(
            flag_data=flag_data,
            flag_name=flag_name,
            tag=tag,
            write_mode=write_mode,
            bucket=bucket,
            prefix=prefix,
        )

        if plot:
            if save_plots:
                s3_key = f"{prefix}/_plots/{flag_name}_{tag}.png"
            else:
                s3_key = None
            plot_flags(
                flags=flag_data,
                time_varying=True,
                separate_low_high=False,
                bucket=bucket,
                s3_key=s3_key,
            )
            plt.show()
            plt.close()


def run_flag_loop_temperature_inconsistencies(
    tags: list[str],
    trees: dict[str, xr.DataTree],
    bucket: str,
    prefix: str,
    is_downscaled: bool,
    plot: bool = True,
    save_plots: bool = False,
    flag_name: str = "temperature_inconsistency",
    write_mode: str = "a",
) -> None:
    """Flag tas/tasmin/tasmax physical inconsistencies (tasmax < tas or tasmin > tas).

    Iterates the `tas` tags in `tags`; for each one whose matching tasmin and
    tasmax tags are also present, computes all three inconsistency flags
    (flag_tasmax_tas_inconsistency, flag_tasmin_tas_inconsistency, and their
    union for tas) and writes each to its own tag's intermediate store under
    `flag_name`. `tas` tags with no matching tasmin/tasmax tag are skipped.

    Parameters
    ----------
    tags : list[str]
        Tags to iterate, as produced by discover_leaves.
    trees : dict[str, xr.DataTree]
    plot : bool
        If True, call plot_flags on the combined tas flag.
    save_plots : bool
        If True (and plot is also True), also upload the plotted figure to
        S3 under ``{bucket}/{prefix}/_plots/{flag_name}_{tag}.png``.
    flag_name : str
        Name to write all three computed flags under.
    write_mode : {"w", "a"}
        Passed through to write_individual_flags.
    bucket, prefix : str
        Passed through to write_individual_flags.
    is_downscaled : bool
        Passed through to get_data.
    """
    logger.info("%d tags to process", len(tags))
    for tag in tags:
        gcm, var, scenario, ens, method = parse_tag(tag)
        if var != "tas":
            continue

        tag_tasmin = f"{gcm}_tasmin_{scenario}_{ens}_{method}"
        tag_tasmax = f"{gcm}_tasmax_{scenario}_{ens}_{method}"
        if (tag_tasmin in tags) and (tag_tasmax in tags):
            logger.info(tag)

            tas = get_data(tag=tag, trees=trees, is_downscaled=is_downscaled)
            tasmin = get_data(tag=tag_tasmin, trees=trees, is_downscaled=is_downscaled)
            tasmax = get_data(tag=tag_tasmax, trees=trees, is_downscaled=is_downscaled)

            flag_tas_tasmax = flag_tasmax_tas_inconsistency(tas, tasmax)
            flag_tas_tasmin = flag_tasmin_tas_inconsistency(tas, tasmin)

            flag_tas = (flag_tas_tasmax + flag_tas_tasmin) > 0
            flag_tasmin = flag_tas_tasmin
            flag_tasmax = flag_tas_tasmax

            write_individual_flags(
                flag_data=flag_tas,
                flag_name=flag_name,
                tag=tag,
                write_mode=write_mode,
                bucket=bucket,
                prefix=prefix,
            )

            write_individual_flags(
                flag_data=flag_tasmax,
                flag_name=flag_name,
                tag=tag_tasmax,
                write_mode=write_mode,
                bucket=bucket,
                prefix=prefix,
            )

            write_individual_flags(
                flag_data=flag_tasmin,
                flag_name=flag_name,
                tag=tag_tasmin,
                write_mode=write_mode,
                bucket=bucket,
                prefix=prefix,
            )

            if plot:
                if save_plots:
                    s3_key = f"{prefix}/_plots/{flag_name}_{tag}.png"
                else:
                    s3_key = None
                plot_flags(
                    flags=flag_tas,
                    time_varying=True,
                    separate_low_high=False,
                    bucket=bucket,
                    s3_key=s3_key,
                )
                plt.show()
                plt.close()


# The catalog now uses exact model names (e.g. "UKESM1-1-LL" instead of "UKESM"). If
# a dataset was generated before this change, the model names will be inconsistent.
# This dictionary maps the pipeline's GCM names to the catalog's GCM names for any exceptions.
# Map the exceptions here; any model name not listed is looked up under its own name.
GCM_CATALOG_NAME_OVERRIDES = {
    "CESM2-WACCM": "CESM2-WACCM6",
    "UKESM": "UKESM1-1-LL",
}


def calculate_ensemble_mean_deltas(
    variable: str,
    tags_scenario1: np.ndarray,
    tags_scenario2: np.ndarray,
    gcm: str,
    scenario1: str,
    scenario2: str,
    scenario1_time_slice: slice,
    scenario2_time_slice: slice,
    trees: dict[str, xr.DataTree],
    is_downscaled: bool,
):
    """
    Compute the ensemble-mean scenario1->scenario2 change in the raw GCM vs.
    the pipeline's (debiased/downscaled) output, for one gcm/variable/method.

    Both scenarios' ensemble means are computed over their own time slice,
    averaged across every ensemble member found in `tags_scenario1`/
    `tags_scenario2`. The pipeline's ensemble means are also re-coarsened
    onto the raw GCM's grid (when `is_downscaled`) so the two deltas are
    directly comparable; this coarse-grid comparison is what
    calculate_trend_distortion_flags flags for distortion/sign-flip.

    Parameters
    ----------
    variable : str
        Variable name as stored in the raw GCM catalog.
    tags_scenario1, tags_scenario2 : array-like of str
        Tags (one per ensemble member) for scenario1 and scenario2, already
        filtered to one gcm/variable/method; must be non-empty.
    gcm, scenario1, scenario2 : str
    scenario1_time_slice, scenario2_time_slice : slice
        Time ranges to average over within each scenario.
    trees : dict[str, xr.DataTree]
    is_downscaled : bool
        If True, coarsen the pipeline's ensemble means onto the raw GCM grid
        before differencing. If False, the pipeline data is already on the
        coarse grid (e.g. debiased_coarse output).

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]
        ``(delta_raw, delta_raw_pct, delta_ds_coarse, delta_ds_coarse_pct, delta_ds)``:
        the raw-GCM and pipeline (coarse-grid) scenario2-scenario1 deltas, as
        absolute and percent change, plus the pipeline delta on its native
        (fine) grid.

    Raises
    ------
    ValueError
        If `tags_scenario1` or `tags_scenario2` is empty.
    """
    # Construct data arrays including all relevant ensemble members
    das_scenario1 = []
    ens_scenario1 = []
    for tag in tags_scenario1:
        da = get_data(tag=tag, trees=trees, is_downscaled=is_downscaled)
        [thisgcm, thisvar, thisscenario, thisens, thismethod] = parse_tag(tag)
        das_scenario1.append(da)
        ens_scenario1.append(thisens)

    if not das_scenario1:
        raise ValueError(
            f"no leaves found for gcm={gcm!r}, scenario={scenario1!r}, variable={variable!r}"
        )

    var_scenario1 = xr.concat(
        das_scenario1, dim=pd.Index(ens_scenario1, name="ensemble_member"), join="outer"
    )

    das_scenario2 = []
    ens_scenario2 = []
    for tag in tags_scenario2:
        da = get_data(tag=tag, trees=trees, is_downscaled=is_downscaled)
        [thisgcm, thisvar, thisscenario, thisens, thismethod] = parse_tag(tag)
        das_scenario2.append(da)
        ens_scenario2.append(thisens)

    if not das_scenario2:
        raise ValueError(
            f"no leaves found for gcm={gcm!r}, scenario={scenario2!r}, variable={variable!r}"
        )

    var_scenario2 = xr.concat(
        das_scenario2, dim=pd.Index(ens_scenario2, name="ensemble_member"), join="outer"
    )

    # Take ensemble mean across comparison time periods
    ens_mean_var_scenario1 = var_scenario1.sel(time=scenario1_time_slice).mean(
        dim=["time", "ensemble_member"]
    )
    ens_mean_var_scenario2 = var_scenario2.sel(time=scenario2_time_slice).mean(
        dim=["time", "ensemble_member"]
    )

    # Temporary fix to deal with catalog inconsistency vs. model names used in old runs
    gcm_catalog_name = GCM_CATALOG_NAME_OVERRIDES.get(gcm, gcm)

    # Take ensemble mean across comparison time periods in raw data
    raw_ds = catalog.get(gcm_catalog_name).to_xarray()
    raw_scenario1_mean = (
        raw_ds[scenario1][variable]
        .sel(time=scenario1_time_slice, ensemble_member=ens_scenario1)
        .mean(dim=["time", "ensemble_member"])
    )
    raw_scenario2_mean = (
        raw_ds[scenario2][variable]
        .sel(time=scenario2_time_slice, ensemble_member=ens_scenario2)
        .mean(dim=["time", "ensemble_member"])
    )

    # Coarsen the downscaled ensemble means
    if is_downscaled:
        ens_mean_var_scenario1_coarsened = interpolate_fine_to_coarse_grid(
            da_fine_to_coarsen=ens_mean_var_scenario1, da_coarse_grid=raw_scenario1_mean
        )
        ens_mean_var_scenario2_coarsened = interpolate_fine_to_coarse_grid(
            da_fine_to_coarsen=ens_mean_var_scenario2, da_coarse_grid=raw_scenario1_mean
        )
    else:
        ens_mean_var_scenario1_coarsened = ens_mean_var_scenario1
        ens_mean_var_scenario2_coarsened = ens_mean_var_scenario2

    # Calculate scenario comparison (annual mean) in downscaled and raw
    delta_raw = raw_scenario2_mean - raw_scenario1_mean
    delta_raw_pct = delta_raw * 100 / raw_scenario1_mean

    delta_ds_coarse = ens_mean_var_scenario2_coarsened - ens_mean_var_scenario1_coarsened
    delta_ds_coarse_pct = delta_ds_coarse * 100 / ens_mean_var_scenario1_coarsened

    delta_ds = ens_mean_var_scenario2 - ens_mean_var_scenario1

    return delta_raw, delta_raw_pct, delta_ds_coarse, delta_ds_coarse_pct, delta_ds


def calculate_trend_distortion_flags(
    trees: dict,
    gcms: list[str],
    variables: list[str],
    methods: list[str],
    tags_np: np.ndarray,
    gcms_np: np.ndarray,
    scenarios_np: np.ndarray,
    variables_np: np.ndarray,
    methods_np: np.ndarray,
    bucket: str,
    prefix: str,
    is_downscaled: bool,
    scenario_comparisons: dict = SCENARIO_COMPARISONS,
    plot: bool = True,
    save_plots: bool = False,
) -> None:
    """
    Compute and write the time-invariant trend-distortion/sign-flip flags.

    For every (scenario_comparisons entry) x gcm x variable x method
    combination, computes the raw-GCM vs. debiased/downscaled scenario
    change at coarse resolution (calculate_ensemble_mean_deltas), flags grid
    cells where that change is distorted beyond tolerance
    (TREND_VARIABLE_SETTINGS, via calculate_distortion_flags) or flips sign
    (sign_flip_mask), regrids both flags back to the fine grid, and writes
    them -- named ``"trend_distortion_{scenario1}_{scenario2}"`` and
    ``"flipped_sign_{scenario1}_{scenario2}"`` -- to every ensemble-member
    tag involved in that comparison (see FLAG_LIST_TIME_INVARIANT).

    A (gcm, variable, method, comparison) combination with no tags for either
    scenario (e.g. a GCM that never ran the comparison's second scenario) is
    skipped with a logged warning rather than raising -- those tags simply
    never get this comparison's flag written, which combine_intermediate_flags
    already treats as "not applicable" rather than "known issue".

    Parameters
    ----------
    trees : dict[str, xr.DataTree]
    gcms, variables, methods : list[str]
        Values to loop over for each scenario comparison.
    tags_np, gcms_np, scenarios_np, variables_np, methods_np : np.ndarray
        Parallel per-leaf arrays returned by discover_leaves, used to select
        the tags for each gcm/variable/method/scenario combination.
    bucket, prefix : str
        Passed through to write_individual_flags.
    is_downscaled : bool
        Passed through to calculate_ensemble_mean_deltas.
    scenario_comparisons : dict
        See SCENARIO_COMPARISONS.
    plot : bool
        If True, call plot_flags on each computed flag.
    save_plots : bool
        If True (and plot is also True), also upload each plotted figure to
        S3 under
        ``{bucket}/{prefix}/_plots/{gcm}_{var}_{method}_trend_distortion_{scenario1}_{scenario2}.png``
        (and the ``flipped_sign_`` equivalent).
    """
    for key in scenario_comparisons:
        logger.info("scenario comparison: %s", key)
        comparison_dict = scenario_comparisons[key]
        scenario1 = comparison_dict["scenario1"]
        scenario2 = comparison_dict["scenario2"]
        scenario1_time_slice = comparison_dict["scenario1_time_slice"]
        scenario2_time_slice = comparison_dict["scenario2_time_slice"]

        for gcm in gcms:
            for var in variables:
                for method in methods:
                    logger.info("variable: %s", var)
                    # Find tags to use in scenario comparison (all ensemble members for this variable and gcm for the two comparison scenarios)
                    tags_scenario1 = tags_np[
                        (gcms_np == gcm)
                        * (scenarios_np == scenario1)
                        * (variables_np == var)
                        * (methods_np == method)
                    ]
                    tags_scenario2 = tags_np[
                        (gcms_np == gcm)
                        * (scenarios_np == scenario2)
                        * (variables_np == var)
                        * (methods_np == method)
                    ]

                    if len(tags_scenario1) == 0 or len(tags_scenario2) == 0:
                        logger.warning(
                            "Skipping trend-distortion check for gcm=%s, var=%s, method=%s, "
                            "comparison=%s (%s->%s): %d tags found for %s, %d tags found for %s",
                            gcm,
                            var,
                            method,
                            key,
                            scenario1,
                            scenario2,
                            len(tags_scenario1),
                            scenario1,
                            len(tags_scenario2),
                            scenario2,
                        )
                        continue

                    # Calculate distortions
                    [delta_raw, delta_raw_pct, delta_ds_coarse, delta_ds_coarse_pct, delta_ds] = (
                        calculate_ensemble_mean_deltas(
                            variable=var,
                            tags_scenario1=tags_scenario1,
                            tags_scenario2=tags_scenario2,
                            gcm=gcm,
                            scenario1=scenario1,
                            scenario2=scenario2,
                            scenario1_time_slice=scenario1_time_slice,
                            scenario2_time_slice=scenario2_time_slice,
                            trees=trees,
                            is_downscaled=is_downscaled,
                        )
                    )

                    distortion_absolute = delta_ds_coarse - delta_raw
                    distortion_pct = delta_ds_coarse_pct - delta_raw_pct

                    # Flag distortions at coarse scale
                    abs_tol = TREND_VARIABLE_SETTINGS[var]["abs_tol"]
                    pct_tol = TREND_VARIABLE_SETTINGS[var]["pct_tol"]
                    sign_flip_tol = TREND_VARIABLE_SETTINGS[var]["sign_flip"]
                    scale = TREND_VARIABLE_SETTINGS[var]["scale"]

                    trend_distortion_flag = calculate_distortion_flags(
                        distortion_absolute=distortion_absolute * scale,
                        distortion_pct=distortion_pct,
                        tolerance_absolute=abs_tol,
                        tolerance_pct=pct_tol,
                    )

                    flipped_sign_flag = sign_flip_mask(
                        delta_ds_coarse * scale, delta_raw * scale, threshold=sign_flip_tol
                    )

                    # Propagate coarse distortion flags to fine scale
                    trend_distortion_flag_fine_frac = interpolate_coarse_to_fine_grid(
                        da_coarse_to_regrid=trend_distortion_flag.astype("float32"),
                        da_fine_grid=delta_ds,
                    )
                    trend_distortion_flag_fine = trend_distortion_flag_fine_frac > 0

                    flipped_sign_flag_fine_frac = interpolate_coarse_to_fine_grid(
                        da_coarse_to_regrid=flipped_sign_flag.astype("float32"),
                        da_fine_grid=delta_ds,
                    )
                    flipped_sign_flag_fine = flipped_sign_flag_fine_frac > 0

                    if plot:
                        if save_plots:
                            name_prefix = f"{gcm}_{var}_{method}"
                            s3_key_trend_distortion = (
                                f"{prefix}/_plots/{name_prefix}_trend_distortion_"
                                f"{scenario1}_{scenario2}.png"
                            )
                            s3_key_sign_flip = (
                                f"{prefix}/_plots/{name_prefix}_flipped_sign_"
                                f"{scenario1}_{scenario2}.png"
                            )
                        else:
                            s3_key_trend_distortion = None
                            s3_key_sign_flip = None

                        plot_flags(
                            flags=trend_distortion_flag_fine,
                            time_varying=False,
                            separate_low_high=False,
                            bucket=bucket,
                            s3_key=s3_key_trend_distortion,
                        )
                        plt.show()
                        plt.close()

                        plot_flags(
                            flags=flipped_sign_flag_fine,
                            time_varying=False,
                            separate_low_high=False,
                            bucket=bucket,
                            s3_key=s3_key_sign_flip,
                        )
                        plt.show()
                        plt.close()

                    # Write out flags
                    tags_to_flag = np.concat([tags_scenario1, tags_scenario2])
                    for tag in tags_to_flag:
                        logger.info(tag)
                        gcm, var, scenario, ens, method = parse_tag(tag)
                        write_individual_flags(
                            flag_data=trend_distortion_flag_fine,
                            flag_name="trend_distortion_" + scenario1 + "_" + scenario2,
                            tag=tag,
                            write_mode="a",
                            bucket=bucket,
                            prefix=prefix,
                        )

                        write_individual_flags(
                            flag_data=flipped_sign_flag_fine,
                            flag_name="flipped_sign_" + scenario1 + "_" + scenario2,
                            tag=tag,
                            write_mode="a",
                            bucket=bucket,
                            prefix=prefix,
                        )


def get_intermediate_flags(
    tag: str,
    bucket: str,
    prefix: str,
    branch: str = "main",
) -> xr.Dataset:
    """
    Open the intermediate-flags icechunk store for `tag` and return it as a Dataset.

    This is the store written by write_individual_flags; each data variable
    in it is one intermediate flag (see FLAG_LIST_TIME_VARYING /
    FLAG_LIST_TIME_INVARIANT), later combined by combine_intermediate_flags.

    Parameters
    ----------
    tag : str
        Tag identifying the store, as produced by discover_leaves.
    bucket, prefix : str
        S3 location of the intermediate-flags stores.
    branch : str
        icechunk branch to read from.

    Returns
    -------
    xr.Dataset
        One data variable per intermediate flag written for `tag`.
    """
    storage = icechunk.s3_storage(bucket=bucket, prefix=f"{prefix}/{tag}.icechunk", from_env=True)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch)
    flag_data = xr.open_zarr(session.store, consolidated=False)
    return flag_data


def combine_intermediate_flags(
    tag: str,
    flag_list_time_varying: list[str],
    flag_list_time_invariant: list[str],
    bucket: str,
    prefix: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Combine a tag's intermediate flags into one time-varying and one time-invariant flag.

    Each name in `flag_list_time_varying`/`flag_list_time_invariant` that is
    present in `tag`'s intermediate-flags store (some checks don't apply to
    every variable) is OR-ed together; a name that isn't present is silently
    skipped rather than treated as all-False.

    Parameters
    ----------
    tag : str
        Tag identifying the intermediate-flags store to read, as produced by
        discover_leaves.
    flag_list_time_varying : list[str]
        Intermediate flag names to OR together into the combined time-varying
        flag, e.g. FLAG_LIST_TIME_VARYING.
    flag_list_time_invariant : list[str]
        Intermediate flag names to OR together into the combined
        time-invariant flag, e.g. FLAG_LIST_TIME_INVARIANT.
    bucket, prefix : str
        S3 location of the intermediate-flags stores; passed to
        get_intermediate_flags.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        ``(overall_flag_time_varying, overall_flag_time_invariant)``, each
        boolean, True wherever any contributing intermediate flag was True.
    """

    flags = get_intermediate_flags(tag=tag, bucket=bucket, prefix=prefix)

    ind = 0
    for flag_time_varying in flag_list_time_varying:
        if flag_time_varying in flags.variables:
            flag = flags[flag_time_varying]
            if ind == 0:
                overall_flag_time_varying = flag
            else:
                overall_flag_time_varying = overall_flag_time_varying + flag
            ind = ind + 1
    overall_flag_time_varying = overall_flag_time_varying > 0

    ind = 0
    for flag_time_invariant in flag_list_time_invariant:
        if flag_time_invariant in flags.variables:
            flag = flags[flag_time_invariant]
            if ind == 0:
                overall_flag_time_invariant = flag
            else:
                overall_flag_time_invariant = overall_flag_time_invariant + flag
            ind = ind + 1
    overall_flag_time_invariant = overall_flag_time_invariant > 0

    return overall_flag_time_varying, overall_flag_time_invariant


FLAG_SINGLE_CHUNK_MAX_BYTES = 16 * 1024**2


def _flag_encoding(existing: xr.Dataset, group: str, flag_data: xr.DataArray) -> dict:
    """Chunk/shard shape for ``flag_data``, copied from the group's data variable.

    Parameters
    ----------
    existing : xr.Dataset
        The production store's group already opened (via `xr.open_zarr`),
        used to find a data variable to copy chunk/shard shape from.
    group : str
        The group path `flag_data` will be written into, e.g.
        ``"{method}/{scenario}/{var}/{ens}"``; the second-to-last path
        segment is taken as the data-variable name to prefer.
    flag_data : xr.DataArray
        The flag about to be written; only its dims/shape/nbytes are used.

    Returns
    -------
    dict
        ``{"chunks": tuple, "shards": tuple | None}`` in `flag_data`'s dim
        order.
    """
    dims = flag_data.dims

    if flag_data.nbytes <= FLAG_SINGLE_CHUNK_MAX_BYTES:
        return {"chunks": tuple(flag_data.shape), "shards": None}

    var_name = group.strip("/").split("/")[-2]
    candidates = [existing[var_name]] if var_name in existing.data_vars else []
    candidates += [var for name, var in existing.data_vars.items() if not name.startswith("flag_")]

    for var in candidates:
        chunks = var.encoding.get("chunks")
        if chunks is None or not set(dims) <= set(var.dims):
            continue
        idx = [var.dims.index(dim) for dim in dims]
        shards = var.encoding.get("shards")
        return {
            "chunks": tuple(chunks[i] for i in idx),
            "shards": tuple(shards[i] for i in idx) if shards else None,
        }

    return {
        "chunks": tuple(FLAG_CHUNKS[dim] for dim in dims),
        "shards": tuple(FLAG_SHARDS[dim] for dim in dims),
    }


def write_final_qa_flags(
    session: icechunk.Session,
    group: str,
    flag_data: xr.DataArray,
    flag_name: str,
    attrs: dict,
    overwrite: bool = False,
) -> None:
    """Write one QA flag variable into an existing group of the final output store.

    Refuses to replace a flag that already exists in the group unless ``overwrite=True``
    is passed explicitly -- this writes into the released production store.

    Parameters
    ----------
    session : icechunk.Session
        Writable session on the production store; not committed here.
    group : str
        Group path to write into, e.g. ``"{method}/{scenario}/{var}/{ens}"``.
    flag_data : xr.DataArray
        Boolean flag array; cast to uint8 before writing.
    flag_name : str
        Name to give the written variable, e.g. ``ATTRS_TIME_VARYING["short_name"]``.
    attrs : dict
        Attributes to attach to the written variable, e.g. ATTRS_TIME_VARYING
        or ATTRS_TIME_INVARIANT.
    overwrite : bool
        If False (default) and `flag_name` already exists in `group`, raises
        instead of replacing it.

    Raises
    ------
    TypeError
        If `flag_data` is not boolean.
    ValueError
        If `flag_name` already exists in `group` and `overwrite` is False.

    Notes
    -----
    Doesn't commit the icechunk session -- that's the caller's responsibility.
    """
    if flag_data.dtype != bool:
        raise TypeError(f"flag_data must be boolean before casting to uint8, got {flag_data.dtype}")
    flag_data = flag_data.rename(flag_name).astype(np.uint8)
    flag_data.attrs = attrs

    existing = xr.open_zarr(session.store, group=group, consolidated=False)
    variable_exists = flag_name in existing.variables

    if variable_exists and not overwrite:
        raise ValueError(
            f"{flag_name!r} already exists in group {group!r}; pass overwrite=True to replace it."
        )

    # Drop the coords carried over from the intermediate flag store. mode="a" writes
    # every variable in the dataset, so leaving them on could re-write the production
    # group's time/lat/lon arrays with the intermediate store's encoding.
    flag_data = flag_data.drop_vars(flag_data.coords)

    dims = flag_data.dims
    layout = _flag_encoding(existing, group, flag_data)

    if flag_data.chunks is not None:
        dask_chunks = layout["shards"] or layout["chunks"]
        flag_data = flag_data.chunk(dict(zip(dims, dask_chunks, strict=True)))

    if variable_exists:
        # overwrite path should already have variable level encoding. Note this means an
        # overwrite cannot change the chunk shape -- reset the branch and rewrite for that.
        encoding = {}
    else:
        encoding = {
            flag_name: {
                "compressors": [COMPRESSOR],
                **{key: value for key, value in layout.items() if value is not None},
            }
        }

    flags = flag_data.to_dataset()
    flags.attrs = dict(existing.attrs)

    to_icechunk(
        flags,
        session,
        group=group,
        mode="a",
        encoding=encoding,
    )


def discover_leaves(
    gcms: list[str], branch: str, root_dir: str, store_subset_id: str, is_downscaled: bool
):
    """Open each GCM's icechunk store and enumerate its (scenario, variable, ensemble) leaves.

    A GCM whose store fails to open (e.g. a run still in flight) is skipped
    rather than failing the whole call, so this can be run against a
    partially-landed set of runs.

    Parameters
    ----------
    gcms : list[str]
        GCM names to open, matching the ``{gcm}-ERA5-{store_subset_id}.icechunk``
        store naming under `root_dir`.
    branch : str
        icechunk branch to read.
    root_dir : str
        Directory containing each GCM's icechunk store.
    store_subset_id : str
        Spatial-subset identifier used in the store filename (e.g. "global").
    is_downscaled : bool
        If True, keep only fine-grid (non ``debiased_coarse``) leaves; if
        False, keep only ``debiased_coarse`` leaves.

    Returns
    -------
    tuple[dict, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(trees, tags, gcms_np, scenarios_np, variables_np, tags_np, methods_np, debiased_coarse_flags_np)``:
        `trees` maps gcm name to its opened DataTree; the remaining arrays
        are parallel, one entry per discovered leaf.
    """

    trees: dict[str, xr.DataTree] = {}
    open_errors: dict[str, str] = {}

    for gcm in gcms:
        uri = f"{root_dir}{gcm}-ERA5-{store_subset_id}.icechunk"
        try:
            repo = icechunk.Repository.open(_icechunk_storage_for_path(uri))
            session = repo.readonly_session(branch) if branch else repo.readonly_session()
            trees[gcm] = xr.open_datatree(session.store, engine="zarr", chunks=INPUT_CHUNKS)
        except Exception as exc:  # noqa: BLE001  # report every failure, do not stop at the first
            open_errors[gcm] = f"{type(exc).__name__}: {exc}"

    for gcm, err in open_errors.items():
        logger.warning("FAILED to open %s: %s", gcm, err)

    # GCMs whose store actually opened. A GCM whose run is still in flight is skipped here rather
    # than failing the whole notebook, so this check can be run against a partially-landed run.
    open_gcms = tuple(gcm for gcm in gcms if gcm in trees)
    logger.info(
        "opened %d/%d stores on branch %r: %s",
        len(open_gcms),
        len(gcms),
        branch,
        ", ".join(open_gcms),
    )

    leaf_gcms, leaf_methods, leaf_debiased_coarse_flags = [], [], []
    leaf_scenarios, leaf_variables, leaf_ensembles = [], [], []

    for gcm in open_gcms:
        tree = trees[gcm]
        for node in tree.subtree:
            if node.children:
                continue  # not a leaf
            parts = node.path.strip("/").split("/")
            if len(parts) < 3:
                continue  # malformed/incomplete path, skip defensively

            scenario, variable, ensemble = parts[-3], parts[-2], parts[-1]
            prefix = parts[:-3]  # everything above scenario/variable/ensemble

            debiased_coarse_flag = "debiased_coarse" if "debiased_coarse" in prefix else ""
            method_parts = [p for p in prefix if p != "debiased_coarse"]
            method = "/".join(
                method_parts
            )  # "" if no method wrapper (old tree, no debiased_coarse)

            leaf_gcms.append(gcm)
            leaf_methods.append(method)
            leaf_debiased_coarse_flags.append(debiased_coarse_flag)
            leaf_scenarios.append(scenario)
            leaf_variables.append(variable)
            leaf_ensembles.append(ensemble)

    logger.info("%d leaves across %d GCMs", len(leaf_gcms), len(open_gcms))

    if is_downscaled:
        keep_idx = [i for i, s in enumerate(leaf_debiased_coarse_flags) if s != "debiased_coarse"]
        leaf_gcms = [leaf_gcms[i] for i in keep_idx]
        leaf_scenarios = [leaf_scenarios[i] for i in keep_idx]
        leaf_variables = [leaf_variables[i] for i in keep_idx]
        leaf_ensembles = [leaf_ensembles[i] for i in keep_idx]
        leaf_methods = [leaf_methods[i] for i in keep_idx]
        leaf_debiased_coarse_flags = [leaf_debiased_coarse_flags[i] for i in keep_idx]

    keep_idx = [i for i, v in enumerate(leaf_variables) if v != "dtr"]
    leaf_gcms = [leaf_gcms[i] for i in keep_idx]
    leaf_scenarios = [leaf_scenarios[i] for i in keep_idx]
    leaf_variables = [leaf_variables[i] for i in keep_idx]
    leaf_ensembles = [leaf_ensembles[i] for i in keep_idx]
    leaf_methods = [leaf_methods[i] for i in keep_idx]
    leaf_debiased_coarse_flags = [leaf_debiased_coarse_flags[i] for i in keep_idx]

    keep_idx = [i for i, v in enumerate(leaf_variables) if v != "hurs"]
    leaf_gcms = [leaf_gcms[i] for i in keep_idx]
    leaf_scenarios = [leaf_scenarios[i] for i in keep_idx]
    leaf_variables = [leaf_variables[i] for i in keep_idx]
    leaf_ensembles = [leaf_ensembles[i] for i in keep_idx]
    leaf_methods = [leaf_methods[i] for i in keep_idx]
    leaf_debiased_coarse_flags = [leaf_debiased_coarse_flags[i] for i in keep_idx]

    tags = []
    for i, gcm in enumerate(leaf_gcms):
        method = leaf_methods[i]
        var = leaf_variables[i]
        scenario = leaf_scenarios[i]
        ens = leaf_ensembles[i]
        # This if statement is to accommodate different data tree structures.
        # Data generated when we only had one method does not have a method tier of the data tree
        if method == "":
            tags.append(f"{gcm}_{var}_{scenario}_{ens}_no-method-specified")
        else:
            tags.append(f"{gcm}_{var}_{scenario}_{ens}_{method}")

    gcms_np = np.array(leaf_gcms)
    scenarios_np = np.array(leaf_scenarios)
    variables_np = np.array(leaf_variables)
    tags_np = np.array(tags)
    methods_np = np.array(leaf_methods)
    debiased_coarse_flags_np = np.array(leaf_debiased_coarse_flags)

    return (
        trees,
        tags,
        gcms_np,
        scenarios_np,
        variables_np,
        tags_np,
        methods_np,
        debiased_coarse_flags_np,
    )


# Using old model names temporarily
GRID_TYPES = ("downscaled", "CESM2-WACCM6", "UKESM1-1-LL")


def load_rsds_lims(grid_type: str, key: str = "zonal_doy_max_rsds") -> xr.DataArray:
    """
    Load the zonal/day-of-year maximum rsds values (from step 1 notebook) for use in the rsds-specific flag.

    Parameters
    ----------
    grid_type : str
        Which grid's precomputed limits to load; one of GRID_TYPES.
    key : str
        Group name within the zarr store to load.

    Returns
    -------
    xr.DataArray
        Maximum plausible rsds by latitude and day of year.

    Raises
    ------
    ValueError
        If `grid_type` is not one of GRID_TYPES.
    """
    if grid_type not in GRID_TYPES:
        raise ValueError(
            f"unsupported grid_type: {grid_type!r}. Valid values include: {GRID_TYPES}"
        )
    elif grid_type == "downscaled":
        fpath = DIR_QA_FLAG_CONSTANT_INPUTS + "zonal_doy_max_rsds.zarr"
    else:
        # step 1 wrote these files under the catalog's GCM name, not the pipeline's
        catalog_name = GCM_CATALOG_NAME_OVERRIDES.get(grid_type, grid_type)
        fpath = DIR_QA_FLAG_CONSTANT_INPUTS + "zonal_doy_max_rsds_" + catalog_name + ".zarr"
    return xr.open_zarr(fpath, group=key)["data"].load()


def prep_annual_threshold_inputs(grid_type: str) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Read the step-1 observational threshold outputs and reduce them to annual bounds.

    Parameters
    ----------
    grid_type : str
        Which grid's precomputed thresholds to load; one of GRID_TYPES.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        ``(outlier_thresh_low_annual, outlier_thresh_high_annual)``, one
        value per variable per pixel, for use by flag_outliers with
        ``timescale="annual"``.

    Raises
    ------
    ValueError
        If `grid_type` is not one of GRID_TYPES.
    """
    # Loading thresholds output from step 1 notebook. There are different stores for different grids
    if grid_type not in GRID_TYPES:
        raise ValueError(
            f"unsupported grid_type: {grid_type!r}. Valid values include: {GRID_TYPES}"
        )
    elif grid_type == "downscaled":
        store = DIR_QA_FLAG_CONSTANT_INPUTS + "doy_obs_thresholds_global.zarr"
    else:
        # step 1 wrote these files under the catalog's GCM name, not the pipeline's
        catalog_name = GCM_CATALOG_NAME_OVERRIDES.get(grid_type, grid_type)
        store = DIR_QA_FLAG_CONSTANT_INPUTS + "doy_obs_thresholds_global_" + catalog_name + ".zarr"

    combined = xr.open_zarr(store)

    def _split(ds, suffix):
        names = [v for v in ds.data_vars if v.endswith(suffix)]
        return ds[names].rename({v: v[: -len(suffix)] for v in names})

    obs_max = _split(combined, "_max")
    obs_min = _split(combined, "_min")
    obs_max_std = _split(combined, "_max_std")
    obs_min_std = _split(combined, "_min_std")

    [outlier_thresh_low, outlier_thresh_high] = calculate_thresholds(
        obs_max, obs_min, obs_max_std, obs_min_std
    )

    outlier_thresh_low_annual = outlier_thresh_low.min(dim="dayofyear")
    outlier_thresh_high_annual = outlier_thresh_high.max(dim="dayofyear")

    # Match the lat/lon chunk grid flag_outliers' `da` is opened with (INPUT_CHUNKS), so
    # comparing against it doesn't force dask to reconcile two different chunk grids --
    # without this, every tag in the annual-outlier flag loop hits a
    # "PerformanceWarning: Increasing number of chunks" from the mismatch.
    outlier_thresh_low_annual = outlier_thresh_low_annual.chunk(
        {"lat": SHARD_LAT, "lon": SHARD_LON}
    )
    outlier_thresh_high_annual = outlier_thresh_high_annual.chunk(
        {"lat": SHARD_LAT, "lon": SHARD_LON}
    )

    return outlier_thresh_low_annual, outlier_thresh_high_annual


def calculate_all_flags(
    variables: list[str],
    gcms: list[str],
    methods: list[str],
    bucket: str,
    prefix: str,
    trees: dict[str, xr.DataTree],
    tags: list[str],
    gcms_np: np.ndarray,
    scenarios_np: np.ndarray,
    variables_np: np.ndarray,
    tags_np: np.ndarray,
    methods_np: np.ndarray,
    grid_type: str,
    scenario_comparisons: dict = SCENARIO_COMPARISONS,
    plot_flag_maps: bool = True,
    save_plots: bool = False,
    verbose: bool = True,
) -> None:
    """
    Run every intermediate QA-flag check -- time-varying and time-invariant --
    over one already-discovered set of leaves, and write the results.

    This is the shared body of run_step2, factored out so it can be called
    once per grid via `grid_type` (see GRID_TYPES) -- e.g. once for the fine
    downscaled output and again for a coarse ``debiased_coarse`` GCM grid --
    since the observational thresholds/rsds limits loaded here
    (prep_annual_threshold_inputs, load_rsds_lims) differ by grid.

    Parameters
    ----------
    variables, gcms, methods : list[str]
        Values to loop over for the trend-distortion/sign-flip checks.
    bucket, prefix : str
        Passed through to write_individual_flags.
    trees, tags, gcms_np, scenarios_np, variables_np, tags_np, methods_np
        Exactly the outputs of discover_leaves for the grid being processed.
    grid_type : str
        Which grid's observational thresholds/rsds limits to load (one of
        GRID_TYPES); should match the grid `trees`/`tags_np`/etc. were
        discovered from.
    scenario_comparisons : dict
        See SCENARIO_COMPARISONS.
    plot_flag_maps : bool
        If True, plot each computed flag.
    save_plots : bool
        If True (and plot_flag_maps is also True), also upload each plotted
        figure to S3 under ``{bucket}/{prefix}/_plots/``.
    verbose : bool
        If True, print progress and timing for each of the five flag loops.
    """

    if verbose:
        t_start = time.time()

    if grid_type == "downscaled":
        is_downscaled = True
    else:
        is_downscaled = False

    ########### Run time-varying flag loops that are the same for all grids ############################################
    # Flag 1. Global exceedances
    if verbose:
        logger.info("Running flag loop 1/5: global exceedance flag...")
        t0 = time.time()
    run_flag_loop(
        tags=tags,
        trees=trees,
        bucket=bucket,
        prefix=prefix,
        flag_name="outside_global_plausible_range",
        compute_flag=lambda da, var: flag_global_exceedances(da=da, var=var),
        write_mode="a",
        is_downscaled=is_downscaled,
        plot=plot_flag_maps,
        save_plots=save_plots,
    )
    if verbose:
        logger.info("  flag loop 1/5 completed in %.1fs", time.time() - t0)

    # Flag 2. Temperature inconsistencies (tas vs. tasmin/tasmax)
    if verbose:
        logger.info("Running flag loop 2/5: temperature inconsistency flag...")
        t0 = time.time()
    run_flag_loop_temperature_inconsistencies(
        tags,
        trees,
        flag_name="temperature_inconsistency",
        bucket=bucket,
        prefix=prefix,
        is_downscaled=is_downscaled,
        plot=plot_flag_maps,
        save_plots=save_plots,
    )
    if verbose:
        logger.info("  flag loop 2/5 completed in %.1fs", time.time() - t0)

    ########### Run time-varying flag loops that use different pre-computed inputs for different grids ###################
    # Flag 3. Outliers based on observations
    if verbose:
        logger.info("Running flag loop 3/5: annual outlier flag...")
        t0 = time.time()
    [outlier_thresh_low_annual, outlier_thresh_high_annual] = prep_annual_threshold_inputs(
        grid_type=grid_type
    )
    run_flag_loop(
        tags=tags,
        trees=trees,
        flag_name="annual_outlier_flag",
        compute_flag=lambda da, var: flag_outliers(
            da=da,
            outlier_thresh_low=outlier_thresh_low_annual[var],
            outlier_thresh_high=outlier_thresh_high_annual[var],
            timescale="annual",
        ),
        bucket=bucket,
        prefix=prefix,
        write_mode="a",
        is_downscaled=is_downscaled,
        plot=plot_flag_maps,
        save_plots=save_plots,
    )
    if verbose:
        logger.info("  flag loop 3/5 completed in %.1fs", time.time() - t0)

    # Flag 4. rsds-specific latitude/day-of-year check
    if verbose:
        logger.info("Running flag loop 4/5: rsds max exceedance flag...")
        t0 = time.time()
    zonal_doy_max_rsds = load_rsds_lims(grid_type=grid_type)
    run_flag_loop(
        tags=tags,
        trees=trees,
        flag_name="rsds_max_exceeded",
        compute_flag=lambda da, var: flag_rsds_above_max(
            da=da, zonal_doy_max_rsds=zonal_doy_max_rsds
        ),
        var_filter=["rsds"],
        bucket=bucket,
        prefix=prefix,
        write_mode="a",
        is_downscaled=is_downscaled,
        plot=plot_flag_maps,
        save_plots=save_plots,
    )
    if verbose:
        logger.info("  flag loop 4/5 completed in %.1fs", time.time() - t0)

    ########### Run time-invariant flag loops ##################################################
    if verbose:
        logger.info("Running flag loop 5/5: trend distortion flag...")
        t0 = time.time()
    calculate_trend_distortion_flags(
        trees=trees,
        gcms=gcms,
        variables=variables,
        methods=methods,
        tags_np=tags_np,
        gcms_np=gcms_np,
        scenarios_np=scenarios_np,
        variables_np=variables_np,
        methods_np=methods_np,
        bucket=bucket,
        prefix=prefix,
        scenario_comparisons=scenario_comparisons,
        is_downscaled=is_downscaled,
        plot=plot_flag_maps,
        save_plots=save_plots,
    )
    if verbose:
        logger.info("  flag loop 5/5 completed in %.1fs", time.time() - t0)
        logger.info("calculate_all_flags total time: %.1fs", time.time() - t_start)


def run_step2(
    variables: list,
    gcms: list,
    methods: list,
    branch: str,
    root_dir: str,
    store_subset_id: str,
    bucket: str,
    prefix: str,
    plot_flag_maps: bool = True,
    save_plots: bool = False,
    verbose: bool = True,
    mode: str = "downscaled_only",
) -> None:
    """
    Run through all the intermediate flag calculations and write them to icechunk stores on scratch.

    Goes through all the steps in the step2 QA-flag notebook, but is designed
    to run as a single script call rather than interactively. The
    intermediate icechunk stores this writes are later read by
    combine_intermediate_flags / write_final_qa_flags to produce the final
    QA flags on the production store.

    Parameters
    ----------
    variables, gcms, methods : list[str]
        Values to discover leaves for and compute flags over.
    branch : str
        icechunk branch to read.
    root_dir : str
        Directory containing each GCM's icechunk store.
    store_subset_id : str
        Spatial-subset identifier used in the store filename (e.g. "global").
    bucket, prefix : str
        S3 location to write intermediate flags to.
    plot_flag_maps : bool
        If True, plot each computed flag.
    save_plots : bool
        If True (and plot_flag_maps is also True), also upload each plotted
        figure to S3 under ``{bucket}/{prefix}/_plots/`` (and the
        ``debiased_coarse`` variant for the coarse-grid branch).
    verbose : bool
        If True, print progress and timing for leaf discovery and each grid's
        calculate_all_flags call.
    mode: str (allowed values: "downscaled_only", "debiased_coarse_only", "both")
    """
    valid_modes = ("downscaled_only", "debiased_coarse_only", "both")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

    ########### Get the leaves of the data tree to traverse and the tags for each leaf ##########
    if verbose:
        t_start = time.time()
        logger.info("Discovering leaves of the data tree...")
        t0 = time.time()
    [
        trees,
        tags,
        gcms_np,
        scenarios_np,
        variables_np,
        tags_np,
        methods_np,
        debiased_coarse_flags_np,
    ] = discover_leaves(
        gcms=gcms,
        branch=branch,
        root_dir=root_dir,
        store_subset_id=store_subset_id,
        is_downscaled=False,
    )
    if verbose:
        logger.info("  leaf discovery completed in %.1fs", time.time() - t0)

    if mode in ["downscaled_only", "both"]:
        keep_idx = [i for i, s in enumerate(debiased_coarse_flags_np) if s != "debiased_coarse"]
        tags_np_downscaled = tags_np[keep_idx]
        gcms_np_downscaled = gcms_np[keep_idx]
        scenarios_np_downscaled = scenarios_np[keep_idx]
        variables_np_downscaled = variables_np[keep_idx]
        tags_np_downscaled = tags_np[keep_idx]
        methods_np_downscaled = methods_np[keep_idx]

        # Calculate the flags for the downscaled output
        if verbose:
            logger.info("Calculating flags on downscaled data...")
            t0 = time.time()
        calculate_all_flags(
            variables=variables,
            gcms=gcms,
            methods=methods,
            bucket=bucket,
            prefix=prefix,
            trees=trees,
            tags=tags_np_downscaled,
            gcms_np=gcms_np_downscaled,
            scenarios_np=scenarios_np_downscaled,
            variables_np=variables_np_downscaled,
            tags_np=tags_np_downscaled,
            methods_np=methods_np_downscaled,
            grid_type="downscaled",
            verbose=verbose,
            plot_flag_maps=plot_flag_maps,
            save_plots=save_plots,
        )
        if verbose:
            logger.info("  flags on downscaled data completed in %.1fs", time.time() - t0)

    if mode in ["debiased_coarse_only", "both"]:
        # Calculate the flags for the coarse debiased output
        if verbose:
            logger.info("Calculating flags on coarse debiased data...")
            t0 = time.time()
        for gcm in gcms:
            mask = (debiased_coarse_flags_np == "debiased_coarse") & (gcms_np == gcm)
            gcms_np_coarse = gcms_np[mask]
            scenarios_np_coarse = scenarios_np[mask]
            variables_np_coarse = variables_np[mask]
            tags_np_coarse = tags_np[mask]
            methods_np_coarse = methods_np[mask]

            if verbose:
                t_gcm = time.time()
            calculate_all_flags(
                variables=variables,
                gcms=[gcm],
                methods=methods,
                bucket=bucket,
                prefix=prefix + "/debiased_coarse",
                trees=trees,
                tags=tags_np_coarse,
                gcms_np=gcms_np_coarse,
                scenarios_np=scenarios_np_coarse,
                variables_np=variables_np_coarse,
                tags_np=tags_np_coarse,
                methods_np=methods_np_coarse,
                grid_type=gcm,
                verbose=verbose,
                plot_flag_maps=plot_flag_maps,
                save_plots=save_plots,
            )
            if verbose:
                logger.info(
                    "  flags on coarse debiased data for %s completed in %.1fs",
                    gcm,
                    time.time() - t_gcm,
                )
        if verbose:
            logger.info(
                "  flags on coarse debiased data (all gcms) completed in %.1fs",
                time.time() - t0,
            )

    if verbose:
        logger.info("run_step2 total time: %.1fs", time.time() - t_start)


def run_step3(
    gcms: list,
    branch: str,
    root_dir: str,
    store_subset_id: str,
    bucket: str,
    prefix: str,
    overwrite: bool = False,
    verbose: bool = True,
    mode: str = "both",
) -> None:
    """
    Combine each tag's intermediate flags and write the two final flags to the production store.

    Goes through all the steps in the step3 QA-flag notebook
    (flag_step3_combine_and_write_to_source_coop.ipynb), but is designed to
    run as a single script call rather than interactively, and covers both
    the downscaled and debiased_coarse intermediate flags written by
    run_step2 (e.g. with mode="both") -- not just downscaled, as the notebook
    currently does.

    combine_intermediate_flags itself needs no changes to support this: it
    already reads from whatever `prefix` it's given, so this function just
    calls it once per tag with `prefix` for the downscaled grid and
    `f"{prefix}/debiased_coarse"` for the coarse grid, matching where
    run_step2 wrote each grid's intermediate flags.

    Parameters
    ----------
    gcms : list[str]
        GCMs to discover leaves for and write final flags for.
    branch : str
        icechunk branch to read intermediate flags from and write final
        flags to.
    root_dir : str
        Directory containing each GCM's production icechunk store.
    store_subset_id : str
        Spatial-subset identifier used in the store filename (e.g. "global").
    bucket, prefix : str
        S3 location of the intermediate flags written by run_step2. The
        debiased_coarse intermediate flags are read from
        ``{prefix}/debiased_coarse``, not `prefix` itself.
    overwrite : bool
        Passed through to write_final_qa_flags; if False (default), refuses
        to replace a flag that has already been written for a given
        tag/group.
    verbose : bool
        If True, print progress and timing for leaf discovery and each
        grid's write loop.
    mode : str (allowed values: "downscaled_only", "debiased_coarse_only", "both")

    Notes
    -----
    A tag with no intermediate-flags store at a given grid's prefix (e.g. a
    gcm/var/scenario/ens/method combination that was only run at one
    resolution) is skipped with a logged warning rather than raising.
    """
    valid_modes = ("downscaled_only", "debiased_coarse_only", "both")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

    flag_time_varying_name = ATTRS_TIME_VARYING["short_name"]
    flag_time_invariant_name = ATTRS_TIME_INVARIANT["short_name"]

    ########### Get the leaves of the data tree to traverse and the tags for each leaf ##########
    if verbose:
        t_start = time.time()
        logger.info("Discovering leaves of the data tree...")
        t0 = time.time()
    [_, _, _, _, _, tags_np, _, debiased_coarse_flags_np] = discover_leaves(
        gcms=gcms,
        branch=branch,
        root_dir=root_dir,
        store_subset_id=store_subset_id,
        is_downscaled=False,
    )
    if verbose:
        logger.info("  leaf discovery completed in %.1fs", time.time() - t0)

    # One writable repo per gcm's production store, opened once and reused across all its tags.
    repos: dict[str, icechunk.Repository] = {}
    for gcm in gcms:
        uri = f"{root_dir}{gcm}-ERA5-{store_subset_id}.icechunk"
        repos[gcm] = icechunk.Repository.open(_icechunk_storage_for_path(uri))

    if mode in ["downscaled_only", "both"]:
        keep_idx = [i for i, s in enumerate(debiased_coarse_flags_np) if s != "debiased_coarse"]
        tags_np_downscaled = tags_np[keep_idx]

        if verbose:
            logger.info("Writing final flags for downscaled data...")
            logger.info("%d tags to process", len(tags_np_downscaled))
            t0 = time.time()

        for tag in tags_np_downscaled:
            [gcm, var, scenario, ens, method] = parse_tag(tag)
            logger.info(tag)
            try:
                [overall_flag_time_varying, overall_flag_time_invariant] = (
                    combine_intermediate_flags(
                        tag=tag,
                        flag_list_time_varying=FLAG_LIST_TIME_VARYING,
                        flag_list_time_invariant=FLAG_LIST_TIME_INVARIANT,
                        bucket=bucket,
                        prefix=prefix,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- a tag with no intermediate-flags store at
                # this prefix (e.g. never ran for this gcm/var combo) shouldn't stop the rest of
                # step 3
                logger.warning("Skipping %s: no intermediate flags at %s: %s", tag, prefix, exc)
                continue

            group = f"{method}/{scenario}/{var}/{ens}"
            session = repos[gcm].writable_session(branch)

            write_final_qa_flags(
                session=session,
                group=group,
                flag_data=overall_flag_time_varying,
                flag_name=flag_time_varying_name,
                attrs=ATTRS_TIME_VARYING,
                overwrite=overwrite,
            )
            write_final_qa_flags(
                session=session,
                group=group,
                flag_data=overall_flag_time_invariant,
                flag_name=flag_time_invariant_name,
                attrs=ATTRS_TIME_INVARIANT,
                overwrite=overwrite,
            )
            commit = session.commit(f"write qa flags for {tag}")
            logger.info("  commit %s", commit)

        if verbose:
            logger.info("  final flags for downscaled data completed in %.1fs", time.time() - t0)

    if mode in ["debiased_coarse_only", "both"]:
        keep_idx = [i for i, s in enumerate(debiased_coarse_flags_np) if s == "debiased_coarse"]
        tags_np_coarse = tags_np[keep_idx]
        coarse_prefix = f"{prefix}/debiased_coarse"

        if verbose:
            logger.info("Writing final flags for coarse debiased data...")
            logger.info("%d tags to process", len(tags_np_coarse))
            t0 = time.time()

        for tag in tags_np_coarse:
            [gcm, var, scenario, ens, method] = parse_tag(tag)
            logger.info(tag)
            try:
                [overall_flag_time_varying, overall_flag_time_invariant] = (
                    combine_intermediate_flags(
                        tag=tag,
                        flag_list_time_varying=FLAG_LIST_TIME_VARYING,
                        flag_list_time_invariant=FLAG_LIST_TIME_INVARIANT,
                        bucket=bucket,
                        prefix=coarse_prefix,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- a tag with no intermediate-flags store at
                # this prefix (e.g. never ran at coarse resolution for this gcm/var combo)
                # shouldn't stop the rest of step 3
                logger.warning(
                    "Skipping %s: no intermediate flags at %s: %s", tag, coarse_prefix, exc
                )
                continue

            # The production tree nests debiased_coarse output between method and scenario
            # (see e.g. srm.cache.debiased_coarse_scenario_loc), so the write group mirrors
            # that layout, not the downscaled group above.
            group = f"{method}/debiased_coarse/{scenario}/{var}/{ens}"
            session = repos[gcm].writable_session(branch)

            write_final_qa_flags(
                session=session,
                group=group,
                flag_data=overall_flag_time_varying,
                flag_name=flag_time_varying_name,
                attrs=ATTRS_TIME_VARYING,
                overwrite=overwrite,
            )
            write_final_qa_flags(
                session=session,
                group=group,
                flag_data=overall_flag_time_invariant,
                flag_name=flag_time_invariant_name,
                attrs=ATTRS_TIME_INVARIANT,
                overwrite=overwrite,
            )
            commit = session.commit(f"write qa flags for {tag}")
            logger.info("  commit %s", commit)

        if verbose:
            logger.info(
                "  final flags for coarse debiased data completed in %.1fs", time.time() - t0
            )

    if verbose:
        logger.info("run_step3 total time: %.1fs", time.time() - t_start)
