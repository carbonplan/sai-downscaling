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
from srm.downscaling_utils import interpolate_fine_to_coarse_grid
from srm.encoding import (
    CHUNK_LAT,
    CHUNK_LON,
    CHUNK_TIME,
    COMPRESSOR,
    SHARD_LAT,
    SHARD_LON,
    SHARD_TIME,
)
from srm.qaqc import VAR_SPATIAL_RANGES

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


def calculate_thresholds(obs_max, obs_min, obs_max_std, obs_min_std):
    outlier_thresh_high = obs_max + (5 * obs_max_std)
    outlier_thresh_low = obs_min - (5 * obs_min_std)

    return outlier_thresh_low, outlier_thresh_high


def flag_outliers(da, outlier_thresh_low, outlier_thresh_high, timescale: str = "annual"):
    """
    Flags outliers based on the observational record
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


def flag_rsds_above_max(da, zonal_doy_max_rsds):
    """
    Flags days when rsds is above expected max (based on latitude and dayofyear)
    """
    aligned_max = zonal_doy_max_rsds.sel(dayofyear=da.time.dt.dayofyear)
    exceeds_max = da > aligned_max

    return exceeds_max


def flag_global_exceedances(da, var: str, var_ranges: dict = VAR_SPATIAL_RANGES):
    """
    Flags days when a variable is outside the globally-defined range of what is plausible.
    """

    var_range = var_ranges[var]
    var_min = var_range["min"][0]
    var_max = var_range["max"][1]

    too_high = da > var_max
    too_low = da < var_min

    outside_range = (too_low | too_high) > 0

    return outside_range


def flag_tasmax_tas_inconsistency(tas, tasmax):
    inconsistent_days = tasmax < tas
    return inconsistent_days


def flag_tasmin_tas_inconsistency(tas, tasmin):
    inconsistent_days = tasmin > tas
    return inconsistent_days


def write_individual_flags(
    flag_data: xr.DataArray,
    flag_name: str,
    tag: str,
    bucket: str = "carbonplan-srm",
    prefix: str = "output/qa-intermediate-flags",
    write_mode: str = "w",
    branch: str = "main",
):
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


def plot_flags(flags, time_varying: bool = True, separate_low_high=True):
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
                count_low_flag.where(count_low_flag > 0).plot(ax=ax1, transform=ccrs.PlateCarree())
                ax1.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.4")
            ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
            if contains_high_flags:
                count_high_flag.where(count_high_flag > 0).plot(
                    ax=ax2, transform=ccrs.PlateCarree()
                )
                ax2.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.4")
            plt.tight_layout()
            plt.show()
    else:
        flag = flags
        if time_varying:
            count_flag = flag.sum(dim="time").load()
        else:
            count_flag = flag

        contains_flags = np.nansum(count_flag)

        # draw the map even when nothing is flagged, so a clean leaf is visibly clean
        # rather than indistinguishable from a leaf that was skipped or errored
        plt.figure(figsize=(5, 3))
        ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        limits = {} if contains_flags else {"vmin": 0, "vmax": 1}
        count_flag.where(count_flag > 0).plot(ax=ax1, transform=ccrs.PlateCarree(), **limits)
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.4")
        unit = "cell-days" if time_varying else "cells"
        ax1.set_title(f"{count_flag.name}: {float(contains_flags):,.0f} flagged {unit}")


def parse_tag(tag):
    parts = tag.split("_")
    gcm, var, ens, method = parts[0], parts[1], parts[-2], parts[-1]
    scenario = "_".join(parts[2:-2])
    return gcm, var, scenario, ens, method


def get_data(tag, trees, var_to_analyze=None):
    [gcm, var, scenario, ens, method] = parse_tag(tag)
    if method == "no-method-specified":
        group_path = f"{scenario}/{var}/{ens}"
    else:
        group_path = f"{method}/{scenario}/{var}/{ens}"
    comparison_ds = trees[gcm][group_path]
    if var_to_analyze is None:
        var_to_analyze = var
    da = comparison_ds[var_to_analyze]

    return da


def run_flag_loop(
    tags,
    trees,
    flag_name,
    compute_flag,
    var_filter=None,
    write_mode: str = "a",
    plot: bool = True,
    bucket: str = "carbonplan-srm",
    prefix: str = "output/qa-intermediate-flags",
):
    """Loop over tags, compute one flag per leaf, write it, optionally plot it.

    compute_flag(da, var) -> flag DataArray. Bind whatever extra fixed arguments a
    specific check needs (e.g. zonal_doy_max_rsds) with a lambda at the call site --
    this loop doesn't need to know what they are.
    """
    print(len(tags))
    for tag in tags:
        [gcm, var, scenario, ens, method] = parse_tag(tag)
        if var_filter is not None and var != var_filter:
            continue
        print(tag)

        da = get_data(tag=tag, trees=trees)
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
            plot_flags(flags=flag_data, time_varying=True, separate_low_high=False)
            plt.show()
            plt.close()


def calculate_ensemble_mean_deltas(
    variable,
    tags_scenario1,
    tags_scenario2,
    gcm,
    scenario1,
    scenario2,
    scenario1_time_slice,
    scenario2_time_slice,
    trees,
):
    # Construct data arrays including all relevant ensemble members
    das_scenario1 = []
    ens_scenario1 = []
    for tag in tags_scenario1:
        da = get_data(tag=tag, trees=trees)
        [thisgcm, thisvar, thisscenario, thisens, thismethod] = parse_tag(tag)
        das_scenario1.append(da)
        ens_scenario1.append(thisens)

    var_scenario1 = xr.concat(das_scenario1, dim=pd.Index(ens_scenario1, name="ensemble_member"))

    das_scenario2 = []
    ens_scenario2 = []
    for tag in tags_scenario2:
        da = get_data(tag=tag, trees=trees)
        [thisgcm, thisvar, thisscenario, thisens, thismethod] = parse_tag(tag)
        das_scenario2.append(da)
        ens_scenario2.append(thisens)

    var_scenario2 = xr.concat(das_scenario2, dim=pd.Index(ens_scenario2, name="ensemble_member"))

    # Take ensemble mean across comparison time periods
    ens_mean_var_scenario1 = var_scenario1.sel(time=scenario1_time_slice).mean(
        dim=["time", "ensemble_member"]
    )
    ens_mean_var_scenario2 = var_scenario2.sel(time=scenario2_time_slice).mean(
        dim=["time", "ensemble_member"]
    )

    # Take ensemble mean across comparison time periods in raw data
    raw_ds = catalog.get(gcm).to_xarray()
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
    ens_mean_var_scenario1_coarsened = interpolate_fine_to_coarse_grid(
        da_fine_to_coarsen=ens_mean_var_scenario1, da_coarse_grid=raw_scenario1_mean
    )
    ens_mean_var_scenario2_coarsened = interpolate_fine_to_coarse_grid(
        da_fine_to_coarsen=ens_mean_var_scenario2, da_coarse_grid=raw_scenario1_mean
    )

    # Calculate scenario comparison (annual mean) in downscaled and raw
    delta_raw = raw_scenario2_mean - raw_scenario1_mean
    delta_raw_pct = delta_raw * 100 / raw_scenario1_mean

    delta_ds_coarse = ens_mean_var_scenario2_coarsened - ens_mean_var_scenario1_coarsened
    delta_ds_coarse_pct = delta_ds_coarse * 100 / ens_mean_var_scenario1_coarsened

    delta_ds = ens_mean_var_scenario2 - ens_mean_var_scenario1

    return delta_raw, delta_raw_pct, delta_ds_coarse, delta_ds_coarse_pct, delta_ds


def get_intermediate_flags(
    tag: str,
    bucket: str = "carbonplan-srm",
    prefix: str = "output/qa-intermediate-flags",
    branch: str = "main",
):
    storage = icechunk.s3_storage(bucket=bucket, prefix=f"{prefix}/{tag}.icechunk", from_env=True)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch)
    flag_data = xr.open_zarr(session.store, consolidated=False)
    return flag_data


def combine_intermediate_flags(
    tag: str,
    flag_list_time_varying: list,
    flag_list_time_invariant: list,
    bucket: str = "carbonplan-srm",
    prefix: str = "output/qa-intermediate-flags",
):
    """
    This calculates two single binary flags from multiple intermediate flags. The intermediate flags

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
    """Chunk/shard shape for ``flag_data``, copied from the group's data variable."""
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

    Note: this doesn't commit the icechunk session, that is up to you.
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


def discover_leaves(gcms: list[str], branch: str, root_dir: str, store_subset_id: str):
    """Open each GCM's icechunk store and enumerate its (scenario, variable, ensemble) leaves.

    Returns (trees, tags, gcms_np, scenarios_np, variables_np, tags_np).
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
        print(f"FAILED to open {gcm}: {err}")

    # GCMs whose store actually opened. A GCM whose run is still in flight is skipped here rather
    # than failing the whole notebook, so this check can be run against a partially-landed run.
    open_gcms = tuple(gcm for gcm in gcms if gcm in trees)
    print(
        f"opened {len(open_gcms)}/{len(gcms)} stores on branch {branch!r}: {', '.join(open_gcms)}"
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

    print(f"{len(leaf_gcms)} leaves across {len(open_gcms)} GCMs")

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
