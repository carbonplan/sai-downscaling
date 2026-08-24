import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from srm.qaqc import VAR_SPATIAL_RANGES

# directory where outputs from step 1 are saved for use in calculating flags in step 2
DIR_QA_FLAG_CONSTANT_INPUTS = "s3://carbonplan-srm/output/qa_flag_inputs/"
# DIR_QA_FLAG_OUTPUTS = "s3://carbonplan-scratch/srm/qaqc/" #previous location for step 1, some outputs still there until rerunning step 1

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

FLAG_LIST_TIME_VARYING = [
    "annual_outlier_flag",
    "rsds_max_exceeded",
    "outside_global_plausible_range",
    "temperature_inconsistency",
]

FLAG_LIST_TIME_INVARIANT = [
    "flipped_signssp245_g6_1p5k",
    "flipped_signhistorical_ssp245",
    "flipped_signhistorical_g6_1p5k",
    "trend_distortion_historical_ssp245",
    "trend_distortion_ssp245_g6_1p5k",
    "trend_distortion_historical_g6_1p5k",
    "flipped_signg6_1p5k_g6_1p5k_end",
    "trend_distortion_g6_1p5k_g6_1p5k_end",
]

ATTRS_TIME_INVARIANT = {
    "long_name": "Quality flag (time-invariant)",
    "description": "This quality flag flags specific locations where debiasing/downscaling meaningfully changes how scenarios compare to each other in the annual mean, compared to the raw GCM input.",
    "possible_values": "This is a binary flag: 0=no known issue; 1=known issue",
    "short_name": "qa_flag_time_invariant",
}

ATTRS_TIME_VARYING = {
    "long_name": "Quality flag (time-varying)",
    "description": "This quality flag flags specific days where results are highly sensitive to debiasing/downscaling method choice (e.g. treatment of outliers) and/or where output is physically unrealistic (e.g. tasmax < tas) or unlikely.",
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
    gcm: str,
    var: str,
    scenario: str,
    ens: str,
    flag_dir: str,
    write_mode: str = "w",
    time_varying: bool = True,
):
    flag_data = flag_data.rename(flag_name)
    flag_data = flag_data.astype(np.uint8)
    flag_data.attrs = {
        "long_name": "Quality flag",
        "description": "0=no known issue; 1=known issue",
        "short_name": flag_name,
    }

    if time_varying:
        flag_data = flag_data.chunk({"lat": 100, "lon": 100, "time": 8000})
    else:
        flag_data = flag_data.chunk({"lat": 100, "lon": 100})

    tag = f"{gcm}_{var}_{scenario}_{ens}"
    store_path = flag_dir + tag + ".zarr"

    # encoding is only valid the first time flag_name is written to this store;
    # xarray errors if encoding is passed for a variable that already exists there
    variable_exists = False
    if write_mode != "w":
        try:
            variable_exists = flag_name in xr.open_zarr(store_path, consolidated=False).variables
        except Exception:
            variable_exists = False  # store or group doesn't exist yet

    encoding = {} if variable_exists else {flag_name: {"_FillValue": None}}

    flag_data.to_zarr(
        store_path,
        mode=write_mode,
        consolidated=False,
        align_chunks=True,
        encoding=encoding,
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

        if contains_flags:
            plt.figure(figsize=(5, 3))
            ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            count_flag.where(count_flag > 0).plot(ax=ax1, transform=ccrs.PlateCarree())
            ax1.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.4")


def parse_tag(tag):
    parts = tag.split("_")
    gcm, var, ens = parts[0], parts[1], parts[-1]
    scenario = "_".join(parts[2:-1])
    return gcm, var, scenario, ens


def get_data(tag, trees):
    [gcm, var, scenario, ens] = parse_tag(tag)
    comparison_ds = trees[gcm][scenario][var][ens]
    da = comparison_ds[var]

    return da


def run_flag_loop(
    tags,
    trees,
    flag_name,
    compute_flag,
    flag_dir,
    var_filter=None,
    write_mode: str = "a",
    plot: bool = True,
):
    """Loop over tags, compute one flag per leaf, write it, optionally plot it.

    compute_flag(da, var) -> flag DataArray. Bind whatever extra fixed arguments a
    specific check needs (e.g. zonal_doy_max_rsds) with a lambda at the call site --
    this loop doesn't need to know what they are.
    """
    print(len(tags))
    for tag in tags:
        gcm, var, scenario, ens = parse_tag(tag)
        if var_filter is not None and var != var_filter:
            continue
        print(tag)

        da = get_data(tag=tag, trees=trees)
        flag_data = compute_flag(da, var)

        write_individual_flags(
            flag_data=flag_data,
            flag_name=flag_name,
            gcm=gcm,
            var=var,
            scenario=scenario,
            ens=ens,
            write_mode=write_mode,
            flag_dir=flag_dir,
        )

        if plot:
            plot_flags(flags=flag_data, time_varying=True, separate_low_high=False)
            plt.show()
            plt.close()


def get_intermediate_flags(flag_dir: str, tag: str):
    flag_data = xr.open_zarr(flag_dir + tag + ".zarr", consolidated=False)
    return flag_data


def combine_intermediate_flags(
    tag: str,
    flag_dir: str,
    flag_list_time_varying: list,
    flag_list_time_invariant: list,
):
    """
    This calculates two single binary flags from multiple intermediate flags. The intermediate flags

    """

    flags = get_intermediate_flags(flag_dir=flag_dir, tag=tag)

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


def write_final_qa_flags(
    dataset_path: str,
    flag_data: xr.DataArray,
    flag_name: str,
    attrs: dict,
    write_mode: str = "a",
    time_varying: bool = True,
):
    flag_data = flag_data.rename(flag_name)
    flag_data = flag_data.fillna(0).astype(np.uint8)
    flag_data.attrs = attrs

    if time_varying:
        flag_data = flag_data.chunk({"lat": 100, "lon": 100, "time": 8000})
    else:
        flag_data = flag_data.chunk({"lat": 100, "lon": 100})

    # encoding is only valid the first time flag_name is written to this store;
    # xarray errors if encoding is passed for a variable that already exists there
    variable_exists = False
    if write_mode != "w":
        try:
            variable_exists = flag_name in xr.open_zarr(dataset_path, consolidated=False).variables
        except Exception:
            variable_exists = False  # store or group doesn't exist yet

    encoding = {} if variable_exists else {flag_name: {"_FillValue": None}}

    # TO DO: change this to write to icechunk
    flag_data.to_zarr(
        dataset_path,
        mode=write_mode,
        consolidated=False,
        align_chunks=True,
        # zarr defaults to skipping the on-disk write for all-fill-value chunks, which
        # leaves stale flagged data from a prior run in place when a chunk is
        # recomputed as all-unflagged; force every chunk to be written so overwrites
        # are complete.
        write_empty_chunks=True,
        encoding=encoding,
    )
