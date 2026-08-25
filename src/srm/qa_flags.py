import cartopy.crs as ccrs
import cartopy.feature as cfeature
import icechunk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from icechunk.xarray import to_icechunk

from srm import catalog
from srm.downscaling_utils import interpolate_fine_to_coarse_grid
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
    bucket: str = "carbonplan-srm",
    prefix: str = "output/qa-intermediate-flags",
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
    storage = icechunk.s3_storage(bucket=bucket, prefix=f"{prefix}/{tag}.icechunk", from_env=True)
    repo = icechunk.Repository.open_or_create(storage)  # one repo per gcm/var/scenario/ens tag
    session = repo.writable_session("main")

    # encoding is only valid the first time flag_name is written to this store;
    # xarray errors if encoding is passed for a variable that already exists there
    variable_exists = False
    if write_mode != "w":
        try:
            variable_exists = flag_name in xr.open_zarr(session.store, consolidated=False).variables
        except Exception:
            variable_exists = False  # store doesn't exist yet

    encoding = {} if variable_exists else {flag_name: {"_FillValue": None}}

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
        thisgcm, thisvar, thisscenario, thisens = parse_tag(tag)
        das_scenario1.append(da)
        ens_scenario1.append(thisens)

    var_scenario1 = xr.concat(das_scenario1, dim=pd.Index(ens_scenario1, name="ensemble_member"))

    das_scenario2 = []
    ens_scenario2 = []
    for tag in tags_scenario2:
        da = get_data(tag=tag, trees=trees)
        thisgcm, thisvar, thisscenario, thisens = parse_tag(tag)
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
    ).compute()
    ens_mean_var_scenario2_coarsened = interpolate_fine_to_coarse_grid(
        da_fine_to_coarsen=ens_mean_var_scenario2, da_coarse_grid=raw_scenario1_mean
    ).compute()

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
):
    storage = icechunk.s3_storage(bucket=bucket, prefix=f"{prefix}/{tag}.icechunk", from_env=True)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
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
