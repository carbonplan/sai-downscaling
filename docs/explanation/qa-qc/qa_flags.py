import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

INDIVIDUAL_FLAG_DIR = "s3://carbonplan-scratch/srm/qaqc/flags/"

PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "tas": (150, 400),
    "tasmax": (150, 400),
    "tasmin": (150, 400),
    "pr": (0, 0.03),
    "rsds": (0, 500),
    "hurs": (0, 400),
    "dtr": (0, 150),
}


def flag_outliers(da, outlier_thresh_low, outlier_thresh_high, timescale: str = "annual"):
    """
    Flags outliers based on the observational record
    """
    if timescale == "dayofyear":
        doy = da["time"].dt.dayofyear
        high_outlier = da > outlier_thresh_high.sel(dayofyear=doy)
        low_outlier = da < outlier_thresh_low.sel(dayofyear=doy)
    elif timescale == "annual":
        high_outlier = da > outlier_thresh_high
        low_outlier = da < outlier_thresh_low

    any_outlier = (low_outlier + high_outlier) > 0

    return any_outlier


def flag_rsds_above_max(da, zonal_doy_max_rsds):
    """
    Flags days when rsds is above expected max (based on latitude and dayofyear)
    """
    aligned_max = zonal_doy_max_rsds.sel(dayofyear=da.time.dt.dayofyear)
    exceeds_max = da > aligned_max

    return exceeds_max


def flag_global_exceedances(da, var: str, var_ranges: dict = PLAUSIBLE_RANGES):
    """
    Flags days when a variable is outside the globally-defined range of what is plausible.
    """

    var_range = var_ranges[var]
    var_min = var_range[0]
    var_max = var_range[1]

    too_high = da > var_max
    too_low = da < var_min

    outside_range = (too_low + too_high) > 0

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
):
    flag_data = flag_data.rename(flag_name)
    flag_data.attrs = {
        "long_name": "Quality flag",
        "description": "0=no known issue; 1=known issue",
        "short_name": flag_name,
    }

    flag_data = flag_data.chunk({"lat": 100, "lon": 100, "time": 8000})

    tag = f"{gcm}_{var}_{scenario}_{ens}"
    flag_data.to_zarr(
        flag_dir + tag + ".zarr", mode=write_mode, consolidated=False, align_chunks=True
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
