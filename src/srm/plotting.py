import calendar
import random
import typing

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from xclim.indices import dry_days, growing_degree_days, hot_days, tx_max

StatName = typing.Literal["mean", "99p", "dry_days", "hottest_day", "gdd", "days_over_30C"]
StatisticToPlot = tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray | None]
BiasMode = typing.Literal["absolute", "percentage"]


def _collect_datasets(
    raw: xr.Dataset, era5: xr.Dataset, ds1: xr.Dataset, ds2: xr.Dataset | None = None
) -> list[xr.Dataset]:
    ds_list = [raw, era5, ds1]
    if ds2 is not None:
        ds_list.append(ds2)
    return ds_list


def _with_default_units(da: xr.DataArray, default_units: str) -> xr.DataArray:
    if "units" in da.attrs:
        return da
    return da.assign_attrs(units=default_units)


def _ensure_inferable_freq(da: xr.DataArray) -> xr.DataArray:
    """Convert to a noleap CFTimeIndex when xr.infer_freq returns None.

    Model output on a 365-day calendar is often stored as datetime64[ns] with
    Feb 29 absent. xclim calls xr.infer_freq() internally and raises if it
    returns None. Converting to "noleap" fixes the gaps without altering data.
    Gregorian data (Feb 29 present, infer_freq == "D") is left unchanged.
    """
    if xr.infer_freq(da.time) is None:
        da = da.convert_calendar("noleap")
    return da


def _calc_mean(ds: xr.Dataset, variable: str) -> xr.DataArray:
    return ds[variable].mean(dim="time")


def _calc_99p(ds: xr.Dataset, variable: str) -> xr.DataArray:
    return ds[variable].quantile(0.99, dim="time").rename("99p")


def _calc_dry_days(ds: xr.Dataset, _variable: str) -> xr.DataArray:
    pr = _ensure_inferable_freq(_with_default_units(ds["pr"], "mm/day"))
    return dry_days(pr, thresh="1 mm/day", freq="YS").sum("time").rename("dry_days")


def _calc_hottest_day(ds: xr.Dataset, variable: str) -> xr.DataArray:
    tasmax = _ensure_inferable_freq(_with_default_units(ds[variable], "K"))
    # This calculates annual maxima by calendar year and can split austral summers.
    return tx_max(tasmax, freq="YS").mean("time").rename("hottest_day")


def _calc_gdd(ds: xr.Dataset, variable: str) -> xr.DataArray:
    tas = _ensure_inferable_freq(_with_default_units(ds[variable], "K"))
    return growing_degree_days(tas, thresh="10 degC", freq="YS").mean("time").rename("gdd")


def _calc_days_over_30c(ds: xr.Dataset, variable: str) -> xr.DataArray:
    tasmax = _ensure_inferable_freq(_with_default_units(ds[variable], "K"))
    return hot_days(tasmax, thresh="30 degC", freq="YS").mean("time").rename("days_over_30C")


_STAT_CALCULATORS: dict[StatName, typing.Callable[[xr.Dataset, str], xr.DataArray]] = {
    "mean": _calc_mean,
    "99p": _calc_99p,
    "dry_days": _calc_dry_days,
    "hottest_day": _calc_hottest_day,
    "gdd": _calc_gdd,
    "days_over_30C": _calc_days_over_30c,
}


def _calculate_stat(ds: xr.Dataset, stat: StatName, variable: str) -> xr.DataArray:
    try:
        calculator = _STAT_CALCULATORS[stat]
    except KeyError as exc:
        valid = ", ".join(_STAT_CALCULATORS.keys())
        raise ValueError(f"Unknown stat: {stat}. Expected one of: {valid}") from exc
    return calculator(ds, variable)


def _plot_primary_comparison_panels(
    axarr: np.ndarray,
    obs: xr.DataArray,
    raw: xr.DataArray,
    ds1: xr.DataArray,
    ds1_name: str,
    ds2: xr.DataArray | None,
    ds2_name: str | None,
) -> tuple[float, float]:
    cax = raw.plot(ax=axarr[0, 0], robust=True)
    axarr[0, 0].set_title("GCM raw output")

    vmin, vmax = cax.get_clim()
    obs.plot(ax=axarr[1, 0], vmin=vmin, vmax=vmax)
    axarr[1, 0].set_title("ERA5")

    ds1.plot(ax=axarr[0, 1], vmin=vmin, vmax=vmax)
    axarr[0, 1].set_title(ds1_name)

    if ds2 is not None:
        ds2.plot(ax=axarr[0, 2], vmin=vmin, vmax=vmax)
        axarr[0, 2].set_title(ds2_name)
    else:
        axarr[0, 2].axis("off")

    axarr[0, 3].axis("off")
    return vmin, vmax


def _plot_absolute_bias_panels(
    axarr: np.ndarray,
    obs: xr.DataArray,
    ds1: xr.DataArray,
    ds1_name: str,
    ds2: xr.DataArray | None,
    ds2_name: str | None,
) -> None:
    (ds1 - obs).plot(ax=axarr[1, 1], robust=True)
    axarr[1, 1].set_title(f"{ds1_name} - ERA5")

    if ds2 is not None:
        (ds2 - obs).plot(ax=axarr[1, 2], robust=True)
        axarr[1, 2].set_title(f"{ds2_name} - ERA5")

        (ds2 - ds1).plot(ax=axarr[1, 3], robust=True)
        axarr[1, 3].set_title(f"{ds2_name} - {ds1_name}")
    else:
        axarr[1, 2].axis("off")
        axarr[1, 3].axis("off")


def _plot_percentage_bias_panels(
    axarr: np.ndarray,
    obs: xr.DataArray,
    ds1: xr.DataArray,
    ds1_name: str,
    ds2: xr.DataArray | None,
    ds2_name: str | None,
) -> None:
    (((ds1 - obs) / obs) * 100).plot(ax=axarr[1, 1], robust=True)
    axarr[1, 1].set_title(f"{ds1_name} - ERA5 (%)")

    if ds2 is not None:
        (((ds2 - obs) / obs) * 100).plot(ax=axarr[1, 2])
        axarr[1, 2].set_title(f"{ds2_name} - ERA5 (%)")

        (((ds2 - ds1) / ds1) * 100).plot(ax=axarr[1, 3])
        axarr[1, 3].set_title(f"{ds2_name} - {ds1_name} (%)")
    else:
        axarr[1, 2].axis("off")
        axarr[1, 3].axis("off")


def _plot_bias_panels(
    axarr: np.ndarray,
    obs: xr.DataArray,
    ds1: xr.DataArray,
    ds1_name: str,
    ds2: xr.DataArray | None,
    ds2_name: str | None,
    bias: BiasMode,
) -> None:
    if bias == "absolute":
        _plot_absolute_bias_panels(axarr, obs, ds1, ds1_name, ds2, ds2_name)
    elif bias == "percentage":
        _plot_percentage_bias_panels(axarr, obs, ds1, ds1_name, ds2, ds2_name)
    else:
        raise ValueError(f"Unknown bias: {bias}. Expected one of: absolute, percentage")


def plot_comparisons(
    obs: xr.DataArray,
    raw: xr.DataArray,
    ds1: xr.DataArray,
    ds1_name: str,
    ds2: xr.DataArray | None = None,
    bias: BiasMode = "absolute",
    ds2_name: str | None = None,
    title: str = "",
) -> None:
    fig, axarr = plt.subplots(figsize=(20, 8), nrows=2, ncols=4)

    _ = str(obs.name) if obs.name is not None else ""
    fig.suptitle(title, fontsize=16, y=0.98)

    _plot_primary_comparison_panels(axarr, obs, raw, ds1, ds1_name, ds2, ds2_name)
    _plot_bias_panels(axarr, obs, ds1, ds1_name, ds2, ds2_name, bias)

    plt.tight_layout()


def calculate_statistic_to_plot(
    raw: xr.Dataset,
    era5: xr.Dataset,
    ds1: xr.Dataset,
    stat: StatName,
    variable: str,
    ds2: xr.Dataset | None = None,
) -> StatisticToPlot:
    ds_list = _collect_datasets(raw, era5, ds1, ds2)
    out = [_calculate_stat(ds, stat, variable).compute() for ds in ds_list]

    raw_toplot, era5_toplot, ds1_toplot = out[:3]
    ds2_toplot = out[3] if ds2 is not None else None

    return raw_toplot, era5_toplot, ds1_toplot, ds2_toplot


# 4 subregions to focus on
REGIONS_4 = {
    "India": dict(lat=(5, 35), lon=(68, 97)),
    "South Africa": dict(lat=(-35, -20), lon=(16, 33)),
    "Brazil": dict(lat=(-35, 6), lon=(-75, -34)),
    "West Africa": dict(lat=(0, 20), lon=(-20, 15)),
}


def subset_latlon(ds, lat_bounds, lon_bounds, lat_name="lat", lon_name="lon"):
    """Subset an xarray Dataset or DataArray to lat/lon bounds."""
    lat0, lat1 = lat_bounds
    lon0, lon1 = lon_bounds

    # handle lat ordering
    lat = ds[lat_name]
    lat_slice = slice(lat0, lat1) if lat[0] < lat[-1] else slice(lat1, lat0)

    # handle 0–360 vs -180–180 longitude
    lon = ds[lon_name]
    uses_360 = (lon.min() >= 0) and (lon.max() > 180)

    def to_360(x):
        return (x + 360) % 360

    if uses_360:
        lon0c, lon1c = to_360(lon0), to_360(lon1)
    else:
        lon0c, lon1c = lon0, lon1

    if uses_360 and lon0c > lon1c:
        a = ds.sel({lat_name: lat_slice, lon_name: slice(lon0c, 360)})
        b = ds.sel({lat_name: lat_slice, lon_name: slice(0, lon1c)})
        return xr.concat([a, b], dim=lon_name)

    return ds.sel({lat_name: lat_slice, lon_name: slice(lon0c, lon1c)})


def get_4_subregions(ds, regions=REGIONS_4):
    """Return dictionary with 4 subset datasets/dataarrays."""
    return {
        name: subset_latlon(ds, bounds["lat"], bounds["lon"]) for name, bounds in regions.items()
    }


def plot_4regions_comparisons(
    raw, era5, ds1, stat, variable, ds1_title, ds2=None, regions=REGIONS_4
):
    """
    One figure:
      rows = 4 regions
      cols = ERA5 | raw | ds1 | ds1-ERA5
    """
    # subset the datasets
    raw_sub = get_4_subregions(raw, regions)
    era5_sub = get_4_subregions(era5, regions)
    ds1_sub = get_4_subregions(ds1, regions)
    ds2_sub = get_4_subregions(ds2, regions) if ds2 is not None else None

    region_names = list(regions.keys())
    nrows, ncols = len(region_names), 4
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 4.2 * nrows))

    # compute statistic and plot per region
    for r, reg in enumerate(region_names):
        raw_r = raw_sub[reg]
        era5_r = era5_sub[reg]
        ds1_r = ds1_sub[reg]
        ds2_r = ds2_sub[reg] if ds2 is not None else None

        raw_toplot, era5_toplot, ds1_toplot, _ = calculate_statistic_to_plot(
            raw_r, era5_r, ds1_r, stat, variable, ds2=ds2_r
        )

        # ERA5 defines clim for the first 3 cols
        cax = era5_toplot.plot(ax=axarr[r, 0], add_colorbar=True)
        vmin, vmax = cax.get_clim()

        raw_toplot.plot(ax=axarr[r, 1], vmin=vmin, vmax=vmax, add_colorbar=True)
        ds1_toplot.plot(ax=axarr[r, 2], vmin=vmin, vmax=vmax, add_colorbar=True)
        (ds1_toplot - era5_toplot).plot(ax=axarr[r, 3], add_colorbar=True)

        # titles
        axarr[r, 0].set_title(f"{reg}: ERA5")
        axarr[r, 1].set_title(f"{reg}: Raw output")
        axarr[r, 2].set_title(f"{reg}: {ds1_title}")
        axarr[r, 3].set_title(f"{reg}: {ds1_title} minus ERA5")

    fig.suptitle(f"{stat} ({variable}) — 4 subregions", fontsize=16, y=0.995)
    plt.tight_layout()
    return fig


def prep_funky_calendar(ds, ds_timeindex_to_match, time_slice):
    ds_subset = ds.sel(time=time_slice)
    # overwrite the time index because some calendars are weird and won't play nice in plotting
    # this will only work if we're not in a leap year! (if we have any 360 day calendar models
    # we'll have to change this as well)
    ds_subset["time"] = ds_timeindex_to_match.sel(time=time_slice)["time"]
    return ds_subset


def sel_point(ds, lat, lon):
    return ds.sel(lat=lat, lon=lon, method="nearest")


def random_non_leap_year(start=1984, end=2014):
    """pick a random year in the range that isn't a leap year -
    we'll use this to plot random years in the record.
    """
    years = [y for y in np.arange(start, end + 1) if not calendar.isleap(y)]
    chosen_year = random.choice(years)
    return chosen_year


def prep_datasets_for_daily_timeseries_plotting(era5, raw, ds1, time_slice, lat, lon, ds2=None):
    era5_toplot = sel_point(era5.sel(time=time_slice), lat, lon)
    raw_toplot = sel_point(prep_funky_calendar(raw, era5, time_slice), lat, lon)
    ds1_toplot = sel_point(prep_funky_calendar(ds1, era5, time_slice), lat, lon)
    if ds2 is not None:
        ds2_toplot = sel_point(prep_funky_calendar(ds2, era5, time_slice), lat, lon)
        return era5_toplot, raw_toplot, ds1_toplot, ds2_toplot
    else:
        return era5_toplot, raw_toplot, ds1_toplot


def prep_datasets_for_seasonal_cycle_plotting(
    era5, raw, ds1, time_slice, lat, lon, variable, ds2=None
):
    era5_toplot, raw_toplot, ds1_toplot = [
        ds[variable]
        .sel(time=time_slice)
        .sel(lat=lat, lon=lon, method="nearest")
        .groupby("time.dayofyear")
        .mean()
        for ds in [era5, raw, ds1]
    ]
    if ds2 is not None:
        ds2_toplot = (
            ds2[variable]
            .sel(time=time_slice)
            .sel(lat=lat, lon=lon, method="nearest")
            .groupby("time.dayofyear")
            .mean()
        )
        return era5_toplot, raw_toplot, ds1_toplot, ds2_toplot
    else:
        return era5_toplot, raw_toplot, ds1_toplot


def plot_timeseries(ax, era5_toplot, raw_toplot, ds1_toplot, location, ds2_toplot=None):
    era5_toplot.plot(ax=ax, color="grey", alpha=0.5)
    raw_toplot.plot(ax=ax, color="k")
    ds1_toplot.plot(ax=ax, color="firebrick")
    if ds2_toplot is not None:
        ds2_toplot.plot(ax=ax, color="royalblue")
    ax.set_title(location)


def plot_pdf(era5, raw, ds1, var, ds2=None, title=None, xlabel=None):
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 11

    sns.kdeplot(
        era5,
        label="Observations (ERA5)",
        color="gray",
        linewidth=7,
        alpha=0.3,
    )
    sns.kdeplot(
        raw,
        label="Modeled historical (raw)",
        color="black",
    )
    sns.kdeplot(
        ds1,
        label="Downscaled v1",
        color="firebrick",
        linestyle="-",
    )
    if ds2 is not None:
        sns.kdeplot(
            ds2,
            label="Downscaled v2",
            color="royalblue",
            linestyle="-",
        )

    plt.title(
        title,
        fontsize=13,
        fontweight="bold",
    )

    plt.legend()
    plt.xlabel(xlabel)


def plot_cdf(era5, raw, ds1, var=None, ds2=None, title=None, xlabel=None):
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 11

    def _prep(x):
        if hasattr(x, "values"):
            v = x.values.ravel()
            return v[~np.isnan(v)]
        return x

    sns.kdeplot(
        _prep(era5),
        label="Observations (ERA5)",
        color="gray",
        linewidth=7,
        alpha=0.3,
        cumulative=True,
    )
    sns.kdeplot(
        _prep(raw),
        label="Modeled historical (raw)",
        color="black",
        cumulative=True,
    )
    sns.kdeplot(
        _prep(ds1),
        label="Downscaled v1",
        color="firebrick",
        linestyle="-",
        cumulative=True,
    )
    if ds2 is not None:
        sns.kdeplot(
            _prep(ds2),
            label="Downscaled v2",
            color="royalblue",
            linestyle="-",
            cumulative=True,
        )

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative probability")
    plt.legend()
    plt.tight_layout()


locations = {
    "Cape Town": (-33.9221, 18.4231),
    "Addis Ababa": (9.0192, 38.7525),
    "Mumbai": (19.0728, 72.8826),
    "Sao Paulo": (-23.5558, -46.6396),
    "Berlin": (52.5200, 13.4050),
    "New York": (40.7128, -74.0060),
    "Sydney": (-33.8727, 151.2057),
    "Kyiv": (50.4504, 30.5245),
    "Beijing": (39.9042, 116.4074),
    "Accra": (5.5593, -0.1974),
    "Kualar Lumpur": (3.1319, 101.6841),
    "Lagos": (6.6137, 3.3553),
    "Rio de Janerio": (-22.9068, -43.1729),
    "Ankara": (39.9334, 32.8597),
    "Cairo": (30.0444, 31.2357),
    "Melbourne": (-37.8136, 144.9631),
    "Paris": (48.8575, 2.3514),
    "Tokyo": (35.6764, 139.6500),
    "Seoul": (37.5503, 126.9971),
    "Chicago": (41.8832, -87.6324),
    # TODO Temi: add in 9 more major cities around the world
    # TODO Temi: add in 10 locations from a variety of climatic zones around the world
}
Climatezones = {
    "Kualar Lumpur": (3.1319, 101.6841),
    "Lagos": (6.6137, 3.3553),
    "Rio de Janerio": (-22.9068, -43.1729),
    "Ankara": (39.9334, 32.8597),
    "Cairo": (30.0444, 31.2357),
    "Melbourne": (-37.8136, 144.9631),
    "Paris": (48.8575, 2.3514),
    "Tokyo": (35.6764, 139.6500),
    "Seoul": (37.5503, 126.9971),
    "Chicago": (41.8832, -87.6324),
}
