import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from xclim.indices import dry_days, growing_degree_days, hot_days, tx_max


def plot_comparisons(obs, raw, ds1, ds2=None, bias="absolute"):
    fig, axarr = plt.subplots(figsize=(20, 8), nrows=2, ncols=4)

    varname = str(obs.name) if obs.name is not None else ""
    fig.suptitle(f"1978 mean {varname}", fontsize=16, y=0.98)

    cax = obs.plot(ax=axarr[0, 0])
    axarr[0, 0].set_title("ERA5")

    raw.plot(ax=axarr[0, 1], vmin=cax.get_clim()[0], vmax=cax.get_clim()[1])
    axarr[0, 1].set_title("GCM raw output")

    ds1.plot(ax=axarr[0, 2], vmin=cax.get_clim()[0], vmax=cax.get_clim()[1])
    axarr[0, 2].set_title("GCM BCSD nonparametric")

    if ds2 is not None:
        ds2.plot(ax=axarr[0, 3], vmin=cax.get_clim()[0], vmax=cax.get_clim()[1])
        axarr[0, 3].set_title("GCM BCSD parametric")
    else:
        axarr[0, 3].axis("off")

    if bias == "absolute":
        (ds1 - obs).plot(ax=axarr[1, 0])
        axarr[1, 0].set_title("Nonparametric minus ERA5")

        if ds2 is not None:
            (ds2 - obs).plot(ax=axarr[1, 1])
            axarr[1, 1].set_title("Parametric minus ERA5")

            (ds2 - ds1).plot(ax=axarr[1, 2])
            axarr[1, 2].set_title("Parametric minus Nonparametric")
        else:
            axarr[1, 1].axis("off")
            axarr[1, 2].axis("off")

    elif bias == "percentage":
        (((ds1 - obs) / obs) * 100).plot(ax=axarr[1, 0])
        axarr[1, 0].set_title("Nonparametric − ERA5 (%)")

        if ds2 is not None:
            (((ds2 - obs) / obs) * 100).plot(ax=axarr[1, 1])
            axarr[1, 1].set_title("Parametric − ERA5 (%)")

            (((ds2 - ds1) / ds1) * 100).plot(ax=axarr[1, 2])
            axarr[1, 2].set_title("Parametric − Nonparametric (%)")
        else:
            axarr[1, 1].axis("off")
            axarr[1, 2].axis("off")

    axarr[1, 3].axis("off")

    plt.tight_layout()


def calculate_statistic_to_plot(raw, era5, ds1, stat, variable, ds2=None):
    ds_list = [raw, era5, ds1]
    if ds2 is not None:
        ds_list.append(ds2)

    if stat == "mean":
        out = [ds[variable].mean(dim="time").compute() for ds in ds_list]

    elif stat == "99p":
        out = [ds[variable].quantile(0.99, dim="time").compute() for ds in ds_list]
        out = [da.rename("99p") for da in out]

    elif stat == "dry_days":
        out = []
        for ds in ds_list:
            pr = ds["pr"]
            pr.attrs.setdefault("units", "mm/day")
            out.append(dry_days(pr, thresh="1 mm/day", freq="YS").sum("time").compute())
        out = [da.rename("dry_days") for da in out]

    elif stat == "hottest_day":
        out = []
        for ds in ds_list:
            tasmax = ds[variable]
            tasmax.attrs.setdefault("units", "K")
            out.append(tx_max(tasmax, freq="YS").mean("time").compute())
        out = [da.rename("hottest_day") for da in out]

    elif stat == "gdd":
        out = []
        for ds in ds_list:
            tas = ds[variable]
            tas.attrs.setdefault("units", "K")
            out.append(growing_degree_days(tas, thresh="10 degC", freq="YS").mean("time").compute())
        out = [da.rename("gdd") for da in out]

    elif stat == "days_over_30C":
        out = []
        for ds in ds_list:
            tasmax = ds[variable]
            tasmax.attrs.setdefault("units", "K")
            out.append(hot_days(tasmax, thresh="30 degC", freq="YS").mean("time").compute())
        out = [da.rename("days_over_30C") for da in out]

    else:
        raise ValueError(f"Unknown stat: {stat}")

    if ds2 is None:
        raw_toplot, era5_toplot, ds1_toplot = out
        ds2_toplot = None
    else:
        raw_toplot, era5_toplot, ds1_toplot, ds2_toplot = out

    return raw_toplot, era5_toplot, ds1_toplot, ds2_toplot
    
def get_4_subregions(ds, regions=REGIONS_4):
    """Return dictionary with 4 subset datasets/dataarrays."""
    return {
        name: subset_latlon(ds, bounds["lat"], bounds["lon"])
        for name, bounds in regions.items()

def plot_4regions_comparisons(raw, era5, ds1, stat, variable, ds2=None, regions=None):
    """
    One figure:
      rows = 4 regions
      cols = ERA5 | raw | ds1 | ds1-ERA5
    """
    # subset the datasets 
    raw_sub  = get_4_subregions(raw,  regions)
    era5_sub = get_4_subregions(era5, regions)
    ds1_sub  = get_4_subregions(ds1,  regions)
    ds2_sub  = get_4_subregions(ds2,  regions) if ds2 is not None else None

    region_names = list(regions.keys())
    nrows, ncols = len(region_names), 4
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 4.2*nrows))

    # compute statistic and plot per region
    for r, reg in enumerate(region_names):
        raw_r  = raw_sub[reg]
        era5_r = era5_sub[reg]
        ds1_r  = ds1_sub[reg]
        ds2_r  = ds2_sub[reg] if ds2 is not None else None

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
        axarr[r, 2].set_title(f"{reg}: BCSD")
        axarr[r, 3].set_title(f"{reg}: BCSD minus ERA5")

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
