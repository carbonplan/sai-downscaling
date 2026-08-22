"""
Visualization utilities for comparing BCSD downscaling outputs.

Provides functions for plotting raw vs. debiased vs. observed distributions and
spatial maps across pipeline stages. Intended for use in QA and diagnostic notebooks.
"""

import calendar
import random

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from xclim.indices import dry_days, growing_degree_days, hot_days, tx_max

from srm.bcsd_config import METHOD_SEGMENTS


def plot_comparisons(obs, raw, ds1, ds1_name, ds2=None, bias="absolute", ds2_name=None, title=""):
    fig, axarr = plt.subplots(figsize=(20, 8), nrows=2, ncols=4)

    _ = str(obs.name) if obs.name is not None else ""
    fig.suptitle(title, fontsize=16, y=0.98)

    cax = obs.plot(ax=axarr[0, 0])
    axarr[0, 0].set_title("Observations")

    raw.plot(ax=axarr[0, 1], vmin=cax.get_clim()[0], vmax=cax.get_clim()[1])
    axarr[0, 1].set_title("GCM raw output")

    ds1.plot(ax=axarr[0, 2], vmin=cax.get_clim()[0], vmax=cax.get_clim()[1])
    axarr[0, 2].set_title(ds1_name)

    if ds2 is not None:
        ds2.plot(ax=axarr[0, 3], vmin=cax.get_clim()[0], vmax=cax.get_clim()[1])
        axarr[0, 3].set_title(ds2_name)
    else:
        axarr[0, 3].axis("off")

    if bias == "absolute":
        cax2 = (ds1 - obs).plot(ax=axarr[1, 0])
        axarr[1, 0].set_title(f"{ds1_name} minus Obs")

        if ds2 is not None:
            (ds2 - obs).plot(ax=axarr[1, 1], vmax=cax2.get_clim()[1])
            axarr[1, 1].set_title(f"{ds2_name} minus Obs")

            (ds2 - ds1).plot(ax=axarr[1, 2], vmax=cax2.get_clim()[1])
            axarr[1, 2].set_title(f"{ds2_name} minus {ds1_name}")
        else:
            axarr[1, 1].axis("off")
            axarr[1, 2].axis("off")

    elif bias == "percentage":
        (((ds1 - obs) / obs) * 100).plot(ax=axarr[1, 0])
        axarr[1, 0].set_title("Nonparametric − Obs (%)")

        if ds2 is not None:
            (((ds2 - obs) / obs) * 100).plot(ax=axarr[1, 1])
            axarr[1, 1].set_title("Parametric − Obs (%)")

            (((ds2 - ds1) / ds1) * 100).plot(ax=axarr[1, 2])
            axarr[1, 2].set_title("Parametric − Nonparametric (%)")
        else:
            axarr[1, 1].axis("off")
            axarr[1, 2].axis("off")

    axarr[1, 3].axis("off")

    plt.tight_layout()


def calculate_statistic_to_plot(ds_list, stat, variable):
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
            # tx_max doesn't fail
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

    return out


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
    raw, obs, ds1, stat, variable, ds1_title, ds2=None, regions=REGIONS_4
):
    """
    One figure:
      rows = 4 regions
      cols = Obs | raw | ds1 | ds1-Obs
    """
    # subset the datasets
    raw_sub = get_4_subregions(raw, regions)
    obs_sub = get_4_subregions(obs, regions)
    ds1_sub = get_4_subregions(ds1, regions)

    region_names = list(regions.keys())
    nrows, ncols = len(region_names), 4
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 4.2 * nrows))

    # compute statistic and plot per region
    for r, reg in enumerate(region_names):
        raw_r = raw_sub[reg]
        obs_r = obs_sub[reg]
        ds1_r = ds1_sub[reg]

        raw_toplot, obs_toplot, ds1_toplot = calculate_statistic_to_plot(
            ds_list=[raw_r, obs_r, ds1_r],
            stat=stat,
            variable=variable,
        )

        # obs defines clim for the first 3 cols
        cax = obs_toplot.plot(ax=axarr[r, 0], add_colorbar=True)
        vmin, vmax = cax.get_clim()

        raw_toplot.plot(ax=axarr[r, 1], vmin=vmin, vmax=vmax, add_colorbar=True)
        ds1_toplot.plot(ax=axarr[r, 2], vmin=vmin, vmax=vmax, add_colorbar=True)
        (ds1_toplot - obs_toplot).plot(ax=axarr[r, 3], add_colorbar=True)

        # titles
        axarr[r, 0].set_title(f"{reg}: Observations")
        axarr[r, 1].set_title(f"{reg}: Raw output")
        axarr[r, 2].set_title(f"{reg}: {ds1_title}")
        axarr[r, 3].set_title(f"{reg}: {ds1_title} minus Obs")

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


def prep_datasets_for_daily_timeseries_plotting(obs, raw, ds1, time_slice, lat, lon, ds2=None):
    obs_toplot = sel_point(obs.sel(time=time_slice), lat, lon)
    raw_toplot = sel_point(prep_funky_calendar(raw, obs, time_slice), lat, lon)
    ds1_toplot = sel_point(prep_funky_calendar(ds1, obs, time_slice), lat, lon)
    if ds2 is not None:
        ds2_toplot = sel_point(prep_funky_calendar(ds2, obs, time_slice), lat, lon)
        return obs_toplot, raw_toplot, ds1_toplot, ds2_toplot
    else:
        return obs_toplot, raw_toplot, ds1_toplot


def prep_datasets_for_seasonal_cycle_plotting(
    obs, raw, ds1, time_slice, lat, lon, variable, ds2=None
):
    obs_toplot, raw_toplot, ds1_toplot = [
        ds[variable]
        .sel(time=time_slice)
        .sel(lat=lat, lon=lon, method="nearest")
        .groupby("time.dayofyear")
        .mean()
        for ds in [obs, raw, ds1]
    ]
    if ds2 is not None:
        ds2_toplot = (
            ds2[variable]
            .sel(time=time_slice)
            .sel(lat=lat, lon=lon, method="nearest")
            .groupby("time.dayofyear")
            .mean()
        )
        return obs_toplot, raw_toplot, ds1_toplot, ds2_toplot
    else:
        return obs_toplot, raw_toplot, ds1_toplot


def plot_timeseries(ax, obs_toplot, raw_toplot, ds1_toplot, location, ds2_toplot=None):
    obs_toplot.plot(ax=ax, color="grey", alpha=0.5)
    raw_toplot.plot(ax=ax, color="k")
    ds1_toplot.plot(ax=ax, color="firebrick")
    if ds2_toplot is not None:
        ds2_toplot.plot(ax=ax, color="royalblue")
    ax.set_title(location)


def plot_pdf(obs, raw, ds1, var, ds2=None, title=None, xlabel=None):
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 11

    sns.kdeplot(
        obs,
        label="Observations",
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


def plot_cdf(obs, raw, ds1, var=None, ds2=None, title=None, xlabel=None):
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 11

    def _prep(x):
        if hasattr(x, "values"):
            v = x.values.ravel()
            return v[~np.isnan(v)]
        return x

    sns.kdeplot(
        _prep(obs),
        label="Observations",
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


def plot_cdf_by_location(
    debiased_downscaled,
    coarse_debiased,
    locations,
    obs_fine=None,
    obs_coarse=None,
    raw_gcm=None,
    var=None,
    ncols=5,
):
    """Grid of per-location CDFs comparing debiased_downscaled (red),
    coarse_debiased (blue), and raw_gcm (green), each against the obs at its
    own resolution (obs_fine, obs_coarse) if provided. Every trace reports the
    percent of dry days (zero precipitation); when an obs series is supplied,
    each model trace's subplot annotation also reports mean bias and percent
    bias relative to that obs.
    """

    def _prep(da, lat, lon):
        v = da.sel(lat=lat, lon=lon, method="nearest").values.ravel()
        return v[~np.isnan(v)]

    # each model series is compared against the obs at matching resolution, if given
    model_series = [
        dict(
            label="debiased_downscaled",
            da=debiased_downscaled,
            resolution="fine",
            color="red",
            linestyle="-",
        ),
        dict(
            label="coarse_debiased",
            da=coarse_debiased,
            resolution="coarse",
            color="blue",
            linestyle="-",
        ),
        dict(
            label="raw_gcm",
            da=raw_gcm,
            resolution="coarse",
            color="green",
            linestyle="-",
        ),
    ]
    model_series = [series for series in model_series if series["da"] is not None]
    obs_series = {
        "fine": dict(label="obs_fine", da=obs_fine, color="black", linestyle="-"),
        "coarse": dict(label="obs_coarse", da=obs_coarse, color="dimgrey", linestyle="--"),
    }
    obs_series = {res: cfg for res, cfg in obs_series.items() if cfg["da"] is not None}

    names = list(locations.keys())
    nrows = int(np.ceil(len(names) / ncols))
    all_series = [*model_series, *obs_series.values()]
    n_series = len(all_series)
    # stats live under each panel, so reserve vertical room proportional to series count
    row_height = 3.5 + 0.16 * n_series
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, row_height * nrows))
    axarr = np.atleast_1d(axarr).ravel()
    label_width = max(len(s["label"]) for s in all_series)

    for ax, name in zip(axarr, names):
        lat, lon = locations[name]
        obs_vals = {res: _prep(cfg["da"], lat, lon) for res, cfg in obs_series.items()}
        # inset in the lower-right quarter of the panel, zoomed to the top 1% of the CDF
        ax_inset = ax.inset_axes([0.52, 0.06, 0.46, 0.46])

        stats_lines = []
        all_vals = []
        for series in all_series:
            vals = _prep(series["da"], lat, lon)
            all_vals.append(vals)
            for target_ax in (ax, ax_inset):
                sns.kdeplot(
                    vals,
                    ax=target_ax,
                    label=series["label"],
                    color=series["color"],
                    linestyle=series["linestyle"],
                    cumulative=True,
                )
            dry_pct = 100 * np.mean(vals == 0)
            line = f"{series['label']:<{label_width}}  dry={dry_pct:5.1f}%"
            resolution = series.get("resolution")
            if resolution in obs_vals:
                obs_mean = obs_vals[resolution].mean()
                mean_bias = vals.mean() - obs_mean
                pct_bias = 100 * mean_bias / obs_mean
                line += f"  bias={mean_bias:+.2e}  %bias={pct_bias:+7.2f}%"
            stats_lines.append((line, series["color"]))

        zoom_lo = min(np.percentile(v, 99) for v in all_vals)
        zoom_hi = max(v.max() for v in all_vals)
        ax_inset.set_xlim(zoom_lo, zoom_hi)
        ax_inset.set_ylim(0.99, 1.0)
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")
        ax_inset.set_title("top 1%", fontsize=6)
        ax_inset.tick_params(labelsize=6)
        ax_inset.xaxis.set_major_locator(mticker.MaxNLocator(3, prune="upper"))
        ax_inset.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax_inset.xaxis.get_offset_text().set_fontsize(5)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel(var or "")
        ax.set_ylabel("")
        # keep x tick labels from colliding: few ticks, shared sci-notation offset
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="upper"))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(7)

        # stats below the axes, one colored row per series, so nothing overlaps the curves
        for i, (line, color) in enumerate(stats_lines):
            ax.text(
                0.0,
                -0.30 - 0.07 * i,
                line,
                transform=ax.transAxes,
                fontsize=6,
                family="monospace",
                color=color,
                va="top",
                ha="left",
            )

    for ax in axarr[len(names) :]:
        ax.axis("off")

    handles, labels = axarr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_series, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("CDF comparison by location", fontsize=16, y=1.01)
    plt.tight_layout()
    # tight_layout ignores the stats text drawn outside the axes; open up the rows for it
    fig.subplots_adjust(hspace=0.35 + 0.09 * n_series)
    return fig


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


def plot_region_zoom(
    exceedance: xr.DataArray,
    regions: pd.DataFrame,
    *,
    direction: str,
    units: str,
    scale: float = 1.0,
    pad: float = 8.0,
    ncols: int = 3,
) -> None:
    """Zoomed maps of the ranked exceedance regions, one panel each.

    The global maps show *that* a leaf is flagged; these show what each flagged patch looks like
    locally, with a red cross marking the worst cell, which is the point the drill-down inspects.
    """
    if len(regions) == 0:
        print(f"no {direction} regions to plot")
        return

    flagged = exceedance.where(exceedance > 0 if direction == "high" else exceedance < 0) * scale
    cmap = "viridis_r" if direction == "high" else "viridis"
    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    for ax, region in zip(axes.flat, regions.itertuples(), strict=False):
        # A region straddling the antimeridian has no single bounding box, so centre the view on
        # its circular-mean centroid instead of on a lon_min/lon_max that spans the whole globe.
        if region.wraps_lon:
            lon_lo, lon_hi = region.centroid_lon - 30.0, region.centroid_lon + 30.0
        else:
            lon_lo, lon_hi = region.lon_min - pad, region.lon_max + pad
        ax.set_extent(
            [lon_lo, lon_hi, max(-90.0, region.lat_min - pad), min(90.0, region.lat_max + pad)],
            crs=ccrs.PlateCarree(),
        )
        flagged.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            cbar_kwargs={"label": units, "shrink": 0.7},
        )
        ax.plot(
            region.worst_lon,
            region.worst_lat,
            "x",
            color="red",
            ms=10,
            mew=2,
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.3")
        ax.set_title(
            f"region {region.region}: {region.worst_value:.1f} {units} at "
            f"({region.worst_lat:.2f}, {region.worst_lon:.2f}), {region.n_cells} cells"
            + (" [wraps lon]" if region.wraps_lon else ""),
            fontsize=9,
        )

    for ax in axes.flat[len(regions) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def _debiased_coarse_node(tree, scenario: str, method: str | None = None):
    """Return the ``debiased_coarse`` node for ``scenario``, for either store layout.

    Method-dependent groups are namespaced under a leading downscaling-method segment,
    so the coarse debiased artifact lives at ``{method}/debiased_coarse/{scenario}``.
    Stores written before the namespacing hold it at ``debiased_coarse/{scenario}``.
    Both are resolved here so a drill-down works against either store.

    When ``method`` is given and the store has no method segments at all, the store
    predates the split and holds exactly one method's output at the root, so the
    request resolves to the root regardless of which method was asked for. When the
    store does have method segments, ``method`` must name one that is actually present;
    otherwise the lookup fails and says which segments the store does have.

    Parameters
    ----------
    tree : xr.DataTree
        Open output store.
    scenario : str
        On-disk scenario group name, e.g. ``"ssp245"``.
    method : str, optional
        Downscaling method whose segment to read, case-insensitive (``"BCSD"`` or
        ``"QDMSD"``). ``None`` (the default) auto-detects the single layout present.

    Returns
    -------
    xr.DataTree
        The ``debiased_coarse/{scenario}`` node.

    Raises
    ------
    KeyError
        If ``method`` names a segment the store does not have (and the store does have
        other method segments, so it postdates the split and a miss is genuine); if no
        candidate group holds ``scenario``; or if ``method`` is ``None`` and the store
        holds the scenario under more than one method segment. A store carrying both
        methods is genuinely ambiguous, so it must be disambiguated rather than silently
        paired against whichever segment happens to sort first.
    """
    # Roots to look under, keyed by the method segment ("" for the pre-namespace root).
    # A store can hold both forms, so every candidate is probed and the ambiguity is
    # reported rather than resolved arbitrarily.
    present_segments = {n: c for n, c in tree.children.items() if n in METHOD_SEGMENTS}
    roots = {"": tree} | present_segments
    if method is not None:
        segment = method.lower()
        if segment in tree.children:
            roots = {segment: tree.children[segment]}
        elif not present_segments:
            # Pre-namespace store: no method segments exist, so it predates the split
            # and holds exactly one method's output at the root. Any method request is
            # trivially satisfied there.
            roots = {"": tree}
        else:
            raise KeyError(
                f"store has no {segment!r} method segment; present segments: "
                f"{sorted(present_segments)}"
            )
    found = {
        segment: root.children["debiased_coarse"].children[scenario]
        for segment, root in sorted(roots.items())
        if "debiased_coarse" in root.children
        and scenario in root.children["debiased_coarse"].children
    }
    if not found:
        raise KeyError(
            f"no debiased_coarse/{scenario} group in this store; looked under "
            f"{[segment or '<root>' for segment in sorted(roots)]}"
        )
    if len(found) > 1:
        raise KeyError(
            f"debiased_coarse/{scenario} exists under {sorted(found)}; pass method= to choose one"
        )
    return next(iter(found.values()))


def point_series(
    key: tuple[str, str, str],
    lat: float,
    lon: float,
    low_bound: xr.DataArray,
    high_bound: xr.DataArray,
    tree,
    leaves,
    obs_fine,
    gcm,
    historical_member,
) -> dict[str, xr.DataArray]:
    """Every series the drill-down plots for one grid cell, all lazy and in stored units.

    Nothing is computed here. Collect these dicts for every point of interest and pass them to a
    SINGLE dask.compute so all the reads share one graph and one scheduler round-trip.

    `low_bound`/`high_bound` are the leaf's fine-grid plausible bounds from `leaf_bounds`, so the
    caller decides whether the expensive observed envelope underneath them was cached or rebuilt.
    """
    scenario, var, member = key
    at_point = {"lat": lat, "lon": lon, "method": "nearest"}
    coarse = _debiased_coarse_node(tree, scenario)[f"{var}/{member}"].dataset[var]
    return {
        "downscaled": leaves[key].sel(**at_point),
        "bound_low": low_bound.sel(**at_point),
        "bound_high": high_bound.sel(**at_point),
        "obs": obs_fine(var).sel(**at_point),
        "raw_scenario": gcm[scenario][var].sel(ensemble_member=member).sel(**at_point),
        "raw_historical": (
            gcm["historical"][var].sel(ensemble_member=historical_member(*key)).sel(**at_point)
        ),
        "coarse_debiased": coarse.sel(**at_point),
    }


def plot_point_panels(
    series: dict[str, xr.DataArray],
    key: tuple[str, str, str],
    lat: float,
    lon: float,
    *,
    units: str,
    scale: float = 1.0,
) -> int:
    """Four-panel drill-down for one grid cell, from already-computed point series.

    Returns the number of days that leave the envelope at this point. Every series is stored in
    the same units, so one scale converts the whole panel and the axes agree with the summary
    table. (ERA5 `pr` comes from mean_total_precipitation_rate and the raw GCM PRECT is CMORized
    to kg m-2 s-1 during ETL, so no series needs a different factor.)
    """
    scenario, var, member = key
    values = series["downscaled"] * scale
    bound_low = series["bound_low"] * scale
    bound_high = series["bound_high"] * scale
    obs = series["obs"] * scale
    raw_scenario = series["raw_scenario"] * scale
    raw_historical = series["raw_historical"] * scale
    coarse_debiased = series["coarse_debiased"] * scale

    doy = values["time.dayofyear"]
    too_high = values > bound_high.sel(dayofyear=doy)
    too_low = values < bound_low.sel(dayofyear=doy)
    obs_doy_mean = obs.groupby("time.dayofyear").mean()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    def _series(ax, da, **kw):
        """Scatter a point time series against its day of year."""
        ax.plot(da["time.dayofyear"], da, ".", ms=3, **kw)

    def _envelope(ax):
        """Overlay the plausible day-of-year envelope as dashed lines."""
        bound_high.plot(ax=ax, color="black", ls="--", label="Plausible envelope")
        bound_low.plot(ax=ax, color="black", ls="--")

    def _outliers(ax):
        """Downscaled values, with red rings on the days that leave the envelope."""
        ax.plot(doy, values, ".", ms=3, label="Downscaled")
        ax.plot(doy, values.where(too_high), "o", mfc="none", color="red", label="Outside envelope")
        ax.plot(doy, values.where(too_low), "o", mfc="none", color="red")

    # (0,0) Downscaled output vs envelope, with all inputs overlaid.
    ax = axes[0, 0]
    _outliers(ax)
    _series(ax, raw_historical, label="Raw historical GCM")
    _series(ax, raw_scenario, label="Raw scenario GCM")
    _series(ax, obs, color="black", label="ERA5 obs")
    _envelope(ax)
    ax.set_title("Downscaled vs inputs")
    ax.legend(fontsize=7)

    # (0,1) Downscaled output and envelope, with the observed day-of-year mean.
    ax = axes[0, 1]
    _outliers(ax)
    ax.plot(obs_doy_mean["dayofyear"], obs_doy_mean, "-k", label="Obs day-of-year mean")
    _envelope(ax)
    ax.set_title("Downscaled vs envelope")
    ax.legend(fontsize=7)

    # (1,0) Historical: obs and raw GCM.
    ax = axes[1, 0]
    _series(ax, raw_historical, color="C1", label="Raw historical GCM")
    _series(ax, obs, color="black", label="ERA5 obs")
    ax.plot(obs_doy_mean["dayofyear"], obs_doy_mean, "-k", label="Obs day-of-year mean")
    _envelope(ax)
    ax.set_title("Historical: obs and raw GCM")
    ax.legend(fontsize=7)

    # (1,1) Scenario: raw GCM and coarse debiased.
    ax = axes[1, 1]
    _series(ax, raw_scenario, color="C2", label="Raw scenario GCM")
    _series(ax, coarse_debiased, color="C3", label="Coarse debiased")
    _envelope(ax)
    ax.set_title("Scenario: raw and coarse debiased")
    ax.legend(fontsize=7)

    for ax in axes.flat:
        ax.set_ylabel(f"{var} ({units})")

    fig.suptitle(f"{scenario}/{var}/{member} at lat={lat}, lon={lon}")
    plt.tight_layout()
    plt.show()
    return int((too_high | too_low).sum())
