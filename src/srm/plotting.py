import matplotlib.pyplot as plt

def plot_comparisons(obs, raw, ds1, ds2=None, bias='absolute'):
    fig, axarr = plt.subplots(figsize=(20,8), nrows=2, ncols=4)
    cax =  obs.plot(ax=axarr[1,0])
    # plot raw and ds1 with same colorbar limits as obs
    raw.plot(ax=axarr[0,0], vmin=cax.get_clim()[0], vmax=cax.get_clim()[1])
    ds1.plot(ax=axarr[0,1], vmin=cax.get_clim()[0], vmax=cax.get_clim()[1])

    if bias=='absolute':
        (ds1-obs).plot(ax=axarr[1,1])
    elif bias=='percentage':
        (((ds1-obs)/obs)*100).plot(ax=axarr[1,1])
    if ds2 is not None:
        ds1.plot(ax=axarr[0,2], vmin=cax.get_clim()[0], vmax=cax.get_clim()[1])
        if bias=='absolute':
            (ds2-obs).plot(ax=axarr[1,2])
            (ds2-ds1).plot(ax=axarr[0,3])
        elif bias=='percentage':
            (((ds2-obs)/obs)*100).plot(ax=axarr[1,2])
            (((ds2-obs)/obs)*100).plot(ax=axarr[0,3])
    axarr[1,3].axis('off')
    plt.tight_layout()

def calculate_statistic_to_plot(raw, era5, ds1, stat, variable, ds2=None):
    # input datasets should be already subset to the time periods of interest
    # this function will collapse them down into a 2d map
    # todo: make this work for the following set of statistics: mean, 99p, 1p, 
    ds_list = [raw, era5, ds1]
    if ds2 is not None:
        ds_list.append(ds2)
    if stat=='mean':
        [raw_toplot, era5_toplot, ds1_toplot, ds2_toplot] = [ds.mean(dim='time')[variable].compute() for ds in ds_list]
    elif stat=='99p':
        [raw_toplot, era5_toplot, ds1_toplot, ds2_toplot] = [ds.quantile(.99, dim='time')[variable].compute() for ds in ds_list]
    return raw_toplot, era5_toplot, ds1_toplot, ds2_toplot
    
def prep_funky_calendar(ds, ds_timeindex_to_match, time_slice):
    ds_subset = ds.sel(time=time_slice)
    # overwrite the time index because some calendars are weird and won't play nice in plotting
    # this will only work if we're not in a leap year! (if we have any 360 day calendar models 
    # we'll have to change this as well)
    ds_subset['time'] = ds_timeindex_to_match.sel(time=time_slice)['time']
    return ds_subset

def sel_point(ds, lat, lon):
    return ds.sel(latitude=lat, longitude=lon, method='nearest')

def prep_datasets_for_daily_timeseries_plotting(ds_list, time_slice, lat, lon, variable):
    era5_subset = sel_point(era5.sel(time=time_slice), lat, lon)
    raw_subset = sel_point(prep_funky_calendar(raw, era5, time_slice), lat, lon)
    ds1_subset = sel_point(prep_funky_calendar(ds1, era5, time_slice), lat, lon)
    if ds2 is not None:
        ds2_subset = sel_point(prep_funky_calendar(ds2, era5, time_slice), lat, lon)
        return era5_toplot, raw_toplot, ds1_toplot, ds2_toplot
    else:
        return era5_toplot, raw_toplot, ds1_toplot

def prep_datasets_for_seasonal_cycle_plotting(era5, raw, ds1, time_slice, lat, lon, variable, ds2=None):

    era5_toplot, raw_toplot, ds1_toplot = [ds[variable].sel(time=time_slice).sel(latitude=lat, longitude=lon, method='nearest').groupby('time.dayofyear').mean() for ds in [era5, raw, ds1]]
    if ds2 is not None:
        ds2_toplot = ds2[variable].sel(time=time_slice).sel(latitude=lat, longitude=lon, method='nearest').groupby('time.dayofyear').mean()
        return era5_toplot, raw_toplot, ds1_toplot, ds2_toplot
    else:
        return era5_toplot, raw_toplot, ds1_toplot



def plot_timeseries(ax, era5_toplot, raw_toplot, ds1_toplot, location, ds2_toplot=None):
    era5_toplot.plot(ax=ax, color='grey', alpha=0.5)
    raw_toplot.plot(ax=ax, color='k')
    ds1_toplot.plot(ax=ax, color='firebrick')
    if ds2_toplot is not None:
        ds2_toplot.plot(ax=ax, color='royalblue')
    ax.set_title(location)


def plot_pdf(era5, raw, ds1, var, ds2=ds2, title=None, xlabel=None):
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