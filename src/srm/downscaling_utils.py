import icechunk
import xarray as xr
import pandas as pd
import rasterix  # noqa: F401  # side-effect import: registers .proj/.rio accessors

from srm import catalog


def get_experiment(
    gcm: str = "CESM2-WACCM",
    scenario: str = "SSP245",
    var: str = "tas",
    coord_bounds_list: list = None,
):
    cat_name = gcm + "-" + scenario + "-icechunk"
    scenario_cat = catalog.get(cat_name)

    scenario_storage = icechunk.s3_storage(
        bucket=scenario_cat.bucket,
        prefix=scenario_cat.prefix,
        from_env=True,
    )

    scenario_repo = icechunk.Repository.open(scenario_storage)
    scenario_session = scenario_repo.readonly_session("main")

    ds_scenario = xr.open_zarr(scenario_session.store, consolidated=False)

    ds_scenario = ds_scenario.proj.assign_crs(spatial_ref="epsg:4326")

    da = ds_scenario[var]

    if coord_bounds_list is not None:
        print("Subsetting spatial domain")
        da = subset_space(da, coord_bounds_list)

    return da


def get_obs(var: str = "tas", coord_bounds_list: list = None):
    era5_cat = catalog.get("ERA5")

    era5_storage = icechunk.s3_storage(
        bucket=era5_cat.bucket,
        prefix=era5_cat.prefix,
        from_env=True,
    )
    era5_repo = icechunk.Repository.open(era5_storage)
    era5_session = era5_repo.readonly_session("main")
    era5 = xr.open_zarr(era5_session.store)
    era5 = era5.proj.assign_crs(spatial_ref="epsg:4326")

    da = era5[var]

    if coord_bounds_list is not None:
        da = subset_space(da, coord_bounds_list)
    return da


def subset_space(da: xr.DataArray, coord_bounds_list: list):
    [lat_min, lat_max, lon_min, lon_max] = coord_bounds_list
    da_subset = da.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max),
    )
    return da_subset


def subset_time(da: xr.DataArray, run_parameters: dict, time_period: str = "train"):
    if time_period == "train":
        start_year = run_parameters["TRAIN_PERIOD_START"]
        end_year = run_parameters["TRAIN_PERIOD_END"]

    elif time_period == "predict":
        start_year = run_parameters["PREDICT_PERIOD_START"]
        end_year = run_parameters["PREDICT_PERIOD_END"]

    da = da.where(da["time.year"] >= start_year, drop=True)
    da = da.where(da["time.year"] <= end_year, drop=True)

    return da


def rechunk(da: xr.DataArray, pattern: str):
    if pattern == "full_space":
        da_rechunk = da.chunk(time=5, lat=-1, lon=-1)
    elif pattern == "full_time":
        da_rechunk = da.chunk(time=-1, lat=7, lon=14)

    return da_rechunk


def calculate_baseline_climatology(
    da_baseline: xr.DataArray,
    baseline_period_start: int = 1978,
    baseline_period_end: int = 2014,
):
    da_baseline = da_baseline.drop_vars("spatial_ref", errors="ignore")
    da_baseline = da_baseline.where(da_baseline['time.year']>=baseline_period_start)
    da_baseline = da_baseline.where(da_baseline['time.year']<=baseline_period_end)
    da_baseline_clim = da_baseline.groupby("time.month").mean(dim="time")

    return da_baseline_clim


def detrend(da: xr.DataArray, da_baseline_clim: xr.DataArray):
    # Calculate monthly averages
    da_mon = da.resample(time="1MS").mean("time")
    da_mon = da_mon.chunk({"time": 120})

    # Group by month
    g = da_mon.groupby("time.month")

    # Apply a 9-year rolling mean within each month group
    da_mon_avg = g.map(lambda x: x.rolling(time=9, center=True, min_periods=1).mean())

    da_mon_trend = da_mon_avg.groupby("time.month").map(
        lambda x: x - da_baseline_clim.sel(month=x["time.month"][0].item())
    )

    # Project that monthly trend onto the daily timestep
    trend_on_daily_timestep = (
        da_mon_trend.resample(time="1D").ffill().reindex(time=da.time).ffill(dim="time")
    ).compute()

    # Calculate detrended timeseries
    detrended = da - trend_on_daily_timestep

    return detrended, trend_on_daily_timestep

def retrend(bias_corrected_detrended: xr.DataArray,
           trend_on_daily_timestep: xr.DataArray,
           detrending='additive'):
    
    if detrending=='additive':
        retrended= bias_corrected_detrended + trend_on_daily_timestep

    return retrended


def interpolate_to_coarse_grid(
    da_fine_to_coarsen: xr.DataArray, da_coarse_grid: xr.DataArray
):
    da_fine_to_coarsen = da_fine_to_coarsen.persist()

    da_coarse = da_fine_to_coarsen.interp(
        lon=da_coarse_grid.lon,
        lat=da_coarse_grid.lat,
        method="linear",
    )

    return da_coarse


def calculate_error_map(obs_coarse: xr.DataArray, obs_fine: xr.DataArray):
    def calculate_doy_means(ds):
        ds_xr = xr.DataArray(
            ds.data,
            dims=ds.dims,
            coords={k: v for k, v in ds.coords.items() if k != "spatial_ref"},
        )

        ds_xr = ds_xr.assign_coords(time=("time", pd.to_datetime(ds["time"].values)))

        ds_xr_doy_mean = ds_xr.groupby("time.dayofyear").mean("time")

        return ds_xr_doy_mean

    obs_coarse_on_fine_grid = obs_coarse.interp(
        lon=obs_fine["lon"],
        lat=obs_fine["lat"],
        method="linear",
    )
    error_map = obs_fine.mean(dim="time") - obs_coarse_on_fine_grid.mean(dim="time")

    obs_fine_doy_means = calculate_doy_means(obs_fine)

    obs_coarse_on_fine_grid_doy_means = calculate_doy_means(obs_coarse_on_fine_grid)

    error_map = obs_fine_doy_means - obs_coarse_on_fine_grid_doy_means

    return error_map


def downscale_from_coarse(
    da: xr.DataArray, error_map: xr.DataArray, fine_grid: xr.DataArray
):
    da_fine_grid = da.interp(
        lon=fine_grid["lon"],
        lat=fine_grid["lat"],
        method="linear",
    )

    return da_fine_grid.groupby("time.dayofyear") + error_map
