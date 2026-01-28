import icechunk
import numpy as np
import pandas as pd
import rasterix  # noqa: F401  # side-effect import: registers .proj/.rio accessors
import xarray as xr
import xarray_regrid  # noqa: F401  # side-effect import: registers .regrid namespace

from srm import catalog


def subset_space(da: xr.DataArray, coord_bounds_list: list):
    [lat_min, lat_max, lon_min, lon_max] = coord_bounds_list
    da_subset = da.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max),
    )
    return da_subset


def subset_time(
    da: xr.DataArray,
    train_period_start: int = 1978,
    train_period_end: int = 2014,
    predict_period_start: int = 2015,
    predict_period_end: int = 2100,
    time_period: str = "train",
):
    if time_period == "train":
        start_year = train_period_start
        end_year = train_period_end
    elif time_period == "predict":
        start_year = predict_period_start
        end_year = predict_period_end
    da = da.sel(time=slice(f"{start_year}", f"{end_year}"))
    return da


def rechunk(da: xr.DataArray, pattern: str):
    if pattern == "full_space":
        da_rechunk = da.chunk(time=5, lat=-1, lon=-1)
    elif pattern == "full_time":
        da_rechunk = da.chunk(time=-1, lat=7, lon=14)

    return da_rechunk


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


def calculate_baseline_climatology(
    da_baseline: xr.DataArray,
    baseline_period_start: int = 1978,
    baseline_period_end: int = 2014,
):
    da_baseline = da_baseline.drop_vars("spatial_ref", errors="ignore")
    da_baseline = da_baseline.sel(time=slice(f"{baseline_period_start}", f"{baseline_period_end}"))
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


def retrend(
    bias_corrected_detrended: xr.DataArray,
    trend_on_daily_timestep: xr.DataArray,
    detrending="additive",
):
    valid_values = ["additive"]
    if detrending not in valid_values:
        raise ValueError(
            f"{detrending} is currently not supported. valid values are: {valid_values}"
        )

    if detrending == "additive":
        retrended = bias_corrected_detrended + trend_on_daily_timestep

    return retrended


def interpolate_fine_to_coarse_grid(da_fine_to_coarsen: xr.DataArray, da_coarse_grid: xr.DataArray):
    da_fine_to_coarsen = da_fine_to_coarsen.persist()

    # `.regrid` namespace comes from xarray_regrid; assumes rectilinear, which is same as NCL and good enough for us.
    da_coarse = da_fine_to_coarsen.regrid.conservative(da_coarse_grid).as_numpy().persist()

    return da_coarse


def interpolate_coarse_to_fine_grid(da_coarse_to_regrid: xr.DataArray, da_fine_grid: xr.DataArray):
    coarse_on_fine_grid = da_coarse_to_regrid.interp(
        lon=da_fine_grid["lon"],
        lat=da_fine_grid["lat"],
        method="linear",
    )

    return coarse_on_fine_grid


def fft_smooth_3harmonics(data):
    """Apply FFT and retain only mean + 3 harmonics"""
    # Handle NaN values
    if np.all(np.isnan(data)):
        return data

    # Compute FFT
    Z = np.fft.fft(data)

    # Create filtered version: keep mean (0) + first 3 harmonics (1,2,3 and -3,-2,-1)
    Z_filtered = np.zeros_like(Z)
    Z_filtered[0] = Z[0]  # mean (DC component)
    Z_filtered[1:4] = Z[1:4]  # positive frequencies (harmonics 1-3)
    Z_filtered[-3:] = Z[-3:]  # negative frequencies (harmonics 1-3)

    # Inverse FFT to get smoothed time series
    smoothed = np.real(np.fft.ifft(Z_filtered))

    return smoothed


def calculate_doy_means(ds, clim_method: str = "simple"):
    """
    Calculate the daily climatology of high-res observations.
    """
    ds_xr = xr.DataArray(
        ds.data,
        dims=ds.dims,
        coords={k: v for k, v in ds.coords.items() if k != "spatial_ref"},
    )

    ds_xr = ds_xr.assign_coords(time=("time", pd.to_datetime(ds["time"].values)))

    ds_xr_doy_mean = ds_xr.groupby("time.dayofyear").mean("time")

    if clim_method == "simple":
        doy_means = ds_xr_doy_mean
    elif clim_method == "fft":
        # Apply FFT smoothing along the time dimension
        obs_fine_doy_means_smoothed = xr.apply_ufunc(
            fft_smooth_3harmonics,
            ds_xr_doy_mean.load(),
            input_core_dims=[["dayofyear"]],
            output_core_dims=[["dayofyear"]],
            vectorize=True,
            # dask='parallelized',
            output_dtypes=[float],
        )

        obs_fine_doy_means_smoothed = obs_fine_doy_means_smoothed.transpose(
            "dayofyear", "lat", "lon"
        )

        doy_means = obs_fine_doy_means_smoothed
    return doy_means


def downscale_from_coarse(
    da: xr.DataArray,
    obs_coarse: xr.DataArray,
    obs_fine: xr.DataArray,
    method: str = "subtract",
    clim_method: str = "simple",
):
    valid_clim_methods = ["simple", "fft"]
    if clim_method not in valid_clim_methods:
        raise ValueError(
            f"{method} is currently not supported. valid values are: {valid_clim_methods}"
        )

    # Step 1: calculate the daily climatology of high-res observations
    obs_fine_doy_means = calculate_doy_means(obs_fine, clim_method=clim_method)

    # Step 2: Aggregate daily climatology to the low-resolution grid of the GCM being processed
    obs_coarse_doy_means = interpolate_fine_to_coarse_grid(
        da_fine_to_coarsen=obs_fine_doy_means, da_coarse_grid=obs_coarse
    )

    # Step 3: Remove coarsened daily climatology from the bias-corrected fields
    valid_values = ["subtract", "divide"]
    if method not in valid_values:
        raise ValueError(f"{method} is currently not supported. valid values are: {valid_values}")

    if method == "subtract":
        residuals = da.groupby("time.dayofyear") - obs_coarse_doy_means
    elif method == "divide":
        residuals = da.groupby("time.dayofyear") / obs_coarse_doy_means

    # Step 4: Bilinearly interpolate residuals to the high-res grid
    residuals_fine = interpolate_coarse_to_fine_grid(
        da_coarse_to_regrid=residuals, da_fine_grid=obs_fine
    )

    # Step 5: Return high-res climatology
    if method == "subtract":
        downscaled = residuals_fine.groupby("time.dayofyear") + obs_fine_doy_means
    elif method == "divide":
        downscaled = residuals_fine.groupby("time.dayofyear") * obs_fine_doy_means

    return downscaled
