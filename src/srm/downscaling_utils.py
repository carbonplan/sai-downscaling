import typing

import dask.base
import icechunk
import icechunk.xarray
import numpy as np
import rasterix  # noqa: F401  # side-effect import: registers .proj/.rio accessors
import xarray as xr
import xarray_regrid  # noqa: F401  # side-effect import: registers .regrid namespace

from srm import catalog


def subset_space(da: xr.DataArray, coord_bounds_list: list) -> xr.DataArray:
    [lat_min, lat_max, lon_min, lon_max] = coord_bounds_list
    da_subset = da.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max),
    )
    return da_subset


_TARGET_CHUNK_BYTES = 100 * 1024 * 1024  # 100 MB


def rechunk(da: xr.DataArray, pattern: typing.Literal["full_space", "full_time"]) -> xr.DataArray:
    if pattern == "full_space":
        already_chunked = (
            "time" in da.chunksizes
            and len(da.chunksizes["time"]) > 1
            and "lat" in da.chunksizes
            and len(da.chunksizes["lat"]) == 1
            and "lon" in da.chunksizes
            and len(da.chunksizes["lon"]) == 1
        )
        if not already_chunked:
            n_lat = da.sizes["lat"]
            n_lon = da.sizes["lon"]
            time_chunk = max(1, int(_TARGET_CHUNK_BYTES / (n_lat * n_lon * da.dtype.itemsize)))
            da = da.chunk(time=time_chunk, lat=-1, lon=-1)
    elif pattern == "full_time":
        already_chunked = (
            "time" in da.chunksizes
            and len(da.chunksizes["time"]) == 1
            and "lat" in da.chunksizes
            and len(da.chunksizes["lat"]) > 1
            and "lon" in da.chunksizes
            and len(da.chunksizes["lon"]) > 1
        )
        if not already_chunked:
            n_time = da.sizes["time"]
            n_lat = da.sizes["lat"]
            n_lon = da.sizes["lon"]
            total_spatial = _TARGET_CHUNK_BYTES / (n_time * da.dtype.itemsize)
            # split spatial pixels proportionally to preserve the lat/lon aspect ratio
            lat_chunk = max(1, int(np.sqrt(total_spatial * n_lat / n_lon)))
            lon_chunk = max(1, int(np.sqrt(total_spatial * n_lon / n_lat)))
            da = da.chunk(time=-1, lat=lat_chunk, lon=lon_chunk)
    da_rechunk = dask.base.optimize(da)[0]
    return da_rechunk


def get_experiment(
    gcm: str = "CESM2-WACCM",
    scenario: str = "SSP245",
    var: str = "tas",
    coord_bounds_list: list | None = None,
):
    cat_name = gcm + "-" + scenario + "-icechunk"
    ds_scenario = catalog.get(cat_name).to_xarray()

    ds_scenario = ds_scenario.proj.assign_crs(spatial_ref="epsg:4326")

    da = ds_scenario[var]

    if coord_bounds_list is not None:
        print("Subsetting spatial domain")
        da = subset_space(da, coord_bounds_list)

    return da


def get_obs(var: str = "tas", coord_bounds_list: list | None = None):
    era5 = catalog.get("ERA5").to_xarray()
    era5 = era5.proj.assign_crs(spatial_ref="epsg:4326")

    da = era5[var]

    if coord_bounds_list is not None:
        da = subset_space(da, coord_bounds_list)
    return da


def calculate_baseline_climatology(
    da_baseline: xr.DataArray,
    baseline_period_start: int = 1978,
    baseline_period_end: int = 2014,
) -> xr.DataArray:
    da_baseline = da_baseline.drop_vars("spatial_ref", errors="ignore")
    da_baseline = da_baseline.sel(time=slice(f"{baseline_period_start}", f"{baseline_period_end}"))
    da_baseline_clim = da_baseline.groupby("time.month").mean(dim="time")

    return da_baseline_clim.astype(da_baseline.dtype)


def detrend(
    da: xr.DataArray,
    da_baseline_clim: xr.DataArray,
    detrend_method: typing.Literal["additive", "multiplicative"] = "additive",
) -> tuple[xr.DataArray, xr.DataArray]:
    valid_values = ["additive", "multiplicative"]
    if detrend_method not in valid_values:
        raise ValueError(
            f"{detrend_method} is currently not supported. valid values are: {valid_values}"
        )
    # Calculate monthly averages
    da_mon = da.resample(time="1MS").mean("time")
    da_mon = da_mon.chunk({"time": 120})

    # Group by month
    g = da_mon.groupby("time.month")

    # Apply a 9-year rolling mean within each month group
    da_mon_avg = g.map(lambda x: x.rolling(time=9, center=True, min_periods=1).mean())

    if detrend_method == "additive":
        da_mon_trend = da_mon_avg.groupby("time.month").map(
            lambda x: x - da_baseline_clim.sel(month=x["time.month"][0].item())
        )
    elif detrend_method == "multiplicative":
        da_mon_trend = da_mon_avg.groupby("time.month").map(
            lambda x: x / da_baseline_clim.sel(month=x["time.month"][0].item())
        )

    # Project that monthly trend onto the daily timestep
    trend_on_daily_timestep = (
        da_mon_trend.resample(time="1D").ffill().reindex(time=da.time).ffill(dim="time")
    ).compute()

    # Calculate detrended timeseries
    if detrend_method == "additive":
        detrended = da - trend_on_daily_timestep
    elif detrend_method == "multiplicative":
        detrended = da / trend_on_daily_timestep

    return detrended.astype(da.dtype), trend_on_daily_timestep.astype(da.dtype)


def retrend(
    bias_corrected_detrended: xr.DataArray,
    trend_on_daily_timestep: xr.DataArray,
    detrend_method: typing.Literal["additive", "multiplicative"] = "additive",
) -> xr.DataArray:
    valid_values = ["additive", "multiplicative"]
    if detrend_method not in valid_values:
        raise ValueError(
            f"{detrend_method} is currently not supported. valid values are: {valid_values}"
        )

    if detrend_method == "additive":
        retrended = bias_corrected_detrended + trend_on_daily_timestep
    elif detrend_method == "multiplicative":
        retrended = bias_corrected_detrended * trend_on_daily_timestep

    return retrended


def interpolate_fine_to_coarse_grid(
    da_fine_to_coarsen: xr.DataArray, da_coarse_grid: xr.DataArray
) -> xr.DataArray:
    # `.regrid` namespace comes from xarray_regrid; assumes rectilinear, which is same as NCL and good enough for us.
    da_coarse = da_fine_to_coarsen.regrid.conservative(da_coarse_grid).as_numpy().persist()

    return da_coarse.astype(da_fine_to_coarsen.dtype)


def interpolate_coarse_to_fine_grid(
    da_coarse_to_regrid: xr.DataArray, da_fine_grid: xr.DataArray
) -> xr.DataArray:
    # Using slinear instead of linear because linear can produce very small negative numbers even when input dataset is all positive
    coarse_on_fine_grid = da_coarse_to_regrid.interp(
        lon=da_fine_grid["lon"],
        lat=da_fine_grid["lat"],
        method="slinear",
    )

    return coarse_on_fine_grid.astype(da_coarse_to_regrid.dtype)


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
    smoothed = np.real(np.fft.ifft(Z_filtered)).astype(data.dtype)

    return smoothed


def calculate_doy_means(
    da: xr.DataArray, clim_method: typing.Literal["simple", "fft"] = "simple"
) -> xr.DataArray:
    """
    Calculate the daily climatology of high-res observations.
    """

    da_xr_doy_mean = da.groupby("time.dayofyear").mean("time")

    if clim_method == "simple":
        doy_means = da_xr_doy_mean
    elif clim_method == "fft":
        # Apply FFT smoothing along the time dimension
        obs_fine_doy_means_smoothed = xr.apply_ufunc(
            fft_smooth_3harmonics,
            da_xr_doy_mean.load(),
            input_core_dims=[["dayofyear"]],
            output_core_dims=[["dayofyear"]],
            vectorize=True,
            # dask='parallelized',
            output_dtypes=[da.dtype],
        )

        # transpose from ["lat", "lon", "dayofyear"] to original order of ["dayofyear", "lat", "lon"]
        obs_fine_doy_means_smoothed = obs_fine_doy_means_smoothed.transpose(
            "dayofyear", "lat", "lon"
        )

        doy_means = obs_fine_doy_means_smoothed
    return doy_means


def downscale_from_coarse(
    da: xr.DataArray,
    obs_coarse: xr.DataArray,
    obs_fine: xr.DataArray,
    method: typing.Literal["additive", "multiplicative"] = "additive",
    clim_method: typing.Literal["simple", "fft"] = "simple",
) -> xr.DataArray:
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
    valid_values = ["additive", "multiplicative"]
    if method not in valid_values:
        raise ValueError(f"{method} is currently not supported. valid values are: {valid_values}")

    if method == "additive":
        residuals = da.groupby("time.dayofyear") - obs_coarse_doy_means
    elif method == "multiplicative":
        residuals = da.groupby("time.dayofyear") / obs_coarse_doy_means

    # Step 4: Bilinearly interpolate residuals to the high-res grid
    residuals_fine = interpolate_coarse_to_fine_grid(
        da_coarse_to_regrid=residuals, da_fine_grid=obs_fine
    )

    # Step 5: Return high-res climatology
    # Add or multiply a constant value to the residuals based on DOY
    if method == "additive":
        downscaled = residuals_fine.groupby("time.dayofyear") + obs_fine_doy_means
    elif method == "multiplicative":
        downscaled = residuals_fine.groupby("time.dayofyear") * obs_fine_doy_means

    return downscaled


def save_data(
    dict_data: dict,
    fname_key: str,
    var_name: str,
    output_suffix: str = "zarr",
    s3_bucket: str = "s3://carbonplan-scratch/",
    prefix: str = "srm-scratch/v0.3_SouthAfrica/",
    print_fpath: bool = True,
    chunks: dict = {"time": "100MB", "lat": -1, "lon": -1},
):
    s3_path = f"{s3_bucket + prefix}{fname_key}.{output_suffix}"

    if print_fpath:
        print(f"Saving to {s3_path}")

    dict_dsets = {key: dict_data[key].to_dataset(name=var_name) for key in dict_data}
    # clear encoding to avoid issues when saving
    for ds in dict_dsets.values():
        for var in ds.data_vars:
            ds[var].encoding = {}
    datatree = xr.DataTree.from_dict(dict_dsets)

    if output_suffix == "zarr":
        # instead of chunking here, we could let the user specify chunking earlier?
        if chunks is not None:
            datatree = datatree.chunk(chunks)
        datatree.to_zarr(s3_path, mode="w")

    elif output_suffix == "nc":
        datatree.to_netcdf(s3_path)

    elif output_suffix == "icechunk":
        # Option 2: save as icechunk. Example:
        storage = icechunk.s3_storage(
            bucket=s3_bucket,
            prefix=prefix + fname_key + ".icechunk",
            from_env=True,
        )
        repo = icechunk.Repository.create(storage)

        session = repo.writable_session("main")

        icechunk.xarray.to_icechunk(datatree, session)
        session.commit("write data")

    else:
        raise ValueError("Invalid output format. Please choose 'zarr' or 'netcdf'.")


def get_output_data(
    fname_key: str,
    dtree_key: str = "data.zarr/",
    s3_bucket: str = "s3://carbonplan-scratch/",
    prefix: str = "srm-scratch/v0.3_SouthAfrica/",
):
    dt = xr.open_datatree(s3_bucket + prefix + dtree_key, engine="zarr", chunks={})
    ds = dt[fname_key]

    return ds
