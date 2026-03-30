import typing

import icechunk
import icechunk.xarray
import numpy as np
import rasterix  # noqa: F401  # side-effect import: registers .proj/.rio accessors
import xarray as xr
import xarray_regrid  # noqa: F401  # side-effect import: registers .regrid namespace

from srm import catalog
from srm.bcsd_config import DetrendMethod, DownscalingClimMethod, DownscalingMethod


def subset_space(
    da: xr.DataArray,
    coord_bounds_list: typing.Sequence[float] | None = None,
    *,
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
) -> xr.DataArray:
    """
    Subset a DataArray to a latitude/longitude bounding box.

    Parameters
    ----------
    da : xr.DataArray
        Input array with ``lat`` and ``lon`` coordinates.
    coord_bounds_list : sequence[float] | None, optional
        Legacy positional bounds in the order
        ``[lat_min, lat_max, lon_min, lon_max]``.
    lat_bounds : tuple[float, float] | None, optional
        Latitude bounds as ``(lat_min, lat_max)``.
    lon_bounds : tuple[float, float] | None, optional
        Longitude bounds as ``(lon_min, lon_max)``.

    Returns
    -------
    xr.DataArray
        Spatially subsetted array.

    Raises
    ------
    ValueError
        If both legacy and named bounds are provided, required bounds are missing,
        the legacy bounds are not length 4, or min/max ordering is invalid.
    """
    using_legacy_bounds = coord_bounds_list is not None
    using_named_bounds = lat_bounds is not None or lon_bounds is not None

    if using_legacy_bounds and using_named_bounds:
        raise ValueError("Provide either coord_bounds_list or lat_bounds/lon_bounds, not both.")

    if using_legacy_bounds:
        if len(coord_bounds_list) != 4:
            raise ValueError(
                "coord_bounds_list must contain four values in order: "
                "[lat_min, lat_max, lon_min, lon_max]."
            )
        lat_min, lat_max, lon_min, lon_max = coord_bounds_list
        lat_bounds = (lat_min, lat_max)
        lon_bounds = (lon_min, lon_max)
    else:
        if lat_bounds is None or lon_bounds is None:
            raise ValueError(
                "Provide both lat_bounds and lon_bounds when coord_bounds_list is not used."
            )

    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    if lat_min >= lat_max:
        raise ValueError(f"lat_bounds must be (min, max) with min < max, got {lat_bounds}.")
    if lon_min >= lon_max:
        raise ValueError(f"lon_bounds must be (min, max) with min < max, got {lon_bounds}.")

    return da.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max),
    )


_TARGET_CHUNK_BYTES = 100 * 1024 * 1024  # 100 MB


def rechunk(da: xr.DataArray, pattern: typing.Literal["full_space", "full_time"]) -> xr.DataArray:
    """
    Rechunk a gridded DataArray for common BCSD workflow access patterns.

    Parameters
    ----------
    da : xr.DataArray
        Input array with ``time``, ``lat``, and ``lon`` dimensions.
    pattern : {"full_space", "full_time"}
        Target chunk layout:
        - ``"full_space"``: chunk across time while keeping full lat/lon in each chunk.
        - ``"full_time"``: keep full time in each chunk while splitting lat/lon chunks.

    Returns
    -------
    xr.DataArray
        Rechunked array. If the current chunking already matches the requested
        pattern, the input is returned unchanged.
    """
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
    return da


def get_experiment(
    gcm: str,
    scenario: str,
    var: str,
    coord_bounds_list: list | None = None,
    ensemble_member: str | None = None,
):
    """
    Load in a GCM simulation.

    Parameters
    ----------
    gcm : str
        Name of the GCM, e.g. "CESM2-WACCM"
    scenario : str
        Scenario of experiment, e.g. "SSP245"
    var : str
        Variable to load, e.g. "tas"

    Returns
    -------
    xr.DataArray
        Xarray data array for requested simulation

    Raises
    ------
    ValueError
        If invalid ensemble member requested

    """
    cat_name = gcm + "-" + scenario + "-icechunk"
    dataset = catalog.get(cat_name)
    # confirm that the requested ensemble member is available
    if ensemble_member is not None and dataset.ensemble_members is not None:
        if ensemble_member not in dataset.ensemble_members:
            raise ValueError(
                f"Invalid ensemble_member '{ensemble_member}' for '{cat_name}'. "
                f"Valid options: {dataset.ensemble_members}"
            )

    ds_scenario = dataset.to_xarray()

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
    """
    Compute monthly baseline climatology over a selected time period.

    Parameters
    ----------
    da_baseline : xr.DataArray
        Input time series with a ``time`` coordinate.
    baseline_period_start : int, default: 1978
        Inclusive start year for the climatology window.
    baseline_period_end : int, default: 2014
        Inclusive end year for the climatology window.

    Returns
    -------
    xr.DataArray
        Monthly climatology with ``month`` coordinate (1-12).

    Notes
    -----
    ``groupby(...).mean()`` can promote values (for example ``float32`` to
    ``float64``). We cast back to the original dtype to keep memory usage and
    downstream dtype expectations consistent across the workflow.
    """
    da_baseline = da_baseline.drop_vars("spatial_ref", errors="ignore")
    da_baseline = da_baseline.sel(time=slice(f"{baseline_period_start}", f"{baseline_period_end}"))
    da_baseline_clim = da_baseline.groupby("time.month").mean(dim="time")

    # Keep output dtype stable (mean can upcast float32 -> float64).
    return da_baseline_clim.astype(da_baseline.dtype)


def detrend(
    da: xr.DataArray,
    da_baseline_clim: xr.DataArray,
    detrend_method: DetrendMethod = "additive",
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Remove a smoothed monthly trend from a daily time series.

    The trend is estimated by:
    1. computing monthly means,
    2. applying a 9-year rolling mean within each calendar month, and
    3. expressing that trend relative to baseline monthly climatology.

    Parameters
    ----------
    da : xr.DataArray
        Daily (or higher-frequency) input time series with a ``time`` coordinate.
    da_baseline_clim : xr.DataArray
        Monthly climatology indexed by ``month`` (1-12), typically from
        :func:`calculate_baseline_climatology`.
    detrend_method : {"additive", "multiplicative"}, default: "additive"
        Trend-removal method:
        - ``"additive"`` subtracts the trend signal.
        - ``"multiplicative"`` divides by the trend signal.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        ``(detrended, trend_on_daily_timestep)`` where the trend has been
        expanded back to daily resolution and aligned to ``da.time``.
    """
    # Calculate monthly averages
    da_mon = da.resample(time="1MS").mean("time")
    # Keep chunks contiguous in time for rolling/groupby operations while
    # avoiding very small chunks that create excessive Dask task overhead.
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

    # Translate the monthly trend timeseries into a daily timeseries
    # where every day in that month is the same value. This will produce
    # jumps from month to month (for example, if January was high but February
    # was low, it would go from a positive adjustment for january 31 (and entire month before) to a negative
    #  adjustment for february 1 (and the entire month after). thus, there could be noticeable
    # artificial discontinuities inserted into the timeseries between 1/31 and 2/1.
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
    detrend_method: DetrendMethod = "additive",
) -> xr.DataArray:
    """
    Reincorporate a previously removed trend into a detrended time series.

    Parameters
    ----------
    bias_corrected_detrended : xr.DataArray
        Detrended (and typically bias-corrected) data.
    trend_on_daily_timestep : xr.DataArray
        Trend term aligned to the same daily time axis as
        ``bias_corrected_detrended``.
    detrend_method : {"additive", "multiplicative"}, default: "additive"
        Method used during detrending:
        - ``"additive"`` adds trend back.
        - ``"multiplicative"`` multiplies trend back.

    Returns
    -------
    xr.DataArray
        Retrended time series on the original scale.

    Raises
    ------
    ValueError
        If ``detrend_method`` is not one of ``"additive"`` or
        ``"multiplicative"``.
    """
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
    # ensure da_coarse_grid consists of only lat/lon coordinates and a single time step (if time coordinate exists) to avoid issues with xarray_regrid
    target_grid = da_coarse_grid.reset_coords(drop=True)
    if "time" in target_grid.coords:
        target_grid = target_grid.isel(time=[0])
    # use conservative remapping.
    da_coarse = da_fine_to_coarsen.regrid.conservative(target_grid, latitude_coord="lat")
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
    da: xr.DataArray, clim_method: DownscalingClimMethod = "simple"
) -> xr.DataArray:
    """
    Calculate the daily climatology of high-res observations.
    """

    da_xr_doy_mean = da.groupby("time.dayofyear").mean("time")

    if clim_method == "simple":
        return da_xr_doy_mean
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

    return obs_fine_doy_means_smoothed


def downscale_from_coarse(
    da: xr.DataArray,
    obs_coarse: xr.DataArray,
    obs_fine: xr.DataArray,
    method: DownscalingMethod = "additive",
    clim_method: DownscalingClimMethod = "simple",
) -> xr.DataArray:
    # Step 1: calculate the daily climatology of high-res observations
    obs_fine_doy_means = calculate_doy_means(obs_fine, clim_method=clim_method)

    # Step 2: Aggregate daily climatology to the low-resolution grid of the GCM being processed
    obs_coarse_doy_means = interpolate_fine_to_coarse_grid(
        da_fine_to_coarsen=obs_fine_doy_means, da_coarse_grid=obs_coarse
    )

    # Step 3: Remove coarsened daily climatology from the bias-corrected fields
    if method == "additive":
        residuals = da.groupby("time.dayofyear") - obs_coarse_doy_means
    elif method == "multiplicative":
        residuals = da.groupby("time.dayofyear") / obs_coarse_doy_means

    # Step 4: Bilinearly interpolate residuals to the high-res grid
    # this creates a smooth layer of how different the particular simulated february 10 is
    # from the average february 10.
    residuals_fine = interpolate_coarse_to_fine_grid(
        da_coarse_to_regrid=residuals, da_fine_grid=obs_fine
    )

    # Step 5: Return high-res climatology
    # Add or multiply a constant value to the residuals based on DOY
    # this step adds back in the day-of-year spatial texture saying,
    # "let's combine (a) how different February 10 is from the typical February 10 at the coarse scale
    # with the typical spatial structure of February 10"
    if method == "additive":
        downscaled = residuals_fine.groupby("time.dayofyear") + obs_fine_doy_means
    elif method == "multiplicative":
        downscaled = residuals_fine.groupby("time.dayofyear") * obs_fine_doy_means

    return downscaled


# TODO: this is currently unused. we should delete it
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


# TODO: this is currently unused. we should delete it
def get_output_data(
    fname_key: str,
    dtree_key: str = "data.zarr/",
    s3_bucket: str = "s3://carbonplan-scratch/",
    prefix: str = "srm-scratch/v0.3_SouthAfrica/",
):
    dt = xr.open_datatree(s3_bucket + prefix + dtree_key, engine="zarr", chunks={})
    ds = dt[fname_key]

    return ds
