"""
Pure functions for spatial and statistical downscaling operations.

Contains the core algorithmic building blocks used by
:class:`~srm.pipeline.BCSDPipeline`: detrending and retrend operations, spatial
interpolation between coarse and fine grids, FFT climatology smoothing, and helpers
for loading observation and GCM datasets.
"""

import typing

import numpy as np
import rasterix  # noqa: F401  # side-effect import: registers .proj/.rio accessors
import xarray as xr
import xarray_regrid  # noqa: F401  # side-effect import: registers .regrid namespace

from srm.bcsd_config import DetrendMethod, DownscalingClimMethod, DownscalingMethod
from srm.config import SCENARIO_TO_GROUP
from srm.datasets import catalog
from srm.qa_checks import assert_no_nans
from srm.utils import get_variable

_dt_cache: dict[str, xr.DataTree] = {}


def _gcm_datatree(gcm: str) -> xr.DataTree:
    if gcm not in _dt_cache:
        _dt_cache[gcm] = catalog.get(gcm).to_xarray()
    return _dt_cache[gcm]


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
        time_chunks = da.chunksizes.get("time", ())
        # zarr requires last chunk ≤ first; inherited concat chunks can violate this
        # (e.g. ssp-bridge piece 7305 + g6 piece 18250 after predict-period slice)
        zarr_valid = len(time_chunks) <= 1 or time_chunks[-1] <= time_chunks[0]
        already_chunked = (
            "time" in da.chunksizes
            and len(time_chunks) > 1
            and zarr_valid
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
    Load a GCM simulation variable from the unified per-GCM icechunk store.

    Parameters
    ----------
    gcm : str
        Name of the GCM, e.g. "CESM2-WACCM"
    scenario : str
        Scenario of experiment, e.g. "SSP245"
    var : str
        Variable to load, e.g. "tas"
    coord_bounds_list : list, optional
        Spatial subset bounds [lat_min, lat_max, lon_min, lon_max].
    ensemble_member : str, optional
        Ensemble member to select.

    Returns
    -------
    xr.DataArray
        Xarray data array for the requested simulation.
    """
    group = SCENARIO_TO_GROUP[scenario]
    ds = _gcm_datatree(gcm)[group].to_dataset()
    ds = ds.proj.assign_crs(spatial_ref="epsg:4326")
    da = get_variable(ds, var)
    if ensemble_member is not None:
        da = da.sel(ensemble_member=ensemble_member)
    if coord_bounds_list is not None:
        da = subset_space(da, coord_bounds_list)
    return da


def get_historical_experiment(gcm: str, member: str, var: str) -> xr.DataArray:
    """Load a single historical ensemble member from the unified per-GCM icechunk store.

    The unified historical group merges both NCAR and pangeo member families for CESM2-WACCM,
    so no per-member routing is needed.
    """
    ds = _gcm_datatree(gcm)["historical"].to_dataset()
    ds = ds.proj.assign_crs(spatial_ref="epsg:4326")
    return get_variable(ds, var).sel(ensemble_member=member)


def get_obs(var: str = "tas", coord_bounds_list: list | None = None, dataset_name: str = "ERA5"):
    obs = catalog.get(dataset_name).to_xarray()
    obs = obs.proj.assign_crs(spatial_ref="epsg:4326")

    da = get_variable(obs, var)

    if coord_bounds_list is not None:
        da = subset_space(da, coord_bounds_list)
    return da


def select_training_window(
    da: xr.DataArray, train_period_start: int, train_period_end: int
) -> xr.DataArray:
    """Narrow a series to the bias-correction reference window.

    This is the single point where the quantile-mapping reference pool is established.
    Callers hand over whatever record they hold and this function decides the window,
    so no call site can widen the pool (issue #518).

    The scenario loader deliberately returns the historical record from
    ``train_period_start`` onward, because ``stitch_historical_scenario`` needs the
    years between ``train_period_end`` and the scenario start to build a continuous
    timeseries for detrending. Bias correction must not see those years.

    Parameters
    ----------
    da : xr.DataArray
        Series with a ``time`` coordinate, covering at least the training window.
    train_period_start, train_period_end : int
        Inclusive year bounds of the training window.

    Returns
    -------
    xr.DataArray
        ``da`` restricted to ``train_period_start``–``train_period_end``.

    Raises
    ------
    ValueError
        If the window selects no timesteps at all, which means the series and the
        configured window do not overlap.

    Notes
    -----
    This deliberately does not assert that ``da`` *spans* the window. Partial coverage is
    a config-versus-data mismatch, caught before any compute by
    ``srm.validation.check_train_period_coverage`` for the historical GCM member and by
    ``srm.validation.check_obs_compatibility`` for the observation dataset. Re-checking it
    here would make every caller responsible for modelling a full-length record while
    adding nothing those deploy-gate checks have not already rejected.
    """
    out = da.sel(time=slice(f"{train_period_start}", f"{train_period_end}"))
    if out["time"].size == 0:
        raise ValueError(
            f"Training window {train_period_start}-{train_period_end} selects no timesteps. "
            f"Series spans {str(da['time'].values[0])[:10]} to {str(da['time'].values[-1])[:10]}."
        )
    return out


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
    # resample("1D") anchors at midnight; MIROC use noon timestamps.
    # Floor da.time to midnight for reindex, then restore original coords to fix nan issue in #361
    da_time_midnight = da.time.values.astype("datetime64[D]").astype("datetime64[ns]")
    trend_on_daily_timestep = (
        da_mon_trend.resample(time="1D")
        .ffill()
        .reindex(time=da_time_midnight)
        .ffill(dim="time")
        .assign_coords(time=da.time)
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
    """
    Remap a fine-resolution field onto a coarser target grid.

    Parameters
    ----------
    da_fine_to_coarsen : xr.DataArray
        Fine-resolution data with ``lat``/``lon`` coordinates.
    da_coarse_grid : xr.DataArray
        DataArray defining the target coarse grid coordinates.

    Returns
    -------
    xr.DataArray
        Fine data conservatively remapped to the coarse grid, cast back to the
        input dtype.

    Notes
    -----
    Uses conservative remapping via ``xarray_regrid``. Any non-spatial coords
    (for example ``time``) are dropped from the target grid to avoid ambiguity
    in regrid operations. We pass ``latitude_coord="lat"`` explicitly so
    xarray-regrid can apply latitude-aware weighting (accounting for spherical
    area distortion toward the poles) when building conservative remapping
    weights.
    """
    # `.regrid` namespace is registered by xarray_regrid and assumes a rectilinear grid.
    target_grid = da_coarse_grid.reset_coords(drop=True)
    if "time" in target_grid.coords:
        # Intentionally use .isel(time=0)`)so
        # `time` remains a length-1 dimension rather than being dropped to a scalar.
        target_grid = target_grid.isel(time=[0])
    # Explicitly identify the latitude coordinate so conservative weights include
    # latitude-based area correction.
    da_coarse = da_fine_to_coarsen.regrid.conservative(target_grid, latitude_coord="lat")
    return da_coarse.astype(da_fine_to_coarsen.dtype)


def _lon_spans_globe(lon_vals: np.ndarray) -> bool:
    """Whether an ascending longitude coordinate wraps the whole planet.

    A global grid covers the whole planet minus the gap between its last cell center
    and the wrap-around back to the first (~one grid step, dlon), so its span
    (last - first) is roughly 360 - dlon. A regional grid spans much less. The
    1.5 * dlon just leaves room for floating-point / uneven-spacing wobble; it is not
    tied to any particular resolution.
    """
    if lon_vals.size <= 1:
        return False
    dlon = np.median(np.diff(lon_vals))
    return bool((lon_vals[-1] - lon_vals[0]) >= 360.0 - 1.5 * dlon)


def coarse_domain_mask(
    da_coarse_to_regrid: xr.DataArray, da_fine_grid: xr.DataArray
) -> xr.DataArray:
    """Boolean fine-grid mask marking cells inside the coarse interpolation domain.

    This mirrors the domain that :func:`interpolate_coarse_to_fine_grid` interpolates
    over, including its periodic longitude pad. The result is therefore exactly
    ``~isnan(...)`` of that function's output whenever the coarse input is itself
    NaN-free.

    Parameters
    ----------
    da_coarse_to_regrid : xr.DataArray
        Coarse-resolution source, supplying the interpolation domain.
    da_fine_grid : xr.DataArray
        DataArray providing target fine-grid ``lat``/``lon`` coordinates.

    Returns
    -------
    xr.DataArray
        Boolean array over ``(lat, lon)`` of the fine grid, True where a fine cell
        falls inside the coarse domain.

    Notes
    -----
    Global runs come out all-True, so a NaN check masked by this is a strict zero-NaN
    assertion in production. Regional (subset) runs exclude the frame of fine cells
    lying outside the outermost coarse cell centers, which interpolate to NaN by
    design — see the Notes of :func:`interpolate_coarse_to_fine_grid`.
    """
    coarse = da_coarse_to_regrid
    if not coarse.indexes["lon"].is_monotonic_increasing:
        coarse = coarse.sortby("lon")

    lon_vals = coarse["lon"].values
    lat_vals = coarse["lat"].values
    fine_lat = da_fine_grid["lat"].values
    fine_lon = da_fine_grid["lon"].values

    lat_ok = (fine_lat >= lat_vals.min()) & (fine_lat <= lat_vals.max())
    # The periodic pad appends the first coarse cell at +360, extending the domain to
    # a full turn starting from the first coarse longitude.
    lon_hi = lon_vals[0] + 360.0 if _lon_spans_globe(lon_vals) else lon_vals[-1]
    lon_ok = (fine_lon >= lon_vals[0]) & (fine_lon <= lon_hi)

    return xr.DataArray(
        lat_ok[:, np.newaxis] & lon_ok[np.newaxis, :],
        dims=["lat", "lon"],
        coords={"lat": da_fine_grid["lat"], "lon": da_fine_grid["lon"]},
    )


def interpolate_coarse_to_fine_grid(
    da_coarse_to_regrid: xr.DataArray, da_fine_grid: xr.DataArray
) -> xr.DataArray:
    """
    Interpolate a coarse field onto a finer target grid.

    Parameters
    ----------
    da_coarse_to_regrid : xr.DataArray
        Coarse-resolution input data.
    da_fine_grid : xr.DataArray
        DataArray providing target fine-grid ``lat``/``lon`` coordinates.

    Returns
    -------
    xr.DataArray
        Coarse data interpolated to the fine grid, cast back to the input
        dtype.

    Notes
    -----
    Uses ``slinear`` interpolation. This is preferred over ``linear`` here
    because ``linear`` can introduce tiny negative artifacts for strictly
    positive variables.


    Fix for issue: https://github.com/carbonplan/srm-downscaling/issues/462

    Global coarse grids (lon end-gap ~ one grid step) get their first cell
    periodically wrapped to +360 before interpolating, since -180..180 lon has
    no source point at exactly +180 and fine cells past the last coarse center
    would otherwise fall outside the interpolation domain and return NaN.
    Regional (subset) domains have a large end-gap, are detected as such, and
    are left unpadded so out-of-domain fine cells correctly stay NaN.

    References
    ----------
    xarray has no native cyclic-longitude support and padding seems to be the standard
    workaround:
    https://discourse.pangeo.io/t/interpolating-2d-data-with-periodic-boundaries-to-points-using-xarray/2702
    https://github.com/pydata/xarray/issues/623
    """
    # Padding math and slinear interp both assume ascending lon. Check, and if not apply sortby
    if not da_coarse_to_regrid.indexes["lon"].is_monotonic_increasing:
        da_coarse_to_regrid = da_coarse_to_regrid.sortby("lon")

    lon = da_coarse_to_regrid["lon"]
    lon_vals = lon.values
    spans_globe = _lon_spans_globe(lon_vals)

    if spans_globe:
        # Right-only wrap relies on -180 already being an exact coarse lon point,
        # which holds for even lon-point-count grids (all current GCMs: CESM2,
        # MIROC-ES2H, UKESM). An odd-count grid would need a left pad too.
        # Wrap the first cell to just past the last (+360) so fine points between
        # the last coarse center and +180 interpolate across the dateline instead
        # of falling outside the domain and returning NaN.
        right = da_coarse_to_regrid.isel(lon=[0]).assign_coords(lon=lon.isel(lon=[0]) + 360)
        da_to_interp = xr.concat([da_coarse_to_regrid, right], dim="lon")
    else:
        da_to_interp = da_coarse_to_regrid

    coarse_on_fine_grid = da_to_interp.interp(
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
    da: xr.DataArray,
    clim_method: DownscalingClimMethod = "simple",
    allow_negative_values: bool = True,
) -> xr.DataArray:
    """
    Compute day-of-year climatology on the fine-resolution observation grid.

    Parameters
    ----------
    da : xr.DataArray
        Input observation time series with ``time``, ``lat``, and ``lon``.
    clim_method : {"simple", "fft"}, default: "simple"
        Climatology smoothing method:
        - ``"simple"`` returns raw day-of-year means.
        - ``"fft"`` smooths the day-of-year cycle with mean + first 3 harmonics.
    allow_negative_values : bool, default: True
        Whether to allow negative values in the output.

    Returns
    -------
    xr.DataArray
        Day-of-year climatology with dimensions ordered as
        ``("dayofyear", "lat", "lon")``.
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
        # apply_ufunc moves input_core_dims to the last position, so the output has
        # dims (lat, lon, dayofyear).  Transpose back to the canonical
        # (dayofyear, lat, lon) order that matches the simple-path output and
        # the xarray groupby() convention.
        obs_fine_doy_means_smoothed = obs_fine_doy_means_smoothed.transpose(
            "dayofyear", "lat", "lon"
        )

        # It is possible for the FFT smoothing to introduce small negative artifacts for variables that are strictly positive (e.g., precipitation)
        # If allow_negative_values is False, we set any negative values to zero here.
        if not allow_negative_values:
            obs_fine_doy_means_smoothed = obs_fine_doy_means_smoothed.where(
                obs_fine_doy_means_smoothed >= 0, 0
            )

    return obs_fine_doy_means_smoothed


def downscale_from_coarse(
    da: xr.DataArray,
    obs_coarse: xr.DataArray,
    obs_fine: xr.DataArray,
    method: DownscalingMethod = "additive",
    clim_method: DownscalingClimMethod = "simple",
    allow_negative_values: bool = True,
    max_residual: float = 100,
    enforce_conservation: bool = False,
) -> xr.DataArray:
    """
    Spatially disaggregate bias-corrected coarse data to the fine observation grid.

    Parameters
    ----------
    da : xr.DataArray
        Bias-corrected coarse-resolution simulation to downscale.
    obs_coarse : xr.DataArray
        Observations remapped to the same coarse grid as ``da``.
    obs_fine : xr.DataArray
        Native fine-resolution observations used to define high-res climatology.
    method : {"additive", "multiplicative"}, default: "additive"
        Residual formulation:
        - ``"additive"`` uses anomalies from coarse climatology.
        - ``"multiplicative"`` uses ratios to coarse climatology.
    clim_method : {"simple", "fft"}, default: "simple"
        Method used to estimate fine-grid day-of-year climatology.
    max_residual : float
        The maximum value possible for the residuals used for building the
        relationship between coarse data and fine.
    enforce_conservation: bool, default: False
        Whether to enforce conservation of the coarse-scale mean after downscaling.
        - True: the downscaled field is adjusted to ensure that its coarse-scale mean matches the debiased, coarse input
        - False: the downscaled field is returned without adjustment, which may result in an added bias at the coarse scale.

    Returns
    -------
    xr.DataArray
        Downscaled data on the fine ``obs_fine`` grid.

    Notes
    -----
    Workflow:
    1. Compute fine-grid day-of-year climatology.
    2. Coarsen that climatology to the model grid.
    3. Compute coarse residuals (difference or ratio).
    4. Interpolate residuals to fine grid.
    5. Reapply fine-grid climatology (add or multiply).
    """
    # Step 1: calculate the daily climatology of high-res observations
    obs_fine_doy_means = calculate_doy_means(
        obs_fine, clim_method=clim_method, allow_negative_values=allow_negative_values
    )
    assert_no_nans(obs_fine_doy_means, name="obs_fine_doy_means")

    # Step 2: Aggregate daily climatology to the low-resolution grid of the GCM being processed
    obs_coarse_doy_means = interpolate_fine_to_coarse_grid(
        da_fine_to_coarsen=obs_fine_doy_means, da_coarse_grid=obs_coarse
    )
    assert_no_nans(obs_coarse_doy_means, name="obs_coarse_doy_means")

    # Step 3: Remove coarsened daily climatology from the bias-corrected fields
    if method == "additive":
        residuals = da.groupby("time.dayofyear") - obs_coarse_doy_means
    elif method == "multiplicative":
        # Guard the denominator: where coarse climatology is zero (dry cells/days),
        # the NCL reference forces the ratio to 0 rather than producing inf/NaN.
        # Replace exact zeros with NaN so the division yields NaN, then fill those
        # specific locations with 0 after dividing.
        zero_clim = obs_coarse_doy_means == 0
        safe_clim = obs_coarse_doy_means.where(~zero_clim)  # zeros -> NaN

        residuals = da.groupby("time.dayofyear") / safe_clim

        # Force ratio to 0 exactly where the coarse climatology was zero.
        # Broadcast the per-DOY zero mask back onto the time axis.
        zero_clim_on_time = zero_clim.sel(dayofyear=da["time"].dt.dayofyear)
        residuals = residuals.where(~zero_clim_on_time, 0.0).clip(max=max_residual)

    assert_no_nans(residuals, name="residuals")

    # Step 4: Bilinearly interpolate residuals to the high-res grid
    # this creates a smooth layer of how different the particular simulated february 10 is
    # from the average february 10.
    residuals_fine = interpolate_coarse_to_fine_grid(
        da_coarse_to_regrid=residuals, da_fine_grid=obs_fine
    )
    # Global runs mask to all-True, making this a strict zero-NaN assertion. Regional
    # runs exclude the frame of fine cells outside the coarse domain, which
    # interpolate to NaN by design; interior NaNs still abort the run.
    fine_domain_mask = coarse_domain_mask(residuals, obs_fine)
    assert_no_nans(residuals_fine, name="residuals_fine", where=fine_domain_mask)

    # Step 5: Return high-res climatology
    # Add or multiply a constant value to the residuals based on DOY
    # this step adds back in the day-of-year spatial texture saying,
    # "let's combine (a) how different February 10 is from the typical February 10 at the coarse scale
    # with the typical spatial structure of February 10"
    if method == "additive":
        downscaled = residuals_fine.groupby("time.dayofyear") + obs_fine_doy_means
    elif method == "multiplicative":
        downscaled = residuals_fine.groupby("time.dayofyear") * obs_fine_doy_means

    # Optional Step 6: Enforce conservation of the coarse-scale mean after downscaling
    if enforce_conservation:
        recoarsened = interpolate_fine_to_coarse_grid(
            downscaled.drop_vars("dayofyear", errors="ignore").unify_chunks(), da
        )
        if method == "additive":
            correction_coarse = da - recoarsened
            correction_fine = correction_coarse.interp(
                lat=downscaled.lat, lon=downscaled.lon, method="nearest"
            )
            downscaled_corrected = downscaled + correction_fine
        elif method == "multiplicative":
            ratio_coarse = da / recoarsened.where(recoarsened != 0, 1.0)
            ratio_fine = ratio_coarse.interp(
                lat=downscaled.lat, lon=downscaled.lon, method="nearest"
            )
            downscaled_corrected = downscaled * ratio_fine
        downscaled = downscaled_corrected
        # This branch is the one NaN source downstream of the residual gates. Recoarsening
        # yields NaN for any coarse cell with no valid fine contributors, and the nearest
        # neighbor interpolation then spreads it across interior cells the earlier checks
        # already certified clean. Step 5 needs no such check, since it only combines two
        # arrays this function has already asserted NaN-free.
        assert_no_nans(downscaled, name="downscaled", where=fine_domain_mask)

    return downscaled


def derive_tasmin(tasmax: xr.DataArray, dtr: xr.DataArray) -> xr.DataArray:
    """Return ``tasmin = tasmax - dtr``, requiring identical time axes.

    ``dtr`` is produced by a separate (non-detrended) scenario path. For SAI
    scenarios it was historically truncated to the SAI simulation period (~2035+)
    while ``tasmax`` spans the full predict window (issue #363). Subtracting
    mismatched axes silently NaN-fills ``tasmin`` on the non-overlapping days, so
    fail loudly here instead — the upstream extent bug should be surfaced, not
    shipped as scattered NaNs.

    Parameters
    ----------
    tasmax, dtr : xr.DataArray
        Debiased-coarse maximum temperature and diurnal temperature range, which
        must share an identical ``time`` axis.

    Returns
    -------
    xr.DataArray
        ``tasmax - dtr`` named ``"tasmin"``.
    """
    if not tasmax.indexes["time"].equals(dtr.indexes["time"]):
        tmax_t, dtr_t = tasmax["time"].values, dtr["time"].values
        raise ValueError(
            f"cannot derive tasmin: tasmax spans {tasmax.sizes['time']} timesteps "
            f"({str(tmax_t.min())[:10]}..{str(tmax_t.max())[:10]}) but dtr spans "
            f"{dtr.sizes['time']} ({str(dtr_t.min())[:10]}..{str(dtr_t.max())[:10]}); "
            f"their time axes must be identical (issue #363 — a truncated dtr would "
            f"silently NaN-fill tasmin)."
        )
    return (tasmax - dtr).rename("tasmin")


def swap_temperature_extremes(
    tasmax: xr.DataArray, tasmin: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Enforce ``tasmax >= tasmin`` by swapping values where ``tasmax < tasmin``.

    Independent spatial disaggregation of ``tasmax`` and ``tasmin`` can leave a
    small number of cells where the downscaled ``tasmax`` falls below ``tasmin``.
    Following the NEX-GDDP-CMIP6 v2 final sweep (issue #331), swap the two values
    at those cells so the physical constraint ``tasmax >= tasmin`` holds
    everywhere.

    The comparison is NaN-safe: cells where either input is NaN (e.g. ocean under
    the land mask) compare ``False`` and are left unchanged. The operation is lazy
    and idempotent.

    Inputs must share identical coordinates. Alignment is enforced with
    ``join="exact"`` so a mismatched grid raises loudly rather than being silently
    inner/outer-joined into dropped or NaN-filled cells, regardless of the global
    ``arithmetic_join`` option.

    Parameters
    ----------
    tasmax, tasmin : xr.DataArray
        Downscaled daily maximum / minimum near-surface air temperature on the
        same grid and time axis.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        ``(tasmax_corrected, tasmin_corrected)`` with names and attrs preserved.
    """
    tasmax, tasmin = xr.align(tasmax, tasmin, join="exact")
    swap = tasmax < tasmin
    tasmax_corrected = xr.where(swap, tasmin, tasmax).astype(tasmax.dtype).rename(tasmax.name)
    tasmin_corrected = xr.where(swap, tasmax, tasmin).astype(tasmin.dtype).rename(tasmin.name)
    tasmax_corrected.attrs = dict(tasmax.attrs)
    tasmin_corrected.attrs = dict(tasmin.attrs)
    return tasmax_corrected, tasmin_corrected
