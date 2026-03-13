"""
BCSD pipeline with three-stage architecture and automatic caching.

Stages:
1. prepare_observations: Regrid observations to GCM grid (once per GCM/variable)
2.fit_historical: Downscale historical period (once per GCM/variable/ensemble)
3. transform_scenario: Downscale future scenario (many times, reuses cached artifacts)
"""

from __future__ import annotations

import logging
import warnings

import dask.system
import icechunk
import numpy as np
import scipy.stats
import xarray as xr
from icechunk.xarray import to_icechunk

from srm.bcsd_config import BCSDConfig
from srm.cache import ArtifactCache
from srm.downscaling_utils import (
    calculate_baseline_climatology,
    detrend,
    downscale_from_coarse,
    get_experiment,
    get_obs,
    interpolate_fine_to_coarse_grid,
    rechunk,
    retrend,
    subset_space,
)
from srm.utils import Timer

logger = logging.getLogger(__name__)


class BCSDPipeline:
    """
    Three-stage BCSD downscaling pipeline with automatic caching.

    This class orchestrates the BCSD workflow, automatically caching intermediate
    artifacts to enable efficient reuse across multiple runs. Each stage checks
    for cached outputs before computing, and validates dependencies exist.

    Example
    -------
    >>> config = BCSDConfig(
    ...     gcm="CESM2-WACCM",
    ...     variable="tas",
    ...     ensemble_member=0,
    ...     scenario="ssp245",
    ...     predict_period_start=2015,
    ...     predict_period_end=2100,
    ... )
    >>> pipeline = BCSDPipeline(config)
    >>> # Run all stages
    >>> pipeline.prepare_observations()
    >>> pipeline.fit_historical()
    >>> result = pipeline.transform_scenario()
    """

    def __init__(self, config: BCSDConfig):
        """
        Initialize pipeline with configuration.

        Parameters
        ----------
        config : BCSDConfig
            Configuration for the BCSD run
        """
        self.config = config
        self.cache = ArtifactCache(
            base_path=config.cache_dir,
            environment=config.environment,
            version=config.version,
            output_dir=config.output_dir,
        )

        # State dictionary for intermediate results (mostly for debugging)
        self._state = {}

    @staticmethod
    def _icechunk_storage(path: str):
        """Create icechunk Storage from an S3 or local path."""
        if path.startswith("s3://"):
            path_no_scheme = path[len("s3://") :]
            bucket, _, prefix = path_no_scheme.partition("/")
            return icechunk.s3_storage(bucket=bucket, prefix=prefix)
        else:
            return icechunk.local_filesystem_storage(path=path)

    def _write_to_icechunk(self, da: xr.DataArray, path: str, commit_message: str) -> str:
        """Write a DataArray to an icechunk store and commit atomically."""
        storage = self._icechunk_storage(path)
        repo = icechunk.Repository.open_or_create(storage)
        session = repo.writable_session("main")
        to_icechunk(da.to_dataset(), session, mode="w")
        return session.commit(commit_message, rebase_with=icechunk.ConflictDetector())

    def _open_from_icechunk(self, path: str) -> xr.Dataset:
        """Open a dataset from an icechunk store."""
        storage = self._icechunk_storage(path)
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        return xr.open_dataset(session.store, engine="zarr", consolidated=False)

    def prepare_observations(self, force: bool = False) -> str:
        """
        Stage 1: Regrid observations to GCM grid.

        This stage loads ERA5 observations and regrids them to the coarse GCM
        grid using local area averaging. The result is cached and reused across
        all ensemble members and scenarios for this GCM/variable combination.

        Parameters
        ----------
        force : bool, optional
            Force recomputation even if cached artifact exists

        Returns
        -------
        str
            S3 path to cached artifact
        """
        output_path = self.cache.get_obs_path(
            self.config.gcm, self.config.variable, self.config.subset_bounds
        )

        # Check cache
        if self.cache.exists(output_path) and not force:
            if self.config.verbose:
                logger.info(f"✓ Using cached observations: {output_path}")
            return output_path

        if self.config.verbose:
            logger.info(
                f"Computing observation regridding for {self.config.gcm}/{self.config.variable}"
            )

        with Timer("Loaded observations", verbose=self.config.verbose):
            # Load fine-resolution observations
            obs_fine = get_obs(var=self.config.variable)
            obs_fine = obs_fine.drop_vars("spatial_ref", errors="ignore")

            # Load GCM grid for target
            model_grid = get_experiment(
                gcm=self.config.gcm, scenario="Historical", var=self.config.variable
            )
            model_grid = model_grid.drop_vars("spatial_ref", errors="ignore")

            # Subset spatially if requested
            if self.config.subset_bounds:
                lat_min, lat_max, lon_min, lon_max = self.config.subset_bounds
                obs_fine = subset_space(obs_fine, [lat_min, lat_max, lon_min, lon_max])
                model_grid = subset_space(model_grid, [lat_min, lat_max, lon_min, lon_max])

        # Regrid to coarse grid
        with Timer("Regridded observations to coarse grid", verbose=self.config.verbose):
            # Suppress expected warnings from sparse array operations during regridding
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in divide")
                warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
                obs_coarse = interpolate_fine_to_coarse_grid(
                    da_fine_to_coarsen=obs_fine, da_coarse_grid=model_grid
                )

        # Rechunk for efficient cache writes and downstream spatial operations
        if self.config.rechunk_workflow:
            obs_coarse = rechunk(obs_coarse, pattern="full_space")

        # Save to cache
        with Timer("Saved to cache", verbose=self.config.verbose):
            obs_coarse.name = self.config.variable
            obs_coarse.attrs = obs_fine.attrs  # Preserve units and metadata
            self._write_to_icechunk(obs_coarse, output_path, "write complete")

        if self.config.verbose:
            logger.info(f"✓ Cached observations: {output_path}")

        return output_path

    def fit_historical(self, force: bool = False) -> str:
        """
        Stage 2: Downscale historical period.

        This stage performs the complete historical downscaling workflow:
        - Load coarse obs (from cache) and historical GCM data
        - Quantile map (debias) historical GCM to match obs distribution
        - Spatially disaggregate to fine resolution

        The result is cached and reused for all scenarios with this GCM/variable/ensemble.

        Parameters
        ----------
        force : bool, optional
            Force recomputation even if cached artifact exists

        Returns
        -------
        str
            S3 path to cached artifact

        Raises
        ------
        ValueError
            If obs_regridded dependency is missing
        """
        # Validate dependencies
        self.cache.validate_dependencies("fit_historical", self.config)

        output_path = self.cache.get_historical_path(
            self.config.gcm,
            self.config.variable,
            self.config.ensemble_member,
            self.config.subset_bounds,
        )

        # Check cache
        if self.cache.exists(output_path) and not force:
            if self.config.verbose:
                logger.info(f"✓ Using cached historical: {output_path}")
            return output_path

        if self.config.verbose:
            logger.info(
                f"Computing historical downscaling for "
                f"{self.config.gcm}/{self.config.variable}/{self.config.ensemble_member}"
            )

        with Timer("Loaded data", verbose=self.config.verbose):
            # Load cached coarse observations
            deps = self.cache.check_dependencies("fit_historical", self.config)
            obs_coarse_path = deps["obs_regridded"][1]
            obs_coarse = self._open_from_icechunk(obs_coarse_path)[self.config.variable]

            # Load fine observations
            obs_fine = get_obs(var=self.config.variable)
            obs_fine = obs_fine.drop_vars("spatial_ref", errors="ignore")

            # Load historical GCM
            model_hist = get_experiment(
                gcm=self.config.gcm, scenario="Historical", var=self.config.variable
            )
            # Historical data may not have ensemble_member dimension
            if "ensemble_member" in model_hist.dims:
                model_hist = model_hist.sel(ensemble_member=self.config.ensemble_member)
            model_hist = model_hist.drop_vars("spatial_ref", errors="ignore")

            # Subset spatially if requested
            if self.config.subset_bounds:
                lat_min, lat_max, lon_min, lon_max = self.config.subset_bounds
                obs_fine = subset_space(obs_fine, [lat_min, lat_max, lon_min, lon_max])
                model_hist = subset_space(model_hist, [lat_min, lat_max, lon_min, lon_max])

            # Subset time to training period
            obs_coarse = obs_coarse.sel(
                time=slice(f"{self.config.train_period_start}", f"{self.config.train_period_end}")
            )
            obs_fine = obs_fine.sel(
                time=slice(f"{self.config.train_period_start}", f"{self.config.train_period_end}")
            )
            model_hist = model_hist.sel(
                time=slice(f"{self.config.train_period_start}", f"{self.config.train_period_end}")
            )

        # Bias correct (quantile mapping)
        with Timer("Bias corrected historical", verbose=self.config.verbose):
            from ibicus.debias import QuantileMapping

            def _make_debiaser(**kwargs):
                if self.config.variable == "rsds":
                    return QuantileMapping(distribution=scipy.stats.beta, **kwargs)
                return QuantileMapping.from_variable(variable=self.config.variable, **kwargs)

            # For detrending the historical and doing a nonparametric/parametric hybrid quantile mapping
            # method, just use the nonparametric version because by definition the modeled historical period will
            # always be within the range of the modeled historical, so it's never necessary to 
            # use the parametric version for out of range modeled values.
            if self.config.mapping_type == "nonparametric_hybrid":
                debiaser = _make_debiaser(
                    mapping_type="nonparametric",
                    detrending="no_detrending",
                    running_window_mode=self.config.do_windowing,
                    running_window_length=31,
                    running_window_step_length=1,
                    running_window_mode_over_years_of_cm_future=False,
                )
            else:
                debiaser = _make_debiaser(
                    mapping_type=self.config.mapping_type,
                    detrending="no_detrending",
                    running_window_mode=self.config.do_windowing,
                    running_window_length=31,
                    running_window_step_length=1,
                    running_window_mode_over_years_of_cm_future=False,
                )

            # Convert to numpy for ibicus
            obs_np = obs_coarse.as_numpy().values
            cm_hist_np = model_hist.as_numpy().values

            # Apply quantile mapping
            model_hist_debiased_np = debiaser.apply(
                obs=obs_np,
                cm_hist=cm_hist_np,
                cm_future=cm_hist_np,  # Debias historical with itself
                time_obs=obs_coarse["time"].values,
                time_cm_hist=model_hist["time"].values,
                time_cm_future=model_hist["time"].values,
                parallel=True,
                nr_processes=dask.system.CPU_COUNT,
                progressbar=False,
            )

            # Convert back to xarray
            model_hist_debiased = xr.DataArray(
                data=model_hist_debiased_np,
                coords={
                    "lat": model_hist["lat"],
                    "lon": model_hist["lon"],
                    "time": model_hist["time"],
                },
                dims=["time", "lat", "lon"],
            )

            if self.config.save_intermediate:
                with Timer("Saved coarse debiased to cache", verbose=self.config.verbose):
                    debiased_path = self.cache.get_debiased_historical_path(
                        gcm=self.config.gcm,
                        variable=self.config.variable,
                        subset_bounds=self.config.subset_bounds,
                        ensemble=self.config.ensemble_member,
                    )
                    model_hist_debiased.name = self.config.variable
                    model_hist_debiased.attrs = (
                        model_hist_debiased.attrs
                    )  # Preserve units and metadata
                    self._write_to_icechunk(model_hist_debiased, debiased_path, "write complete")
                    if self.config.verbose:
                        logger.info(f"✓ Saved debiased historical: {debiased_path}")

        # Spatially disaggregate
        with Timer("Spatially disaggregated", verbose=self.config.verbose):
            model_hist_downscaled = downscale_from_coarse(
                da=model_hist_debiased,
                obs_coarse=obs_coarse.as_numpy(),
                obs_fine=obs_fine.as_numpy(),
                method=self.config.downscaling_method,
                clim_method=self.config.downscaling_clim_method,
            )
            model_hist_downscaled = rechunk(model_hist_downscaled, pattern="full_space")

        # Save to cache
        with Timer("Saved to cache", verbose=self.config.verbose):
            model_hist_downscaled.name = self.config.variable
            model_hist_downscaled.attrs = model_hist.attrs  # Preserve units and metadata
            self._write_to_icechunk(model_hist_downscaled, output_path, "write complete")

        if self.config.verbose:
            logger.info(f"✓ Cached historical: {output_path}")

        return output_path

    def transform_scenario(self, force: bool = False) -> str:
        """
        Stage 3: Downscale future scenario.

        This stage performs scenario downscaling by:
        - Loading cached obs and historical artifacts
        - Loading scenario GCM data (SSP or SAI)
        - Optionally detrending scenario data
        - Quantile mapping to remove biases
        - Re-trending (if detrending was applied)
        - Spatially disaggregating to fine resolution

        Parameters
        ----------
        force : bool, optional
            Force recomputation even if cached artifact exists

        Returns
        -------
        str
            S3 path to output zarr store

        Raises
        ------
        ValueError
            If obs_regridded or historical dependencies are missing,
            or if scenario is not specified in config
        """
        if self.config.scenario is None:
            raise ValueError("scenario must be specified in config for transform_scenario")

        # Validate dependencies
        self.cache.validate_dependencies("transform_scenario", self.config)

        output_path = self.cache.get_scenario_path(
            self.config.gcm,
            self.config.variable,
            self.config.ensemble_member,
            self.config.scenario,
            self.config.subset_bounds,
        )

        # Check cache
        if self.cache.exists(output_path) and not force:
            if self.config.verbose:
                logger.info(f"✓ Using cached scenario: {output_path}")
            return output_path

        if self.config.verbose:
            logger.info(
                f"Computing scenario downscaling for "
                f"{self.config.gcm}/{self.config.variable}/{self.config.ensemble_member}/{self.config.scenario}"
            )

        # Load data
        with Timer("Loaded data", verbose=self.config.verbose):
            # Load cached artifacts
            deps = self.cache.check_dependencies("transform_scenario", self.config)
            obs_coarse = self._open_from_icechunk(deps["obs_regridded"][1])[self.config.variable]

            # Load fine observations
            obs_fine = get_obs(var=self.config.variable)
            obs_fine = obs_fine.drop_vars("spatial_ref", errors="ignore")

            # Load historical for training
            model_hist = get_experiment(
                gcm=self.config.gcm, scenario="Historical", var=self.config.variable
            )
            # Historical data may not have ensemble_member dimension
            if "ensemble_member" in model_hist.dims:
                model_hist = model_hist.sel(ensemble_member=self.config.ensemble_member)
            model_hist = model_hist.drop_vars("spatial_ref", errors="ignore")

            # Load scenario
            model_scenario = get_experiment(
                gcm=self.config.gcm, scenario=self.config.scenario, var=self.config.variable
            )
            model_scenario = model_scenario.sel(ensemble_member=self.config.ensemble_member)
            model_scenario = model_scenario.drop_vars("spatial_ref", errors="ignore")

            if self.config.is_sai_scenario:
                ssp_timeseries = get_experiment(
                    gcm=self.config.gcm, scenario="SSP245", var=self.config.variable
                )
                ssp_timeseries = ssp_timeseries.sel(ensemble_member=self.config.ensemble_member)
                ssp_timeseries = ssp_timeseries.drop_vars("spatial_ref", errors="ignore")

            # Subset spatially if requested
            if self.config.subset_bounds:
                lat_min, lat_max, lon_min, lon_max = self.config.subset_bounds
                obs_fine = subset_space(obs_fine, [lat_min, lat_max, lon_min, lon_max])
                model_hist = subset_space(model_hist, [lat_min, lat_max, lon_min, lon_max])
                model_scenario = subset_space(model_scenario, [lat_min, lat_max, lon_min, lon_max])
                if self.config.is_sai_scenario:
                    ssp_timeseries = subset_space(
                        ssp_timeseries, [lat_min, lat_max, lon_min, lon_max]
                    )

            # Subset time periods
            obs_coarse = obs_coarse.sel(
                time=slice(f"{self.config.train_period_start}", f"{self.config.train_period_end}")
            )
            obs_fine = obs_fine.sel(
                time=slice(f"{self.config.train_period_start}", f"{self.config.train_period_end}")
            )
            model_hist = model_hist.sel(
                time=slice(f"{self.config.train_period_start}", f"{self.config.train_period_end}")
            )
            model_scenario = model_scenario.sel(
                time=slice(
                    f"{self.config.predict_period_start}", f"{self.config.predict_period_end}"
                )
            )

        # Detrend if needed
        scenario_detrended = model_scenario
        scenario_trend = None

        if self.config.detrend_data:
            # Rechunk for temporal operations
            if self.config.rechunk_workflow:
                with Timer("Rechunked for detrending", verbose=self.config.verbose):
                    model_hist = rechunk(model_hist, pattern="full_time").persist()
                    model_scenario = rechunk(model_scenario, pattern="full_time").persist()
                    if self.config.is_sai_scenario:
                        ssp_timeseries = rechunk(ssp_timeseries, pattern="full_time").persist()

            with Timer("Detrended scenario", verbose=self.config.verbose):
                # Splice historical + scenario for smooth detrending
                if self.config.is_sai_scenario:
                    # SAI simulations run from 2035 to 2084.
                    # But historical ends in 2014/2015, so we stitch in SSP data for the gap when detrending

                    historical_and_ssp = xr.concat(
                        [
                            model_hist.sel(
                                time=model_hist["time.year"] < self.config.train_period_end
                            ),
                            ssp_timeseries.sel(
                                time=ssp_timeseries["time.year"] >= self.config.train_period_end
                            ),
                        ],
                        dim="time",
                    )
                    historical_scenario = xr.concat(
                        [
                            historical_and_ssp.sel(
                                time=historical_and_ssp["time.year"]
                                < self.config.predict_period_start
                            ),
                            model_scenario.sel(
                                time=model_scenario["time.year"] >= self.config.predict_period_start
                            ),
                        ],
                        dim="time",
                    )
                else:
                    historical_scenario = xr.concat(
                        [
                            model_hist.sel(
                                time=model_hist["time.year"] < self.config.predict_period_start
                            ),
                            model_scenario.sel(
                                time=model_scenario["time.year"] >= self.config.predict_period_start
                            ),
                        ],
                        dim="time",
                    )

                # Calculate baseline climatology
                da_baseline_clim = calculate_baseline_climatology(
                    da_baseline=model_hist,
                    baseline_period_start=self.config.train_period_start,
                    baseline_period_end=self.config.train_period_end,
                )

                # Detrend
                scenario_detrended, scenario_trend = detrend(
                    da=historical_scenario,
                    da_baseline_clim=da_baseline_clim,
                    detrend_method=self.config.detrend_method,
                )

                # Extract just scenario period
                scenario_detrended = scenario_detrended.sel(
                    time=slice(
                        f"{self.config.predict_period_start}", f"{self.config.predict_period_end}"
                    )
                )
                scenario_trend = scenario_trend.sel(
                    time=slice(
                        f"{self.config.predict_period_start}", f"{self.config.predict_period_end}"
                    )
                )

                if self.config.save_intermediate:
                    with Timer("Saved detrended to cache", verbose=self.config.verbose):
                        detrended_path = self.cache.get_detrended_scenario_path(
                            gcm=self.config.gcm,
                            variable=self.config.variable,
                            subset_bounds=self.config.subset_bounds,
                            scenario=self.config.scenario,
                            ensemble=self.config.ensemble_member,
                        )
                        scenario_detrended.name = self.config.variable
                        scenario_detrended.attrs = (
                            scenario_detrended.attrs
                        )  # Preserve units and metadata
                        self._write_to_icechunk(
                            scenario_detrended, detrended_path, "write complete"
                        )
                        if self.config.verbose:
                            logger.info(f"✓ Saved detrended scenario: {detrended_path}")

                    with Timer("Saved trend to cache", verbose=self.config.verbose):
                        trend_path = self.cache.get_trend_scenario_path(
                            gcm=self.config.gcm,
                            variable=self.config.variable,
                            subset_bounds=self.config.subset_bounds,
                            scenario=self.config.scenario,
                            ensemble=self.config.ensemble_member,
                        )
                        scenario_trend.name = self.config.variable
                        scenario_trend.attrs = scenario_trend.attrs  # Preserve units and metadata
                        self._write_to_icechunk(scenario_trend, trend_path, "write complete")
                        if self.config.verbose:
                            logger.info(f"✓ Saved scenario trend: {trend_path}")

        # Bias correct
        with Timer("Bias corrected scenario", verbose=self.config.verbose):
            from ibicus.debias import QuantileMapping

            # Convert to numpy
            obs_np = obs_coarse.as_numpy().values
            cm_hist_np = model_hist.as_numpy().values
            cm_future_np = scenario_detrended.load().values

            def _make_debiaser(**kwargs):
                if self.config.variable == "rsds":
                    return QuantileMapping(distribution=scipy.stats.beta, **kwargs)
                return QuantileMapping.from_variable(variable=self.config.variable, **kwargs)

            # If method is parametric or nonparametric, only debias one time. If hybrid, debias twice (one time parametric and one time nonparametric) and blend results.
            if self.config.mapping_type in ["parametric", "nonparametric"]:
                debiaser = _make_debiaser(
                    mapping_type=self.config.mapping_type,
                    detrending="no_detrending",
                    running_window_mode=self.config.do_windowing,
                    running_window_length=31,
                    running_window_step_length=1,
                    running_window_mode_over_years_of_cm_future=False,
                )

                # Apply quantile mapping
                scenario_debiased_np = debiaser.apply(
                    obs=obs_np,
                    cm_hist=cm_hist_np,
                    cm_future=cm_future_np,
                    time_obs=obs_coarse["time"].values,
                    time_cm_hist=model_hist["time"].values,
                    time_cm_future=scenario_detrended["time"].values,
                    parallel=True,
                    nr_processes=dask.system.CPU_COUNT,
                    progressbar=False,
                )

            elif self.config.mapping_type == "nonparametric_hybrid":
                debiaser_parametric = _make_debiaser(
                    mapping_type="parametric",
                    detrending="no_detrending",
                    running_window_mode=self.config.do_windowing,
                    running_window_length=31,
                    running_window_step_length=1,
                    running_window_mode_over_years_of_cm_future=False,
                )
                debiaser_nonparametric = _make_debiaser(
                    mapping_type="nonparametric",
                    detrending="no_detrending",
                    running_window_mode=self.config.do_windowing,
                    running_window_length=31,
                    running_window_step_length=1,
                    running_window_mode_over_years_of_cm_future=False,
                )

                # Apply quantile mapping twice
                scenario_debiased_parametric_np = debiaser_parametric.apply(
                    obs=obs_np,
                    cm_hist=cm_hist_np,
                    cm_future=cm_future_np,
                    time_obs=obs_coarse["time"].values,
                    time_cm_hist=model_hist["time"].values,
                    time_cm_future=scenario_detrended["time"].values,
                    parallel=True,
                    nr_processes=dask.system.CPU_COUNT,
                    progressbar=False,
                )

                scenario_debiased_nonparametric_np = debiaser_nonparametric.apply(
                    obs=obs_np,
                    cm_hist=cm_hist_np,
                    cm_future=cm_future_np,
                    time_obs=obs_coarse["time"].values,
                    time_cm_hist=model_hist["time"].values,
                    time_cm_future=scenario_detrended["time"].values,
                    parallel=True,
                    nr_processes=dask.system.CPU_COUNT,
                    progressbar=False,
                )

                # Blend results
                # Nonparametric mapping when in range of the modeled historical
                # Parametric mapping when out of range of the modeled historical

                def _calculate_out_of_range_mask(
                    model_hist, scenario_detrended, center_window=31, pad=15
                ):
                    """
                    Calculate mask of where scenario is out of range of modeled historical
                    on a day-of-year basis. The historical range for any day-of-year is the max and min
                    of modeled historical values that fall within a centered window around that day-of-year.
                    This should be the same window size used in the debiaser if using running_window_mode.
                    """
                    doy_max = model_hist.groupby("time.dayofyear").max()
                    doy_min = model_hist.groupby("time.dayofyear").min()

                    # Pad the dayofyear dimension to handle the rolling window at the edges, using values from the opposite end of the year
                    doy_max_padded = xr.concat(
                        [
                            doy_max.isel(dayofyear=slice(-pad, None)),
                            doy_max,
                            doy_max.isel(dayofyear=slice(None, pad)),
                        ],
                        dim="dayofyear",
                    )

                    rolling_doy_max = doy_max_padded.rolling(
                        dayofyear=center_window, center=True
                    ).max()
                    rolling_doy_max = rolling_doy_max.isel(
                        dayofyear=slice(pad, pad + len(doy_max.dayofyear))
                    )

                    doy_min_padded = xr.concat(
                        [
                            doy_min.isel(dayofyear=slice(-pad, None)),
                            doy_min,
                            doy_min.isel(dayofyear=slice(None, pad)),
                        ],
                        dim="dayofyear",
                    )

                    rolling_doy_min = doy_min_padded.rolling(
                        dayofyear=center_window, center=True
                    ).min()
                    rolling_doy_min = rolling_doy_min.isel(
                        dayofyear=slice(pad, pad + len(doy_min.dayofyear))
                    )

                    doy = scenario_detrended["time.dayofyear"]

                    out_of_range = (scenario_detrended > rolling_doy_max.sel(dayofyear=doy)) | (
                        scenario_detrended < rolling_doy_min.sel(dayofyear=doy)
                    )

                    return out_of_range

                out_of_range = _calculate_out_of_range_mask(
                    model_hist=model_hist, scenario_detrended=scenario_detrended
                )

                # Use parametric quantile mapping when out_of_range is True 
                # and nonparametric where out_of_range is False
                scenario_debiased_np = np.where(
                    out_of_range.values,
                    scenario_debiased_parametric_np,
                    scenario_debiased_nonparametric_np,
                )

            # Convert back to xarray
            scenario_debiased = xr.DataArray(
                data=scenario_debiased_np,
                coords={
                    "lat": scenario_detrended["lat"],
                    "lon": scenario_detrended["lon"],
                    "time": scenario_detrended["time"],
                },
                dims=["time", "lat", "lon"],
            )

        if self.config.save_intermediate:
            with Timer("Saved coarse debiased scenario to cache", verbose=self.config.verbose):
                debiased_path = self.cache.get_debiased_scenario_path(
                    gcm=self.config.gcm,
                    variable=self.config.variable,
                    subset_bounds=self.config.subset_bounds,
                    scenario=self.config.scenario,
                    ensemble=self.config.ensemble_member,
                )
                scenario_debiased.name = self.config.variable
                scenario_debiased.attrs = scenario_debiased.attrs  # Preserve units and metadata
                self._write_to_icechunk(scenario_debiased, debiased_path, "write complete")
                if self.config.verbose:
                    logger.info(f"✓ Saved debiased scenario: {debiased_path}")

        # Re-trend if needed
        if self.config.detrend_data and scenario_trend is not None:
            with Timer("Re-trended scenario", verbose=self.config.verbose):
                scenario_debiased = retrend(
                    bias_corrected_detrended=scenario_debiased,
                    trend_on_daily_timestep=scenario_trend,
                    detrend_method=self.config.detrend_method,
                )

        if self.config.save_intermediate:
            with Timer("Saved coarse debiased, retrended to cache", verbose=self.config.verbose):
                debiased_path = self.cache.get_debiased_retrended_scenario_path(
                    gcm=self.config.gcm,
                    variable=self.config.variable,
                    subset_bounds=self.config.subset_bounds,
                    scenario=self.config.scenario,
                    ensemble=self.config.ensemble_member,
                )
                scenario_debiased.name = self.config.variable
                scenario_debiased.attrs = scenario_debiased.attrs  # Preserve units and metadata
                self._write_to_icechunk(scenario_debiased, debiased_path, "write complete")
                if self.config.verbose:
                    logger.info(f"✓ Saved debiased scenario: {debiased_path}")

        # Spatially disaggregate
        with Timer("Spatially disaggregated", verbose=self.config.verbose):
            scenario_downscaled = downscale_from_coarse(
                da=scenario_debiased,
                obs_coarse=obs_coarse.as_numpy(),
                obs_fine=obs_fine.as_numpy(),
                method=self.config.downscaling_method,
                clim_method=self.config.downscaling_clim_method,
            )
            scenario_downscaled = rechunk(scenario_downscaled, pattern="full_space")

        # Save output
        with Timer("Saved output", verbose=self.config.verbose):
            scenario_downscaled.name = self.config.variable
            scenario_downscaled.attrs = model_scenario.attrs  # Preserve units and metadata
            self._write_to_icechunk(scenario_downscaled, output_path, "write complete")

        if self.config.verbose:
            logger.info(f"✓ Saved scenario output: {output_path}")

        return output_path

    def run_full_pipeline(self, force: bool = False) -> str:
        """
        Run all three stages in sequence.

        Convenience method that executes prepare_observations, fit_historical,
        and transform_scenario in order.

        Parameters
        ----------
        force : bool, optional
            Force recomputation of all stages

        Returns
        -------
        str
            S3 path to final scenario output
        """
        self.prepare_observations(force=force)
        self.fit_historical(force=force)
        return self.transform_scenario(force=force)
