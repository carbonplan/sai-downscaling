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
import xarray as xr

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
            base_path=config.cache_dir, environment=config.environment, output_dir=config.output_dir
        )

        # State dictionary for intermediate results (mostly for debugging)
        self._state = {}

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

        # Rechunk for spatial operations
        if self.config.rechunk_workflow:
            with Timer("Rechunked to full space", verbose=self.config.verbose):
                obs_fine = rechunk(obs_fine, pattern="full_space")
                obs_fine = obs_fine.persist()

        # Regrid to coarse grid
        with Timer("Regridded observations to coarse grid", verbose=self.config.verbose):
            # Suppress expected warnings from sparse array operations during regridding
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in divide")
                warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
                obs_coarse = interpolate_fine_to_coarse_grid(
                    da_fine_to_coarsen=obs_fine, da_coarse_grid=model_grid
                )

        # Save to cache
        with Timer("Saved to cache", verbose=self.config.verbose):
            obs_coarse.name = self.config.variable
            obs_coarse.attrs = obs_fine.attrs  # Preserve units and metadata
            obs_coarse.chunk({"time": "100MB"}).to_zarr(output_path, mode="w")

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
                f"{self.config.gcm}/{self.config.variable}/{self.config.ensemble_member:03d}"
            )

        # Load data
        with Timer("Loaded data", verbose=self.config.verbose):
            # Load cached coarse observations
            deps = self.cache.check_dependencies("fit_historical", self.config)
            obs_coarse_path = deps["obs_regridded"][1]
            obs_coarse = xr.open_zarr(obs_coarse_path)[self.config.variable]

            # Load fine observations
            obs_fine = get_obs(var=self.config.variable)
            obs_fine = obs_fine.drop_vars("spatial_ref", errors="ignore")

            # Load historical GCM
            model_hist = get_experiment(
                gcm=self.config.gcm, scenario="Historical", var=self.config.variable
            )
            # Historical data may not have ensemble_member dimension
            if "ensemble_member" in model_hist.dims:
                model_hist = model_hist.isel(ensemble_member=self.config.ensemble_member)
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

            if self.config.do_windowing:
                debiaser = QuantileMapping.from_variable(
                    variable=self.config.variable,
                    mapping_type=self.config.mapping_type,
                    detrending="no_detrending",
                    running_window_mode=True,
                    running_window_length=31,
                    running_window_step_length=1,
                    running_window_mode_over_years_of_cm_future=False,
                )
            else:
                debiaser = QuantileMapping.from_variable(
                    variable=self.config.variable,
                    mapping_type=self.config.mapping_type,
                    detrending="no_detrending",
                    running_window_mode=False,
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

        # Rechunk for spatial operations
        if self.config.rechunk_workflow:
            with Timer("Rechunked for downscaling", verbose=self.config.verbose):
                model_hist_debiased = rechunk(model_hist_debiased, pattern="full_space")
                model_hist_debiased = model_hist_debiased.persist()

        # Spatially disaggregate
        with Timer("Spatially disaggregated", verbose=self.config.verbose):
            model_hist_downscaled = downscale_from_coarse(
                da=model_hist_debiased,
                obs_coarse=obs_coarse.as_numpy(),
                obs_fine=obs_fine.as_numpy(),
                method=self.config.downscaling_method,
                clim_method=self.config.downscaling_clim_method,
            )

        # Save to cache
        with Timer("Saved to cache", verbose=self.config.verbose):
            model_hist_downscaled.name = self.config.variable
            model_hist_downscaled.attrs = model_hist.attrs  # Preserve units and metadata
            model_hist_downscaled.chunk({"time": "100MB"}).to_zarr(output_path, mode="w")

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
                f"{self.config.gcm}/{self.config.variable}/{self.config.ensemble_member:03d}/{self.config.scenario}"
            )

        # Load data
        with Timer("Loaded data", verbose=self.config.verbose):
            # Load cached artifacts
            deps = self.cache.check_dependencies("transform_scenario", self.config)
            obs_coarse = xr.open_zarr(deps["obs_regridded"][1])[self.config.variable]
            _ = xr.open_zarr(deps["historical"][1])

            # Load fine observations
            obs_fine = get_obs(var=self.config.variable)
            obs_fine = obs_fine.drop_vars("spatial_ref", errors="ignore")

            # Load historical for training
            model_hist = get_experiment(
                gcm=self.config.gcm, scenario="Historical", var=self.config.variable
            )
            # Historical data may not have ensemble_member dimension
            if "ensemble_member" in model_hist.dims:
                model_hist = model_hist.isel(ensemble_member=self.config.ensemble_member)
            model_hist = model_hist.drop_vars("spatial_ref", errors="ignore")

            # Load scenario
            model_scenario = get_experiment(
                gcm=self.config.gcm, scenario=self.config.scenario, var=self.config.variable
            )
            model_scenario = model_scenario.isel(ensemble_member=self.config.ensemble_member)
            model_scenario = model_scenario.drop_vars("spatial_ref", errors="ignore")

            # Subset spatially if requested
            if self.config.subset_bounds:
                lat_min, lat_max, lon_min, lon_max = self.config.subset_bounds
                obs_fine = subset_space(obs_fine, [lat_min, lat_max, lon_min, lon_max])
                model_hist = subset_space(model_hist, [lat_min, lat_max, lon_min, lon_max])
                model_scenario = subset_space(model_scenario, [lat_min, lat_max, lon_min, lon_max])

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
                    model_hist = rechunk(model_hist, pattern="full_time")
                    model_scenario = rechunk(model_scenario, pattern="full_time")
                    model_hist = model_hist.persist()
                    model_scenario = model_scenario.persist()

            with Timer("Detrended scenario", verbose=self.config.verbose):
                # Splice historical + scenario for smooth detrending
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
                    detrend=self.config.detrend_method,
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

        # Bias correct
        with Timer("Bias corrected scenario", verbose=self.config.verbose):
            from ibicus.debias import QuantileMapping

            if self.config.do_windowing:
                debiaser = QuantileMapping.from_variable(
                    variable=self.config.variable,
                    mapping_type=self.config.mapping_type,
                    detrending="no_detrending",
                    running_window_mode=True,
                    running_window_length=31,
                    running_window_step_length=1,
                    running_window_mode_over_years_of_cm_future=False,
                )
            else:
                debiaser = QuantileMapping.from_variable(
                    variable=self.config.variable,
                    mapping_type=self.config.mapping_type,
                    detrending="no_detrending",
                    running_window_mode=False,
                    running_window_mode_over_years_of_cm_future=False,
                )

            # Convert to numpy
            obs_np = obs_coarse.as_numpy().values
            cm_hist_np = model_hist.as_numpy().values
            cm_future_np = scenario_detrended.load().values

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

        # Re-trend if needed
        if self.config.detrend_data and scenario_trend is not None:
            with Timer("Re-trended scenario", verbose=self.config.verbose):
                scenario_debiased = retrend(
                    bias_corrected_detrended=scenario_debiased,
                    trend_on_daily_timestep=scenario_trend,
                    detrend_method=self.config.detrend_method,
                )

        # Rechunk for spatial operations
        if self.config.rechunk_workflow:
            with Timer("Rechunked for downscaling", verbose=self.config.verbose):
                scenario_debiased = rechunk(scenario_debiased, pattern="full_space")
                scenario_debiased = scenario_debiased.persist()

        # Spatially disaggregate
        with Timer("Spatially disaggregated", verbose=self.config.verbose):
            scenario_downscaled = downscale_from_coarse(
                da=scenario_debiased,
                obs_coarse=obs_coarse.as_numpy(),
                obs_fine=obs_fine.as_numpy(),
                method=self.config.downscaling_method,
                clim_method=self.config.downscaling_clim_method,
            )

        # Save output
        with Timer("Saved output", verbose=self.config.verbose):
            scenario_downscaled.name = self.config.variable
            scenario_downscaled.attrs = model_scenario.attrs  # Preserve units and metadata
            scenario_downscaled.chunk({"time": "100MB"}).to_zarr(output_path, mode="w")

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
