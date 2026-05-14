"""
BCSD pipeline with three-stage architecture and automatic caching.

Stages:
1. prepare_observations: Regrid observations to GCM grid (once per GCM/variable)
2. fit_historical: Downscale historical period (once per GCM/variable/ensemble)
3. transform_scenario: Downscale future scenario (many times, reuses cached artifacts)
"""

from __future__ import annotations

import importlib.metadata
import logging
import time
import warnings
from datetime import UTC, datetime

import dask.system
import icechunk
import numpy as np
import scipy.stats
import xarray as xr
from ibicus.debias import QuantileMapping
from icechunk.xarray import to_icechunk

from srm.bcsd_config import BCSDConfig
from srm.cache import ArtifactCache
from srm.datasets import BaseDataset, catalog as _catalog
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
from srm.encoding import SHARD_LAT, SHARD_LON, SHARD_TIME, make_encoding

logger = logging.getLogger(__name__)


def _make_debiaser(variable: str, **kwargs):
    if variable == "rsds":
        return QuantileMapping(distribution=scipy.stats.beta, **kwargs)
    return QuantileMapping.from_variable(variable=variable, **kwargs)


def calculate_out_of_range_mask(
    model_hist: xr.DataArray,
    scenario_detrended: xr.DataArray,
    center_window: int = 31,
) -> xr.DataArray:
    """
    Calculate mask of where scenario is out of range of modeled historical
    on a day-of-year basis. The historical range for any day-of-year is the max and min
    of modeled historical values that fall within a centered window around that day-of-year.
    This should be the same window size used in the debiaser if using running_window_mode.

    Parameters
    ----------
    model_hist : xr.DataArray
        Historical GCM data with a time dimension. This is used to compute the
        day-of-year min/max range.
    scenario_detrended : xr.DataArray
        Detrended scenario data to test against the historical range.
    center_window : int, optional
        Size of the centered rolling window (in days) used to compute the
        historical range per day-of-year. Should match the debiaser's
        running_window_length. Default is 31.
    pad : int, optional
        Number of days to pad at each end of the day-of-year dimension to
        handle edge effects in the rolling window. Default is 15.

    Returns
    -------
    xr.DataArray
        Boolean DataArray with the same shape as scenario_detrended. True where
        the scenario value falls outside the historical range for that day-of-year,
        False otherwise.

    """
    grouped_by_dayofyear = model_hist.groupby("time.dayofyear")
    doy_max = grouped_by_dayofyear.max()
    doy_min = grouped_by_dayofyear.min()

    # Pad the dayofyear dimension to handle the rolling window at the edges, using values from the opposite end of the year
    pad = center_window // 2
    # slice(-pad, None) takes the last `pad` values, and slice(None, pad) takes the first `pad` values
    # This wraps around the dayofyear dimension for the rolling window
    doy_max_padded = xr.concat(
        [
            doy_max.isel(dayofyear=slice(-pad, None)),
            doy_max,
            doy_max.isel(dayofyear=slice(None, pad)),
        ],
        dim="dayofyear",
    )

    rolling_doy_max = doy_max_padded.rolling(dayofyear=center_window, center=True).max()
    rolling_doy_max = rolling_doy_max.isel(dayofyear=slice(pad, pad + len(doy_max.dayofyear)))

    doy_min_padded = xr.concat(
        [
            doy_min.isel(dayofyear=slice(-pad, None)),
            doy_min,
            doy_min.isel(dayofyear=slice(None, pad)),
        ],
        dim="dayofyear",
    )

    rolling_doy_min = doy_min_padded.rolling(dayofyear=center_window, center=True).min()
    rolling_doy_min = rolling_doy_min.isel(dayofyear=slice(pad, pad + len(doy_min.dayofyear)))

    doy = scenario_detrended["time.dayofyear"]

    out_of_range = (scenario_detrended > rolling_doy_max.sel(dayofyear=doy)) | (
        scenario_detrended < rolling_doy_min.sel(dayofyear=doy)
    )

    return out_of_range


def _assert_stitched_continuity(result: xr.DataArray) -> None:
    """Raise ValueError if the stitched timeseries has duplicate timestamps or year-level gaps.

    Day-level gaps within a year are tolerated (some GCMs, e.g. UKESM, are
    missing a single day at the historical boundary). The checks are:

    1. No duplicate timestamps – the same calendar day must not appear twice.
    2. No year-level gaps – every integer year between the first and last year
       must be represented by at least one timestep.
    """
    times = result["time"].values
    unique_times, counts = np.unique(times, return_counts=True)
    duplicates = unique_times[counts > 1]
    if len(duplicates):
        raise ValueError(
            f"Stitched timeseries contains {len(duplicates)} duplicate timestamp(s); "
            f"first duplicate: {duplicates[0]}"
        )

    years = np.unique(result["time.year"].values)
    gaps = [(int(y1), int(y2)) for y1, y2 in zip(years, years[1:]) if y2 - y1 > 1]
    if gaps:
        raise ValueError(f"Stitched timeseries has year-level gap(s): {gaps}")


def stitch_historical_scenario(
    model_hist: xr.DataArray,
    model_scenario: xr.DataArray,
    train_period_end: int,
    predict_period_start: int,
    ssp_timeseries: xr.DataArray | None = None,
) -> xr.DataArray:
    """Stitch historical and scenario data into a continuous timeseries for detrending.

    For SAI scenarios (``ssp_timeseries`` provided), historical data is first
    concatenated with SSP245 to bridge the gap between the historical period end
    (``train_period_end``) and the SAI simulation start
    (``predict_period_start``). The combined series is then concatenated with
    the SAI scenario.

    For non-SAI scenarios, historical and scenario data are concatenated
    directly at the ``predict_period_start`` boundary.

    All supported GCMs share the same historical/SSP breakpoint:
    - historical ends  2014-12-31  (``train_period_end`` = 2014)
    - SSP245 begins    2015-01-01  (``predict_period_start`` = 2015)

    Parameters
    ----------
    model_hist : xr.DataArray
        Historical GCM data.
    model_scenario : xr.DataArray
        Future scenario GCM data (SAI or non-SAI).
    train_period_end : int
        Last year of the historical training period (inclusive).
    predict_period_start : int
        First year of the prediction period.
    ssp_timeseries : xr.DataArray, optional
        SSP245 data used to bridge the historical-to-SAI gap. When provided,
        the SAI stitching path is taken; otherwise the non-SAI path is used.

    Returns
    -------
    xr.DataArray
        Continuous timeseries spanning from the start of historical data
        through the end of the scenario period.
    """
    if ssp_timeseries is not None:
        # SAI: historical ≤ train_period_end, then SSP from train_period_end+1,
        # then SAI from predict_period_start onward.
        historical_and_ssp = xr.concat(
            [
                model_hist.sel(time=model_hist["time.year"] <= train_period_end),
                ssp_timeseries.sel(time=ssp_timeseries["time.year"] >= train_period_end + 1),
            ],
            dim="time",
        )
        result = xr.concat(
            [
                historical_and_ssp.sel(time=historical_and_ssp["time.year"] < predict_period_start),
                model_scenario.sel(time=model_scenario["time.year"] >= predict_period_start),
            ],
            dim="time",
        )
    else:
        # Non-SAI: historical up to predict_period_start, then scenario.
        result = xr.concat(
            [
                model_hist.sel(time=model_hist["time.year"] < predict_period_start),
                model_scenario.sel(time=model_scenario["time.year"] >= predict_period_start),
            ],
            dim="time",
        )

    _assert_stitched_continuity(result)
    return result


class BCSDPipeline:
    """
    Three-stage BCSD downscaling pipeline with automatic caching.

    Stages:
    1. Prepare (i.e. coarsen) training dataset to be at same model resolution as GCM
    2. Fit model between coarsened training dataset and historical GCM simulation
    3. Apply model on GCM simulation (whether historical or future)

    This class orchestrates the BCSD workflow, automatically caching intermediate
    artifacts to enable efficient reuse across multiple runs. Each stage checks
    for cached outputs before computing.

    Cache/force behavior summary
    ----------------------------
    - ``force=False`` (default): a stage returns immediately when its output
      artifact already exists.
    - ``force=True``: a stage recomputes and overwrites its own output artifact.
    - ``fit_historical`` and ``transform_scenario`` validate dependency
      artifacts before any cache-hit early return.
    - Cache checks are existence checks only; cached content is not validated
      for integrity or schema compatibility at read time.

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
        self.cache = ArtifactCache.from_config(config)
        self._state = {}

        if config.historical_ensemble_member is not None:
            parts = [
                f"ensemble_member={config.ensemble_member!r}",
                f"historical={config.historical_ensemble_member!r}",
            ]
            if config.ssp245_ensemble_member is not None:
                parts.append(f"ssp245_bridge={config.ssp245_ensemble_member!r}")
            logger.info("Lineage resolved — %s", "  ".join(parts))

    @staticmethod
    def _icechunk_storage(path: str):
        """Create icechunk Storage from an S3 or local path."""
        if path.startswith("s3://"):
            path_no_scheme = path[len("s3://") :]
            bucket, _, prefix = path_no_scheme.partition("/")
            return icechunk.s3_storage(bucket=bucket, prefix=prefix)
        else:
            return icechunk.local_filesystem_storage(path=path)

    def _build_output_attrs(
        self,
        source_dataset: BaseDataset | None,
    ) -> dict:
        """Adds attrs to output datasets"""
        dataset_attrs = {
            "author": "CarbonPlan",
            "processing": "BCSD (quantile mapping bias correction + spatial downscaling)",
            "bias_correction_method": self.config.mapping_type,
            "downscaling_method": self.config.downscaling_method,
            "train_period": f"{self.config.train_period_start}-{self.config.train_period_end}",
            "observation_dataset": "ERA5",
            "creation_date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "srm_version": importlib.metadata.version("srm"),
            "model": self.config.gcm,
            "scenario": self.config.scenario or "historical",
            "variable": self.config.variable,
            "ensemble_member": self.config.ensemble_member,
            "historical_ensemble_member": self.config.historical_ensemble_member
            or self.config.ensemble_member,
            "ssp245_ensemble_member": self.config.ssp245_ensemble_member
            or self.config.ensemble_member,
        }

        if source_dataset is not None:
            dataset_attrs["license"] = source_dataset.license
            dataset_attrs["citation"] = source_dataset.citation

        return dataset_attrs

    def _write_to_icechunk(
        self,
        da: xr.DataArray,
        path: str,
        commit_message: str,
        encoding: dict | None = None,
        dataset_attrs: dict | None = None,
    ) -> str:
        """Write a DataArray to an icechunk store and commit atomically."""
        storage = self._icechunk_storage(path)
        repo = icechunk.Repository.open_or_create(storage)
        session = repo.writable_session("main")
        ds = da.to_dataset()
        if dataset_attrs is not None:
            ds.attrs = dataset_attrs
        to_icechunk(ds, session, mode="w", encoding=encoding or {})
        return session.commit(commit_message, rebase_with=icechunk.ConflictDetector())

    @staticmethod
    def _build_ocean_mask(da: xr.DataArray) -> xr.DataArray:
        """Compute a land/ocean mask aligned to da's spatial grid.

        Returns a boolean DataArray where True = land (keep) and False = ocean (mask).
        Uses GSHHS high-resolution coastline boundaries from the catalog. Ocean pixels
        in the output should be set to NaN via ``da.where(mask)``.
        """
        import xproj  # noqa
        from rasterix.rasterize import geometry_mask

        from srm.datasets import catalog

        coast = catalog.get("ocean-mask").to_geodataframe()
        # Sort lat descending — required by rusterize; .where() re-aligns by coordinate
        template = (
            da.isel(time=0).sortby("lat", ascending=False).proj.assign_crs(spatial_ref="epsg:4326")
        )
        return ~geometry_mask(
            template, coast[["geom"]], all_touched=True, engine="rusterize", xdim="lon", ydim="lat"
        ).drop_vars("spatial_ref", errors="ignore")

    def _open_from_icechunk(self, path: str) -> xr.Dataset:
        """Open a dataset from an icechunk store."""
        storage = self._icechunk_storage(path)
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        return xr.open_dataset(session.store, engine="zarr", consolidated=False, chunks="auto")

    def prepare_observations(self, force: bool = False) -> str:
        """
        Stage 1: Regrid observations to GCM grid.

        This stage loads ERA5 observations and regrids them to the coarse GCM
        grid using local area averaging. The result is cached and reused across
        all ensemble members and scenarios for this GCM/variable combination.

        Parameters
        ----------
        force : bool, optional
            If ``False``, return the cached stage output when present.
            If ``True``, recompute this stage and overwrite cached output.

        Returns
        -------
        str
            Path to the stage output artifact.

        Notes
        -----
        This stage does not depend on prior stage artifacts. Cache-hit behavior
        is based on artifact existence at ``self.cache.obs_path``.
        """
        output_path = self.cache.obs_path

        # Check whether regridded dataset already exists, if so (and you don't
        # have the force flag enabled which allows overwrite) use the existing dataset.
        # Note: this does not check anything about the data at the output_path -
        # if it is corrupted in any way or doesn't match the attributes of the
        # config it won't fail.
        if self.cache.exists(output_path) and not force:
            logger.info("✓ Using cached observations: %s", output_path)
            return output_path

        logger.info(
            "Computing observation regridding for %s/%s", self.config.gcm, self.config.variable
        )

        t0 = time.perf_counter()
        obs_fine = get_obs(var=self.config.variable)
        obs_fine = obs_fine.drop_vars("spatial_ref", errors="ignore")
        model_grid = get_experiment(
            gcm=self.config.gcm, scenario="historical", var=self.config.variable
        )
        model_grid = model_grid.drop_vars("spatial_ref", errors="ignore")
        if self.config.subset_bounds:
            lat_min, lat_max, lon_min, lon_max = self.config.subset_bounds
            lat_bounds = (lat_min, lat_max)
            lon_bounds = (lon_min, lon_max)
            obs_fine = subset_space(obs_fine, lat_bounds=lat_bounds, lon_bounds=lon_bounds)
            model_grid = subset_space(model_grid, lat_bounds=lat_bounds, lon_bounds=lon_bounds)
        logger.info("Loaded observations (%.2fs)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
            obs_coarse = interpolate_fine_to_coarse_grid(
                da_fine_to_coarsen=obs_fine, da_coarse_grid=model_grid
            )
        logger.info("Regridded observations to coarse grid (%.2fs)", time.perf_counter() - t0)

        if self.config.rechunk_workflow:
            obs_coarse = rechunk(obs_coarse, pattern="full_space")

        t0 = time.perf_counter()
        obs_coarse.name = self.config.variable
        hist_dataset = _catalog.datasets.get(f"{self.config.gcm}-historical-icechunk")
        dataset_attrs = self._build_output_attrs(hist_dataset)
        self._write_to_icechunk(
            obs_coarse, output_path, "write complete", dataset_attrs=dataset_attrs
        )
        logger.info("✓ Cached observations: %s (%.2fs)", output_path, time.perf_counter() - t0)

        return output_path

    def _load_gcm_obs(self) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Load obs_coarse (from cache), obs_fine, and model_hist, subsetted to training period."""
        deps = self.cache.check_dependencies("fit_historical", self.config)
        obs_coarse = self._open_from_icechunk(deps["obs_regridded"][1])[self.config.variable]

        obs_fine = get_obs(var=self.config.variable)
        obs_fine = obs_fine.drop_vars("spatial_ref", errors="ignore")

        model_hist = get_experiment(
            gcm=self.config.gcm, scenario="historical", var=self.config.variable
        )
        model_hist = model_hist.sel(
            ensemble_member=self.config.historical_ensemble_member or self.config.ensemble_member
        )
        model_hist = model_hist.drop_vars("spatial_ref", errors="ignore")

        if self.config.subset_bounds:
            lat_min, lat_max, lon_min, lon_max = self.config.subset_bounds
            lat_bounds = (lat_min, lat_max)
            lon_bounds = (lon_min, lon_max)
            obs_fine = subset_space(obs_fine, lat_bounds=lat_bounds, lon_bounds=lon_bounds)
            model_hist = subset_space(model_hist, lat_bounds=lat_bounds, lon_bounds=lon_bounds)

        train_slice = slice(f"{self.config.train_period_start}", f"{self.config.train_period_end}")
        obs_coarse = obs_coarse.sel(time=train_slice)
        obs_fine = obs_fine.sel(time=train_slice)
        model_hist = model_hist.sel(time=train_slice)

        return obs_coarse, obs_fine, model_hist

    def _apply_bias_correction(
        self,
        obs_coarse: xr.DataArray,
        model_hist: xr.DataArray,
    ) -> xr.DataArray:
        """Apply quantile mapping to historical GCM data.

        Uses nonparametric mapping for the nonparametric_hybrid case because modeled
        historical is always within its own range, making the parametric tail unnecessary.
        """
        mapping_type = (
            "nonparametric"
            if self.config.mapping_type == "nonparametric_hybrid"
            else self.config.mapping_type
        )
        debiaser = _make_debiaser(
            variable=self.config.variable,
            mapping_type=mapping_type,
            detrending="no_detrending",
            running_window_mode=self.config.do_windowing,
            running_window_length=self.config.running_window_length,
            running_window_step_length=1,
            running_window_mode_over_years_of_cm_future=False,
        )

        obs_np = obs_coarse.as_numpy().values
        cm_hist_np = model_hist.as_numpy().values

        debiased_np = debiaser.apply(
            obs=obs_np,
            cm_hist=cm_hist_np,
            cm_future=cm_hist_np,  # debias historical with itself
            time_obs=obs_coarse["time"].values,
            time_cm_hist=model_hist["time"].values,
            time_cm_future=model_hist["time"].values,
            parallel=True,
            nr_processes=dask.system.CPU_COUNT,
            progressbar=False,
            failsafe=True,  # ocean pixels have NaN obs; fill with NaN rather than crash
        )

        return xr.DataArray(
            data=debiased_np,
            coords={"lat": model_hist["lat"], "lon": model_hist["lon"], "time": model_hist["time"]},
            dims=["time", "lat", "lon"],
        )

    def _apply_spatial_downscaling(
        self,
        debiased: xr.DataArray,
        obs_coarse: xr.DataArray,
        obs_fine: xr.DataArray,
    ) -> xr.DataArray:
        """Spatially disaggregate coarse debiased data to fine resolution."""
        downscaled = downscale_from_coarse(
            da=debiased,
            obs_coarse=obs_coarse.as_numpy(),
            obs_fine=obs_fine.as_numpy(),
            method=self.config.downscaling_method,
            clim_method=self.config.downscaling_clim_method,
        )
        return downscaled.chunk({"time": SHARD_TIME, "lat": SHARD_LAT, "lon": SHARD_LON})

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
            If ``False``, return the cached stage output when present.
            If ``True``, recompute this stage and overwrite cached output.

        Returns
        -------
        str
            Path to the stage output artifact.

        Raises
        ------
        ValueError
            If obs_regridded dependency is missing

        Notes
        -----
        The output (fully downscaled historical data) is written to the cache as a
        data artifact for the historical period. It is also used as a completion gate:
        ``transform_scenario`` checks that this artifact exists before it will run, but
        does *not* load it as an input (scenario runs re-load the raw GCM historical data
        for their own bias-correction training). Setting ``force=True`` reruns all three
        computation steps and overwrites the cached artifact; ``force=False`` skips all
        three and returns the existing path immediately.

        Dependency validation is always performed before checking this stage's
        cache-hit short-circuit.
        """
        self.cache.validate_dependencies("fit_historical", self.config)

        output_path = self.cache.historical_path

        if self.cache.exists(output_path) and not force:
            logger.info("✓ Using cached historical: %s", output_path)
            return output_path

        logger.info(
            "Computing historical downscaling for %s/%s/%s",
            self.config.gcm,
            self.config.variable,
            self.config.ensemble_member,
        )

        t0 = time.perf_counter()
        obs_coarse, obs_fine, model_hist = self._load_gcm_obs()
        logger.info("Loaded data (%.2fs)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        model_hist_debiased = self._apply_bias_correction(obs_coarse, model_hist)
        logger.info("Bias corrected historical (%.2fs)", time.perf_counter() - t0)

        if self.config.save_intermediate:
            t0 = time.perf_counter()
            debiased_path = self.cache.get_debiased_historical_path(self.config)
            model_hist_debiased.name = self.config.variable
            model_hist_debiased.attrs = model_hist.attrs
            self._write_to_icechunk(model_hist_debiased, debiased_path, "write complete")
            logger.info(
                "✓ Saved debiased historical: %s (%.2fs)", debiased_path, time.perf_counter() - t0
            )

        t0 = time.perf_counter()
        model_hist_downscaled = self._apply_spatial_downscaling(
            model_hist_debiased, obs_coarse, obs_fine
        )
        logger.info("Spatially disaggregated (%.2fs)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        model_hist_downscaled.name = self.config.variable
        hist_dataset = _catalog.datasets.get(f"{self.config.gcm}-historical-icechunk")
        dataset_attrs = self._build_output_attrs(hist_dataset)
        self._write_to_icechunk(
            da=model_hist_downscaled,
            path=output_path,
            commit_message="write complete",
            encoding=make_encoding(self.config.variable),
            dataset_attrs=dataset_attrs,
        )
        logger.info("✓ Cached historical: %s (%.2fs)", output_path, time.perf_counter() - t0)

        return output_path

    def _load_scenario_data(
        self,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray | None]:
        """Load obs_coarse, obs_fine, model_hist, model_scenario, and optionally ssp_timeseries.

        Returns (obs_coarse, obs_fine, model_hist, model_scenario, ssp_timeseries).
        obs_coarse/obs_fine/model_hist are subsetted to the training period;
        model_scenario to the predict period. ssp_timeseries is None for non-SAI scenarios.
        """
        deps = self.cache.check_dependencies("transform_scenario", self.config)
        obs_coarse = self._open_from_icechunk(deps["obs_regridded"][1])[self.config.variable]

        obs_fine = get_obs(var=self.config.variable)
        obs_fine = obs_fine.drop_vars("spatial_ref", errors="ignore")

        model_hist = get_experiment(
            gcm=self.config.gcm, scenario="historical", var=self.config.variable
        )
        model_hist = model_hist.sel(
            ensemble_member=self.config.historical_ensemble_member or self.config.ensemble_member
        )
        model_hist = model_hist.drop_vars("spatial_ref", errors="ignore")

        model_scenario = get_experiment(
            gcm=self.config.gcm, scenario=self.config.scenario, var=self.config.variable
        )
        model_scenario = model_scenario.sel(ensemble_member=self.config.ensemble_member)
        model_scenario = model_scenario.drop_vars("spatial_ref", errors="ignore")

        # SAI scenarios need an SSP245 bridge to fill the gap between historical and SAI start
        ssp_timeseries: xr.DataArray | None = None
        if self.config.is_sai_scenario:
            ssp_timeseries = get_experiment(
                gcm=self.config.gcm, scenario="SSP245", var=self.config.variable
            )
            ssp_timeseries = ssp_timeseries.sel(
                ensemble_member=self.config.ssp245_ensemble_member or self.config.ensemble_member
            )
            ssp_timeseries = ssp_timeseries.drop_vars("spatial_ref", errors="ignore")

        if self.config.subset_bounds:
            lat_min, lat_max, lon_min, lon_max = self.config.subset_bounds
            lat_bounds = (lat_min, lat_max)
            lon_bounds = (lon_min, lon_max)
            obs_fine = subset_space(obs_fine, lat_bounds=lat_bounds, lon_bounds=lon_bounds)
            model_hist = subset_space(model_hist, lat_bounds=lat_bounds, lon_bounds=lon_bounds)
            model_scenario = subset_space(
                model_scenario, lat_bounds=lat_bounds, lon_bounds=lon_bounds
            )
            if ssp_timeseries is not None:
                ssp_timeseries = subset_space(
                    ssp_timeseries, lat_bounds=lat_bounds, lon_bounds=lon_bounds
                )

        train_slice = slice(f"{self.config.train_period_start}", f"{self.config.train_period_end}")
        obs_coarse = obs_coarse.sel(time=train_slice)
        obs_fine = obs_fine.sel(time=train_slice)
        model_hist = model_hist.sel(time=train_slice)
        model_scenario = model_scenario.sel(
            time=slice(f"{self.config.predict_period_start}", f"{self.config.predict_period_end}")
        )

        return obs_coarse, obs_fine, model_hist, model_scenario, ssp_timeseries

    def _detrend_scenario(
        self,
        model_hist: xr.DataArray,
        model_scenario: xr.DataArray,
        ssp_timeseries: xr.DataArray | None,
    ) -> tuple[xr.DataArray, xr.DataArray | None]:
        """Optionally detrend the scenario timeseries.

        Returns (scenario_detrended, scenario_trend). When detrending is disabled,
        returns (model_scenario, None) and scenario_trend will be None.

        For SAI scenarios, stitches in SSP245 data to bridge the gap between the end of
        historical (2014/2015) and the SAI simulation start (~2035) before detrending,
        ensuring a smooth baseline for trend removal.
        """
        if not self.config.detrend_data:
            return model_scenario, None

        if self.config.rechunk_workflow:
            t0 = time.perf_counter()
            model_hist = rechunk(model_hist, pattern="full_time").persist()
            model_scenario = rechunk(model_scenario, pattern="full_time").persist()
            if ssp_timeseries is not None:
                ssp_timeseries = rechunk(ssp_timeseries, pattern="full_time").persist()
            logger.info("Rechunked for detrending (%.2fs)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        historical_scenario = stitch_historical_scenario(
            model_hist=model_hist,
            model_scenario=model_scenario,
            train_period_end=self.config.train_period_end,
            predict_period_start=self.config.predict_period_start,
            ssp_timeseries=ssp_timeseries if self.config.is_sai_scenario else None,
        )

        da_baseline_clim = calculate_baseline_climatology(
            da_baseline=model_hist,
            baseline_period_start=self.config.train_period_start,
            baseline_period_end=self.config.train_period_end,
        )

        scenario_detrended, scenario_trend = detrend(
            da=historical_scenario,
            da_baseline_clim=da_baseline_clim,
            detrend_method=self.config.detrend_method,
        )

        predict_slice = slice(
            f"{self.config.predict_period_start}", f"{self.config.predict_period_end}"
        )
        scenario_detrended = scenario_detrended.sel(time=predict_slice)
        scenario_trend = scenario_trend.sel(time=predict_slice)
        logger.info("Detrended scenario (%.2fs)", time.perf_counter() - t0)

        if self.config.save_intermediate:
            t0 = time.perf_counter()
            detrended_path = self.cache.get_detrended_scenario_path(self.config)
            scenario_detrended.name = self.config.variable
            self._write_to_icechunk(
                rechunk(scenario_detrended, pattern="full_space"),
                detrended_path,
                "write complete",
            )
            logger.info(
                "✓ Saved detrended scenario: %s (%.2fs)", detrended_path, time.perf_counter() - t0
            )

            t0 = time.perf_counter()
            trend_path = self.cache.get_trend_scenario_path(self.config)
            scenario_trend.name = self.config.variable
            scenario_trend.attrs = model_scenario.attrs
            self._write_to_icechunk(
                rechunk(scenario_trend, pattern="full_space"),
                trend_path,
                "write complete",
            )
            logger.info("✓ Saved scenario trend: %s (%.2fs)", trend_path, time.perf_counter() - t0)

        return scenario_detrended, scenario_trend

    def _apply_bias_correction_scenario(
        self,
        obs_coarse: xr.DataArray,
        model_hist: xr.DataArray,
        scenario_detrended: xr.DataArray,
    ) -> xr.DataArray:
        """Apply quantile mapping to the (optionally detrended) scenario.

        For nonparametric_hybrid: runs both parametric and nonparametric debiasers and
        blends them — parametric where the scenario falls outside the historical range,
        nonparametric everywhere else.
        """
        obs_np = obs_coarse.as_numpy().values
        cm_hist_np = model_hist.as_numpy().values
        cm_future_np = scenario_detrended.load().values

        common_kwargs = dict(
            variable=self.config.variable,
            detrending="no_detrending",
            running_window_mode=self.config.do_windowing,
            running_window_length=self.config.running_window_length,
            running_window_step_length=1,
            running_window_mode_over_years_of_cm_future=False,
        )
        apply_kwargs = dict(
            obs=obs_np,
            cm_hist=cm_hist_np,
            cm_future=cm_future_np,
            time_obs=obs_coarse["time"].values,
            time_cm_hist=model_hist["time"].values,
            time_cm_future=scenario_detrended["time"].values,
            parallel=True,
            nr_processes=dask.system.CPU_COUNT,
            progressbar=False,
            failsafe=True,  # ocean pixels have NaN obs; fill with NaN rather than crash
        )

        if self.config.mapping_type in ["parametric", "nonparametric"]:
            debiased_np = _make_debiaser(
                mapping_type=self.config.mapping_type, **common_kwargs
            ).apply(**apply_kwargs)

        elif self.config.mapping_type == "nonparametric_hybrid":
            parametric_np = _make_debiaser(mapping_type="parametric", **common_kwargs).apply(
                **apply_kwargs
            )
            nonparametric_np = _make_debiaser(mapping_type="nonparametric", **common_kwargs).apply(
                **apply_kwargs
            )

            out_of_range = calculate_out_of_range_mask(
                model_hist=model_hist,
                scenario_detrended=scenario_detrended,
                center_window=self.config.running_window_length,
            )
            debiased_np = np.where(out_of_range.values, parametric_np, nonparametric_np)

        else:
            raise ValueError(
                "mapping_type must be 'parametric', 'nonparametric', or 'nonparametric_hybrid'."
            )

        return xr.DataArray(
            data=debiased_np,
            coords={
                "lat": scenario_detrended["lat"],
                "lon": scenario_detrended["lon"],
                "time": scenario_detrended["time"],
            },
            dims=["time", "lat", "lon"],
        )

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
            If ``False``, return the cached stage output when present.
            If ``True``, recompute this stage and overwrite cached output.

        Returns
        -------
        str
            Path to the stage output artifact.

        Raises
        ------
        ValueError
            If obs_regridded or historical dependencies are missing,
            or if scenario is not specified in config

        Notes
        -----
        Dependency validation is always performed before checking this stage's
        cache-hit short-circuit.
        """
        if self.config.scenario is None:
            raise ValueError("scenario must be specified in config for transform_scenario")

        self.cache.validate_dependencies("transform_scenario", self.config)

        output_path = self.cache.scenario_path

        if self.cache.exists(output_path) and not force:
            logger.info("✓ Using cached scenario: %s", output_path)
            return output_path

        logger.info(
            "Computing scenario downscaling for %s/%s/%s/%s",
            self.config.gcm,
            self.config.variable,
            self.config.ensemble_member,
            self.config.scenario,
        )

        t0 = time.perf_counter()
        obs_coarse, obs_fine, model_hist, model_scenario, ssp_timeseries = (
            self._load_scenario_data()
        )
        logger.info("Loaded data (%.2fs)", time.perf_counter() - t0)

        scenario_detrended, scenario_trend = self._detrend_scenario(
            model_hist, model_scenario, ssp_timeseries
        )

        t0 = time.perf_counter()
        scenario_debiased = self._apply_bias_correction_scenario(
            obs_coarse, model_hist, scenario_detrended
        )
        logger.info("Bias corrected scenario (%.2fs)", time.perf_counter() - t0)

        if self.config.save_intermediate:
            t0 = time.perf_counter()
            debiased_path = self.cache.get_debiased_scenario_path(self.config)
            scenario_debiased.name = self.config.variable
            self._write_to_icechunk(scenario_debiased, debiased_path, "write complete")
            logger.info(
                "✓ Saved debiased scenario: %s (%.2fs)", debiased_path, time.perf_counter() - t0
            )

        if scenario_trend is not None:
            t0 = time.perf_counter()
            scenario_debiased = retrend(
                bias_corrected_detrended=scenario_debiased,
                trend_on_daily_timestep=scenario_trend,
                detrend_method=self.config.detrend_method,
            )
            logger.info("Re-trended scenario (%.2fs)", time.perf_counter() - t0)

        if self.config.save_intermediate:
            t0 = time.perf_counter()
            debiased_path = self.cache.get_debiased_retrended_scenario_path(self.config)
            scenario_debiased.name = self.config.variable
            scenario_debiased.attrs = model_scenario.attrs
            self._write_to_icechunk(scenario_debiased, debiased_path, "write complete")
            logger.info(
                "✓ Saved debiased retrended scenario: %s (%.2fs)",
                debiased_path,
                time.perf_counter() - t0,
            )

        t0 = time.perf_counter()
        scenario_downscaled = self._apply_spatial_downscaling(
            scenario_debiased, obs_coarse, obs_fine
        )
        logger.info("Spatially disaggregated (%.2fs)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        if self.config.apply_ocean_mask:
            scenario_downscaled = scenario_downscaled.where(
                self._build_ocean_mask(scenario_downscaled)
            ).chunk({"time": SHARD_TIME, "lat": SHARD_LAT, "lon": SHARD_LON})
        scenario_downscaled.name = self.config.variable
        scenario_dataset = _catalog.datasets.get(
            f"{self.config.gcm}-{self.config.scenario}-icechunk"
        )
        dataset_attrs = self._build_output_attrs(scenario_dataset)
        self._write_to_icechunk(
            scenario_downscaled,
            output_path,
            "write complete",
            dataset_attrs=dataset_attrs,
            encoding=make_encoding(self.config.variable),
        )
        logger.info("✓ Saved scenario output: %s (%.2fs)", output_path, time.perf_counter() - t0)

        return output_path

    def run_full_pipeline(self, force: bool = False) -> str:
        """
        Run all three stages in sequence.

        Convenience method that executes prepare_observations, fit_historical,
        and transform_scenario in order.

        Parameters
        ----------
        force : bool, optional
            Passed through to each stage:
            - ``False``: each stage may short-circuit on its own cache hit.
            - ``True``: all stages recompute and overwrite their outputs.

        Returns
        -------
        str
            Path to final scenario stage output artifact.
        """
        self.prepare_observations(force=force)
        self.fit_historical(force=force)
        # `transform_scenario` depends on fit_historical only as a completion
        # gate (artifact existence); it does not read the cached historical
        # output as data input.
        return self.transform_scenario(force=force)
