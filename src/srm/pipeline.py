"""
BCSD pipeline with three-stage architecture and automatic caching.

Implements the full bias-correction spatial disaggregation workflow via
:class:`BCSDPipeline`. Stages run in order: ``prepare_observations`` (once per
GCM/variable), ``fit_historical`` (once per GCM/variable/ensemble), and
``transform_scenario`` (once per GCM/variable/ensemble/scenario).
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
from ibicus.utils import PrecipitationHurdleModelGamma
from icechunk.xarray import to_icechunk

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.cache import ArtifactCache, StoreLocation
from srm.config import _ensure_root_group, _icechunk_storage_for_path
from srm.datasets import catalog as _catalog
from srm.downscaling_utils import (
    calculate_baseline_climatology,
    derive_tasmin,
    detrend,
    downscale_from_coarse,
    get_experiment,
    get_historical_experiment,
    get_obs,
    interpolate_fine_to_coarse_grid,
    rechunk,
    retrend,
    subset_space,
    swap_temperature_extremes,
)
from srm.encoding import (
    SHARD_LAT,
    SHARD_LAT_COARSE,
    SHARD_LON,
    SHARD_LON_COARSE,
    SHARD_TIME,
    SHARD_TIME_COARSE,
    make_coarse_encoding,
    make_encoding,
)
from srm.utils import get_variable

logger = logging.getLogger(__name__)


def _make_debiaser(variable: str, distribution=None, **kwargs):
    if distribution is None:
        if variable in ["tas", "tasmax"]:
            distribution = scipy.stats.norm
        elif variable in ["hurs", "rsds", "dtr"]:
            distribution = scipy.stats.beta
        elif variable == "pr":
            distribution = PrecipitationHurdleModelGamma
    return QuantileMapping(distribution=distribution, **kwargs)


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

    out_of_range_low = scenario_detrended < rolling_doy_min.sel(dayofyear=doy)
    out_of_range_high = scenario_detrended > rolling_doy_max.sel(dayofyear=doy)

    out_of_range = out_of_range_low | out_of_range_high

    return out_of_range, out_of_range_low, out_of_range_high


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
    (``train_period_end``) and the first available year of the SAI simulation.
    The combined series is then concatenated with the SAI scenario. The SAI
    splice point is inferred from the first timestep in ``model_scenario``
    rather than from ``predict_period_start``, because the scenario data may
    start later than the requested prediction window (e.g. G6-1.5K begins in
    2035 even when ``predict_period_start`` is 2015).

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
        First year of the prediction period. Used as the stitch boundary for
        non-SAI scenarios. For SAI scenarios the splice point is inferred from
        the first year present in ``model_scenario``.
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
        if model_scenario.time.size == 0:
            raise ValueError(
                "model_scenario contains no timesteps for the requested predict period. "
                "Verify that predict_period_start/predict_period_end overlap the scenario's "
                "available time range."
            )
        # Use the first year present in the (already-sliced) scenario array as
        # the splice point.  predict_period_start may be earlier than the SAI
        # data start (e.g. predict_period_start=2015 but G6-1.5K begins 2035),
        # so we cannot rely on it here.
        sai_start_year = int(model_scenario["time.year"].min())
        ssp_min_year = int(ssp_timeseries["time.year"].min())
        if ssp_min_year >= sai_start_year:
            raise ValueError(
                f"SSP245 bridge data starts at {ssp_min_year} but the SAI scenario starts "
                f"at {sai_start_year}. The bridge must cover the gap "
                f"{train_period_end + 1}–{sai_start_year - 1}. "
                f"Use a bridge dataset that spans this period."
            )
        # SAI: historical ≤ train_period_end, then SSP from train_period_end+1
        # up to (but not including) the SAI start, then SAI data.
        # Drop ensemble_member scalar coords before concat — historical and bridge may carry
        # different member values (e.g. r1i1p4f2 vs r01), which causes MergeError in older xarray.
        # Re-attach from model_scenario afterward: the stitched array represents the scenario's
        # timeseries, so model_scenario.ensemble_member is the authoritative identity.
        scenario_member_coord = model_scenario.coords.get("ensemble_member")
        historical_and_ssp = xr.concat(
            [
                model_hist.sel(time=model_hist["time.year"] <= train_period_end).drop_vars(
                    "ensemble_member", errors="ignore"
                ),
                ssp_timeseries.sel(
                    time=ssp_timeseries["time.year"] >= train_period_end + 1
                ).drop_vars("ensemble_member", errors="ignore"),
            ],
            dim="time",
        )
        result = xr.concat(
            [
                historical_and_ssp.sel(time=historical_and_ssp["time.year"] < sai_start_year),
                model_scenario.sel(time=model_scenario["time.year"] >= sai_start_year).drop_vars(
                    "ensemble_member", errors="ignore"
                ),
            ],
            dim="time",
        )
        if scenario_member_coord is not None:
            result = result.assign_coords(ensemble_member=scenario_member_coord)
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

    def __init__(self, config: BCSDConfig, options: PipelineOptions):
        """
        Initialize pipeline with configuration and operational options.

        Ensemble member lineage is resolved automatically from the lineage
        table. Unknown GCM / scenario / member combinations are silently
        skipped (lineage members fall back to ensemble_member).

        Parameters
        ----------
        config : BCSDConfig
            Run-identity configuration for the BCSD run
        options : PipelineOptions
            Operational settings (storage paths, runtime flags)
        """
        self.config = config
        self.options = options
        self.cache = ArtifactCache.from_config(config, options)
        self._state = {}

        self._hist_member = config.ensemble_member
        self._ssp245_member = config.ensemble_member
        self._ssp245_esgf_member: str | None = None
        if config.scenario is not None:
            from srm.lineage import resolve_member_lineage

            try:
                self._hist_member, self._ssp245_member, self._ssp245_esgf_member = (
                    resolve_member_lineage(
                        config.gcm, config.scenario, config.ensemble_member, config.variable
                    )
                )
            except KeyError:
                pass

        if self._hist_member != config.ensemble_member:
            parts = [
                f"ensemble_member={config.ensemble_member!r}",
                f"historical={self._hist_member!r}",
            ]
            if self._ssp245_member != config.ensemble_member:
                parts.append(f"ssp245_bridge={self._ssp245_member!r}")
            logger.info("Lineage resolved — %s", "  ".join(parts))

    def _build_output_attrs(self) -> dict:
        """Build dataset-level attributes for pipeline output artifacts."""
        version = importlib.metadata.version("srm")
        return {
            # CF-standard — flat
            "Conventions": "CF-1.8",
            "institution": "CarbonPlan",
            "history": (
                f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}: "
                f"BCSD downscaling by srm v{version}"
            ),
            # Pipeline provenance — namespaced
            "srm_downscaling:version": version,
            "srm_downscaling:gcm": self.config.gcm,
            "srm_downscaling:scenario": self.config.scenario or "historical",
            "srm_downscaling:variable": self.config.variable,
            "srm_downscaling:ensemble_member": self.config.ensemble_member,
            "srm_downscaling:historical_ensemble_member": self._hist_member,
            "srm_downscaling:ssp245_ensemble_member": self._ssp245_member,
            "srm_downscaling:observation_dataset": self.config.obs_dataset,
            "srm_downscaling:bias_correction_method": self.config.debias_approach,
            "srm_downscaling:downscaling_method": self.config.variable_config.downscaling_method,
            "srm_downscaling:train_period": (
                f"{self.config.train_period_start}-{self.config.train_period_end}"
            ),
            "srm_downscaling:config_hash": self.config.config_hash,
            "srm_downscaling:config_json": self.config.model_dump_json(),
            "srm_downscaling:creation_date": datetime.now(UTC).strftime("%Y-%m-%d"),
        }

    def _write_to_icechunk(
        self,
        da: xr.DataArray,
        loc: StoreLocation,
        encoding: dict | None = None,
        dataset_attrs: dict | None = None,
        force: bool = False,
    ) -> str:
        """Write a DataArray to an icechunk group and commit atomically.

        Retries up to 3 times on RebaseFailedError. Concurrent VMs writing to
        sibling groups race to create their shared parent zarr group for the
        first time, producing a structural conflict. On retry the parent exists
        and the commit succeeds cleanly.
        """
        branch = self.cache._branch_for()
        storage = _icechunk_storage_for_path(loc.store_path)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                repo = icechunk.Repository.open_or_create(storage)
                root_snapshot_id = _ensure_root_group(repo)
                if branch not in repo.list_branches():
                    repo.create_branch(branch, root_snapshot_id)
                session = repo.writable_session(branch)
                ds = da.to_dataset()
                if dataset_attrs is not None:
                    ds.attrs = dataset_attrs
                # Drop non-index auxiliary coords that leak from intermediate ops:
                # dayofyear - broadcast residual from .sel(dayofyear=...) in bias correction
                # ensemble_member - string scalar from raw GCM source (already in attrs),
                #   causes NotImplementedError when opening with chunks="auto" (object dtype)
                _drop = [c for c in ("dayofyear", "ensemble_member") if c in ds.coords]
                if _drop:
                    ds = ds.drop_vars(_drop)
                # fix incompatible dask chunk sizes in encoding
                for coord in list(ds.coords):
                    ds[coord].encoding.pop("chunks", None)
                    ds[coord].encoding.pop("shards", None)
                to_icechunk(ds, session, mode="w", encoding=encoding or {}, group=loc.group)
                commit_id = session.commit(loc.group, rebase_with=icechunk.ConflictDetector())
                break
            except icechunk.RebaseFailedError:
                if attempt == max_attempts - 1:
                    raise
                logger.warning(
                    "Rebase conflict on %s (attempt %d/%d), retrying",
                    loc.group,
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(0.5 * (attempt + 1))

        if force:
            history = list(repo.ancestry(branch=branch))
            if len(history) > 2:
                keep_from = history[1].written_at
                n_expired = len(repo.expire_snapshots(older_than=keep_from))
                gc_result = repo.garbage_collect(keep_from)
                logger.info(
                    "GC after force overwrite of %s: expired %d snapshots, collected %s",
                    loc.group,
                    n_expired,
                    gc_result,
                )

        return commit_id

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

    def _open_from_icechunk(self, loc: StoreLocation, chunks="auto") -> xr.Dataset:
        """Open a zarr group from an icechunk store.

        Parameters
        ----------
        loc : StoreLocation
            Store path and group to open.
        chunks : str or dict, optional
            Dask chunking for the opened dataset. Defaults to ``"auto"``. Pass an
            explicit shard-aligned dict (e.g. ``{"time": SHARD_TIME, ...}``) when the
            caller needs per-shard streaming reads rather than auto-sized chunks — the
            store-to-store reconcile relies on this to avoid materialising the full
            global fine array (reconcile OOM).
        """
        branch = self.cache._branch_for()
        storage = _icechunk_storage_for_path(loc.store_path)
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session(branch=branch)
        return xr.open_dataset(
            session.store, engine="zarr", consolidated=False, chunks=chunks, group=loc.group
        )

    def reconcile_temperature_extremes(
        self,
        tasmin_loc: StoreLocation,
        tasmax_loc: StoreLocation,
        *,
        tasmin_fine: xr.DataArray | None = None,
        force: bool = False,
    ) -> None:
        """Dedicated reconcile step: enforce ``tasmax >= tasmin`` on the fine outputs.

        Independent spatial disaggregation of tasmax and tasmin can leave a few fine
        cells with ``tasmax < tasmin`` (issue #331). Following the NEX-GDDP-CMIP6 v2
        final sweep, this step reads the sibling fine tasmax, swaps the offending
        cells against tasmin, and writes *both* corrected fields back to their own
        stores. It runs after both fine outputs are produced (the #363 wave-gating
        guarantees tasmax is final before tasmin), and can also be re-run standalone
        against the persisted outputs.

        The reconciliation is idempotent and structurally monotone (see
        :func:`swap_temperature_extremes`); ``qaqc.validate_temp_consistency`` is the
        output-QA gate that catches any residual inversion (e.g. from a tasmax-only
        rerun that has not yet been re-reconciled).

        Parameters
        ----------
        tasmin_loc : StoreLocation
            Location of this stage's fine tasmin output (written here, corrected).
        tasmax_loc : StoreLocation
            Location of the sibling fine tasmax output (rewritten here, corrected).
        tasmin_fine : xr.DataArray, optional
            The freshly downscaled tasmin. If omitted, tasmin is read back from
            ``tasmin_loc`` (standalone reconcile of already-persisted outputs).
        force : bool, optional
            Threaded to the *tasmin* write only (see below).
        """
        if not self.cache.exists(tasmax_loc):
            raise ValueError(
                f"tasmin reconciliation needs the fine tasmax output "
                f"{tasmax_loc.store_path}/{tasmax_loc.group}, which is missing. "
                f"tasmax must complete before tasmin."
            )
        # Read both fields shard-aligned so the swap+write slices per-shard from clean
        # sharded zarr (no interp graph) instead of pulling the whole global fine array.
        # With the pipeline persisting the raw tasmin before this step, tasmin_fine is
        # None here in production and both inputs are plain sharded reads (reconcile OOM).
        shard = {"time": SHARD_TIME, "lat": SHARD_LAT, "lon": SHARD_LON}
        if tasmin_fine is None:
            if not self.cache.exists(tasmin_loc):
                raise ValueError(
                    f"standalone reconcile needs the fine tasmin output "
                    f"{tasmin_loc.store_path}/{tasmin_loc.group}, which is missing."
                )
            tasmin_fine = self._open_from_icechunk(tasmin_loc, chunks=shard)[self.config.variable]

        tasmax_ds = self._open_from_icechunk(tasmax_loc, chunks=shard)
        # swap_temperature_extremes enforces exact grid alignment and is structurally
        # monotone, so tasmax >= tasmin holds by construction; the second full-array
        # pass is left to output-QA rather than gated here (issue #331).
        tasmax_corrected, tasmin_corrected = swap_temperature_extremes(
            tasmax_ds["tasmax"], tasmin_fine
        )
        tasmax_corrected = tasmax_corrected.chunk(shard)
        tasmin_corrected = tasmin_corrected.chunk(shard)

        # Bound peak memory: the swap+writes are memory-bound. A synchronous scheduler
        # keeps only a few shards resident at once (measured flat vs. array size) rather
        # than the threaded scheduler's whole-array co-residency that OOMs at global
        # scale (reconcile OOM; see project_fit_historical_single_threaded).
        with dask.config.set(scheduler="synchronous"):
            # tasmax first, force=False: tasmin_corrected still lazily reads this
            # pre-rewrite tasmax snapshot, so a force GC now would collect those chunks
            # before the tasmin write below materialises them (data-loss hazard).
            self._write_to_icechunk(
                tasmax_corrected,
                tasmax_loc,
                encoding=make_encoding("tasmax"),
                dataset_attrs=dict(tasmax_ds.attrs),
                force=False,
            )
            # tasmin last, force-threaded: it is materialised before its own commit, so
            # the trailing GC safely sweeps the now-superseded tasmax snapshot too.
            tasmin_corrected.name = self.config.variable
            self._write_to_icechunk(
                tasmin_corrected,
                tasmin_loc,
                encoding=make_encoding(self.config.variable),
                dataset_attrs=self._build_output_attrs(),
                force=force,
            )

    def prepare_observations(self, force: bool = False) -> str:
        """
        Stage 1: Regrid observations to GCM grid.

        This stage loads observations and regrids them to the coarse GCM
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
        is based on artifact existence at ``self.cache.obs_loc``.
        """
        loc = self.cache.obs_loc

        # Check whether regridded dataset already exists, if so (and you don't
        # have the force flag enabled which allows overwrite) use the existing dataset.
        # Note: this does not check anything about the data at the output_path -
        # if it is corrupted in any way or doesn't match the attributes of the
        # config it won't fail.
        if self.cache.exists(loc) and not force:
            logger.info("✓ Using cached observations: %s/%s", loc.store_path, loc.group)
            return loc.store_path

        logger.info(
            "Computing observation regridding for %s/%s", self.config.gcm, self.config.variable
        )

        t0 = time.perf_counter()
        obs_fine = get_obs(var=self.config.variable, dataset_name=self.config.obs_dataset)
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

        if self.options.rechunk_workflow:
            obs_coarse = rechunk(obs_coarse, pattern="full_space")

        t0 = time.perf_counter()
        obs_coarse.name = self.config.variable
        self._write_to_icechunk(
            obs_coarse, loc, dataset_attrs=self._build_output_attrs(), force=force
        )
        logger.info(
            "✓ Cached observations: %s/%s (%.2fs)",
            loc.store_path,
            loc.group,
            time.perf_counter() - t0,
        )

        return loc.store_path

    def _load_gcm_obs(self) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Load obs_coarse (from cache), obs_fine, and model_hist, subsetted to training period."""
        deps = self.cache.check_dependencies(
            "fit_historical", self.config, hist_member=self._hist_member
        )
        obs_coarse = self._open_from_icechunk(deps["obs_regridded"][1])[
            self.config.variable
        ]  # [1] is StoreLocation

        obs_fine = get_obs(var=self.config.variable, dataset_name=self.config.obs_dataset)
        obs_fine = obs_fine.drop_vars("spatial_ref", errors="ignore")

        model_hist = get_historical_experiment(
            gcm=self.config.gcm, member=self._hist_member, var=self.config.variable
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
            if self.config.debias_approach
            in ["nonparametric_hybrid", "nonparametric_hybrid_2sided"]
            else self.config.debias_approach
        )
        debiaser = _make_debiaser(
            variable=self.config.variable,
            mapping_type=mapping_type,
            detrending="no_detrending",
            running_window_mode=self.config.variable_config.do_windowing,
            running_window_length=self.config.variable_config.running_window_length,
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

        # We don't want negative values for any of the variables we are downscaling (tas, tasmin, tasmax, rsds, hurs, pr)

        downscaled = downscale_from_coarse(
            da=debiased,
            obs_coarse=obs_coarse.as_numpy(),
            obs_fine=obs_fine.as_numpy(),
            method=self.config.variable_config.downscaling_method,
            clim_method=self.config.variable_config.downscaling_clim_method,
            allow_negative_values=False,
        )
        return downscaled.chunk({"time": SHARD_TIME, "lat": SHARD_LAT, "lon": SHARD_LON})

    def fit_historical_tasmin(self, force: bool = False) -> str:
        """
        Stage 2: Downscale historical period for tasmin. This differs from normal fit_historical
        because it loads debiased coarse tasmax and dtr to compute debiased coarse tasmin,
        which is then spatially disaggregated to fine resolution.
        """
        loc = self.cache.historical_loc(self._hist_member)
        coarse_loc = self.cache.debiased_coarse_historical_loc(self._hist_member)
        tasmax_fine_loc = self.cache.historical_loc(self._hist_member, variable="tasmax")

        if not force and self.cache.exists(loc) and self.cache.exists(coarse_loc):
            logger.info("✓ Using existing historical: %s/%s", loc.store_path, loc.group)
            return loc.store_path

        # Validate inputs only when we are actually going to compute, so a cache hit
        # is never blocked by a reaped upstream (issue #363). tasmin reads the
        # debiased-coarse tasmax/dtr and the fine tasmax; fail fast if any is missing.
        self.cache.validate_dependencies(
            "fit_historical", self.config, hist_member=self._hist_member
        )

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
        debiased_dtr_loc = self.cache.debiased_coarse_historical_loc(
            self._hist_member, variable="dtr"
        )
        debiased_tasmax_loc = self.cache.debiased_coarse_historical_loc(
            self._hist_member, variable="tasmax"
        )
        debiased_dtr = self._open_from_icechunk(debiased_dtr_loc)["dtr"]
        debiased_tasmax = self._open_from_icechunk(debiased_tasmax_loc)["tasmax"]
        model_hist_debiased = derive_tasmin(debiased_tasmax, debiased_dtr)
        logger.info("Bias corrected historical (%.2fs)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        model_hist_debiased.name = self.config.variable
        # derive_tasmin runs on auto-chunked coarse inputs, so its dask chunks (sub-shard on
        # time, e.g. ~600-step) need not tile the coarse shard grid. rechunk(full_space) only
        # fixes lat/lon; it leaves those sub-shard time chunks straddling SHARD_TIME_COARSE and
        # trips xarray's safe_chunks check. Chunk to the full coarse shard grid instead (as the
        # fine-res write does) so every dask chunk maps to exactly one shard.
        self._write_to_icechunk(
            model_hist_debiased.chunk(
                {"time": SHARD_TIME_COARSE, "lat": SHARD_LAT_COARSE, "lon": SHARD_LON_COARSE}
            ),
            coarse_loc,
            encoding=make_coarse_encoding(self.config.variable),
            dataset_attrs=self._build_output_attrs(),
        )
        logger.info(
            "✓ Saved debiased coarse historical: %s/%s (%.2fs)",
            coarse_loc.store_path,
            coarse_loc.group,
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        model_hist_downscaled = self._apply_spatial_downscaling(
            model_hist_debiased, obs_coarse, obs_fine
        )
        model_hist_downscaled.name = self.config.variable
        logger.info("Spatially disaggregated (%.2fs)", time.perf_counter() - t0)

        # Persist the raw (un-reconciled) tasmin FIRST, then reconcile store-to-store
        # (tasmin_fine=None). This keeps the sibling fine tasmax and this fresh interp
        # graph from being full-array-resident at the same time — the reconcile then
        # re-reads both fields shard-aligned from clean sharded zarr (reconcile OOM).
        t0 = time.perf_counter()
        self._write_to_icechunk(
            model_hist_downscaled,
            loc,
            encoding=make_encoding(self.config.variable),
            dataset_attrs=self._build_output_attrs(),
            force=False,
        )
        logger.info(
            "✓ Saved raw tasmin: %s/%s (%.2fs)", loc.store_path, loc.group, time.perf_counter() - t0
        )

        # Dedicated reconcile step: swap any tasmax < tasmin left by independent
        # disaggregation and write both corrected fields (issue #331).
        t0 = time.perf_counter()
        self.reconcile_temperature_extremes(loc, tasmax_fine_loc, tasmin_fine=None, force=force)
        logger.info(
            "✓ Reconciled + saved historical: %s/%s (%.2fs)",
            loc.store_path,
            loc.group,
            time.perf_counter() - t0,
        )

        return loc.store_path

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
        The output (fully downscaled historical data) is written to the output store as
        a deliverable for the historical period — the analog of the fine scenario output.
        It is also used as a completion gate: ``transform_scenario`` checks that this
        artifact exists before it will run, but does *not* load it as an input (scenario
        runs re-load the raw GCM historical data for their own bias-correction training).
        Setting ``force=True`` reruns all three computation steps and overwrites the
        existing artifact; ``force=False`` skips all three and returns the existing path
        immediately.

        Dependency validation is always performed before checking this stage's
        cache-hit short-circuit.
        """
        # tasmin is derived (tasmax - dtr) and reconciled against tasmax; route it to
        # the dedicated method from here so every entry point — including the
        # distributed batch_runner, which calls this method directly — gets the
        # correct path (issues #363/#331).
        if self.config.variable == "tasmin":
            return self.fit_historical_tasmin(force=force)

        self.cache.validate_dependencies("fit_historical", self.config)

        loc = self.cache.historical_loc(self._hist_member)
        coarse_loc = self.cache.debiased_coarse_historical_loc(self._hist_member)

        if self.cache.exists(loc) and self.cache.exists(coarse_loc) and not force:
            logger.info("✓ Using existing historical: %s/%s", loc.store_path, loc.group)
            return loc.store_path

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

        t0 = time.perf_counter()
        model_hist_debiased.name = self.config.variable
        self._write_to_icechunk(
            model_hist_debiased,
            coarse_loc,
            encoding=make_coarse_encoding(self.config.variable),
            dataset_attrs=self._build_output_attrs(),
        )
        logger.info(
            "✓ Saved debiased coarse historical: %s/%s (%.2fs)",
            coarse_loc.store_path,
            coarse_loc.group,
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        model_hist_downscaled = self._apply_spatial_downscaling(
            model_hist_debiased, obs_coarse, obs_fine
        )
        logger.info("Spatially disaggregated (%.2fs)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        model_hist_downscaled.name = self.config.variable
        self._write_to_icechunk(
            da=model_hist_downscaled,
            loc=loc,
            encoding=make_encoding(self.config.variable),
            dataset_attrs=self._build_output_attrs(),
            force=force,
        )
        logger.info(
            "✓ Saved historical: %s/%s (%.2fs)",
            loc.store_path,
            loc.group,
            time.perf_counter() - t0,
        )

        return loc.store_path

    def _load_ssp245_bridge(self) -> xr.DataArray:
        """Load the SSP245 bridge timeseries for SAI detrending.

        For most GCMs, returns the primary SSP245 dataset directly. For MIROC-ES2H
        G6-1.5K, the primary (GeoMIP) SSP245 starts in 2020, leaving a 2015–2019 gap.
        When _ssp245_esgf_member is set, ESGF SSP245 data fills that gap before the
        GeoMIP data begins. The primary is already in proleptic_gregorian; the ESGF
        dataset is converted via to_proleptic_gregorian before concat.
        """
        primary = get_experiment(self.config.gcm, "SSP245", self.config.variable)
        primary = primary.sel(ensemble_member=self._ssp245_member)

        if self._ssp245_esgf_member is None:
            return primary

        primary_start_year = int(primary.time.dt.year.min())
        if primary_start_year <= self.config.train_period_end + 1:
            return primary

        # Gap detected: prepend ESGF data for the missing years before the GeoMIP start.
        # ESGF may use a different calendar — convert to proleptic_gregorian (primary's calendar).
        from srm.utils import to_proleptic_gregorian

        esgf_ds = to_proleptic_gregorian(
            _catalog.get(self.config.gcm).to_xarray(group="esgf_ssp245")
        )
        esgf_bridge = get_variable(esgf_ds, self.config.variable).sel(
            ensemble_member=self._ssp245_esgf_member
        )
        esgf_gap = esgf_bridge.isel(time=(esgf_bridge.time.dt.year < primary_start_year).values)

        if esgf_gap.time.size == 0:
            logger.warning(
                "_load_ssp245_bridge: ESGF dataset for %s has no data before year %d; "
                "returning primary GeoMIP dataset only — 2015–%d gap will remain",
                self._ssp245_esgf_member,
                primary_start_year,
                primary_start_year - 1,
            )
            return primary

        esgf_gap_years = (int(esgf_gap.time.dt.year.min()), int(esgf_gap.time.dt.year.max()))
        primary_years = (primary_start_year, int(primary.time.dt.year.max()))
        logger.info(
            "_load_ssp245_bridge: stitching ESGF %s %d–%d + GeoMIP %s %d–%d",
            self._ssp245_esgf_member,
            *esgf_gap_years,
            self._ssp245_member,
            *primary_years,
        )

        # Drop the scalar ensemble_member coord before concat — the two slices carry
        # different values (r1i1p4f2 vs r01) and xr.concat refuses to merge mismatched
        # scalar coords. Re-attach the primary member value so the bridge is transparent
        # to any downstream code that reads ensemble_member.
        esgf_clean = esgf_gap.drop_vars("ensemble_member", errors="ignore")
        primary_clean = primary.drop_vars("ensemble_member", errors="ignore")
        bridge = xr.concat([esgf_clean, primary_clean], dim="time")
        bridge = bridge.assign_coords(ensemble_member=primary.coords["ensemble_member"])
        bridge.attrs.update(
            {
                "bridge_type": "esgf_geomip_stitch",
                "bridge_esgf_member": self._ssp245_esgf_member,
                "bridge_esgf_years": f"{esgf_gap_years[0]}-{esgf_gap_years[1]}",
                "bridge_geomip_member": self._ssp245_member,
                "bridge_geomip_years": f"{primary_years[0]}-{primary_years[1]}",
                "bridge_gcm": self.config.gcm,
                "bridge_variable": self.config.variable,
            }
        )
        return bridge

    def _load_scenario_data(
        self,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray | None]:
        """Load obs_coarse, obs_fine, model_hist, model_scenario, and optionally ssp_timeseries.

        Returns (obs_coarse, obs_fine, model_hist, model_scenario, ssp_timeseries).
        obs_coarse/obs_fine/model_hist are subsetted to the training period;
        model_scenario to the predict period. ssp_timeseries is None for non-SAI scenarios.
        """
        deps = self.cache.check_dependencies(
            "transform_scenario", self.config, hist_member=self._hist_member
        )
        obs_coarse = self._open_from_icechunk(deps["obs_regridded"][1])[
            self.config.variable
        ]  # [1] is StoreLocation

        obs_fine = get_obs(var=self.config.variable, dataset_name=self.config.obs_dataset)
        obs_fine = obs_fine.drop_vars("spatial_ref", errors="ignore")

        model_hist = get_historical_experiment(
            gcm=self.config.gcm, member=self._hist_member, var=self.config.variable
        )
        model_hist = model_hist.drop_vars("spatial_ref", errors="ignore")

        model_scenario = get_experiment(
            gcm=self.config.gcm, scenario=self.config.scenario, var=self.config.variable
        )
        model_scenario = model_scenario.sel(ensemble_member=self.config.ensemble_member)
        model_scenario = model_scenario.drop_vars("spatial_ref", errors="ignore")

        # Non-SAI scenarios whose primary dataset starts after predict_period_start
        # (e.g. MIROC-ES2H GeoMIP SSP245 starts 2020) need ESGF data prepended to close the gap.
        if not self.config.is_sai_scenario and self._ssp245_esgf_member is not None:
            scenario_start_year = int(model_scenario.time.dt.year.min())
            if scenario_start_year > self.config.predict_period_start:
                from srm.utils import to_proleptic_gregorian

                esgf_ds = to_proleptic_gregorian(
                    _catalog.get(self.config.gcm).to_xarray(group="esgf_ssp245")
                )
                esgf_data = get_variable(esgf_ds, self.config.variable).sel(
                    ensemble_member=self._ssp245_esgf_member
                )
                esgf_pre = esgf_data.isel(
                    time=(
                        (esgf_data.time.dt.year >= self.config.predict_period_start)
                        & (esgf_data.time.dt.year < scenario_start_year)
                    ).values
                )
                logger.info(
                    "Non-SAI scenario starts at %d; prepending ESGF SSP245 %s for %d–%d",
                    scenario_start_year,
                    self._ssp245_esgf_member,
                    int(esgf_pre.time.dt.year.min()),
                    int(esgf_pre.time.dt.year.max()),
                )
                esgf_pre = esgf_pre.drop_vars("ensemble_member", errors="ignore")
                scenario_clean = model_scenario.drop_vars("ensemble_member", errors="ignore")
                model_scenario = xr.concat([esgf_pre, scenario_clean], dim="time")
                model_scenario = model_scenario.assign_coords(
                    ensemble_member=self.config.ensemble_member
                )
                model_scenario = model_scenario.drop_vars("spatial_ref", errors="ignore")

        # SAI scenarios need an SSP245 bridge to fill the gap between historical and SAI start
        ssp_timeseries: xr.DataArray | None = None
        if self.config.is_sai_scenario:
            ssp_timeseries = self._load_ssp245_bridge()
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
        model_hist = model_hist.sel(
            time=slice(
                f"{self.config.train_period_start}", f"{self.config.predict_period_start - 1}"
            )
        )
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
        returns (scenario, None) and scenario_trend will be None.

        For SAI scenarios, stitches in SSP245 data to bridge the gap between the end of
        historical (2014/2015) and the SAI simulation start (~2035). This bridge is
        applied even when detrending is disabled, so that non-detrended variables
        (dtr, pr, rsds, hurs) still span the full predict window rather than starting
        at the SAI simulation year — otherwise ``tasmin = tasmax - dtr`` breaks against
        the bridged (full-length) tasmax on the missing days (issue #363).
        """
        if not self.config.variable_config.detrend_data:
            if self.config.is_sai_scenario:
                # No detrending, but a SAI scenario still needs the SSP245 bridge so
                # the debiased-coarse output spans predict_period_start..end (#363).
                predict_slice = slice(
                    f"{self.config.predict_period_start}", f"{self.config.predict_period_end}"
                )
                bridged = stitch_historical_scenario(
                    model_hist=model_hist,
                    model_scenario=model_scenario,
                    train_period_end=self.config.train_period_end,
                    predict_period_start=self.config.predict_period_start,
                    ssp_timeseries=ssp_timeseries,
                )
                return bridged.sel(time=predict_slice), None
            return model_scenario, None

        if self.options.rechunk_workflow:
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
            detrend_method=self.config.variable_config.detrend_method,
        )

        predict_slice = slice(
            f"{self.config.predict_period_start}", f"{self.config.predict_period_end}"
        )
        scenario_detrended = scenario_detrended.sel(time=predict_slice)
        scenario_trend = scenario_trend.sel(time=predict_slice)
        logger.info("Detrended scenario (%.2fs)", time.perf_counter() - t0)

        if self.options.save_intermediate:
            t0 = time.perf_counter()
            detrended_loc = self.cache.detrended_scenario_loc()
            scenario_detrended.name = self.config.variable
            self._write_to_icechunk(
                rechunk(scenario_detrended, pattern="full_space"),
                detrended_loc,
            )
            logger.info(
                "✓ Saved detrended scenario: %s/%s (%.2fs)",
                detrended_loc.store_path,
                detrended_loc.group,
                time.perf_counter() - t0,
            )

            t0 = time.perf_counter()
            trend_loc = self.cache.trend_scenario_loc()
            scenario_trend.name = self.config.variable
            scenario_trend.attrs = model_scenario.attrs
            self._write_to_icechunk(
                rechunk(scenario_trend, pattern="full_space"),
                trend_loc,
            )
            logger.info(
                "✓ Saved scenario trend: %s/%s (%.2fs)",
                trend_loc.store_path,
                trend_loc.group,
                time.perf_counter() - t0,
            )

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
            running_window_mode=self.config.variable_config.do_windowing,
            running_window_length=self.config.variable_config.running_window_length,
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

        if self.config.debias_approach in ["parametric", "nonparametric"]:
            debiased_np = _make_debiaser(
                mapping_type=self.config.debias_approach, **common_kwargs
            ).apply(**apply_kwargs)

        elif self.config.debias_approach == "nonparametric_hybrid":
            parametric_np = _make_debiaser(mapping_type="parametric", **common_kwargs).apply(
                **apply_kwargs
            )
            nonparametric_np = _make_debiaser(mapping_type="nonparametric", **common_kwargs).apply(
                **apply_kwargs
            )

            out_of_range, _, _ = calculate_out_of_range_mask(
                model_hist=model_hist,
                scenario_detrended=scenario_detrended,
                center_window=self.config.variable_config.running_window_length,
            )
            debiased_np = np.where(out_of_range.values, parametric_np, nonparametric_np)

        elif self.config.debias_approach == "nonparametric_hybrid_2sided":
            # Use one parametric debiaser for low out-of-range values, another for high, and nonparametric everywhere else

            if self.config.variable in ["pr", "rsds", "hurs", "dtr"]:
                # Use different parametric distributions for low vs. high tails
                low_dist = scipy.stats.weibull_min
                high_dist = scipy.stats.gumbel_r

                parametric_low_np = _make_debiaser(
                    distribution=low_dist, mapping_type="parametric", **common_kwargs
                ).apply(**apply_kwargs)
                parametric_high_np = _make_debiaser(
                    distribution=high_dist, mapping_type="parametric", **common_kwargs
                ).apply(**apply_kwargs)
            else:
                # Unless explicitly specified, use the same parametric debiaser for both tails even if calling "nonparametric_hybrid_2sided"
                parametric_low_np = _make_debiaser(
                    mapping_type="parametric", **common_kwargs
                ).apply(**apply_kwargs)

                parametric_high_np = parametric_low_np

            nonparametric_np = _make_debiaser(mapping_type="nonparametric", **common_kwargs).apply(
                **apply_kwargs
            )

            _, out_of_range_low, out_of_range_high = calculate_out_of_range_mask(
                model_hist=model_hist,
                scenario_detrended=scenario_detrended,
                center_window=self.config.variable_config.running_window_length,
            )

            debiased_np = np.where(out_of_range_low.values, parametric_low_np, nonparametric_np)
            debiased_np = np.where(out_of_range_high.values, parametric_high_np, debiased_np)

        else:
            raise ValueError(
                "debias_approach must be 'parametric', 'nonparametric', 'nonparametric_hybrid', or 'nonparametric_hybrid_2sided'."
            )

        if self.options.clip_values:
            var = self.config.variable
            if var in self.options.clip_bounds:
                bounds = self.options.clip_bounds[var]
                debiased_np = np.clip(debiased_np, a_min=bounds.min, a_max=bounds.max)

        return xr.DataArray(
            data=debiased_np,
            coords={
                "lat": scenario_detrended["lat"],
                "lon": scenario_detrended["lon"],
                "time": scenario_detrended["time"],
            },
            dims=["time", "lat", "lon"],
        )

    def transform_scenario_tasmin(self, force: bool = False) -> str:
        """
        This is a special version of transform_scenario for tasmin.
        The spatial disaggregation approach is the same as for the normal transform_scenario,
        but the bias correction step is different: it loads debiased coarse tasmax and dtr,
        and then computes debiased coarse tasmin by subtracting debiased coarse dtr from debiased coarse tasmax.

        """
        if self.config.scenario is None:
            raise ValueError("scenario must be specified in config for transform_scenario")

        loc = self.cache.scenario_loc
        coarse_loc = self.cache.debiased_coarse_scenario_loc()
        tasmax_fine_loc = self.cache.scenario_output_loc(variable="tasmax")

        if not force and self.cache.exists(loc) and self.cache.exists(coarse_loc):
            logger.info("✓ Using cached scenario: %s/%s", loc.store_path, loc.group)
            return loc.store_path

        # Validate inputs only when we are actually going to compute, so a cache hit
        # is never blocked by a reaped upstream (issue #363).
        self.cache.validate_dependencies(
            "transform_scenario", self.config, hist_member=self._hist_member
        )

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

        t0 = time.perf_counter()
        debiased_dtr_loc = self.cache.debiased_coarse_scenario_loc(variable="dtr")
        debiased_tasmax_loc = self.cache.debiased_coarse_scenario_loc(variable="tasmax")
        debiased_dtr = self._open_from_icechunk(debiased_dtr_loc)["dtr"]
        debiased_tasmax = self._open_from_icechunk(debiased_tasmax_loc)["tasmax"]
        scenario_debiased = derive_tasmin(debiased_tasmax, debiased_dtr)
        logger.info("Bias corrected scenario (%.2fs)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        scenario_debiased.name = self.config.variable
        # derive_tasmin runs on auto-chunked coarse inputs, so its dask chunks (sub-shard on
        # time, e.g. ~600-step) need not tile the coarse shard grid. rechunk(full_space) only
        # fixes lat/lon; it leaves those sub-shard time chunks straddling SHARD_TIME_COARSE and
        # trips xarray's safe_chunks check on scenarios longer than one shard (>16000 days).
        # Chunk to the full coarse shard grid instead so every dask chunk maps to one shard.
        self._write_to_icechunk(
            scenario_debiased.chunk(
                {"time": SHARD_TIME_COARSE, "lat": SHARD_LAT_COARSE, "lon": SHARD_LON_COARSE}
            ),
            coarse_loc,
            encoding=make_coarse_encoding(self.config.variable),
            dataset_attrs=self._build_output_attrs(),
        )
        logger.info(
            "✓ Saved debiased coarse scenario: %s/%s (%.2fs)",
            coarse_loc.store_path,
            coarse_loc.group,
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        scenario_downscaled = self._apply_spatial_downscaling(
            scenario_debiased, obs_coarse, obs_fine
        )
        logger.info("Spatially disaggregated (%.2fs)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        if self.options.apply_ocean_mask:
            scenario_downscaled = scenario_downscaled.where(
                self._build_ocean_mask(scenario_downscaled)
            ).chunk({"time": SHARD_TIME, "lat": SHARD_LAT, "lon": SHARD_LON})
        scenario_downscaled.name = self.config.variable

        # Persist the raw (un-reconciled) tasmin FIRST — after any ocean masking so both
        # fine fields share aligned NaN cells — then reconcile store-to-store
        # (tasmin_fine=None). Sequencing the writes keeps the sibling fine tasmax and
        # this fresh interp graph off the heap simultaneously; the reconcile re-reads
        # both fields shard-aligned from clean sharded zarr (reconcile OOM).
        self._write_to_icechunk(
            scenario_downscaled,
            loc,
            encoding=make_encoding(self.config.variable),
            dataset_attrs=self._build_output_attrs(),
            force=False,
        )
        logger.info(
            "✓ Saved raw tasmin: %s/%s (%.2fs)", loc.store_path, loc.group, time.perf_counter() - t0
        )

        # Dedicated reconcile step: swap any tasmax < tasmin left by independent
        # disaggregation and write both corrected fields (issue #331).
        t0 = time.perf_counter()
        self.reconcile_temperature_extremes(loc, tasmax_fine_loc, tasmin_fine=None, force=force)
        logger.info(
            "✓ Reconciled + saved scenario output: %s/%s (%.2fs)",
            loc.store_path,
            loc.group,
            time.perf_counter() - t0,
        )

        return loc.store_path

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
        # tasmin is derived (tasmax - dtr) and reconciled against tasmax; route it to
        # the dedicated method from here so every entry point — including the
        # distributed batch_runner, which calls this method directly — gets the
        # correct path (issues #363/#331).
        if self.config.variable == "tasmin":
            return self.transform_scenario_tasmin(force=force)

        if self.config.scenario is None:
            raise ValueError("scenario must be specified in config for transform_scenario")

        self.cache.validate_dependencies(
            "transform_scenario", self.config, hist_member=self._hist_member
        )

        loc = self.cache.scenario_loc
        coarse_loc = self.cache.debiased_coarse_scenario_loc()

        if self.cache.exists(loc) and self.cache.exists(coarse_loc) and not force:
            logger.info("✓ Using cached scenario: %s/%s", loc.store_path, loc.group)
            return loc.store_path

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

        if self.options.save_intermediate:
            t0 = time.perf_counter()
            debiased_loc = self.cache.debiased_scenario_loc()
            scenario_debiased.name = self.config.variable
            self._write_to_icechunk(scenario_debiased, debiased_loc)
            logger.info(
                "✓ Saved debiased scenario: %s/%s (%.2fs)",
                debiased_loc.store_path,
                debiased_loc.group,
                time.perf_counter() - t0,
            )

        if scenario_trend is not None:
            t0 = time.perf_counter()
            scenario_debiased = retrend(
                bias_corrected_detrended=scenario_debiased,
                trend_on_daily_timestep=scenario_trend,
                detrend_method=self.config.variable_config.detrend_method,
            )
            logger.info("Re-trended scenario (%.2fs)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        scenario_debiased.name = self.config.variable
        self._write_to_icechunk(
            scenario_debiased,
            coarse_loc,
            encoding=make_coarse_encoding(self.config.variable),
            dataset_attrs=self._build_output_attrs(),
        )
        logger.info(
            "✓ Saved debiased coarse scenario: %s/%s (%.2fs)",
            coarse_loc.store_path,
            coarse_loc.group,
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        scenario_downscaled = self._apply_spatial_downscaling(
            scenario_debiased, obs_coarse, obs_fine
        )
        logger.info("Spatially disaggregated (%.2fs)", time.perf_counter() - t0)

        t0 = time.perf_counter()
        if self.options.apply_ocean_mask:
            scenario_downscaled = scenario_downscaled.where(
                self._build_ocean_mask(scenario_downscaled)
            ).chunk({"time": SHARD_TIME, "lat": SHARD_LAT, "lon": SHARD_LON})
        scenario_downscaled.name = self.config.variable
        self._write_to_icechunk(
            scenario_downscaled,
            loc,
            dataset_attrs=self._build_output_attrs(),
            encoding=make_encoding(self.config.variable),
            force=force,
        )
        logger.info(
            "✓ Saved scenario output: %s/%s (%.2fs)",
            loc.store_path,
            loc.group,
            time.perf_counter() - t0,
        )

        return loc.store_path

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

        # fit_historical / transform_scenario self-dispatch tasmin to their derived
        # variants, so no variable-specific branching is needed here.
        self.fit_historical(force=force)
        # `transform_scenario` depends on fit_historical only as a completion gate
        # (artifact existence); it does not read the historical output as data input.
        return self.transform_scenario(force=force)
