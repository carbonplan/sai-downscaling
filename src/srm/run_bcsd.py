import typing
import warnings

import dask.system
import rasterix  # noqa: F401  # side-effect import: registers .proj/.rio accessors
import xarray as xr
from ibicus.debias import QuantileMapping

from srm.downscaling_utils import (
    calculate_baseline_climatology,
    detrend,
    downscale_from_coarse,
    get_experiment,
    get_obs,
    interpolate_fine_to_coarse_grid,
    rechunk,
    retrend,
    save_data,
    subset_space,
)
from srm.utils import Timer

# this is here to suppress xarray_regrid warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


BCSD_CONFIG = {
    "pr": {
        "detrend_data": False,
        "do_windowing": True,
        "downscaling_method": "divide",
        "downscaling_clim_method": "simple",
    },
    "tas": {
        "detrend_data": True,
        "do_windowing": True,
        "downscaling_method": "subtract",
        "downscaling_clim_method": "fft",
    },
    "tasmax": {
        "detrend_data": True,
        "do_windowing": True,
        "downscaling_method": "subtract",
        "downscaling_clim_method": "fft",
    },
}


def get_all_data(
    gcm: str,
    var_name: str,
    verbose: bool = True,
):
    with Timer("Loaded data", verbose=verbose):
        model_scenario = get_experiment(gcm=gcm, scenario="SSP245", var=var_name)
        model_scenario = model_scenario.isel(
            ensemble_member=0
        )  # TODO: update to .sel() with string label (e.g. 'r1i1p1f1')
        model_scenario = model_scenario.drop_vars("spatial_ref")

        model_historical = get_experiment(gcm=gcm, scenario="Historical", var=var_name)
        model_historical = model_historical.drop_vars("spatial_ref")

        obs = get_obs(var=var_name)
        obs = obs.drop_vars("spatial_ref")

    dict_all = {}

    dict_all["model_hist"] = model_historical
    dict_all["obs"] = obs
    dict_all["model_scenario"] = model_scenario

    return dict_all


def preprocess_data(
    dict_all: dict,
    train_period_start: int,
    train_period_end: int,
    predict_period_start: int,
    predict_period_end: int,
    verbose: bool = True,
    rechunk_workflow: bool = True,
    detrend_data: bool = True,
):
    ################## Interpolate obs to coarse grid
    if rechunk_workflow:
        with Timer("Rechunked obs to full space", verbose=verbose):
            dict_all["obs"] = rechunk(dict_all["obs"], pattern="full_space")
            # currently, we need this call otherwise xarray_regrid throws an error when we try to access values of
            # the resulting regridded dataset
            dict_all["obs"] = dict_all["obs"].persist()

    with Timer("Interpolated obs to coarse grid", verbose=verbose):
        dict_all["obs_coarse"] = interpolate_fine_to_coarse_grid(
            da_fine_to_coarsen=dict_all["obs"], da_coarse_grid=dict_all["model_hist"]
        )

    ###### Detrend data
    if detrend_data:
        if rechunk_workflow:
            with Timer("Rechunked all to full time", verbose=verbose):
                for key in ["obs_coarse", "model_hist", "model_scenario"]:
                    dict_all[key] = rechunk(dict_all[key], pattern="full_time")
                    dict_all[key] = dict_all[key].persist()

        # Splice together historical and future scenario for calculating smooth 9 year running mean for detrending (otherwise first few years of scenario will be nans)
        with Timer("Detrended data", verbose=verbose):
            dict_all["historical_scenario"] = xr.concat(
                [
                    dict_all["model_hist"].sel(
                        time=dict_all["model_hist"]["time.year"] < predict_period_start
                    ),
                    dict_all["model_scenario"].sel(
                        time=dict_all["model_scenario"]["time.year"] >= predict_period_start
                    ),
                ],
                dim="time",
            )

            da_baseline_clim = calculate_baseline_climatology(
                da_baseline=dict_all["model_hist"],
                baseline_period_start=train_period_start,
                baseline_period_end=train_period_end,
            )

            ssp_detrended, ssp_trend_on_daily_timestep = detrend(
                da=dict_all["historical_scenario"],
                da_baseline_clim=da_baseline_clim,
            )

            dict_all["scenario_detrended"] = ssp_detrended
            dict_all["scenario_trend"] = ssp_trend_on_daily_timestep

    else:
        # Use raw model scenario data if not detrending
        dict_all["scenario_detrended"] = dict_all["model_scenario"]

    ################## Subset time periods
    dict_all["model_hist"] = dict_all["model_hist"].sel(
        time=slice(f"{train_period_start}", f"{train_period_end}")
    )
    dict_all["obs"] = dict_all["obs"].sel(
        time=slice(f"{train_period_start}", f"{train_period_end}")
    )
    dict_all["model_scenario"] = dict_all["model_scenario"].sel(
        time=slice(f"{predict_period_start}", f"{predict_period_end}")
    )
    dict_all["scenario_detrended"] = dict_all["scenario_detrended"].sel(
        time=slice(f"{predict_period_start}", f"{predict_period_end}")
    )

    return dict_all


def bias_correct(
    dict_all: dict,
    var_name: str,
    verbose: bool = True,
    detrend_data: bool = True,
    do_windowing: bool = True,
    mapping_type: str = "parametric",
):
    with Timer("Quantile mapped historical", verbose=verbose):
        if do_windowing:
            debiaser = QuantileMapping.from_variable(
                variable=var_name,
                mapping_type=mapping_type,
                detrending="no_detrending",
                running_window_mode=True,
                running_window_length=31,
                running_window_step_length=1,
                running_window_mode_over_years_of_cm_future=False,
            )
        else:
            debiaser = QuantileMapping.from_variable(
                variable=var_name,
                mapping_type=mapping_type,
                detrending="no_detrending",
                running_window_mode=False,
                running_window_mode_over_years_of_cm_future=False,
            )

        # as_numpy brings from sparse to dense. regridding sparsifies, so bring it back here for downstream tasks.
        obs = dict_all["obs_coarse"].as_numpy().values
        cm_hist = dict_all["model_hist"].as_numpy().values

        var_cm_hist_debiased = debiaser.apply(
            obs=obs,
            cm_hist=cm_hist,
            cm_future=cm_hist,  # cm_future is the scenario being debiased. cm_future=cm_hist because the historical model run is being debiased here
            time_obs=dict_all["obs_coarse"]["time"].values,
            time_cm_hist=dict_all["model_hist"]["time"].values,
            time_cm_future=dict_all["model_hist"]["time"].values,
            parallel=True,
            progressbar=False,  # progress bar doesn't work if parallel=True
            nr_processes=dask.system.CPU_COUNT,
        )

    with Timer("Quantile mapped future", verbose=verbose):
        cm_future = dict_all["scenario_detrended"].load().values
        scenario_fut_debiased = debiaser.apply(
            obs=obs,
            cm_hist=cm_hist,
            cm_future=cm_future,
            time_obs=dict_all["obs_coarse"]["time"].values,
            time_cm_hist=dict_all["model_hist"]["time"].values,
            time_cm_future=dict_all["scenario_detrended"]["time"].values,
            parallel=True,
            progressbar=False,  # progress bar doesn't work if parallel=True
            nr_processes=dask.system.CPU_COUNT,
        )

    ################## Save debiased data to dictionary
    dict_all["model_hist_debiased"] = xr.DataArray(
        data=var_cm_hist_debiased,
        coords={
            "lat": dict_all["model_hist"]["lat"],
            "lon": dict_all["model_hist"]["lon"],
            "time": dict_all["model_hist"]["time"],
        },
        dims=["time", "lat", "lon"],
    )

    scenario_debiased = xr.DataArray(
        data=scenario_fut_debiased,
        coords={
            "lat": dict_all["scenario_detrended"]["lat"],
            "lon": dict_all["scenario_detrended"]["lon"],
            "time": dict_all["scenario_detrended"]["time"],
        },
        dims=["time", "lat", "lon"],
    )

    ################## Add back trend if previously detrended
    if detrend_data:
        dict_all["scenario_debiased_detrended"] = scenario_debiased

        with Timer("Added back trend", verbose=verbose):
            dict_all["scenario_debiased"] = retrend(
                bias_corrected_detrended=dict_all["scenario_debiased_detrended"],
                trend_on_daily_timestep=dict_all["scenario_trend"],
                detrending="additive",
            )

    else:
        dict_all["scenario_debiased"] = scenario_debiased

    return dict_all


def spatially_disaggregate(
    dict_all: dict,
    verbose: bool = True,
    rechunk_workflow: bool = True,
    clim_method: typing.Literal["simple", "fft"] = "fft",
    method: typing.Literal["subtract", "divide"] = "subtract",
):
    if rechunk_workflow:
        with Timer("Rechunked to full space for downscaling", verbose=verbose):
            for key in ["model_hist_debiased", "scenario_debiased"]:
                dict_all[key] = rechunk(dict_all[key], pattern="full_space")
                dict_all[key] = dict_all[key].persist()

    ################## Downscale coarse -> fine
    with Timer("Downscaled historical", verbose=verbose):
        dict_all["model_hist_debiased_downscaled"] = downscale_from_coarse(
            da=dict_all["model_hist_debiased"],
            obs_coarse=dict_all["obs_coarse"].as_numpy(),
            obs_fine=dict_all["obs"].as_numpy(),
            method=method,
            clim_method=clim_method,
        )

    with Timer("Downscaled future", verbose=verbose):
        dict_all["scenario_debiased_downscaled"] = downscale_from_coarse(
            da=dict_all["scenario_debiased"],
            obs_coarse=dict_all["obs_coarse"].as_numpy(),
            obs_fine=dict_all["obs"].as_numpy(),
            method=method,
            clim_method=clim_method,
        )

    return dict_all


def run_bcsd(
    *,
    gcm: str,
    train_period_start: int,
    train_period_end: int,
    predict_period_start: int,
    predict_period_end: int,
    var_name: str,
    verbose: bool = True,
    rechunk_workflow: bool = True,
    subset_bounds: list | None = None,
    save_output: bool = True,
    save_intermediate_output: bool = True,
):
    cfg = BCSD_CONFIG[var_name]
    detrend_data = cfg["detrend_data"]
    do_windowing = cfg["do_windowing"]
    downscaling_method = cfg["downscaling_method"]
    downscaling_clim_method = cfg["downscaling_clim_method"]

    dict_all = get_all_data(
        gcm=gcm,
        var_name=var_name,
        verbose=verbose,
    )

    if subset_bounds is not None:
        [lat_min, lat_max, lon_min, lon_max] = subset_bounds
        for key in dict_all:
            dict_all[key] = subset_space(
                dict_all[key],
                coord_bounds_list=[lat_min, lat_max, lon_min, lon_max],
            )

    dict_all = preprocess_data(
        dict_all=dict_all,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        verbose=verbose,
        rechunk_workflow=rechunk_workflow,
        detrend_data=detrend_data,
    )

    dict_all = bias_correct(
        dict_all=dict_all,
        var_name=var_name,
        verbose=verbose,
        detrend_data=detrend_data,
        do_windowing=do_windowing,
        mapping_type="parametric",
    )

    dict_all = spatially_disaggregate(
        dict_all,
        verbose=verbose,
        rechunk_workflow=rechunk_workflow,
        clim_method=downscaling_clim_method,
        method=downscaling_method,
    )

    if save_output:
        with Timer("Saved all data", verbose=verbose):
            if save_intermediate_output:
                dict_to_save = dict_all
            else:
                dict_to_save = {
                    key: dict_all[key]
                    for key in dict_all
                    if key
                    in [
                        "model_hist_debiased_downscaled",
                        "scenario_debiased_downscaled",
                    ]
                }

            # fname should be all keys joined by underscores
            # fname_key = "_".join(dict_to_save.keys())
            fname_key = var_name + "_data"
            save_data(
                dict_data=dict_to_save,
                fname_key=fname_key,
                var_name=var_name,
                output_suffix="zarr",
                s3_bucket="s3://carbonplan-scratch/",
                prefix="srm-scratch/v0.3_SouthAfrica/",
            )

    return dict_all
