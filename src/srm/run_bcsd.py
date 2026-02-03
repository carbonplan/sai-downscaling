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
    subset_time,
)
from srm.utils import Timer

warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # or do i put this at the top of run_bcsd

RUN_PARAMETERS = {
    "gcm": "CESM2-WACCM",
    "train_period_start": 1978,
    "train_period_end": 2014,
    "predict_period_start": 2015,
    "predict_period_end": 2100,
    "var_name": "tas",
}


def get_all_data(
    gcm: str,
    var_name: str,
    verbose: bool = True,
):
    with Timer("Loaded data", verbose=verbose):
        model_scenario = get_experiment(gcm=gcm, scenario="SSP245", var=var_name)
        model_scenario = model_scenario.isel(ensemble_member=0)
        model_scenario = model_scenario.drop_vars("spatial_ref")

        model_historical = get_experiment(gcm=gcm, scenario="Historical", var=var_name)
        model_historical = model_historical.drop_vars("spatial_ref")

        obs = get_obs(var=var_name)
        obs = obs.drop_vars("spatial_ref")

    ################## Subset time
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

    dict_all["obs_coarse"] = interpolate_fine_to_coarse_grid(
        da_fine_to_coarsen=dict_all["obs"], da_coarse_grid=dict_all["model_hist"]
    )

    with Timer("Interpolated obs to coarse grid", verbose=verbose):
        # `.regrid` namespace comes from xarray_regrid; assumes rectilinear, which is same as NCL and good enough for us.
        dict_all["obs_coarse"] = (
            dict_all["obs"].regrid.conservative(dict_all["model_hist"]).compute()
        )

    ###### Detrend data
    if rechunk_workflow:
        with Timer("Rechunked all to full time", verbose=verbose):
            for key in ["obs_coarse", "model_hist", "model_scenario"]:
                dict_all[key] = rechunk(dict_all[key], pattern="full_time")
                dict_all[key] = dict_all[key].persist()

    if detrend_data:
        with Timer("Detrended data", verbose=verbose):
            dict_all["historical_scenario"] = xr.concat(
                [
                    dict_all["model_hist"].where(
                        dict_all["model_hist"]["time.year"] < predict_period_start,
                        drop=True,
                    ),
                    dict_all["model_scenario"].where(
                        dict_all["model_scenario"]["time.year"] >= predict_period_start,
                        drop=True,
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
    dict_all["model_hist"] = subset_time(
        dict_all["model_hist"],
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        time_period="train",
    )
    dict_all["obs"] = subset_time(
        dict_all["obs"],
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        time_period="train",
    )
    dict_all["model_scenario"] = subset_time(
        dict_all["model_scenario"],
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        time_period="predict",
    )

    dict_all["scenario_detrended"] = subset_time(
        dict_all["scenario_detrended"],
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        time_period="predict",
    )

    return dict_all


def bias_correct(
    dict_all: dict,
    var_name: str,
    verbose: bool = True,
    rechunk_workflow: bool = True,
    detrend_data: bool = True,
    do_windowing: bool = True,
    mapping_type: str = "nonparametric",
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
        cm_future = cm_hist

        with Timer("Quantile mapped historical", verbose=verbose):
            tas_cm_hist_debiased = debiaser.apply(
                obs=obs,
                cm_hist=cm_hist,
                cm_future=cm_future,
                time_obs=dict_all["obs_coarse"]["time"].values,
                time_cm_hist=dict_all["model_hist"]["time"].values,
                parallel=True,
                progressbar=False,
                nr_processes=dask.system.CPU_COUNT,
            )

    # as_numpy brings from sparse to dense. regridding sparsifies, so bring it back here for downstream tasks.
    obs = dict_all["obs_coarse"].as_numpy().values
    cm_hist = dict_all["model_hist"].as_numpy().values
    cm_future = cm_hist

    with Timer("Quantile mapped future", verbose=verbose):
        tas_cm_hist_debiased = debiaser.apply(
            obs=obs,
            cm_hist=cm_hist,
            cm_future=cm_future,
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
        data=tas_cm_hist_debiased,
        coords={
            "lat": dict_all["model_hist"]["lat"],
            "lon": dict_all["model_hist"]["lon"],
            "time": dict_all["model_hist"]["time"],
        },
        dims=["time", "lat", "lon"],
    )

    debiased_scenario = xr.DataArray(
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
        dict_all["scenario_debiased_detrended"] = debiased_scenario

        with Timer("Added back trend", verbose=verbose):
            dict_all["scenario_debiased"] = retrend(
                bias_corrected_detrended=dict_all["scenario_debiased_detrended"],
                trend_on_daily_timestep=dict_all["scenario_trend"],
                detrending="additive",
            )

    else:
        dict_all["scenario_debiased"] = debiased_scenario

    return dict_all


def spatially_disaggregate(
    dict_all: dict,
    verbose: bool = True,
    rechunk_workflow: bool = True,
    clim_method: str = "fft",
    method: str = "subtract",
):
    if rechunk_workflow:
        with Timer("Rechunked to full space", verbose=verbose):
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
    detrend_data: bool = True,
    do_windowing: bool = True,
    subset_bounds: list | None = None,
    save_output: bool = True,
    save_intermediate_output: bool = True,
):
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
        rechunk_workflow=rechunk_workflow,
        detrend_data=detrend_data,
        do_windowing=do_windowing,
    )

    dict_all = spatially_disaggregate(dict_all, verbose=verbose, rechunk_workflow=rechunk_workflow)

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
            fname_key = "data"
            save_data(
                dict_data=dict_to_save,
                fname_key=fname_key,
                var_name=var_name,
                output_suffix="zarr",
                s3_bucket="s3://carbonplan-scratch/",
                prefix="srm-scratch/v0.3_SouthAfrica/",
            )

    return dict_all
