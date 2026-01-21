import time
import warnings

import rasterix  # noqa: F401  # side-effect import: registers .proj/.rio accessors
import xarray as xr
import xarray_regrid  # noqa: F401  # side-effect import: registers .regrid namespace
from ibicus.debias import QuantileMapping

from srm.downscaling_utils import (
    calculate_baseline_climatology,
    calculate_error_map,
    detrend,
    downscale_from_coarse,
    get_experiment,
    get_obs,
    rechunk,
    retrend,
    subset_time,
)

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
    gcm=None,
    var_name=None,
    verbose=True,
):
    start_time = time.time()

    model_scenario = get_experiment(gcm=gcm, scenario="SSP245", var=var_name)
    model_scenario = model_scenario.isel(ensemble_member=0)
    model_scenario = model_scenario.drop_vars("spatial_ref")

    model_historical = get_experiment(gcm=gcm, scenario="Historical", var=var_name)
    model_historical = model_historical.drop_vars("spatial_ref")

    obs = get_obs(var=var_name)
    obs = obs.drop_vars("spatial_ref")

    if verbose:
        elapsed = time.time() - start_time
        print(f"Loaded data: {elapsed:.2f} seconds")

    start_time = time.time()

    ################## Subset time
    step_start_time = time.time()
    dict_all = {}

    if verbose:
        elapsed = time.time() - step_start_time
        print(f"Subset time: {elapsed:.2f} seconds")

    dict_all["model_hist"] = model_historical
    dict_all["obs"] = obs
    dict_all["model_scenario"] = model_scenario

    return dict_all


def preprocess_data(
    dict_all,
    train_period_start=None,
    train_period_end=None,
    predict_period_start=None,
    predict_period_end=None,
    verbose=True,
    rechunk_workflow=True,
    detrend_data=True,
):
    ################## Interpolate obs to coarse grid
    if rechunk_workflow:
        step_start_time = time.time()
        dict_all["obs"] = rechunk(dict_all["obs"], pattern="full_space")
        # currently, we need this call otherwise xarray_regrid throws an error when we try to access values of
        # the resulting regridded dataset
        dict_all["obs"] = dict_all["obs"].persist()

        if verbose:
            elapsed = time.time() - step_start_time
            print(f"Rechunked obs to full space: {elapsed:.2f} seconds")

    step_start_time = time.time()

    # `.regrid` namespace comes from xarray_regrid; assumes rectilinear, which is same as NCL and good enough for us.
    dict_all["obs_coarse"] = dict_all["obs"].regrid.conservative(dict_all["model_hist"]).persist()

    if verbose:
        elapsed = time.time() - step_start_time
        print(f"Interpolated obs to coarse grid: {elapsed:.2f} seconds")

    ###### Detrend data
    if detrend_data:
        step_start_time = time.time()
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
        if verbose:
            elapsed = time.time() - step_start_time
            print(f"Detrended data: {elapsed:.2f} seconds")

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

    return dict_all


def bias_correct(
    dict_all,
    var_name=None,
    verbose=True,
    rechunk_workflow=True,
    detrend_data=True,
    do_windowing=True,
    mapping_type="nonparametric",
):
    step_start_time = time.time()

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

    tas_cm_hist_debiased = debiaser.apply(
        obs=obs,
        cm_hist=cm_hist,
        cm_future=cm_future,
        time_obs=dict_all["obs_coarse"]["time"].values,
        time_cm_hist=dict_all["model_hist"]["time"].values,
        # parallelbool = True,
        # nr_processesint = 1
    )
    if verbose:
        elapsed = time.time() - step_start_time
        print(f"Quantile mapped historical: {elapsed:.2f} seconds")

    step_start_time = time.time()
    cm_future = dict_all["model_scenario"].load().values
    scenario_fut_debiased = debiaser.apply(
        obs=obs,
        cm_hist=cm_hist,
        cm_future=cm_future,
        time_obs=dict_all["obs_coarse"]["time"].values,
        time_cm_hist=dict_all["model_hist"]["time"].values,
        time_cm_future=dict_all["model_scenario"]["time"].values,
    )
    if verbose:
        elapsed = time.time() - step_start_time
        print(f"Quantile mapped future: {elapsed:.2f} seconds")

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

    dict_all["scenario_debiased"] = xr.DataArray(
        data=scenario_fut_debiased,
        coords={
            "lat": dict_all["model_scenario"]["lat"],
            "lon": dict_all["model_scenario"]["lon"],
            "time": dict_all["model_scenario"]["time"],
        },
        dims=["time", "lat", "lon"],
    )

    ################## Add back trend if previously detrended
    if detrend_data:
        step_start_time = time.time()
        dict_all["scenario_debiased"] = retrend(
            bias_corrected_detrended=dict_all["scenario_debiased"],
            trend_on_daily_timestep=dict_all["scenario_trend"],
            detrending="additive",
        )
        if verbose:
            elapsed = time.time() - step_start_time
            print(f"Added back trend: {elapsed:.2f} seconds")
    return dict_all


def spatially_disaggregate(dict_all, verbose=True, rechunk_workflow=True):
    ################## Do Quantile Mapping
    if rechunk_workflow:
        step_start_time = time.time()
        for key in ["obs_coarse", "model_hist", "model_scenario"]:
            dict_all[key] = rechunk(dict_all[key], pattern="full_time")
            dict_all[key] = dict_all[key].persist()
        elapsed = time.time() - step_start_time
        if verbose:
            print(f"Rechunked all to full time: {elapsed:.2f} seconds")

    ################## Calculate error map
    # Calculate a fine-resolution spatial anomaly pattern derived from the observations
    step_start_time = time.time()
    error_map = calculate_error_map(
        obs_coarse=dict_all["obs_coarse"].as_numpy(),
        obs_fine=dict_all["obs"].as_numpy(),
    )
    if verbose:
        elapsed = time.time() - step_start_time
        print(f"Calculate error map for spatial disaggregation: {elapsed:.2f} seconds")

    ################## Downscale coarse -> fine
    step_start_time = time.time()
    dict_all["model_hist_debiased_downscaled"] = downscale_from_coarse(
        dict_all["model_hist_debiased"], error_map=error_map, fine_grid=dict_all["obs"]
    )
    if verbose:
        elapsed = time.time() - step_start_time
        print(f"Downscaled historical: {elapsed:.2f} seconds")

    step_start_time = time.time()
    dict_all["scenario_debiased_downscaled"] = downscale_from_coarse(
        dict_all["scenario_debiased"], error_map=error_map, fine_grid=dict_all["obs"]
    )

    if verbose:
        elapsed = time.time() - step_start_time
        print(f"Downscaled future: {elapsed:.2f} seconds")

    return dict_all


def run_bcsd(
    *,
    gcm=None,
    train_period_start=None,
    train_period_end=None,
    predict_period_start=None,
    predict_period_end=None,
    var_name=None,
    verbose=True,
    rechunk_workflow=True,
    detrend_data=True,
):
    dict_all = get_all_data(
        gcm=gcm,
        var_name=var_name,
        verbose=verbose,
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
    )

    dict_all = spatially_disaggregate(dict_all, verbose=verbose, rechunk_workflow=rechunk_workflow)

    return dict_all
