import time

from ibicus.debias import QuantileMapping
import rasterix  # noqa: F401  # side-effect import: registers .proj/.rio accessors
import xarray as xr
import xarray_regrid

from srm.downscaling_utils import (
    get_experiment,
    get_obs,
    subset_time,
    rechunk,
    calculate_error_map,
    downscale_from_coarse,
)

import warnings

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


def run_bcsd(
    *,
    gcm=None,
    train_period_start=None,
    train_period_end=None,
    predict_period_start=None,
    predict_period_end=None,
    scenario=None,
    var_name=None,
    verbose=True,
    rechunk_workflow=True,
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

    dict_all["model_hist"] = subset_time(
        model_historical,
        start_year=train_period_start,
        end_year=train_period_end,
    )
    dict_all["obs"] = subset_time(
        obs, start_year=train_period_start, end_year=train_period_end
    )
    dict_all["model_scenario"] = subset_time(
        model_scenario, start_year=train_period_start, end_year=predict_period_end
    )

    if verbose:
        elapsed = time.time() - step_start_time
        print(f"Subset time: {elapsed:.2f} seconds")

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
    dict_all["obs_coarse"] = (
        dict_all["obs"].regrid.conservative(dict_all["model_hist"]).persist()
    )

    if verbose:
        elapsed = time.time() - step_start_time
        print(f"Interpolated obs to coarse grid: {elapsed:.2f} seconds")

    ################## Do Quantile Mapping
    if rechunk_workflow:
        step_start_time = time.time()
        for key in ["obs_coarse", "model_hist", "model_scenario"]:
            dict_all[key] = rechunk(dict_all[key], pattern="full_time")
            dict_all[key] = dict_all[key].persist()
        elapsed = time.time() - step_start_time
        if verbose:
            print(f"Rechunked all to full time: {elapsed:.2f} seconds")

    step_start_time = time.time()
    debiaser = QuantileMapping.from_variable(
        variable=var_name, mapping_type="parametric"
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

    ################## Calculate error map
    # Calculate a fine-resolution spatial anomaly pattern derived from the observations
    step_start_time = time.time()
    error_map = calculate_error_map(
        obs_coarse=dict_all["obs_coarse"], obs_fine=dict_all["obs"]
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

        elapsed_total = time.time() - start_time
        print(f"TOTAL TIME: {elapsed_total:.2f} seconds")

    return dict_all
