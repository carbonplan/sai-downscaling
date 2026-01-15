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
warnings.filterwarnings("ignore", category=RuntimeWarning) # or do i put this at the top of run_bcsd

RUN_PARAMETERS = {
    "OBS": "ERA5",
    "GCM": "CESM2-WACCM",
    "TRAIN_PERIOD_START": 1978,
    "TRAIN_PERIOD_END": 2014,
    "PREDICT_PERIOD_START": 2015,
    "PREDICT_PERIOD_END": 2100,
    "VAR": "tas",
}


def main(verbose=True, rechunk_workflow=True, run_parameters=RUN_PARAMETERS):
    start_time = time.time()

    ################## Load data
    ssp245 = get_experiment(
        gcm=run_parameters["GCM"], scenario="SSP245", var=run_parameters["VAR"]
    )
    ssp245 = ssp245.isel(ensemble_member=0)
    ssp245 = ssp245.drop_vars("spatial_ref")

    model_historical = get_experiment(
        gcm=run_parameters["GCM"], scenario="Historical", var=run_parameters["VAR"]
    )
    model_historical = model_historical.drop_vars("spatial_ref")

    obs = get_obs(var=run_parameters["VAR"])
    obs = obs.drop_vars("spatial_ref")

    if verbose:
        elapsed = time.time() - start_time
        print(f"Loaded data: {elapsed:.2f} seconds")

    start_time = time.time()

    ################## Subset time
    step_start_time = time.time()
    dict_all = {}

    dict_all["model_hist"] = subset_time(
        model_historical, run_parameters, time_period="train"
    )
    dict_all["obs"] = subset_time(obs, run_parameters, time_period="train")
    dict_all["ssp245"] = subset_time(ssp245, run_parameters, time_period="predict")

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
    dict_all["obs_coarse"] = dict_all["obs"].regrid.conservative(dict_all['model_hist']).persist()

    if verbose:
        elapsed = time.time() - step_start_time
        print(f"Interpolated obs to coarse grid: {elapsed:.2f} seconds")

    ################## Do Quantile Mapping
    if rechunk_workflow:
        step_start_time = time.time()
        for key in ["obs_coarse", "model_hist", "ssp245"]:
            dict_all[key] = rechunk(dict_all[key], pattern="full_time")
            dict_all[key] = dict_all[key].persist()
        elapsed = time.time() - step_start_time
        if verbose:
            print(f"Rechunked all to full time: {elapsed:.2f} seconds")

    step_start_time = time.time()
    debiaser = QuantileMapping.from_variable(
        variable=run_parameters["VAR"], mapping_type="parametric"
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
    cm_future = dict_all["ssp245"].load().values
    ssp245_fut_debiased = debiaser.apply(
        obs=obs,
        cm_hist=cm_hist,
        cm_future=cm_future,
        time_obs=dict_all["obs_coarse"]["time"].values,
        time_cm_hist=dict_all["model_hist"]["time"].values,
        time_cm_future=dict_all["ssp245"]["time"].values,
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

    dict_all["ssp245_debiased"] = xr.DataArray(
        data=ssp245_fut_debiased,
        coords={
            "lat": dict_all["ssp245"]["lat"],
            "lon": dict_all["ssp245"]["lon"],
            "time": dict_all["ssp245"]["time"],
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
    dict_all["ssp245_debiased_downscaled"] = downscale_from_coarse(
        dict_all["ssp245_debiased"], error_map=error_map, fine_grid=dict_all["obs"]
    )

    if verbose:
        elapsed = time.time() - step_start_time
        print(f"Downscaled future: {elapsed:.2f} seconds")

        elapsed_total = time.time() - start_time
        print(f"TOTAL TIME: {elapsed_total:.2f} seconds")

    return dict_all
