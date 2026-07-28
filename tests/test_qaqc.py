from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import xarray as xr

from srm.qaqc import (
    calculate_reasonable_bounds_doy,
    obs_doy_bounds,
    scenario_delta_doy,
)


def _coarse(seed: int, *, members: int | None = None) -> xr.DataArray:
    """Small chunked coarse-grid DataArray, optionally with an ensemble dim."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2000-01-01", periods=730, freq="D")
    lat = np.linspace(-60, 60, 4)
    lon = np.linspace(0, 300, 5)
    if members is None:
        data = rng.random((time.size, lat.size, lon.size), dtype="float32") * 30 + 270
        da = xr.DataArray(
            data, dims=["time", "lat", "lon"], coords={"time": time, "lat": lat, "lon": lon}
        )
        return da.chunk({"time": 365})
    ens = [f"m{i}" for i in range(members)]
    data = rng.random((members, time.size, lat.size, lon.size), dtype="float32") * 30 + 270
    da = xr.DataArray(
        data,
        dims=["ensemble_member", "time", "lat", "lon"],
        coords={"ensemble_member": ens, "time": time, "lat": lat, "lon": lon},
    )
    return da.chunk({"time": 365})


def _fine(seed: int) -> xr.DataArray:
    """Small chunked fine-grid observation DataArray on a finer lat/lon grid."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2000-01-01", periods=730, freq="D")
    lat = np.linspace(-60, 60, 8)
    lon = np.linspace(0, 300, 10)
    data = rng.random((time.size, lat.size, lon.size), dtype="float32") * 30 + 270
    da = xr.DataArray(
        data, dims=["time", "lat", "lon"], coords={"time": time, "lat": lat, "lon": lon}
    )
    return da.chunk({"time": 365})


def _is_lazy(da: xr.DataArray) -> bool:
    """True when the array is dask-backed and has not been computed."""
    return da.chunks is not None


def test_obs_doy_bounds_is_lazy_with_expected_shape():
    obs = _fine(0)
    low, high = obs_doy_bounds(obs)

    assert _is_lazy(low) and _is_lazy(high)
    assert low.dims == ("dayofyear", "lat", "lon")
    # Fine grid is preserved; day-of-year spans a full leap year.
    assert low.sizes["lat"] == obs.sizes["lat"] and low.sizes["lon"] == obs.sizes["lon"]
    assert low.sizes["dayofyear"] == 366


def test_scenario_delta_doy_is_lazy_on_coarse_grid():
    scenario = _coarse(1)
    historical = _coarse(2, members=3)
    delta_min, delta_max = scenario_delta_doy(scenario, historical)

    assert _is_lazy(delta_min) and _is_lazy(delta_max)
    assert delta_min.dims == ("dayofyear", "lat", "lon")
    assert "ensemble_member" not in delta_min.dims
    assert delta_min.sizes["lat"] == scenario.sizes["lat"]


def test_scenario_delta_doy_accepts_single_member_historical():
    """A scalar-selected historical member (no ensemble_member dim) must work."""
    scenario = _coarse(1)
    historical = _coarse(2, members=3).isel(ensemble_member=0)
    assert "ensemble_member" not in historical.dims

    delta_min, delta_max = scenario_delta_doy(scenario, historical)
    assert _is_lazy(delta_min) and _is_lazy(delta_max)
    assert delta_min.dims == ("dayofyear", "lat", "lon")
    assert np.isfinite(delta_min.compute()).all()


def test_calculate_reasonable_bounds_doy_is_lazy_and_ordered():
    scenario = _coarse(1)
    historical = _coarse(2, members=3)
    obs = _fine(0)

    low, high = calculate_reasonable_bounds_doy(scenario, historical, obs)

    # Lazy: building the bounds must not trigger any computation.
    assert _is_lazy(low) and _is_lazy(high)
    # Bounds live on the fine (observation) grid.
    assert low.sizes["lat"] == obs.sizes["lat"] and low.sizes["lon"] == obs.sizes["lon"]

    low_c, high_c = dask.compute(low, high)
    assert np.isfinite(low_c).all() and np.isfinite(high_c).all()
    # high_bound >= low_bound everywhere the observations are defined.
    assert bool((high_c - low_c >= -1e-4).all())


def test_calculate_reasonable_bounds_doy_window_changes_result():
    scenario = _coarse(1)
    historical = _coarse(2, members=3)
    obs = _fine(0)

    _, high_default = calculate_reasonable_bounds_doy(scenario, historical, obs)
    _, high_narrow = calculate_reasonable_bounds_doy(scenario, historical, obs, window=2)

    # A wider rolling window smooths the envelope differently, so the bounds differ.
    assert not np.allclose(high_default.compute(), high_narrow.compute())
