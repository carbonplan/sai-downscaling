from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from srm.qaqc import (
    calculate_reasonable_bounds_doy,
    find_exceedance_regions,
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


# ---------------------------------------------------------------------------
# Exceedance region finding
# ---------------------------------------------------------------------------


def _map_2d(values: np.ndarray, *, lon_start: float = -180.0) -> xr.DataArray:
    """2-D (lat, lon) exceedance map with a realistic global lon convention."""
    n_lat, n_lon = values.shape
    lat = np.linspace(-80, 80, n_lat)
    lon = np.linspace(lon_start, lon_start + 360 - 360 / n_lon, n_lon)
    return xr.DataArray(values, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name="over")


def _two_blob_map() -> xr.DataArray:
    """A 9-cell patch of magnitude 5 and an isolated 1-cell fleck of magnitude 20.

    The fleck is the larger magnitude but the smaller area, which is what
    separates magnitude ranking from area ranking. Neither blob touches the
    longitude seam, and they are not 8-connected to each other.
    """
    values = np.zeros((10, 12))
    values[2:5, 2:5] = 5.0
    values[7, 8] = 20.0
    return _map_2d(values)


def test_find_exceedance_regions_ranks_by_magnitude_not_area():
    regions = find_exceedance_regions(_two_blob_map(), direction="high")

    assert len(regions) == 2
    # The 1-cell fleck outranks the 9-cell patch because it is the larger exceedance.
    assert regions.loc[0, "n_cells"] == 1
    assert regions.loc[0, "worst_value"] == 20.0
    assert regions.loc[1, "n_cells"] == 9
    assert regions.loc[1, "worst_value"] == 5.0
    # Regions are numbered in rank order.
    assert regions["region"].tolist() == [1, 2]
    # area_frac matches the check's unweighted cell-count convention.
    assert regions.loc[1, "area_frac"] == 9 / 120


def test_find_exceedance_regions_locates_the_worst_cell():
    blob = _two_blob_map()
    regions = find_exceedance_regions(blob, direction="high")

    worst = regions.loc[0]
    assert worst["worst_lat"] == blob["lat"].values[7]
    assert worst["worst_lon"] == blob["lon"].values[8]
    # A single-cell region's centroid and bounding box collapse onto that cell.
    assert worst["centroid_lat"] == worst["worst_lat"]
    assert worst["lat_min"] == worst["lat_max"] == worst["worst_lat"]
    assert not worst["wraps_lon"]


def test_find_exceedance_regions_merges_across_lon_seam():
    """A patch straddling the antimeridian is one region, not two."""
    values = np.zeros((10, 12))
    values[4, 0] = 3.0  # first longitude column
    values[4, 11] = 4.0  # last longitude column, adjacent across the seam
    regions = find_exceedance_regions(_map_2d(values), direction="high")

    assert len(regions) == 1
    assert regions.loc[0, "n_cells"] == 2
    assert regions.loc[0, "wraps_lon"]
    # The two flagged cells sit at lon -180 and lon 150 on this 30-degree grid. The
    # circular mean puts the centroid between them at 165, near the seam; an
    # arithmetic mean would report -15, on the opposite side of the planet.
    assert regions.loc[0, "centroid_lon"] == pytest.approx(165.0)


def test_find_exceedance_regions_merges_diagonally_across_lon_seam():
    """8-connectivity applies across the seam too, not just within the plane."""
    values = np.zeros((10, 12))
    values[4, 0] = 3.0
    values[5, 11] = 4.0  # diagonally adjacent once longitude wraps
    regions = find_exceedance_regions(_map_2d(values), direction="high")

    assert len(regions) == 1
    assert regions.loc[0, "n_cells"] == 2


def test_find_exceedance_regions_merges_a_chain_across_lon_seam():
    """Three components chaining transitively across the seam collapse to one region.

    Component P touches Q across the seam, and Q touches R, but P and R never touch each
    other directly. A union-find that failed to resolve chains to a common root would
    report two or three regions here instead of one.
    """
    values = np.zeros((12, 12))
    values[2, 0] = values[3, 0] = 1.0  # P, west edge
    values[4, 11] = values[5, 11] = 2.0  # Q, east edge: adjacent to P and to R across the seam
    values[6, 0] = values[7, 0] = 3.0  # R, west edge
    regions = find_exceedance_regions(_map_2d(values), direction="high")

    assert len(regions) == 1
    assert regions.loc[0, "n_cells"] == 6
    assert regions.loc[0, "wraps_lon"]


def test_find_exceedance_regions_does_not_merge_across_lat_poles():
    """Latitude is not periodic, so top and bottom rows stay separate regions."""
    values = np.zeros((10, 12))
    values[0, 5] = 3.0
    values[9, 5] = 4.0
    regions = find_exceedance_regions(_map_2d(values), direction="high")

    assert len(regions) == 2


def test_find_exceedance_regions_empty_when_nothing_flagged():
    regions = find_exceedance_regions(_map_2d(np.zeros((10, 12))), direction="high")

    assert regions.empty
    # Columns survive so downstream code can index them unconditionally.
    assert "worst_value" in regions.columns and "n_cells" in regions.columns


def test_find_exceedance_regions_min_cells_filters_flecks():
    regions = find_exceedance_regions(_two_blob_map(), direction="high", min_cells=5)

    assert len(regions) == 1
    assert regions.loc[0, "n_cells"] == 9


def test_find_exceedance_regions_top_n_truncates():
    values = np.zeros((10, 12))
    for i, col in enumerate([1, 4, 7, 10]):
        values[2 * i, col] = float(i + 1)
    regions = find_exceedance_regions(_map_2d(values), direction="high", top_n=2)

    assert len(regions) == 2
    assert regions["worst_value"].tolist() == [4.0, 3.0]


def test_find_exceedance_regions_low_direction_flags_negatives():
    values = np.zeros((10, 12))
    values[3, 3] = 7.0  # positive: must be ignored when direction="low"
    values[6, 6] = -2.0
    values[8, 9] = -9.0
    regions = find_exceedance_regions(_map_2d(values), direction="low")

    assert len(regions) == 2
    # Ranked by magnitude, so the most negative comes first.
    assert regions["worst_value"].tolist() == [-9.0, -2.0]


def test_find_exceedance_regions_ignores_nan():
    values = np.zeros((10, 12))
    values[:] = np.nan
    values[3, 3] = 6.0
    regions = find_exceedance_regions(_map_2d(values), direction="high")

    assert len(regions) == 1
    assert regions.loc[0, "n_cells"] == 1


def test_find_exceedance_regions_scale_applies_to_magnitude_only():
    plain = find_exceedance_regions(_two_blob_map(), direction="high")
    scaled = find_exceedance_regions(_two_blob_map(), direction="high", scale=86400.0)

    assert scaled.loc[0, "worst_value"] == plain.loc[0, "worst_value"] * 86400.0
    # Area metrics are dimensionless and must not be touched.
    assert scaled.loc[0, "area_frac"] == plain.loc[0, "area_frac"]


def test_find_exceedance_regions_rejects_bad_direction():
    with pytest.raises(ValueError, match="direction"):
        find_exceedance_regions(_two_blob_map(), direction="sideways")
