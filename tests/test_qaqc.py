from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from srm.qaqc import (
    DISTORTION_STAGES,
    area_weights,
    calculate_distortion_flags,
    calculate_reasonable_bounds_doy,
    compute_deltas,
    distortion_fields,
    distortion_summary,
    enumerate_scenario_comparisons,
    find_exceedance_regions,
    obs_doy_bounds,
    scenario_delta_doy,
    sign_flip_mask,
    weighted_fraction,
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


# ---------------------------------------------------------------------------
# Trend-distortion analysis
# ---------------------------------------------------------------------------

# Leaves mirroring the v0.12.0 CESM2-WACCM store for the two variables whose g6 -> ssp245 bridge
# members differ: tas bridges ssp245/003, tasmax bridges ssp245/008.
_CESM_LEAVES = [
    ("historical", "tas", "r3i1p1f1"),
    ("historical", "tasmax", "001"),
    ("ssp245", "tas", "003"),
    ("ssp245", "tasmax", "008"),
    ("g6_1p5k", "tas", "003"),
    ("g6_1p5k", "tasmax", "003"),
]


def _grid(values, *, lat, lon=(0.0, 180.0)) -> xr.DataArray:
    """Small 2-D lat/lon grid from a nested list."""
    return xr.DataArray(
        np.asarray(values, dtype="float64"),
        dims=["lat", "lon"],
        coords={"lat": np.asarray(lat, dtype="float64"), "lon": np.asarray(lon, dtype="float64")},
    )


def _stage_grids(value: float, *, lat=(0.0, 30.0)) -> dict[str, xr.DataArray]:
    """One flat grid per pipeline stage, all holding the same value."""
    flat = [[value, value], [value, value]]
    return {stage: _grid(flat, lat=lat) for stage in DISTORTION_STAGES}


def test_enumerate_scenario_comparisons_resolves_the_g6_ssp_bridge_per_variable():
    """The SAI comparison must pair each variable with ITS bridge member, not one fixed member.

    This is issue #448: on CESM2-WACCM, g6_1p5k/003 bridges ssp245/003 for tas but ssp245/008 for
    tasmax. Hardcoding one member is what kept the notebook stuck on a single variable.
    """
    comparisons, skipped = enumerate_scenario_comparisons(_CESM_LEAVES, gcm="CESM2-WACCM")

    g6_ssp = comparisons[comparisons.family == "g6_ssp"].set_index("variable")
    assert g6_ssp.loc["tas", "before_member"] == "003"
    assert g6_ssp.loc["tasmax", "before_member"] == "008"
    # Every g6_ssp row differences the two scenarios over the same window, not against historical.
    assert set(g6_ssp["before_scenario"]) == {"ssp245"}
    assert len(skipped) == 0


def test_enumerate_scenario_comparisons_resolves_historical_baselines():
    """The tasmax family uses the corrected historical run, so the baseline differs from tas."""
    comparisons, _ = enumerate_scenario_comparisons(_CESM_LEAVES, gcm="CESM2-WACCM")

    hist_baselines = {
        (row.family, row.variable): row.before_member
        for row in comparisons.itertuples()
        if row.before_scenario == "historical"
    }
    assert hist_baselines[("ssp_hist", "tas")] == "r3i1p1f1"
    assert hist_baselines[("ssp_hist", "tasmax")] == "001"
    assert hist_baselines[("g6_hist", "tas")] == "r3i1p1f1"
    assert hist_baselines[("g6_hist", "tasmax")] == "001"


def test_enumerate_scenario_comparisons_expected_totals():
    comparisons, _ = enumerate_scenario_comparisons(_CESM_LEAVES, gcm="CESM2-WACCM")

    counts = comparisons.family.value_counts().to_dict()
    # Two scenario leaves per family here: one tas, one tasmax.
    assert counts == {"ssp_hist": 2, "g6_hist": 2, "g6_ssp": 2}
    assert list(comparisons.columns[:3]) == ["comparison_id", "family", "variable"]
    assert comparisons.comparison_id.is_unique


def test_enumerate_scenario_comparisons_skips_when_the_bridge_leaf_is_absent():
    """A store holding only ssp245/tasmax/003 must skip g6_ssp/tasmax, not silently mispair it.

    This is the failure the hardcoded notebook would hit: ssp245/tasmax/003 does not exist in the
    real store, and pairing against it would compare two different realizations.
    """
    leaves = [
        ("historical", "tasmax", "001"),
        ("ssp245", "tasmax", "003"),
        ("g6_1p5k", "tasmax", "003"),
    ]
    comparisons, skipped = enumerate_scenario_comparisons(leaves, gcm="CESM2-WACCM")

    assert "g6_ssp" not in set(comparisons.family)
    reason = str(skipped.set_index("family").loc["g6_ssp", "reason"])
    assert "ssp245/tasmax/008" in reason and "absent" in reason
    # ssp245/tasmax/003 has no registered lineage at all, so its own comparison is skipped too.
    assert "no registered lineage" in " ".join(skipped.reason)


def test_enumerate_scenario_comparisons_filters_variables_and_returns_typed_empties():
    comparisons, _ = enumerate_scenario_comparisons(
        _CESM_LEAVES, gcm="CESM2-WACCM", variables=["tas"]
    )
    assert set(comparisons.variable) == {"tas"}

    empty, empty_skips = enumerate_scenario_comparisons([], gcm="CESM2-WACCM")
    # Columns must survive an empty result so downstream code does not need a special case.
    assert len(empty) == 0 and "comparison_id" in empty.columns
    assert len(empty_skips) == 0 and "reason" in empty_skips.columns


def test_compute_deltas_is_lazy_and_returns_percent_of_before():
    after = {s: da.chunk() for s, da in _stage_grids(12.0).items()}
    before = {s: da.chunk() for s, da in _stage_grids(10.0).items()}

    deltas, deltas_pct = compute_deltas(after, before)

    assert all(_is_lazy(da) for da in deltas.values())
    assert all(_is_lazy(da) for da in deltas_pct.values())
    assert set(deltas) == set(DISTORTION_STAGES)
    assert float(deltas["raw"].max().compute()) == pytest.approx(2.0)
    assert float(deltas_pct["raw"].max().compute()) == pytest.approx(20.0)


def test_compute_deltas_zero_baseline_gives_nan_not_inf():
    """An infinite percent change would propagate into the extremes and flag the cell."""
    after = _stage_grids(5.0)
    before = {s: _grid([[0.0, 10.0], [10.0, 10.0]], lat=(0.0, 30.0)) for s in DISTORTION_STAGES}

    _, deltas_pct = compute_deltas(after, before)
    pct = deltas_pct["raw"]

    assert np.isnan(pct.isel(lat=0, lon=0))
    assert not np.isinf(pct).any()
    assert float(pct.isel(lat=0, lon=1)) == pytest.approx(-50.0)


def test_calculate_distortion_flags_requires_both_tolerances():
    # (large abs, small pct), (small abs, large pct), (large both), (small both)
    absolute = _grid([[100.0, 1.0], [100.0, 1.0]], lat=(0.0, 30.0))
    percent = _grid([[0.1, 50.0], [50.0, 0.1]], lat=(0.0, 30.0))

    flag = calculate_distortion_flags(absolute, percent, tolerance_absolute=10.0, tolerance_pct=2.0)

    assert not bool(flag.isel(lat=0, lon=0))  # large absolute, percent below tolerance
    assert not bool(flag.isel(lat=0, lon=1))  # large percent, absolute below tolerance
    assert bool(flag.isel(lat=1, lon=0))  # both exceeded
    assert not bool(flag.isel(lat=1, lon=1))  # neither exceeded


def test_calculate_distortion_flags_zero_pct_tolerance_disables_the_percent_condition():
    """Temperature in kelvin sets tolerance_pct=0; a NaN percent must not veto the flag."""
    absolute = _grid([[1.0, 0.1], [1.0, 0.1]], lat=(0.0, 30.0))
    percent = _grid([[np.nan, np.nan], [np.nan, np.nan]], lat=(0.0, 30.0))

    flag = calculate_distortion_flags(absolute, percent, tolerance_absolute=0.25, tolerance_pct=0)

    assert bool(flag.isel(lat=0, lon=0)) and bool(flag.isel(lat=1, lon=0))
    assert not bool(flag.isel(lat=0, lon=1))
    # The literal reading, abs(nan) > 0, would have flagged nothing at all.
    assert int(flag.sum()) == 2


def test_sign_flip_mask_needs_both_stages_past_the_threshold():
    reference = _grid([[10.0, 10.0], [1.0, -10.0]], lat=(0.0, 30.0))
    tested = _grid([[-10.0, 10.0], [-1.0, 10.0]], lat=(0.0, 30.0))

    flip = sign_flip_mask(tested, reference, threshold=5.0)

    assert bool(flip.isel(lat=0, lon=0))  # +10 -> -10, both clear the threshold
    assert not bool(flip.isel(lat=0, lon=1))  # same sign
    assert not bool(flip.isel(lat=1, lon=0))  # opposite signs but neither clears 5.0
    assert bool(flip.isel(lat=1, lon=1))  # -10 -> +10


def test_area_weights_follow_cosine_and_never_go_negative():
    lat_values = [-90.0, -60.0, 0.0, 90.0]
    weights = area_weights(xr.DataArray(lat_values, dims="lat", coords={"lat": lat_values}))

    assert float(weights.sel(lat=0.0)) == pytest.approx(1.0)
    assert float(weights.sel(lat=-60.0)) == pytest.approx(0.5)
    assert (weights >= 0).all()


def test_weighted_fraction_discounts_the_poles():
    """A flag confined to the polar rows must not read as two thirds of the planet."""
    lat = (-89.0, 0.0, 89.0)
    mask = _grid([[True, True], [False, False], [True, True]], lat=lat).astype(bool)
    weights = area_weights(mask.lat)

    fraction = weighted_fraction(mask, weights)

    assert float(mask.mean()) == pytest.approx(2 / 3)  # what an unweighted mean would say
    assert fraction == pytest.approx(0.0343, abs=1e-3)


def test_weighted_fraction_excludes_invalid_cells_from_the_denominator():
    """Conservative recoarsening can NaN an edge band; it must leave the question, not answer it."""
    lat = (0.0, 30.0)
    mask = _grid([[True, True], [False, False]], lat=lat).astype(bool)
    valid = _grid([[True, True], [False, False]], lat=lat).astype(bool)

    # Counting the invalid row as "not distorted" halves the answer.
    assert weighted_fraction(mask, area_weights(mask.lat)) < 0.6
    assert weighted_fraction(mask, area_weights(mask.lat), valid=valid) == pytest.approx(1.0)


def test_weighted_fraction_is_nan_for_an_empty_domain():
    mask = _grid([[True, True], [True, True]], lat=(0.0, 30.0)).astype(bool)
    valid = xr.zeros_like(mask, dtype=bool)

    assert np.isnan(weighted_fraction(mask, area_weights(mask.lat), valid=valid))


def test_distortion_summary_reports_signed_extremes_and_weighted_fractions():
    lat = (0.0, 30.0)
    absolute = _grid([[2.0, -3.0], [0.0, 0.0]], lat=lat)
    percent = _grid([[20.0, -30.0], [0.0, 0.0]], lat=lat)
    flag = calculate_distortion_flags(absolute, percent, tolerance_absolute=1.0, tolerance_pct=2.0)

    summary = distortion_summary(
        absolute,
        percent,
        flag,
        weights=area_weights(absolute.lat),
        sign_flip=sign_flip_mask(absolute, percent, threshold=1.0),
    )

    # Extremes are signed, so the minimum is the largest negative distortion, not the smallest one.
    assert summary["distortion_abs_max"] == pytest.approx(2.0)
    assert summary["distortion_abs_min"] == pytest.approx(-3.0)
    assert summary["distortion_pct_max"] == pytest.approx(20.0)
    assert summary["distortion_pct_min"] == pytest.approx(-30.0)
    # Only the equatorial row is flagged, and it carries most of the weight.
    assert summary["frac_area_distorted"] == pytest.approx(0.536, abs=1e-3)
    assert "frac_area_sign_flip" in summary


def test_distortion_summary_omits_sign_flip_when_not_requested():
    lat = (0.0, 30.0)
    absolute = _grid([[2.0, 2.0], [2.0, 2.0]], lat=lat)
    flag = calculate_distortion_flags(absolute, absolute, tolerance_absolute=1.0, tolerance_pct=0)

    summary = distortion_summary(absolute, absolute, flag, weights=area_weights(absolute.lat))

    assert "frac_area_sign_flip" not in summary


def test_distortion_fields_pairs_the_documented_stages():
    deltas = dict(_stage_grids(0.0))
    deltas["raw"] = _grid([[1.0, 1.0], [1.0, 1.0]], lat=(0.0, 30.0))
    deltas["coarse_debiased"] = _grid([[4.0, 4.0], [4.0, 4.0]], lat=(0.0, 30.0))
    deltas["coarsened_downscaled_debiased"] = _grid([[7.0, 7.0], [7.0, 7.0]], lat=(0.0, 30.0))

    absolute, _ = distortion_fields(deltas, deltas, "coarse_debiased_vs_raw")
    assert float(absolute.max()) == pytest.approx(3.0)

    absolute, _ = distortion_fields(deltas, deltas, "downscaled_debiased_vs_raw")
    assert float(absolute.max()) == pytest.approx(6.0)

    # The downscaling-only pair compares the recoarsened output against coarse debiased, which is
    # what isolates downscaling from debiasing.
    absolute, _ = distortion_fields(deltas, deltas, "downscaled_debiased_vs_coarse_debiased")
    assert float(absolute.max()) == pytest.approx(3.0)
