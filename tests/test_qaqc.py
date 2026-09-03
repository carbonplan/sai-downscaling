from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from srm.qaqc import (
    CHECK_AMBER,
    CHECK_RED,
    DISTORTION_STAGES,
    _parse_gap_fill_member_map,
    area_weights,
    calculate_distortion_flags,
    calculate_reasonable_bounds_doy,
    check_ensemble_spread,
    compute_deltas,
    distortion_fields,
    distortion_summary,
    enumerate_scenario_comparisons,
    find_exceedance_regions,
    highlight,
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


# ---------------------------------------------------------------------------
# Ensemble spread / gap-fill seam
# ---------------------------------------------------------------------------


def _spread_ds(member_values: dict[str, list[float]], *, start: str = "2015-01-01") -> xr.Dataset:
    """Build a (member, time, lat, lon) dataset whose global mean per day is given."""
    members = list(member_values)
    n_time = len(next(iter(member_values.values())))
    time = pd.date_range(start, periods=n_time, freq="YS")
    lat = np.array([-30.0, 30.0])
    lon = np.array([0.0, 180.0])
    data = np.array(
        [[[[v, v], [v, v]] for v in member_values[m]] for m in members], dtype="float64"
    )
    return xr.Dataset(
        {"tas": (["ensemble_member", "time", "lat", "lon"], data)},
        coords={"ensemble_member": members, "time": time, "lat": lat, "lon": lon},
    )


def _gap_filled_ds() -> xr.Dataset:
    """A gap-filled ssp245 group in miniature: a 3 -> 1 bridge, then distinct native years."""
    ds = _spread_ds(
        {
            "r01": [1.0, 1.0, 10.0, 20.0],
            "r02": [2.0, 2.0, 11.0, 21.0],
            "r03": [1.0, 1.0, 12.0, 22.0],
            "r04": [1.0, 1.0, 13.0, 23.0],
        }
    )
    ds.attrs.update(
        {
            "gap_fill_period": "2015-2016",
            "gap_fill_member_map": "r01→rA, r02→rB, r03→rA, r04→rA",
        }
    )
    return ds


def test_parse_gap_fill_member_map_accepts_both_arrow_glyphs():
    assert _parse_gap_fill_member_map("r01→rA, r02→rB") == {"r01": "rA", "r02": "rB"}
    assert _parse_gap_fill_member_map("r01->rA, r02->rB") == {"r01": "rA", "r02": "rB"}


def test_parse_gap_fill_member_map_raises_rather_than_passing_vacuously():
    with pytest.raises(ValueError, match="no recognized arrow"):
        _parse_gap_fill_member_map("r01 rA, r02 rB")
    with pytest.raises(ValueError, match="no member pairs"):
        _parse_gap_fill_member_map("   ")


def test_check_ensemble_spread_returns_one_full_row_without_gap_fill_attrs():
    ds = _spread_ds({"r01": [1.0, 2.0], "r02": [3.0, 4.0]})
    rows = check_ensemble_spread(ds, "GCM group")

    assert [r["window"] for r in rows] == ["full"]
    assert rows[0]["ok"] is True
    assert rows[0]["day"] == "2015-01-01"


def test_check_ensemble_spread_still_flags_real_duplication():
    ds = _spread_ds({"r01": [1.0], "r02": [1.0]})
    (row,) = check_ensemble_spread(ds, "GCM group")

    assert row["ok"] is False
    assert row["n_distinct"] == 1
    assert row["n_members"] == 2


def test_check_ensemble_spread_splits_a_gap_filled_group_into_two_windows():
    rows = check_ensemble_spread(_gap_filled_ds(), "CESM2-WACCM ssp245")

    assert [r["window"] for r in rows] == ["bridge", "native"]
    bridge, native = rows

    # Passes despite 4 members collapsing to 2 means: the map predicts exactly that.
    assert bridge["ok"] is True
    assert bridge["n_distinct"] == 2
    assert bridge["day"] == "2015-01-01"
    assert bridge["groups"] == bridge["expected_groups"] == "r01+r03+r04 | r02"

    assert native["ok"] is True
    assert native["n_distinct"] == 4
    assert native["day"] == "2017-01-01"
    assert native["expected_groups"] is None


def test_check_ensemble_spread_bridge_fails_when_the_member_map_is_miswired():
    ds = _gap_filled_ds()
    # Data still groups r01+r03+r04; the map now claims r04 came from rB.
    ds.attrs["gap_fill_member_map"] = "r01→rA, r02→rB, r03→rA, r04→rB"
    bridge, native = check_ensemble_spread(ds, "CESM2-WACCM ssp245")

    assert bridge["ok"] is False
    assert bridge["groups"] == "r01+r03+r04 | r02"
    assert bridge["expected_groups"] == "r01+r03 | r02+r04"
    assert native["ok"] is True


def test_check_ensemble_spread_checks_the_bridge_rather_than_skipping_it():
    ds = _gap_filled_ds()
    # Broken stitch: r04 should match r01/r03 over the bridge. Mid-record never sees it.
    ds["tas"].loc[{"ensemble_member": "r04", "time": ds.time.values[:2]}] = 9.0
    bridge, native = check_ensemble_spread(ds, "CESM2-WACCM ssp245")

    assert bridge["ok"] is False
    assert bridge["groups"] == "r01+r03 | r02 | r04"
    assert native["ok"] is True


def test_check_ensemble_spread_day_index_offsets_within_each_window():
    rows = check_ensemble_spread(_gap_filled_ds(), "CESM2-WACCM ssp245", day_index=1)

    assert [r["day"] for r in rows] == ["2016-01-01", "2018-01-01"]
    assert all(r["ok"] for r in rows)


def test_check_ensemble_spread_treats_an_all_bridge_group_as_a_single_row():
    ds = _spread_ds({"r01": [1.0, 1.0], "r02": [2.0, 2.0]})
    ds.attrs.update({"gap_fill_period": "2015-2016", "gap_fill_member_map": "r01→rA"})

    assert [r["window"] for r in check_ensemble_spread(ds, "GCM group")] == ["full"]


def test_check_ensemble_spread_handles_a_missing_variable():
    ds = _spread_ds({"r01": [1.0]}).rename({"tas": "pr"})
    (row,) = check_ensemble_spread(ds, "GCM group")

    assert row["ok"] is True
    assert row["members"] == [] and row["means"] == []


# --- Check-table highlighting -------------------------------------------------------------------


def _colors(df: pd.DataFrame, **kwargs) -> dict:
    """{(row label, column): "red" | "amber" | "green" | None} for every cell of a styled table."""
    names = {"#f8d7da": "red", "#fff3cd": "amber", "#d1e7dd": "green"}
    colors = {(row, col): None for row in df.index for col in df.columns}
    for (row, col), props in highlight(df, **kwargs)._compute().ctx.items():
        background = next((value for prop, value in props if prop == "background-color"), None)
        colors[(df.index[row], df.columns[col])] = names.get(background)
    return colors


def _summary() -> pd.DataFrame:
    """Stand-in for the QA notebooks' per-leaf Part 1 summary: one clean leaf, one broken."""
    return pd.DataFrame(
        {
            "no_nans": [True, False],
            "no_interior_nans": [True, False],
            "reasonable_range": [True, False],
            "n_irregular_days": [0, 3],
            "bridged_pre_sai": [False, True],
            "stale_end_tail": [False, False],
            "n_time": [31046, 31046],
        },
        index=["clean", "broken"],
    )


def test_highlight_colors_a_boolean_check_by_whether_it_passed():
    colors = _colors(_summary())

    assert colors[("clean", "no_interior_nans")] == "green"
    assert colors[("broken", "no_interior_nans")] == "red"


def test_highlight_colors_sentinel_booleans_red_only_when_they_fire():
    colors = _colors(_summary())

    assert colors[("broken", "bridged_pre_sai")] == "red"
    assert colors[("clean", "bridged_pre_sai")] == "green"
    assert colors[("broken", "stale_end_tail")] == "green"


def test_highlight_reports_irregular_days_amber_rather_than_red():
    colors = _colors(_summary())

    assert colors[("broken", "n_irregular_days")] == "amber"
    assert colors[("clean", "n_irregular_days")] == "green"


def test_highlight_leaves_unregistered_columns_alone():
    colors = _colors(_summary().assign(some_new_metric=[0, 1]))

    assert colors[("clean", "n_time")] is None
    assert colors[("broken", "some_new_metric")] is None


def test_highlight_demotes_a_column_from_red_to_amber():
    plain = _colors(_summary())
    demoted = _colors(_summary(), demote=["no_nans"])

    assert plain[("broken", "no_nans")] == "red"
    assert demoted[("broken", "no_nans")] == "amber"
    assert demoted[("broken", "no_interior_nans")] == "red"  # others keep their severity


def test_highlight_rejects_demoting_an_unregistered_column():
    with pytest.raises(KeyError, match="no_nan"):
        highlight(_summary(), demote=["no_nan"])


def test_highlight_flags_list_columns_only_when_they_are_non_empty():
    df = pd.DataFrame(
        {"missing": [[], [("dtr", "003")]], "unexpected": [[], []], "pass": [True, False]},
        index=["complete", "short"],
    )
    colors = _colors(df)

    assert colors[("complete", "missing")] == "green"
    assert colors[("short", "missing")] == "red"
    assert colors[("short", "unexpected")] == "green"
    assert colors[("short", "pass")] == "red"


def test_highlight_requires_exactly_one_grid_per_family():
    colors = _colors(pd.DataFrame({"n_distinct_grids": [1, 2]}, index=["fine", "coarse"]))

    assert colors[("fine", "n_distinct_grids")] == "green"
    assert colors[("coarse", "n_distinct_grids")] == "red"


def test_highlight_flags_counts_that_should_be_zero():
    df = pd.DataFrame(
        {"days_max<min": [0, 7], "neg_cell_days": [0, 12], "high_cell_days": [0, 0]},
        index=["clean", "regressed"],
    )
    colors = _colors(df)

    assert colors[("clean", "days_max<min")] == "green"
    assert colors[("regressed", "days_max<min")] == "red"
    assert colors[("regressed", "neg_cell_days")] == "red"
    assert colors[("regressed", "high_cell_days")] == "green"


def test_highlight_renders_a_multiindex_table():
    df = _summary()
    df.index = pd.MultiIndex.from_tuples(
        [("CESM2-WACCM", "ssp245", "tas", "003"), ("UKESM", "g6_1p5k", "pr", "r2i1p1f2")],
        names=["gcm", "scenario", "variable", "member"],
    )

    assert "#f8d7da" in highlight(df).to_html()


def test_highlight_registries_do_not_overlap():
    """A column in both would take its color from dict ordering rather than from intent."""
    assert not set(CHECK_RED) & set(CHECK_AMBER)
