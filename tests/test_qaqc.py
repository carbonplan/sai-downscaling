from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from saidownscale.qaqc import (
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

_DOY_TIME = pd.date_range("2000-01-01", periods=366, freq="D")


def _daily(seed: int, lat_n: int, lon_n: int, members: int | None = None) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    coords = {
        "time": _DOY_TIME,
        "lat": np.linspace(-60, 60, lat_n),
        "lon": np.linspace(0, 300, lon_n),
    }
    dims = ["time", "lat", "lon"]
    shape = (_DOY_TIME.size, lat_n, lon_n)
    if members is not None:
        coords = {"ensemble_member": [f"m{i}" for i in range(members)], **coords}
        dims = ["ensemble_member", *dims]
        shape = (members, *shape)
    data = rng.random(shape, dtype="float32") * 30 + 270
    return xr.DataArray(data, dims=dims, coords=coords).chunk({"time": 183})


def _is_lazy(da: xr.DataArray) -> bool:
    return da.chunks is not None


def test_obs_doy_bounds_is_lazy_with_expected_shape():
    obs = _daily(0, 8, 10)
    low, high = obs_doy_bounds(obs)

    assert _is_lazy(low) and _is_lazy(high)
    assert low.dims == ("dayofyear", "lat", "lon")
    assert low.sizes["lat"] == obs.sizes["lat"] and low.sizes["lon"] == obs.sizes["lon"]
    assert low.sizes["dayofyear"] == 366


def test_reasonable_bounds_doy_are_lazy_ordered_and_window_sensitive(subtests):
    scenario = _daily(1, 4, 5)
    historical = _daily(2, 4, 5, members=3)
    obs = _daily(0, 8, 10)
    variants = {"ensemble": historical, "single_member": historical.isel(ensemble_member=0)}

    for name, hist in variants.items():
        with subtests.test(historical=name):
            delta_min, delta_max = scenario_delta_doy(scenario, hist)
            assert _is_lazy(delta_min) and _is_lazy(delta_max)
            assert delta_min.dims == ("dayofyear", "lat", "lon")
            assert delta_min.sizes["lat"] == scenario.sizes["lat"]

    low, high = calculate_reasonable_bounds_doy(scenario, variants["single_member"], obs)
    _, high_narrow = calculate_reasonable_bounds_doy(scenario, historical, obs, window=2)
    _, high_default = calculate_reasonable_bounds_doy(scenario, historical, obs)
    assert _is_lazy(low) and _is_lazy(high)
    assert low.sizes["lat"] == obs.sizes["lat"] and low.sizes["lon"] == obs.sizes["lon"]

    low_c, high_c, narrow_c, default_c = dask.compute(low, high, high_narrow, high_default)
    assert np.isfinite(low_c).all() and np.isfinite(high_c).all()
    assert bool((high_c - low_c >= -1e-4).all())
    assert not np.allclose(default_c, narrow_c)


def _map_2d(values: np.ndarray, *, lon_start: float = -180.0) -> xr.DataArray:
    n_lat, n_lon = values.shape
    lat = np.linspace(-80, 80, n_lat)
    lon = np.linspace(lon_start, lon_start + 360 - 360 / n_lon, n_lon)
    return xr.DataArray(values, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}, name="over")


def _cells(shape=(10, 12), fill=0.0, **cells) -> xr.DataArray:
    values = np.full(shape, fill)
    for (i, j), v in cells.values():
        values[i, j] = v
    return _map_2d(values)


def test_find_exceedance_regions_ranks_a_large_fleck_above_a_wide_patch():
    values = np.zeros((10, 12))
    values[2:5, 2:5] = 5.0
    values[7, 8] = 20.0
    blob = _map_2d(values)

    regions = find_exceedance_regions(blob, direction="high")
    assert regions["n_cells"].tolist() == [1, 9]
    assert regions["worst_value"].tolist() == [20.0, 5.0]
    assert regions["region"].tolist() == [1, 2]
    assert regions.loc[1, "area_frac"] == 9 / 120
    worst = regions.loc[0]
    assert worst["worst_lat"] == blob["lat"].values[7]
    assert worst["worst_lon"] == blob["lon"].values[8]
    assert worst["centroid_lat"] == worst["worst_lat"]
    assert worst["lat_min"] == worst["lat_max"] == worst["worst_lat"]
    assert not worst["wraps_lon"]

    assert find_exceedance_regions(blob, direction="high", min_cells=5)["n_cells"].tolist() == [9]

    scaled = find_exceedance_regions(blob, direction="high", scale=86400.0)
    assert scaled.loc[0, "worst_value"] == 20.0 * 86400.0
    assert scaled.loc[0, "area_frac"] == regions.loc[0, "area_frac"]

    with pytest.raises(ValueError, match="direction"):
        find_exceedance_regions(blob, direction="sideways")


def test_find_exceedance_regions_connectivity_and_filters(subtests):
    cases = {
        "merges_across_lon_seam": (
            _cells(a=((4, 0), 3.0), b=((4, 11), 4.0)),
            {},
            {"n_cells": [2], "wraps_lon": [True], "centroid_lon": [165.0]},
        ),
        "merges_diagonally_across_seam": (
            _cells(a=((4, 0), 3.0), b=((5, 11), 4.0)),
            {},
            {"n_cells": [2]},
        ),
        "merges_a_chain_across_seam": (
            _cells(
                (12, 12),
                p1=((2, 0), 1.0),
                p2=((3, 0), 1.0),
                q1=((4, 11), 2.0),
                q2=((5, 11), 2.0),
                r1=((6, 0), 3.0),
                r2=((7, 0), 3.0),
            ),
            {},
            {"n_cells": [6], "wraps_lon": [True]},
        ),
        "no_merge_across_poles": (
            _cells(a=((0, 5), 3.0), b=((9, 5), 4.0)),
            {},
            {"n_cells": [1, 1]},
        ),
        "empty": (_cells(), {}, {"worst_value": [], "n_cells": []}),
        "top_n_truncates": (
            _cells(a=((0, 1), 1.0), b=((2, 4), 2.0), c=((4, 7), 3.0), d=((6, 10), 4.0)),
            {"top_n": 2},
            {"worst_value": [4.0, 3.0]},
        ),
        "low_direction_flags_negatives": (
            _cells(a=((3, 3), 7.0), b=((6, 6), -2.0), c=((8, 9), -9.0)),
            {"direction": "low"},
            {"worst_value": [-9.0, -2.0]},
        ),
        "ignores_nan": (_cells(fill=np.nan, a=((3, 3), 6.0)), {}, {"n_cells": [1]}),
    }
    for case, (grid, kwargs, expected) in cases.items():
        with subtests.test(case=case):
            regions = find_exceedance_regions(grid, **{"direction": "high", **kwargs})
            for col, values in expected.items():
                assert regions[col].tolist() == pytest.approx(values)


_CESM_LEAVES = [
    ("historical", "tas", "r3i1p1f1"),
    ("historical", "tasmax", "001"),
    ("ssp245", "tas", "003"),
    ("ssp245", "tasmax", "008"),
    ("g6_1p5k", "tas", "003"),
    ("g6_1p5k", "tasmax", "003"),
]


def _grid(values, *, lat=(0.0, 30.0), lon=(0.0, 180.0)) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype="float64"),
        dims=["lat", "lon"],
        coords={"lat": np.asarray(lat, dtype="float64"), "lon": np.asarray(lon, dtype="float64")},
    )


def _stage_grids(value: float) -> dict[str, xr.DataArray]:
    return {stage: _grid([[value, value], [value, value]]) for stage in DISTORTION_STAGES}


def test_enumerate_scenario_comparisons_pairs_each_variable_with_its_own_bridge():
    """#448: g6_1p5k/003 bridges ssp245/003 for tas but ssp245/008 for tasmax."""
    comparisons, skipped = enumerate_scenario_comparisons(_CESM_LEAVES, gcm="CESM2-WACCM6")

    g6_ssp = comparisons[comparisons.family == "g6_ssp"].set_index("variable")
    assert g6_ssp.loc["tas", "before_member"] == "003"
    assert g6_ssp.loc["tasmax", "before_member"] == "008"
    assert set(g6_ssp["before_scenario"]) == {"ssp245"}
    assert len(skipped) == 0

    hist_baselines = {
        (row.family, row.variable): row.before_member
        for row in comparisons.itertuples()
        if row.before_scenario == "historical"
    }
    assert hist_baselines == {
        ("ssp_hist", "tas"): "r3i1p1f1",
        ("ssp_hist", "tasmax"): "001",
        ("g6_hist", "tas"): "r3i1p1f1",
        ("g6_hist", "tasmax"): "001",
    }
    assert comparisons.family.value_counts().to_dict() == {
        "ssp_hist": 2,
        "g6_hist": 2,
        "g6_ssp": 2,
    }
    assert list(comparisons.columns[:3]) == ["comparison_id", "family", "variable"]
    assert comparisons.comparison_id.is_unique

    filtered, _ = enumerate_scenario_comparisons(
        _CESM_LEAVES, gcm="CESM2-WACCM6", variables=["tas"]
    )
    assert set(filtered.variable) == {"tas"}

    empty, empty_skips = enumerate_scenario_comparisons([], gcm="CESM2-WACCM6")
    assert len(empty) == 0 and "comparison_id" in empty.columns
    assert len(empty_skips) == 0 and "reason" in empty_skips.columns


def test_enumerate_scenario_comparisons_skips_when_the_bridge_leaf_is_absent():
    """#448: ssp245/tasmax/003 is not the tasmax bridge, so g6_ssp must skip rather than mispair."""
    leaves = [
        ("historical", "tasmax", "001"),
        ("ssp245", "tasmax", "003"),
        ("g6_1p5k", "tasmax", "003"),
    ]
    comparisons, skipped = enumerate_scenario_comparisons(leaves, gcm="CESM2-WACCM6")

    assert "g6_ssp" not in set(comparisons.family)
    reason = str(skipped.set_index("family").loc["g6_ssp", "reason"])
    assert "ssp245/tasmax/008" in reason and "absent" in reason
    assert "no registered lineage" in " ".join(skipped.reason)


def test_compute_deltas():
    after = {s: da.chunk() for s, da in _stage_grids(12.0).items()}
    before = {s: da.chunk() for s, da in _stage_grids(10.0).items()}
    deltas, deltas_pct = compute_deltas(after, before)

    assert all(_is_lazy(da) for da in [*deltas.values(), *deltas_pct.values()])
    assert set(deltas) == set(DISTORTION_STAGES)
    assert float(deltas["raw"].max().compute()) == pytest.approx(2.0)
    assert float(deltas_pct["raw"].max().compute()) == pytest.approx(20.0)

    zero_base = {s: _grid([[0.0, 10.0], [10.0, 10.0]]) for s in DISTORTION_STAGES}
    pct = compute_deltas(_stage_grids(5.0), zero_base)[1]["raw"]
    assert np.isnan(pct.isel(lat=0, lon=0))
    assert not np.isinf(pct).any()
    assert float(pct.isel(lat=0, lon=1)) == pytest.approx(-50.0)


def test_calculate_distortion_flags():
    absolute = _grid([[100.0, 1.0], [100.0, 1.0]])
    percent = _grid([[0.1, 50.0], [50.0, 0.1]])
    flag = calculate_distortion_flags(absolute, percent, tolerance_absolute=10.0, tolerance_pct=2.0)
    assert flag.values.tolist() == [[False, False], [True, False]]

    absolute = _grid([[1.0, 0.1], [1.0, 0.1]])
    all_nan = _grid([[np.nan, np.nan], [np.nan, np.nan]])
    flag = calculate_distortion_flags(absolute, all_nan, tolerance_absolute=0.25, tolerance_pct=0)
    assert flag.values.tolist() == [[True, False], [True, False]]


def test_sign_flip_mask_needs_both_stages_past_the_threshold():
    reference = _grid([[10.0, 10.0], [1.0, -10.0]])
    tested = _grid([[-10.0, 10.0], [-1.0, 10.0]])

    flip = sign_flip_mask(tested, reference, threshold=5.0)
    assert flip.values.tolist() == [[True, False], [False, True]]


def test_area_weights_follow_cosine_and_never_go_negative():
    lat_values = [-90.0, -60.0, 0.0, 90.0]
    weights = area_weights(xr.DataArray(lat_values, dims="lat", coords={"lat": lat_values}))

    assert float(weights.sel(lat=0.0)) == pytest.approx(1.0)
    assert float(weights.sel(lat=-60.0)) == pytest.approx(0.5)
    assert (weights >= 0).all()


def test_weighted_fraction():
    polar = _grid([[True, True], [False, False], [True, True]], lat=(-89.0, 0.0, 89.0)).astype(bool)
    assert float(polar.mean()) == pytest.approx(2 / 3)
    assert weighted_fraction(polar, area_weights(polar.lat)) == pytest.approx(0.0343, abs=1e-3)

    mask = _grid([[True, True], [False, False]]).astype(bool)
    weights = area_weights(mask.lat)
    assert weighted_fraction(mask, weights) < 0.6
    assert weighted_fraction(mask, weights, valid=mask) == pytest.approx(1.0)
    assert np.isnan(weighted_fraction(mask, weights, valid=xr.zeros_like(mask, dtype=bool)))


def test_distortion_summary_reports_signed_extremes_and_weighted_fractions():
    absolute = _grid([[2.0, -3.0], [0.0, 0.0]])
    percent = _grid([[20.0, -30.0], [0.0, 0.0]])
    flag = calculate_distortion_flags(absolute, percent, tolerance_absolute=1.0, tolerance_pct=2.0)
    weights = area_weights(absolute.lat)

    summary = distortion_summary(
        absolute,
        percent,
        flag,
        weights=weights,
        sign_flip=sign_flip_mask(absolute, percent, threshold=1.0),
    )
    no_flip = distortion_summary(absolute, percent, flag, weights=weights)

    assert summary["distortion_abs_max"] == pytest.approx(2.0)
    assert summary["distortion_abs_min"] == pytest.approx(-3.0)
    assert summary["distortion_pct_max"] == pytest.approx(20.0)
    assert summary["distortion_pct_min"] == pytest.approx(-30.0)
    assert summary["frac_area_distorted"] == pytest.approx(0.536, abs=1e-3)
    assert "frac_area_sign_flip" in summary
    assert "frac_area_sign_flip" not in no_flip


def test_distortion_fields_pairs_the_documented_stages(subtests):
    deltas = dict(_stage_grids(0.0))
    deltas["raw"] = _grid([[1.0, 1.0], [1.0, 1.0]])
    deltas["coarse_debiased"] = _grid([[4.0, 4.0], [4.0, 4.0]])
    deltas["coarsened_downscaled_debiased"] = _grid([[7.0, 7.0], [7.0, 7.0]])

    expected = {
        "coarse_debiased_vs_raw": 3.0,
        "downscaled_debiased_vs_raw": 6.0,
        "downscaled_debiased_vs_coarse_debiased": 3.0,
    }
    for pair, value in expected.items():
        with subtests.test(pair=pair):
            absolute, _ = distortion_fields(deltas, deltas, pair)
            assert float(absolute.max()) == pytest.approx(value)


def _spread_ds(member_values: dict[str, list[float]], *, start: str = "2015-01-01") -> xr.Dataset:
    """(member, time, lat, lon) dataset whose global mean per step is given."""
    members = list(member_values)
    n_time = len(next(iter(member_values.values())))
    time = pd.date_range(start, periods=n_time, freq="YS")
    data = np.array(
        [[[[v, v], [v, v]] for v in member_values[m]] for m in members], dtype="float64"
    )
    return xr.Dataset(
        {"tas": (["ensemble_member", "time", "lat", "lon"], data)},
        coords={
            "ensemble_member": members,
            "time": time,
            "lat": np.array([-30.0, 30.0]),
            "lon": np.array([0.0, 180.0]),
        },
    )


def _gap_filled_ds() -> xr.Dataset:
    """Gap-filled ssp245 group in miniature: a 3 -> 1 bridge, then distinct native years."""
    ds = _spread_ds(
        {
            "r01": [1.0, 1.0, 10.0, 20.0],
            "r02": [2.0, 2.0, 11.0, 21.0],
            "r03": [1.0, 1.0, 12.0, 22.0],
            "r04": [1.0, 1.0, 13.0, 23.0],
        }
    )
    ds.attrs.update(
        {"gap_fill_period": "2015-2016", "gap_fill_member_map": "r01→rA, r02→rB, r03→rA, r04→rA"}
    )
    return ds


def test_parse_gap_fill_member_map_raises_rather_than_passing_vacuously():
    assert _parse_gap_fill_member_map("r01→rA, r02→rB") == {"r01": "rA", "r02": "rB"}
    assert _parse_gap_fill_member_map("r01->rA, r02->rB") == {"r01": "rA", "r02": "rB"}
    with pytest.raises(ValueError, match="no recognized arrow"):
        _parse_gap_fill_member_map("r01 rA, r02 rB")
    with pytest.raises(ValueError, match="no member pairs"):
        _parse_gap_fill_member_map("   ")


def test_check_ensemble_spread_single_window(subtests):
    all_bridge = _spread_ds({"r01": [1.0, 1.0], "r02": [2.0, 2.0]})
    all_bridge.attrs.update({"gap_fill_period": "2015-2016", "gap_fill_member_map": "r01→rA"})
    cases = {
        "no_gap_fill_attrs": (
            _spread_ds({"r01": [1.0, 2.0], "r02": [3.0, 4.0]}),
            {"window": "full", "ok": True, "day": "2015-01-01"},
        ),
        "real_duplication": (
            _spread_ds({"r01": [1.0], "r02": [1.0]}),
            {"ok": False, "n_distinct": 1, "n_members": 2},
        ),
        "all_bridge_group": (all_bridge, {"window": "full"}),
        "missing_variable": (
            _spread_ds({"r01": [1.0]}).rename({"tas": "pr"}),
            {"ok": True, "members": [], "means": []},
        ),
    }
    for case, (ds, expected) in cases.items():
        with subtests.test(case=case):
            (row,) = check_ensemble_spread(ds, "GCM group")
            assert {k: row[k] for k in expected} == expected


def test_check_ensemble_spread_gap_filled_windows(subtests):
    with subtests.test(case="splits_into_bridge_and_native"):
        bridge, native = check_ensemble_spread(_gap_filled_ds(), "CESM2-WACCM6 ssp245")
        assert (bridge["window"], native["window"]) == ("bridge", "native")
        assert bridge["ok"] is True
        assert bridge["n_distinct"] == 2
        assert bridge["day"] == "2015-01-01"
        assert bridge["groups"] == bridge["expected_groups"] == "r01+r03+r04 | r02"
        assert native["ok"] is True
        assert native["n_distinct"] == 4
        assert native["day"] == "2017-01-01"
        assert native["expected_groups"] is None

    with subtests.test(case="miswired_member_map"):
        ds = _gap_filled_ds()
        ds.attrs["gap_fill_member_map"] = "r01→rA, r02→rB, r03→rA, r04→rB"
        bridge, native = check_ensemble_spread(ds, "CESM2-WACCM6 ssp245")
        assert bridge["ok"] is False
        assert bridge["groups"] == "r01+r03+r04 | r02"
        assert bridge["expected_groups"] == "r01+r03 | r02+r04"
        assert native["ok"] is True

    with subtests.test(case="broken_stitch_caught_in_bridge"):
        ds = _gap_filled_ds()
        ds["tas"].loc[{"ensemble_member": "r04", "time": ds.time.values[:2]}] = 9.0
        bridge, native = check_ensemble_spread(ds, "CESM2-WACCM6 ssp245")
        assert bridge["ok"] is False
        assert bridge["groups"] == "r01+r03 | r02 | r04"
        assert native["ok"] is True

    with subtests.test(case="day_index_offsets_within_each_window"):
        rows = check_ensemble_spread(_gap_filled_ds(), "CESM2-WACCM6 ssp245", day_index=1)
        assert [r["day"] for r in rows] == ["2016-01-01", "2018-01-01"]
        assert all(r["ok"] for r in rows)


def _colors(df: pd.DataFrame, **kwargs) -> dict:
    """{(row label, column): "red" | "amber" | "green" | None} for every cell of a styled table."""
    names = {"#f8d7da": "red", "#fff3cd": "amber", "#d1e7dd": "green"}
    colors = {(row, col): None for row in df.index for col in df.columns}
    for (row, col), props in highlight(df, **kwargs)._compute().ctx.items():
        background = next((value for prop, value in props if prop == "background-color"), None)
        colors[(df.index[row], df.columns[col])] = names.get(background)
    return colors


def _summary() -> pd.DataFrame:
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


def test_highlight_summary_table():
    assert not set(CHECK_RED) & set(CHECK_AMBER)

    colors = _colors(_summary().assign(some_new_metric=[0, 1]))
    assert colors[("clean", "no_interior_nans")] == "green"
    assert colors[("broken", "no_interior_nans")] == "red"
    assert colors[("broken", "no_nans")] == "red"
    assert colors[("broken", "bridged_pre_sai")] == "red"
    assert colors[("clean", "bridged_pre_sai")] == "green"
    assert colors[("broken", "stale_end_tail")] == "green"
    assert colors[("broken", "n_irregular_days")] == "amber"
    assert colors[("clean", "n_irregular_days")] == "green"
    assert colors[("clean", "n_time")] is None
    assert colors[("broken", "some_new_metric")] is None

    demoted = _colors(_summary(), demote=["no_nans"])
    assert demoted[("broken", "no_nans")] == "amber"
    assert demoted[("broken", "no_interior_nans")] == "red"
    with pytest.raises(KeyError, match="no_nan"):
        highlight(_summary(), demote=["no_nan"])

    df = _summary()
    df.index = pd.MultiIndex.from_tuples(
        [("CESM2-WACCM6", "ssp245", "tas", "003"), ("UKESM1-1-LL", "g6_1p5k", "pr", "r2i1p1f2")],
        names=["gcm", "scenario", "variable", "member"],
    )
    assert "#f8d7da" in highlight(df).to_html()


def test_highlight_column_families(subtests):
    cases = {
        "list_columns_flag_only_when_non_empty": (
            pd.DataFrame(
                {"missing": [[], [("dtr", "003")]], "unexpected": [[], []], "pass": [True, False]},
                index=["complete", "short"],
            ),
            {
                ("complete", "missing"): "green",
                ("short", "missing"): "red",
                ("short", "unexpected"): "green",
                ("short", "pass"): "red",
            },
        ),
        "exactly_one_grid_per_family": (
            pd.DataFrame({"n_distinct_grids": [1, 2]}, index=["fine", "coarse"]),
            {("fine", "n_distinct_grids"): "green", ("coarse", "n_distinct_grids"): "red"},
        ),
        "counts_that_should_be_zero": (
            pd.DataFrame(
                {"days_max<min": [0, 7], "neg_cell_days": [0, 12], "high_cell_days": [0, 0]},
                index=["clean", "regressed"],
            ),
            {
                ("clean", "days_max<min"): "green",
                ("regressed", "days_max<min"): "red",
                ("regressed", "neg_cell_days"): "red",
                ("regressed", "high_cell_days"): "green",
            },
        ),
    }
    for case, (df, expected) in cases.items():
        with subtests.test(case=case):
            colors = _colors(df)
            assert {k: colors[k] for k in expected} == expected
