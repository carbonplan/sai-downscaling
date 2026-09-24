"""Tests for the store-layout resolution in saidownscale.plotting."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from saidownscale.plotting import _debiased_coarse_node, point_series

_LAYOUTS = {"pre_namespace": "", "bcsd": "/bcsd", "qdmsd": "/qdmsd"}


def _coarse_leaf(value: float = 1.0) -> xr.Dataset:
    time = xr.date_range("2015-01-01", periods=3, freq="D", calendar="proleptic_gregorian")
    return xr.Dataset(
        {"tas": (["time", "lat", "lon"], np.full((3, 2, 2), value))},
        coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )


def _tree(paths: dict[str, float]) -> xr.DataTree:
    return xr.DataTree.from_dict({path: _coarse_leaf(value) for path, value in paths.items()})


def _max_tas(node) -> float:
    return float(node["tas/001"].dataset["tas"].max())


def test_debiased_coarse_node_resolves_either_layout(subtests):
    for layout, prefix in _LAYOUTS.items():
        with subtests.test(layout=layout):
            tree = _tree({f"{prefix}/debiased_coarse/ssp245/tas/001": 7.0})
            assert _max_tas(_debiased_coarse_node(tree, "ssp245")) == 7.0

    pre_namespace = _tree({"/debiased_coarse/ssp245/tas/001": 7.0})
    namespaced = _tree({"/bcsd/ssp245/tas/001": 1.0, "/bcsd/debiased_coarse/ssp245/tas/001": 7.0})
    for method in ["BCSD", "QDMSD", "bcsd", "qdmsd"]:
        with subtests.test(method=method):
            node = _debiased_coarse_node(pre_namespace, "ssp245", method=method)
            assert node.path == "/debiased_coarse/ssp245"
            assert _max_tas(node) == 7.0
    for method in [None, "bcsd", "BCSD"]:
        with subtests.test(namespaced_method=method):
            node = _debiased_coarse_node(namespaced, "ssp245", method=method)
            assert node.path == "/bcsd/debiased_coarse/ssp245"


def test_debiased_coarse_node_rejects_missing_or_ambiguous_groups():
    one = _tree({"/bcsd/debiased_coarse/ssp245/tas/001": 7.0})
    with pytest.raises(KeyError, match="no debiased_coarse/g6_1p5k group"):
        _debiased_coarse_node(one, "g6_1p5k")
    with pytest.raises(KeyError, match="qdmsd") as excinfo:
        _debiased_coarse_node(one, "ssp245", method="qdmsd")
    assert "bcsd" in str(excinfo.value)

    two = _tree(
        {
            "/bcsd/debiased_coarse/ssp245/tas/001": 7.0,
            "/qdmsd/debiased_coarse/ssp245/tas/001": 9.0,
        }
    )
    with pytest.raises(KeyError, match="pass method="):
        _debiased_coarse_node(two, "ssp245")
    assert _max_tas(_debiased_coarse_node(two, "ssp245", method="QDMSD")) == 9.0


def _point_da(value: float) -> xr.DataArray:
    return xr.DataArray(
        np.full((2, 2), value),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        name="tas",
    )


def test_point_series_reads_coarse_series_from_either_layout(subtests):
    key = ("ssp245", "tas", "001")
    raw = xr.Dataset({"tas": _point_da(3.0).expand_dims(ensemble_member=["001"])})
    for layout in ("pre_namespace", "bcsd"):
        with subtests.test(layout=layout):
            tree = _tree({f"{_LAYOUTS[layout]}/debiased_coarse/ssp245/tas/001": 7.0})
            series = point_series(
                key,
                lat=0.0,
                lon=0.0,
                low_bound=_point_da(0.0),
                high_bound=_point_da(10.0),
                tree=tree,
                leaves={key: _point_da(5.0)},
                obs_fine=lambda var: _point_da(4.0),
                gcm={"ssp245": raw, "historical": raw},
                historical_member=lambda *_: "001",
            )
            assert float(series["coarse_debiased"].max()) == 7.0
            assert float(series["downscaled"]) == 5.0
