"""Tests for saidownscale.plotting helpers that read the output store.

Only the store-layout resolution is covered here: the plotting functions themselves
draw figures and are exercised in the QA notebooks, not in the test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from saidownscale.plotting import _debiased_coarse_node, point_series


def _coarse_leaf(value: float = 1.0) -> xr.Dataset:
    """A minimal debiased_coarse member leaf."""
    time = xr.date_range("2015-01-01", periods=3, freq="D", calendar="proleptic_gregorian")
    return xr.Dataset(
        {"tas": (["time", "lat", "lon"], np.full((3, 2, 2), value))},
        coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )


def _tree(paths: dict[str, float]) -> xr.DataTree:
    return xr.DataTree.from_dict({path: _coarse_leaf(value) for path, value in paths.items()})


class TestDebiasedCoarseNode:
    """The coarse drill-down series must resolve under either store layout.

    Method-dependent groups are namespaced under a leading downscaling-method segment,
    so ``debiased_coarse/{scenario}`` moved to ``{method}/debiased_coarse/{scenario}``.
    Published stores on older branches still carry the unprefixed form.
    """

    @pytest.mark.parametrize("method", [None, "bcsd", "qdmsd"])
    def test_resolves_both_layouts(self, method):
        prefix = f"/{method}" if method else ""
        tree = _tree({f"{prefix}/debiased_coarse/ssp245/tas/001": 7.0})
        node = _debiased_coarse_node(tree, "ssp245")
        assert float(node["tas/001"].dataset["tas"].max()) == 7.0

    def test_scenario_groups_beside_debiased_coarse_are_ignored(self):
        tree = _tree(
            {
                "/bcsd/ssp245/tas/001": 1.0,  # the fine deliverable, not the coarse artifact
                "/bcsd/debiased_coarse/ssp245/tas/001": 7.0,
            }
        )
        node = _debiased_coarse_node(tree, "ssp245")
        assert node.path == "/bcsd/debiased_coarse/ssp245"

    def test_missing_scenario_raises(self):
        tree = _tree({"/bcsd/debiased_coarse/ssp245/tas/001": 7.0})
        with pytest.raises(KeyError, match="no debiased_coarse/g6_1p5k group"):
            _debiased_coarse_node(tree, "g6_1p5k")

    def test_two_methods_are_ambiguous_until_disambiguated(self):
        tree = _tree(
            {
                "/bcsd/debiased_coarse/ssp245/tas/001": 7.0,
                "/qdmsd/debiased_coarse/ssp245/tas/001": 9.0,
            }
        )
        with pytest.raises(KeyError, match="pass method="):
            _debiased_coarse_node(tree, "ssp245")
        node = _debiased_coarse_node(tree, "ssp245", method="QDMSD")
        assert float(node["tas/001"].dataset["tas"].max()) == 9.0

    @pytest.mark.parametrize("method", ["BCSD", "QDMSD", "bcsd", "qdmsd"])
    def test_method_against_pre_namespace_store_resolves_to_root(self, method):
        """A pre-namespace store predates the split, so any method request is satisfied."""
        tree = _tree({"/debiased_coarse/ssp245/tas/001": 7.0})
        node = _debiased_coarse_node(tree, "ssp245", method=method)
        assert node.path == "/debiased_coarse/ssp245"
        assert float(node["tas/001"].dataset["tas"].max()) == 7.0

    def test_method_absent_from_namespaced_store_raises_naming_present_segments(self):
        """A store that does have method segments, but not the requested one, is a genuine miss."""
        tree = _tree({"/bcsd/debiased_coarse/ssp245/tas/001": 7.0})
        with pytest.raises(KeyError) as excinfo:
            _debiased_coarse_node(tree, "ssp245", method="qdmsd")
        message = str(excinfo.value)
        assert "qdmsd" in message
        assert "bcsd" in message

    def test_method_lookup_is_case_insensitive(self):
        tree = _tree({"/bcsd/debiased_coarse/ssp245/tas/001": 7.0})
        lower = _debiased_coarse_node(tree, "ssp245", method="bcsd")
        upper = _debiased_coarse_node(tree, "ssp245", method="BCSD")
        assert lower.path == upper.path == "/bcsd/debiased_coarse/ssp245"


def _point_da(value: float) -> xr.DataArray:
    """A lat/lon field the drill-down can select a single point from."""
    return xr.DataArray(
        np.full((2, 2), value),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        name="tas",
    )


class TestPointSeriesStoreLayout:
    """point_series reads the coarse debiased series straight out of the output tree."""

    @pytest.mark.parametrize("method", [None, "bcsd"])
    def test_reads_coarse_series_from_either_layout(self, method):
        key = ("ssp245", "tas", "001")
        prefix = f"/{method}" if method else ""
        tree = _tree({f"{prefix}/debiased_coarse/ssp245/tas/001": 7.0})
        raw = xr.Dataset(
            {"tas": _point_da(3.0).expand_dims(ensemble_member=["001"])},
        )
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
