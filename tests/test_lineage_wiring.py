"""Tests that resolved lineage members reach pipeline loads, cache paths, and output attrs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions
from saidownscale.pipeline import DownscalingPipeline


@pytest.fixture
def pipeline_options(tmp_path) -> PipelineOptions:
    return PipelineOptions(
        scratch_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
        rechunk_workflow=False,
    )


def _config(variable="tas", member="001", scenario="G6-1.5K") -> DownscalingConfig:
    sai = scenario == "G6-1.5K"
    return DownscalingConfig(
        gcm="CESM2-WACCM6",
        downscaling_method="BCSD",
        variable=variable,
        ensemble_member=member,
        scenario=scenario,
        predict_period_start=2035 if sai else 2015,
        predict_period_end=2084 if sai else 2100,
    )


def _make_mock_da() -> MagicMock:
    da = MagicMock()
    da.sel.return_value = da
    da.drop_vars.return_value = da
    return da


def _load(pipeline, method, scenario_da=None):
    """Run a pipeline loader with all I/O patched; return the (hist, bridge) mocks."""
    deps = {"obs_regridded": (True, "/fake/obs"), "historical": (True, "/fake/hist")}
    with (
        patch("saidownscale.pipeline.get_obs", return_value=_make_mock_da()),
        patch(
            "saidownscale.pipeline.get_historical_experiment", return_value=_make_mock_da()
        ) as mock_hist,
        patch("saidownscale.pipeline.get_experiment", return_value=scenario_da or _make_mock_da()),
        patch.object(pipeline, "_load_ssp245_bridge", return_value=_make_mock_da()) as mock_bridge,
        patch.object(pipeline.cache, "check_dependencies", return_value=deps),
        patch.object(pipeline, "_open_from_icechunk", return_value=MagicMock()),
    ):
        getattr(pipeline, method)()
    return mock_hist, mock_bridge


def test_loaders_select_the_resolved_historical_member(subtests, pipeline_options):
    configs = {
        "g6_lineage": _config(),
        "no_lineage_fallback": _config(member="r1i1p1f1", scenario="ssp245"),
    }
    for name, config in configs.items():
        for method in ("_load_gcm_obs", "_load_scenario_data"):
            with subtests.test(config=name, method=method):
                pipeline = DownscalingPipeline(config, pipeline_options)
                assert pipeline._hist_member == "r1i1p1f1"
                mock_hist, _ = _load(pipeline, method)
                mock_hist.assert_called_once_with(gcm="CESM2-WACCM6", member="r1i1p1f1", var="tas")


def test_scenario_load_uses_ensemble_member_and_loads_the_bridge(pipeline_options):
    pipeline = DownscalingPipeline(_config(), pipeline_options)
    scenario_da = _make_mock_da()
    _, mock_bridge = _load(pipeline, "_load_scenario_data", scenario_da)
    scenario_da.sel.assert_any_call(ensemble_member="001")
    mock_bridge.assert_called_once()


def test_historical_loc_follows_the_resolved_parent(pipeline_options):
    def loc(config):
        p = DownscalingPipeline(config, pipeline_options)
        return p.cache.historical_loc(p._hist_member)

    assert loc(_config("tasmax", "001")) == loc(_config("tasmax", "002"))
    assert loc(_config("tas", "001")) != loc(_config("tas", "002"))
    g6_group = loc(_config()).group
    assert "r1i1p1f1" in g6_group and "001" not in g6_group
    assert "r1i1p1f1" in loc(_config(member="r1i1p1f1", scenario="SSP245")).group


def test_output_attrs_carry_resolved_members(subtests, pipeline_options):
    cases = [
        (_config(), "r1i1p1f1", "001"),
        (_config("tasmax", "001"), "001", "009"),
        (_config("tasmax", "002"), "001", "007"),
        (_config(member="r1i1p1f1", scenario="SSP245"), "r1i1p1f1", "r1i1p1f1"),
    ]
    for config, hist, ssp245 in cases:
        with subtests.test(variable=config.variable, member=config.ensemble_member):
            attrs = DownscalingPipeline(config, pipeline_options)._build_output_attrs()
            assert attrs["sai_downscaling:historical_ensemble_member"] == hist
            assert attrs["sai_downscaling:ssp245_ensemble_member"] == ssp245
