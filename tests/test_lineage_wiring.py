"""Integration tests for ensemble member lineage wiring in pipeline and cache.

Tests verify that:
1. A G6 config with mismatched member labels (numeric G6 vs CMIP6-style historical)
   passes the correct member to each .sel() call.
2. Two G6 members sharing a historical parent produce the same historical_path,
   enabling cache reuse.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.pipeline import BCSDPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_options(tmp_path) -> PipelineOptions:
    return PipelineOptions(
        scratch_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
        rechunk_workflow=False,
    )


@pytest.fixture
def g6_001_tas_config() -> BCSDConfig:
    """G6-1.5K member 001, tas — lineage: historical=r1i1p1f1, SSP245 bridge=001."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="tas",
        ensemble_member="001",
        scenario="G6-1.5K",
        predict_period_start=2035,
        predict_period_end=2084,
    )


@pytest.fixture
def g6_002_tas_config() -> BCSDConfig:
    """G6-1.5K member 002, tas — lineage: historical=r2i1p1f1, SSP245 bridge=002."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="tas",
        ensemble_member="002",
        scenario="G6-1.5K",
        predict_period_start=2035,
        predict_period_end=2084,
    )


@pytest.fixture
def g6_001_tasmax_config() -> BCSDConfig:
    """G6-1.5K member 001, tasmax — lineage: historical=001, SSP245 bridge=009."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="tasmax",
        ensemble_member="001",
        scenario="G6-1.5K",
        predict_period_start=2035,
        predict_period_end=2084,
    )


@pytest.fixture
def g6_002_tasmax_config() -> BCSDConfig:
    """G6-1.5K member 002, tasmax — lineage: historical=001, SSP245 bridge=007."""
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="tasmax",
        ensemble_member="002",
        scenario="G6-1.5K",
        predict_period_start=2035,
        predict_period_end=2084,
    )


def _make_mock_da(name: str = "data") -> MagicMock:
    """Return a MagicMock that behaves enough like a DataArray for pipeline loading."""
    da = MagicMock()
    da.sel.return_value = da
    da.drop_vars.return_value = da
    return da


# ---------------------------------------------------------------------------
# Member selection: _load_gcm_obs uses resolved historical member
# ---------------------------------------------------------------------------


class TestLoadGcmObsMemberSelection:
    def test_historical_member_used_for_hist_sel(self, g6_001_tas_config, pipeline_options):
        """_load_gcm_obs must call .sel(ensemble_member='r1i1p1f1'), not '001'."""
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)
        assert pipeline._hist_member == "r1i1p1f1"

        mock_da = _make_mock_da()

        with (
            patch("srm.pipeline.get_obs", return_value=_make_mock_da()),
            patch("srm.pipeline.get_experiment", return_value=mock_da) as mock_get_exp,
            patch.object(pipeline.cache, "check_dependencies") as mock_deps,
            patch.object(pipeline, "_open_from_icechunk", return_value=MagicMock()),
        ):
            mock_deps.return_value = {"obs_regridded": (True, "/fake/obs")}
            pipeline._load_gcm_obs()

        mock_get_exp.assert_called_once_with(gcm="CESM2-WACCM", scenario="historical", var="tas")
        mock_da.sel.assert_any_call(ensemble_member="r1i1p1f1")

    def test_falls_back_to_ensemble_member_when_no_lineage(self, tmp_path, pipeline_options):
        """For an unknown scenario, _hist_member falls back to ensemble_member."""
        config = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="r1i1p1f1",
            scenario="ssp245",  # lowercase — not in lineage table
            predict_period_start=2015,
            predict_period_end=2100,
        )
        pipeline = BCSDPipeline(config, pipeline_options)
        assert pipeline._hist_member == "r1i1p1f1"

        mock_da = _make_mock_da()

        with (
            patch("srm.pipeline.get_obs", return_value=_make_mock_da()),
            patch("srm.pipeline.get_experiment", return_value=mock_da),
            patch.object(pipeline.cache, "check_dependencies") as mock_deps,
            patch.object(pipeline, "_open_from_icechunk", return_value=MagicMock()),
        ):
            mock_deps.return_value = {"obs_regridded": (True, "/fake/obs")}
            pipeline._load_gcm_obs()

        mock_da.sel.assert_any_call(ensemble_member="r1i1p1f1")


# ---------------------------------------------------------------------------
# Member selection: _load_scenario_data uses correct members
# ---------------------------------------------------------------------------


class TestLoadScenarioDataMemberSelection:
    def _run_load_scenario(self, pipeline, mock_get_experiment):
        with (
            patch("srm.pipeline.get_obs", return_value=_make_mock_da()),
            patch("srm.pipeline.get_experiment", side_effect=mock_get_experiment),
            patch.object(pipeline.cache, "check_dependencies") as mock_deps,
            patch.object(pipeline, "_open_from_icechunk", return_value=MagicMock()),
        ):
            mock_deps.return_value = {
                "obs_regridded": (True, "/fake/obs"),
                "historical": (True, "/fake/hist"),
            }
            pipeline._load_scenario_data()

    def test_historical_sel_uses_historical_ensemble_member(
        self, g6_001_tas_config, pipeline_options
    ):
        """Historical load in _load_scenario_data must use resolved historical member."""
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)

        historical_da = _make_mock_da()
        scenario_da = _make_mock_da()
        ssp245_da = _make_mock_da()

        call_index = {"n": 0}
        returns = [historical_da, scenario_da, ssp245_da]

        def get_experiment_side_effect(**kwargs):
            da = returns[call_index["n"]]
            call_index["n"] += 1
            return da

        self._run_load_scenario(pipeline, get_experiment_side_effect)
        historical_da.sel.assert_any_call(ensemble_member="r1i1p1f1")

    def test_scenario_sel_uses_ensemble_member(self, g6_001_tas_config, pipeline_options):
        """Scenario (G6) load must use ensemble_member (not historical/ssp245 override)."""
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)

        historical_da = _make_mock_da()
        scenario_da = _make_mock_da()
        ssp245_da = _make_mock_da()

        call_index = {"n": 0}
        returns = [historical_da, scenario_da, ssp245_da]

        def get_experiment_side_effect(**kwargs):
            da = returns[call_index["n"]]
            call_index["n"] += 1
            return da

        self._run_load_scenario(pipeline, get_experiment_side_effect)
        scenario_da.sel.assert_any_call(ensemble_member="001")

    def test_ssp245_bridge_sel_uses_ssp245_member(self, g6_001_tas_config, pipeline_options):
        """SSP245 bridge load must use resolved ssp245 member."""
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)

        historical_da = _make_mock_da()
        scenario_da = _make_mock_da()
        ssp245_da = _make_mock_da()

        call_index = {"n": 0}
        returns = [historical_da, scenario_da, ssp245_da]

        def get_experiment_side_effect(**kwargs):
            da = returns[call_index["n"]]
            call_index["n"] += 1
            return da

        self._run_load_scenario(pipeline, get_experiment_side_effect)
        ssp245_da.sel.assert_any_call(ensemble_member="001")

    def test_ssp245_bridge_tasmax_uses_different_ssp245_member(
        self, g6_001_tasmax_config, pipeline_options
    ):
        """tasmax G6-001 uses SSP245 bridge 009, not 001."""
        pipeline = BCSDPipeline(g6_001_tasmax_config, pipeline_options)
        assert pipeline._ssp245_member == "009"

        historical_da = _make_mock_da()
        scenario_da = _make_mock_da()
        ssp245_da = _make_mock_da()

        call_index = {"n": 0}
        returns = [historical_da, scenario_da, ssp245_da]

        def get_experiment_side_effect(**kwargs):
            da = returns[call_index["n"]]
            call_index["n"] += 1
            return da

        self._run_load_scenario(pipeline, get_experiment_side_effect)
        ssp245_da.sel.assert_any_call(ensemble_member="009")


# ---------------------------------------------------------------------------
# Cache path: shared historical parent → same historical_path
# ---------------------------------------------------------------------------


class TestHistoricalPathSharedParent:
    def test_g6_members_sharing_historical_parent_produce_same_path(
        self, g6_001_tasmax_config, g6_002_tasmax_config, pipeline_options
    ):
        """G6 members 001 and 002 (tasmax) both map to historical member '001'.

        They must produce identical historical_path values so the cached artifact
        is reused without recomputing.
        """
        p1 = BCSDPipeline(g6_001_tasmax_config, pipeline_options)
        p2 = BCSDPipeline(g6_002_tasmax_config, pipeline_options)
        assert p1._hist_member == "001"
        assert p2._hist_member == "001"
        path_001 = p1.cache.get_historical_path(g6_001_tasmax_config, hist_member=p1._hist_member)
        path_002 = p2.cache.get_historical_path(g6_002_tasmax_config, hist_member=p2._hist_member)
        assert path_001 == path_002

    def test_g6_members_with_different_historical_parents_produce_different_paths(
        self, g6_001_tas_config, g6_002_tas_config, pipeline_options
    ):
        """G6 tas members 001 and 002 map to r1i1p1f1 and r2i1p1f1 respectively."""
        p1 = BCSDPipeline(g6_001_tas_config, pipeline_options)
        p2 = BCSDPipeline(g6_002_tas_config, pipeline_options)
        path_001 = p1.cache.get_historical_path(g6_001_tas_config, hist_member=p1._hist_member)
        path_002 = p2.cache.get_historical_path(g6_002_tas_config, hist_member=p2._hist_member)
        assert path_001 != path_002

    def test_historical_path_contains_resolved_hist_member(
        self, g6_001_tas_config, pipeline_options
    ):
        """historical_path must embed the resolved historical member label."""
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)
        assert pipeline._hist_member == "r1i1p1f1"
        path = pipeline.cache.get_historical_path(
            g6_001_tas_config, hist_member=pipeline._hist_member
        )
        assert "/r1i1p1f1/" in path
        assert "/001/" not in path

    def test_historical_path_fallback_uses_ensemble_member(self, pipeline_options):
        """When no lineage is registered, ensemble_member fills the path."""
        config = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="r1i1p1f1",
            scenario="ssp245",
            predict_period_start=2015,
            predict_period_end=2100,
        )
        pipeline = BCSDPipeline(config, pipeline_options)
        assert pipeline._hist_member == "r1i1p1f1"
        path = pipeline.cache.get_historical_path(config, hist_member=pipeline._hist_member)
        assert "/r1i1p1f1/" in path


# ---------------------------------------------------------------------------
# Output attrs: resolved members written to metadata
# ---------------------------------------------------------------------------


class TestBuildOutputAttrs:
    def test_historical_ensemble_member_in_attrs(self, g6_001_tas_config, pipeline_options):
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)
        attrs = pipeline._build_output_attrs(source_dataset=None)
        assert attrs["historical_ensemble_member"] == "r1i1p1f1"

    def test_ssp245_ensemble_member_in_attrs(self, g6_001_tas_config, pipeline_options):
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)
        attrs = pipeline._build_output_attrs(source_dataset=None)
        assert attrs["ssp245_ensemble_member"] == "001"

    def test_attrs_fall_back_to_ensemble_member_when_no_lineage(self, pipeline_options):
        config = BCSDConfig(
            gcm="CESM2-WACCM",
            variable="tas",
            ensemble_member="r1i1p1f1",
            scenario="ssp245",  # lowercase — not in lineage table
            predict_period_start=2015,
            predict_period_end=2100,
        )
        pipeline = BCSDPipeline(config, pipeline_options)
        attrs = pipeline._build_output_attrs(source_dataset=None)
        assert attrs["historical_ensemble_member"] == "r1i1p1f1"
        assert attrs["ssp245_ensemble_member"] == "r1i1p1f1"

    def test_tasmax_g6_002_attrs(self, g6_002_tasmax_config, pipeline_options):
        """tasmax G6-002: historical=001, ssp245=007."""
        pipeline = BCSDPipeline(g6_002_tasmax_config, pipeline_options)
        attrs = pipeline._build_output_attrs(source_dataset=None)
        assert attrs["historical_ensemble_member"] == "001"
        assert attrs["ssp245_ensemble_member"] == "007"
