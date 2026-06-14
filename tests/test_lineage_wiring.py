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


@pytest.fixture
def miroc_g6_r01_tas_config() -> BCSDConfig:
    """MIROC G6-1.5K member r01, tas — lineage: hist=r1i1p4f2, ssp245=r01, esgf=r1i1p4f2."""
    return BCSDConfig(
        gcm="MIROC-ES2H",
        variable="tas",
        ensemble_member="r01",
        scenario="G6-1.5K",
        predict_period_start=2035,
        predict_period_end=2084,
    )


@pytest.fixture
def miroc_g6_r04_tas_config() -> BCSDConfig:
    """MIROC G6-1.5K member r04, tas — shares hist=r1i1p4f2 with r01."""
    return BCSDConfig(
        gcm="MIROC-ES2H",
        variable="tas",
        ensemble_member="r04",
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
        """_load_gcm_obs must select resolved hist member 'r1i1p1f1', not ensemble_member '001'."""
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)
        assert pipeline._hist_member == "r1i1p1f1"

        mock_hist_da = _make_mock_da()
        with (
            patch("srm.pipeline.get_obs", return_value=_make_mock_da()),
            patch("srm.pipeline.get_experiment", return_value=mock_hist_da) as mock_get_exp,
            patch.object(pipeline.cache, "check_dependencies") as mock_deps,
            patch.object(pipeline, "_open_from_icechunk", return_value=MagicMock()),
        ):
            mock_deps.return_value = {"obs_regridded": (True, "/fake/obs")}
            pipeline._load_gcm_obs()

        mock_get_exp.assert_called_once_with(gcm="CESM2-WACCM", scenario="historical", var="tas")
        mock_hist_da.sel.assert_any_call(ensemble_member="r1i1p1f1")

    def test_falls_back_to_ensemble_member_when_no_lineage(self, tmp_path, pipeline_options):
        """For unknown scenario, _hist_member falls back to ensemble_member."""
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

        mock_hist_da = _make_mock_da()
        with (
            patch("srm.pipeline.get_obs", return_value=_make_mock_da()),
            patch("srm.pipeline.get_experiment", return_value=mock_hist_da) as mock_get_exp,
            patch.object(pipeline.cache, "check_dependencies") as mock_deps,
            patch.object(pipeline, "_open_from_icechunk", return_value=MagicMock()),
        ):
            mock_deps.return_value = {"obs_regridded": (True, "/fake/obs")}
            pipeline._load_gcm_obs()

        mock_get_exp.assert_called_once_with(gcm="CESM2-WACCM", scenario="historical", var="tas")
        mock_hist_da.sel.assert_any_call(ensemble_member="r1i1p1f1")


# ---------------------------------------------------------------------------
# Member selection: _load_scenario_data uses correct members
# ---------------------------------------------------------------------------


class TestLoadScenarioDataMemberSelection:
    def _run_load_scenario(self, pipeline, mock_get_experiment):
        with (
            patch("srm.pipeline.get_obs", return_value=_make_mock_da()),
            patch("srm.pipeline.get_experiment", side_effect=mock_get_experiment),
            patch.object(pipeline, "_load_ssp245_bridge", return_value=_make_mock_da()),
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
        """_load_scenario_data must select resolved hist member 'r1i1p1f1', not '001'."""
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)

        mock_hist_da = _make_mock_da()

        def get_exp_side_effect(**kw):
            return mock_hist_da if kw.get("scenario") == "historical" else _make_mock_da()

        with (
            patch("srm.pipeline.get_obs", return_value=_make_mock_da()),
            patch("srm.pipeline.get_experiment", side_effect=get_exp_side_effect),
            patch.object(pipeline, "_load_ssp245_bridge", return_value=_make_mock_da()),
            patch.object(pipeline.cache, "check_dependencies") as mock_deps,
            patch.object(pipeline, "_open_from_icechunk", return_value=MagicMock()),
        ):
            mock_deps.return_value = {
                "obs_regridded": (True, "/fake/obs"),
                "historical": (True, "/fake/hist"),
            }
            pipeline._load_scenario_data()

        mock_hist_da.sel.assert_any_call(ensemble_member="r1i1p1f1")

    def test_falls_back_to_ensemble_member_when_no_lineage(self, pipeline_options):
        """For unknown scenario, _hist_member falls back to ensemble_member."""
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

        mock_hist_da = _make_mock_da()

        def get_exp_side_effect(**kw):
            return mock_hist_da if kw.get("scenario") == "historical" else _make_mock_da()

        with (
            patch("srm.pipeline.get_obs", return_value=_make_mock_da()),
            patch("srm.pipeline.get_experiment", side_effect=get_exp_side_effect),
            patch.object(pipeline.cache, "check_dependencies") as mock_deps,
            patch.object(pipeline, "_open_from_icechunk", return_value=MagicMock()),
        ):
            mock_deps.return_value = {"obs_regridded": (True, "/fake/obs")}
            pipeline._load_scenario_data()

        mock_hist_da.sel.assert_any_call(ensemble_member="r1i1p1f1")

    def test_scenario_sel_uses_ensemble_member(self, g6_001_tas_config, pipeline_options):
        """G6 scenario data load must use ensemble_member (not hist override)."""
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)
        scenario_da = _make_mock_da()

        self._run_load_scenario(pipeline, lambda **kw: scenario_da)
        scenario_da.sel.assert_any_call(ensemble_member="001")

    def test_ssp245_bridge_loaded_via_load_ssp245_bridge(self, g6_001_tas_config, pipeline_options):
        """_load_scenario_data must call _load_ssp245_bridge for SAI scenarios."""
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)

        with (
            patch("srm.pipeline.get_obs", return_value=_make_mock_da()),
            patch("srm.pipeline.get_experiment", return_value=_make_mock_da()),
            patch.object(
                pipeline, "_load_ssp245_bridge", return_value=_make_mock_da()
            ) as mock_bridge,
            patch.object(pipeline.cache, "check_dependencies") as mock_deps,
            patch.object(pipeline, "_open_from_icechunk", return_value=MagicMock()),
        ):
            mock_deps.return_value = {
                "obs_regridded": (True, "/fake/obs"),
                "historical": (True, "/fake/hist"),
            }
            pipeline._load_scenario_data()

        mock_bridge.assert_called_once()

    def test_ssp245_bridge_tasmax_member_resolved(self, g6_001_tasmax_config, pipeline_options):
        """tasmax G6-001 resolves ssp245_member=009 (not 001)."""
        pipeline = BCSDPipeline(g6_001_tasmax_config, pipeline_options)
        assert pipeline._ssp245_member == "009"


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
        attrs = pipeline._build_output_attrs()
        assert attrs["historical_ensemble_member"] == "r1i1p1f1"

    def test_ssp245_ensemble_member_in_attrs(self, g6_001_tas_config, pipeline_options):
        pipeline = BCSDPipeline(g6_001_tas_config, pipeline_options)
        attrs = pipeline._build_output_attrs()
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
        attrs = pipeline._build_output_attrs()
        assert attrs["historical_ensemble_member"] == "r1i1p1f1"
        assert attrs["ssp245_ensemble_member"] == "r1i1p1f1"

    def test_tasmax_g6_002_attrs(self, g6_002_tasmax_config, pipeline_options):
        """tasmax G6-002: historical=001, ssp245=007."""
        pipeline = BCSDPipeline(g6_002_tasmax_config, pipeline_options)
        attrs = pipeline._build_output_attrs()
        assert attrs["historical_ensemble_member"] == "001"
        assert attrs["ssp245_ensemble_member"] == "007"


# ---------------------------------------------------------------------------
# MIROC-ES2H G6-1.5K lineage wiring
# ---------------------------------------------------------------------------


class TestMirocG6Wiring:
    """MIROC-ES2H G6-1.5K lineage wiring through BCSDPipeline."""

    def test_hist_member_resolved(self, miroc_g6_r01_tas_config, pipeline_options):
        pipeline = BCSDPipeline(miroc_g6_r01_tas_config, pipeline_options)
        assert pipeline._hist_member == "r1i1p4f2"

    def test_ssp245_member_is_geomip_member(self, miroc_g6_r01_tas_config, pipeline_options):
        pipeline = BCSDPipeline(miroc_g6_r01_tas_config, pipeline_options)
        assert pipeline._ssp245_member == "r01"

    def test_ssp245_esgf_member_set(self, miroc_g6_r01_tas_config, pipeline_options):
        pipeline = BCSDPipeline(miroc_g6_r01_tas_config, pipeline_options)
        assert pipeline._ssp245_esgf_member == "r1i1p4f2"

    def test_r04_shares_hist_member_with_r01(
        self, miroc_g6_r01_tas_config, miroc_g6_r04_tas_config, pipeline_options
    ):
        """r01 and r04 share historical parent r1i1p4f2 — same cache path."""
        p1 = BCSDPipeline(miroc_g6_r01_tas_config, pipeline_options)
        p4 = BCSDPipeline(miroc_g6_r04_tas_config, pipeline_options)
        assert p1._hist_member == p4._hist_member == "r1i1p4f2"
        path_r01 = p1.cache.get_historical_path(
            miroc_g6_r01_tas_config, hist_member=p1._hist_member
        )
        path_r04 = p4.cache.get_historical_path(
            miroc_g6_r04_tas_config, hist_member=p4._hist_member
        )
        assert path_r01 == path_r04
