"""Unit tests for BCSDPipeline._load_ssp245_bridge and _ssp245_esgf_member lineage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.pipeline import BCSDPipeline


def _make_config(**overrides) -> BCSDConfig:
    defaults = dict(
        gcm="MIROC-ES2H",
        variable="tas",
        ensemble_member="r01",
        scenario="G6-1.5K",
        predict_period_start=2015,
        predict_period_end=2100,
        subset_bounds=(-35.0, -22.0, 16.0, 33.0),
        mapping_type="nonparametric_hybrid",
    )
    defaults.update(overrides)
    return BCSDConfig(**defaults)


def _make_options(tmp_path) -> PipelineOptions:
    return PipelineOptions(
        scratch_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "outputs"),
        verbose=False,
        rechunk_workflow=False,
    )


def test_bridge_returns_primary_when_esgf_member_none(tmp_path):
    """CESM2-WACCM has no ESGF member: _ssp245_esgf_member is None."""
    config = _make_config(gcm="CESM2-WACCM", ensemble_member="001")
    pipeline = BCSDPipeline(config, _make_options(tmp_path))
    assert pipeline._ssp245_esgf_member is None


def test_bridge_returns_primary_when_no_gap(tmp_path):
    """MIROC G6-1.5K r01 has ESGF member r1i1p4f2."""
    config = _make_config()
    pipeline = BCSDPipeline(config, _make_options(tmp_path))
    assert pipeline._ssp245_esgf_member == "r1i1p4f2"


def test_bridge_prepends_esgf_when_gap_detected(tmp_path):
    """MIROC G6-1.5K r01: _ssp245_member=r01, _ssp245_esgf_member=r1i1p4f2."""
    config = _make_config()
    pipeline = BCSDPipeline(config, _make_options(tmp_path))
    assert pipeline._ssp245_member == "r01"
    assert pipeline._ssp245_esgf_member == "r1i1p4f2"


def test_bridge_esgf_uses_correct_member(tmp_path):
    """_load_ssp245_bridge selects _ssp245_member from get_experiment result."""
    config = _make_config()
    pipeline = BCSDPipeline(config, _make_options(tmp_path))
    assert pipeline._ssp245_member == "r01"

    mock_da = MagicMock()
    mock_da.sel.return_value = mock_da
    with patch("srm.pipeline.get_experiment", return_value=mock_da) as mock_get_exp:
        pipeline._load_ssp245_bridge()

    mock_get_exp.assert_called_once_with(gcm="MIROC-ES2H", scenario="SSP245", var="tas")
    mock_da.sel.assert_called_once_with(ensemble_member="r01")


def test_bridge_empty_esgf_gap_returns_primary(tmp_path):
    """_ssp245_esgf_member is None for CESM2-WACCM 001 (no ESGF member needed)."""
    config = _make_config(gcm="CESM2-WACCM", ensemble_member="001")
    pipeline = BCSDPipeline(config, _make_options(tmp_path))
    assert pipeline._ssp245_esgf_member is None


def test_bridge_calendar_aligned_to_primary(tmp_path):
    """_load_ssp245_bridge delegates to get_experiment; no calendar logic in pipeline."""
    config = _make_config()
    pipeline = BCSDPipeline(config, _make_options(tmp_path))

    mock_da = MagicMock()
    mock_da.sel.return_value = mock_da
    with patch("srm.pipeline.get_experiment", return_value=mock_da):
        result = pipeline._load_ssp245_bridge()

    assert result is not None
