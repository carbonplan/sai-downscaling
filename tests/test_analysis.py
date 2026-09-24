"""Tests for analysis.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from saidownscale.analysis import DownscalingRun, load_cached_data
from saidownscale.cache import ArtifactCache
from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions

matplotlib.use("Agg")


@pytest.fixture
def config() -> DownscalingConfig:
    return DownscalingConfig(
        gcm="CESM2-WACCM6",
        downscaling_method="BCSD",
        variable="tas",
        ensemble_member="r1i1p1f1",
        scenario="ssp245",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def run(config) -> DownscalingRun:
    return DownscalingRun(config)


def _fake_ds(variable: str = "tas") -> xr.Dataset:
    rng = np.random.default_rng(0)
    lat = np.array([-30.0, 0.0, 30.0])
    lon = np.array([10.0, 20.0, 30.0])
    data = rng.random((3, 3)).astype("float32")
    return xr.Dataset(
        {variable: xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})}
    )


def test_load_cached_data_opens_s3_repo_read_only_with_zarr():
    mock_ds = MagicMock(spec=xr.Dataset)
    with (
        patch("saidownscale.analysis.icechunk.s3_storage") as mock_storage,
        patch("saidownscale.analysis.icechunk.Repository.open") as mock_open,
        patch("saidownscale.analysis.xr.open_dataset", return_value=mock_ds) as mock_open_ds,
    ):
        result = load_cached_data("s3://my-bucket/some/path/to/repo")

    mock_storage.assert_called_once_with(
        bucket="my-bucket", prefix="some/path/to/repo", from_env=True
    )
    mock_open.return_value.readonly_session.assert_called_once_with(branch="main")
    assert result is mock_ds
    _, kwargs = mock_open_ds.call_args
    assert kwargs.get("engine") == "zarr"
    assert kwargs.get("chunks") == {}


def test_run_repr_names_identity_and_binds_one_cache(run, config):
    r = repr(run)
    for field in (config.gcm, config.ensemble_member, config.variable, config.scenario):
        assert field in r
    assert isinstance(run._cache, ArtifactCache)
    assert run._cache is run._cache
    assert run._cache.config is config
    assert run._cache.scratch_dir == PipelineOptions().scratch_dir.rstrip("/")
    assert run._location_cache == {}


def test_data_properties_load_their_own_store_once(run, subtests):
    expected = {
        "obs": run._cache.obs_loc.store_path,
        "historical": run._cache.historical_loc(run._hist_member).store_path,
        "scenario": run._cache.scenario_loc.store_path,
    }
    for prop, path in expected.items():
        with subtests.test(prop=prop):
            fake_ds = _fake_ds()
            with patch("saidownscale.analysis.load_cached_data", return_value=fake_ds) as load:
                assert getattr(run, prop) is fake_ds
                assert getattr(run, prop) is fake_ds
            load.assert_called_once_with(path, branch=run._cache.branch)


def test_get_location_data_returns_arrays_cached_per_location(run):
    fake = _fake_ds("tas")
    for prop in ("obs", "historical", "scenario"):
        run.__dict__[prop] = fake

    first = run.get_location_data(0.0, 10.0)
    assert set(first) == {"obs", "historical", "scenario"}
    assert all(isinstance(arr, np.ndarray) for arr in first.values())
    assert run.get_location_data(0.0, 10.0) is first
    other = run.get_location_data(30.0, 20.0)
    assert other is not first
    assert {(0.0, 10.0), (30.0, 20.0)} <= set(run._location_cache)


def test_plot_location_cdf(run, config):
    rng = np.random.default_rng(1)
    fake_data = {k: rng.random(50).astype("float32") for k in ("obs", "historical", "scenario")}
    run.get_location_data = MagicMock(return_value=fake_data)

    fig, axes = run.plot_location_cdf(-33.5, 18.4)
    try:
        run.get_location_data.assert_called_once_with(-33.5, 18.4)
        assert hasattr(fig, "savefig")
        assert len(axes) == 2
        title = fig.texts[0].get_text()
        assert config.variable in title and config.scenario in title
    finally:
        plt.close(fig)
