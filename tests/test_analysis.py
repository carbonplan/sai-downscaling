"""Tests for analysis.py: load_cached_data, BCSDRun construction, caching, and plotting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from srm.analysis import BCSDRun, load_cached_data
from srm.bcsd_config import BCSDConfig
from srm.cache import ArtifactCache

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> BCSDConfig:
    return BCSDConfig(
        gcm="CESM2-WACCM",
        variable="tas",
        ensemble_member="r1i1p1f1",
        scenario="ssp245",
        predict_period_start=2015,
        predict_period_end=2100,
    )


@pytest.fixture
def run(config) -> BCSDRun:
    return BCSDRun(config)


def _fake_ds(variable: str = "tas") -> xr.Dataset:
    """Minimal xarray Dataset with a single variable over lat/lon."""
    rng = np.random.default_rng(0)
    lat = np.array([-30.0, 0.0, 30.0])
    lon = np.array([10.0, 20.0, 30.0])
    data = rng.random((3, 3)).astype("float32")
    return xr.Dataset(
        {variable: xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})}
    )


# ---------------------------------------------------------------------------
# load_cached_data
# ---------------------------------------------------------------------------


class TestLoadCachedData:
    def test_opens_s3_storage_with_correct_bucket_and_prefix(self):
        uri = "s3://my-bucket/some/path/to/repo"
        mock_ds = MagicMock(spec=xr.Dataset)

        with (
            patch("srm.analysis.icechunk.s3_storage") as mock_storage,
            patch("srm.analysis.icechunk.Repository.open") as mock_open,
            patch("srm.analysis.xr.open_dataset", return_value=mock_ds),
        ):
            mock_repo = MagicMock()
            mock_open.return_value = mock_repo
            mock_session = MagicMock()
            mock_repo.readonly_session.return_value = mock_session

            result = load_cached_data(uri)

        mock_storage.assert_called_once_with(
            bucket="my-bucket", prefix="some/path/to/repo", from_env=True
        )
        mock_repo.readonly_session.assert_called_once_with(branch="main")
        assert result is mock_ds

    def test_opens_dataset_with_zarr_engine(self):
        uri = "s3://bucket/prefix"

        with (
            patch("srm.analysis.icechunk.s3_storage"),
            patch("srm.analysis.icechunk.Repository.open") as mock_open,
            patch("srm.analysis.xr.open_dataset") as mock_open_ds,
        ):
            mock_open.return_value.readonly_session.return_value = MagicMock()
            load_cached_data(uri)

        _, kwargs = mock_open_ds.call_args
        assert kwargs.get("engine") == "zarr"
        assert kwargs.get("chunks") == {}


# ---------------------------------------------------------------------------
# BCSDRun construction
# ---------------------------------------------------------------------------


class TestBCSDRunConstruction:
    def test_repr_contains_key_fields(self, run, config):
        r = repr(run)
        assert config.gcm in r
        assert config.ensemble_member in r
        assert config.variable in r
        assert config.scenario in r

    def test_cache_is_artifact_cache(self, run):
        assert isinstance(run._cache, ArtifactCache)

    def test_cache_is_bound_to_config(self, run, config):
        assert run._cache.config is config

    def test_cache_uses_config_cache_dir(self, run, config):
        assert run._cache.base_path == config.cache_dir.rstrip("/")

    def test_cache_is_cached_property(self, run):
        """Accessing _cache twice returns the same object."""
        cache1 = run._cache
        cache2 = run._cache
        assert cache1 is cache2

    def test_location_cache_starts_empty(self, run):
        assert run._location_cache == {}


# ---------------------------------------------------------------------------
# BCSDRun data properties (obs / historical / scenario)
# ---------------------------------------------------------------------------


class TestBCSDRunDataProperties:
    def _patch_load(self, fake_ds):
        return patch("srm.analysis.load_cached_data", return_value=fake_ds)

    def test_obs_calls_load_with_obs_path(self, run):
        fake_ds = _fake_ds()
        with self._patch_load(fake_ds) as mock_load:
            ds = run.obs
        mock_load.assert_called_once_with(run._cache.obs_path)
        assert ds is fake_ds

    def test_historical_calls_load_with_historical_path(self, run):
        fake_ds = _fake_ds()
        with self._patch_load(fake_ds) as mock_load:
            ds = run.historical
        mock_load.assert_called_once_with(run._cache.historical_path)
        assert ds is fake_ds

    def test_scenario_calls_load_with_scenario_path(self, run):
        fake_ds = _fake_ds()
        with self._patch_load(fake_ds) as mock_load:
            ds = run.scenario
        mock_load.assert_called_once_with(run._cache.scenario_path)
        assert ds is fake_ds

    def test_obs_is_cached_property(self, run):
        fake_ds = _fake_ds()
        with self._patch_load(fake_ds) as mock_load:
            _ = run.obs
            _ = run.obs
        mock_load.assert_called_once()  # second access returns cached value


# ---------------------------------------------------------------------------
# BCSDRun.get_location_data
# ---------------------------------------------------------------------------


class TestGetLocationData:
    @pytest.fixture
    def run_with_data(self, run):
        """BCSDRun with obs/historical/scenario replaced by fake datasets."""
        fake = _fake_ds("tas")
        # Inject as cached_property values directly
        run.__dict__["obs"] = fake
        run.__dict__["historical"] = fake
        run.__dict__["scenario"] = fake
        return run

    def test_returns_dict_with_expected_keys(self, run_with_data):
        result = run_with_data.get_location_data(0.0, 10.0)
        assert set(result.keys()) == {"obs", "historical", "scenario"}

    def test_values_are_numpy_arrays(self, run_with_data):
        result = run_with_data.get_location_data(0.0, 10.0)
        for arr in result.values():
            assert isinstance(arr, np.ndarray)

    def test_result_is_cached_on_second_call(self, run_with_data):
        lat, lon = 0.0, 10.0
        first = run_with_data.get_location_data(lat, lon)
        second = run_with_data.get_location_data(lat, lon)
        assert first is second

    def test_different_locations_are_cached_separately(self, run_with_data):
        a = run_with_data.get_location_data(0.0, 10.0)
        b = run_with_data.get_location_data(30.0, 20.0)
        assert (0.0, 10.0) in run_with_data._location_cache
        assert (30.0, 20.0) in run_with_data._location_cache
        assert a is not b


# ---------------------------------------------------------------------------
# BCSDRun.plot_location_cdf
# ---------------------------------------------------------------------------


class TestPlotLocationCdf:
    @pytest.fixture
    def run_with_fake_location_data(self, run):
        """BCSDRun with get_location_data mocked to return predictable arrays."""
        rng = np.random.default_rng(1)
        fake_data = {
            "obs": rng.random(50).astype("float32"),
            "historical": rng.random(50).astype("float32"),
            "scenario": rng.random(50).astype("float32"),
        }
        run.get_location_data = MagicMock(return_value=fake_data)
        return run

    def test_returns_fig_and_two_axes(self, run_with_fake_location_data):
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend for CI
        import matplotlib.pyplot as plt

        fig, axes = run_with_fake_location_data.plot_location_cdf(0.0, 10.0)
        assert hasattr(fig, "savefig")
        assert len(axes) == 2
        plt.close(fig)

    def test_calls_get_location_data_with_correct_coords(self, run_with_fake_location_data):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lat, lon = -33.5, 18.4
        fig, _ = run_with_fake_location_data.plot_location_cdf(lat, lon)
        run_with_fake_location_data.get_location_data.assert_called_once_with(lat, lon)
        plt.close(fig)

    def test_suptitle_contains_variable_and_scenario(self, run_with_fake_location_data, config):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, _ = run_with_fake_location_data.plot_location_cdf(0.0, 10.0)
        title = fig.texts[0].get_text()
        assert config.variable in title
        assert config.scenario in title
        plt.close(fig)
