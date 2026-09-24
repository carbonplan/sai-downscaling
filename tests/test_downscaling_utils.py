from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray_regrid  # noqa: F401  # registers the .regrid accessor
from xarray_regrid.utils import format_for_regrid

from saidownscale import downscaling_utils
from saidownscale.downscaling_utils import (
    calculate_baseline_climatology,
    coarse_domain_mask,
    derive_tasmin,
    detrend,
    downscale_from_coarse,
    fft_smooth_nharmonics,
    get_historical_experiment,
    interpolate_coarse_to_fine_grid,
    is_global_grid,
    rechunk,
    retrend,
    subset_space,
    swap_temperature_extremes,
)
from saidownscale.qa_checks import NaNCheckError


def _grid_da(data, **coords) -> xr.DataArray:
    return xr.DataArray(np.asarray(data), dims=list(coords), coords=coords)


def _daily(start: str, end: str, value: float, n: int = 2) -> xr.DataArray:
    time = np.arange(np.datetime64(start), np.datetime64(end))
    lat, lon = np.arange(n, dtype=float), 10.0 + np.arange(n)
    return _grid_da(
        np.full((time.size, n, n), value, dtype=np.float32), time=time, lat=lat, lon=lon
    )


def _monthly_clim(value: float, like: xr.DataArray) -> xr.DataArray:
    return _grid_da(
        np.full((12, like.lat.size, like.lon.size), value, dtype=np.float32),
        month=np.arange(1, 13),
        lat=like["lat"].values,
        lon=like["lon"].values,
    )


def test_subset_space(subtests):
    da = _grid_da(np.arange(16).reshape(4, 4), lat=np.arange(4.0), lon=10.0 + np.arange(4))
    for name, kwargs in [
        ("legacy_lat_then_lon", {"coord_bounds_list": [1.0, 2.0, 11.0, 12.0]}),
        ("named", {"lat_bounds": (1.0, 2.0), "lon_bounds": (11.0, 12.0)}),
    ]:
        with subtests.test(name):
            result = subset_space(da, **kwargs)
            np.testing.assert_array_equal(result["lat"].values, [1.0, 2.0])
            np.testing.assert_array_equal(result["lon"].values, [11.0, 12.0])

    for match, kwargs in [
        ("must contain four values", {"coord_bounds_list": [1.0, 2.0, 11.0]}),
        ("lat_bounds must be", {"lat_bounds": (2.0, 1.0), "lon_bounds": (11.0, 12.0)}),
        ("lon_bounds must be", {"lat_bounds": (1.0, 2.0), "lon_bounds": (12.0, 11.0)}),
        (
            "not both",
            {
                "coord_bounds_list": [1.0, 2.0, 11.0, 12.0],
                "lat_bounds": (1.0, 2.0),
                "lon_bounds": (11.0, 12.0),
            },
        ),
    ]:
        with subtests.test(match), pytest.raises(ValueError, match=match):
            subset_space(da, **kwargs)


def test_rechunk(monkeypatch, subtests):
    monkeypatch.setattr(downscaling_utils, "_TARGET_CHUNK_BYTES", 128)
    da = _grid_da(
        np.arange(160).reshape(10, 4, 4), time=np.arange(10), lat=np.arange(4.0), lon=np.arange(4.0)
    )

    with subtests.test("full_space"):
        chunks = rechunk(da, pattern="full_space").chunksizes
        assert len(chunks["time"]) > 1 and len(chunks["lat"]) == len(chunks["lon"]) == 1
    with subtests.test("full_time"):
        chunks = rechunk(da, pattern="full_time").chunksizes
        assert len(chunks["time"]) == 1 and len(chunks["lat"]) > 1 and len(chunks["lon"]) > 1
    for pattern, chunking in [
        ("full_space", {"time": 1, "lat": -1, "lon": -1}),
        ("full_time", {"time": -1, "lat": 1, "lon": 1}),
    ]:
        with subtests.test(f"{pattern}_noop"):
            already = da.chunk(chunking)
            assert rechunk(already, pattern=pattern) is already


def test_rechunk_full_space_fixes_concat_chunks_where_last_exceeds_first():
    """Regression: ssp-bridge + g6 concat leaves last chunk > first, which zarr rejects."""
    concat_da = xr.concat(
        [
            _daily("2015-01-01", "2015-01-04", 1.0).chunk(-1),
            _daily("2035-01-01", "2035-01-06", 1.0).chunk(-1),
        ],
        dim="time",
    )
    assert concat_da.chunksizes["time"][-1] > concat_da.chunksizes["time"][0]

    result_chunks = rechunk(concat_da, pattern="full_space").chunksizes["time"]

    assert result_chunks[-1] <= result_chunks[0]


def test_calculate_baseline_climatology_preserves_input_dtype():
    da = _daily("2000-01-01", "2001-01-01", 1.0)

    clim = calculate_baseline_climatology(da, baseline_period_start=2000, baseline_period_end=2000)

    assert clim.dtype == da.dtype
    np.testing.assert_array_equal(np.sort(clim["month"].values), np.arange(1, 13))


def test_detrend_multiplicative_with_zero_january_climatology():
    da = _daily("2001-01-01", "2010-01-01", 10.0)
    clim = _monthly_clim(2.0, da)
    clim.loc[dict(month=1)] = 0.0
    jan = da["time"].dt.month == 1

    detrended, trend = detrend(da=da, da_baseline_clim=clim, detrend_method="multiplicative")

    np.testing.assert_array_equal(trend["time"].values, da["time"].values)
    assert trend.shape == detrended.shape == da.shape
    assert trend.dtype == detrended.dtype == da.dtype
    assert np.isinf(trend.sel(time=jan).values).all()
    np.testing.assert_allclose(trend.sel(time=~jan).values, 5.0)
    np.testing.assert_allclose(detrended.sel(time=jan).values, 0.0)
    np.testing.assert_allclose(detrended.sel(time=~jan).values, 2.0)


def test_detrend_additive_matches_manual_grouped_rolling_for_january():
    time = np.arange(np.datetime64("2001-01-01"), np.datetime64("2013-01-01"))
    t = np.arange(time.size, dtype=np.float32)
    signal = (10.0 + 0.002 * t + 2.0 * np.sin(2.0 * np.pi * t / 365.25)).astype(np.float32)
    da = _grid_da(signal[:, None, None], time=time, lat=[0.0], lon=[10.0])

    _, trend_daily = detrend(
        da=da, da_baseline_clim=_monthly_clim(0.0, da), detrend_method="additive"
    )

    jan_from_detrend = (
        trend_daily.sel(time=trend_daily.time.dt.month == 1)
        .resample(time="YS")
        .first()
        .squeeze(drop=True)
    )
    da_mon = da.resample(time="1MS").mean("time")
    jan_mon = da_mon.sel(time=da_mon.time.dt.month == 1).squeeze(drop=True)
    rolling = jan_mon.rolling(time=9, center=True, min_periods=1)
    np.testing.assert_array_equal(
        rolling.count().values.astype(int), [5, 6, 7, 8, 9, 9, 9, 9, 8, 7, 6, 5]
    )
    jan_manual = rolling.mean()
    np.testing.assert_array_equal(jan_from_detrend["time"].values, jan_manual["time"].values)
    np.testing.assert_allclose(jan_from_detrend.values, jan_manual.values, rtol=1e-6, atol=1e-6)


def test_retrend(subtests):
    detrended = _daily("2001-01-01", "2001-01-11", 2.0)
    trend = _daily("2001-01-01", "2001-01-11", 5.0)
    for method, expected in [("additive", 7.0), ("multiplicative", 10.0)]:
        with subtests.test(method):
            result = retrend(detrended, trend, detrend_method=method)
            np.testing.assert_allclose(result.values, expected)
            np.testing.assert_array_equal(result["time"].values, detrended["time"].values)
    with subtests.test("invalid"), pytest.raises(ValueError, match="currently not supported"):
        retrend(detrended, trend, detrend_method="invalid")  # type: ignore[arg-type]


def test_fft_smooth_nharmonics(subtests):
    n = 365
    t = np.arange(n)
    three_harmonics = (
        5.0
        + 3.0 * np.cos(2 * np.pi * 1 * t / n + 0.50)
        + 2.0 * np.cos(2 * np.pi * 2 * t / n + 1.00)
        + 1.0 * np.cos(2 * np.pi * 3 * t / n + 1.50)
    )

    with subtests.test("all_nan_returned_unchanged"):
        data = np.full(n, np.nan)
        np.testing.assert_array_equal(fft_smooth_nharmonics(data), data)
    for name, data in [("constant", np.full(n, 7.5)), ("three_harmonics", three_harmonics)]:
        with subtests.test(f"{name}_lossless"):
            result = fft_smooth_nharmonics(data)
            assert result.shape == data.shape
            np.testing.assert_allclose(result, data, atol=1e-10)
    for dtype in (np.float32, np.float64):
        with subtests.test(f"dtype_{dtype.__name__}"):
            assert fft_smooth_nharmonics(three_harmonics.astype(dtype)).dtype == dtype
    with subtests.test("seasonal_peak_not_shifted"):
        seasonal = np.cos(2 * np.pi * t / n)
        assert np.argmax(fft_smooth_nharmonics(seasonal)) == np.argmax(seasonal)
    with subtests.test("attenuates_spike"):
        spike = np.zeros(n)
        spike[0] = 1.0
        assert np.sqrt(np.mean(fft_smooth_nharmonics(spike) ** 2)) < np.sqrt(np.mean(spike**2))


def _make_global_coarse_da(dtype: str = "float64") -> xr.DataArray:
    lon = np.arange(-180.0, 180.0, 45.0)
    lat = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    data = np.sin(np.deg2rad(lon))[np.newaxis, :] + 0.1 * np.cos(np.deg2rad(lat))[:, np.newaxis]
    return _grid_da(data.astype(dtype), lat=lat, lon=lon)


def _make_fine_grid() -> xr.DataArray:
    lon = np.array([-179.9, -170.0, -90.0, 0.0, 90.0, 170.0, 179.9])
    lat = np.array([-55.0, 0.0, 55.0])
    return _grid_da(np.zeros((lat.size, lon.size)), lat=lat, lon=lon)


def test_interpolate_coarse_to_fine_grid(subtests):
    coarse = _make_global_coarse_da()
    fine = _make_fine_grid()
    result = interpolate_coarse_to_fine_grid(coarse, fine)

    with subtests.test("wraps_periodically"):
        w = (179.9 - 135.0) / 45.0
        expected = (1 - w) * np.sin(np.deg2rad(135.0)) + w * np.sin(np.deg2rad(-180.0)) + 0.1
        np.testing.assert_allclose(result.sel(lat=0.0, lon=179.9).item(), expected, atol=1e-6)
        truth = (
            np.sin(np.deg2rad(fine["lon"].values))[np.newaxis, :]
            + 0.1 * np.cos(np.deg2rad(fine["lat"].values))[:, np.newaxis]
        )
        np.testing.assert_allclose(result.values, truth, atol=0.1)
    with subtests.test("interior_matches_plain_interp"):
        plain = coarse.interp(lon=fine["lon"], lat=fine["lat"], method="linear")
        interior = ~plain.isnull().values
        np.testing.assert_allclose(result.values[interior], plain.values[interior], atol=1e-12)
    with subtests.test("preserves_float32"):
        f32 = interpolate_coarse_to_fine_grid(_make_global_coarse_da("float32"), fine)
        assert f32.dtype == np.float32
    with subtests.test("dask_backed_no_nan"):
        lazy = coarse.expand_dims(time=pd.date_range("2000-01-01", periods=3)).chunk(
            {"time": 1, "lat": -1, "lon": 4}
        )
        assert not interpolate_coarse_to_fine_grid(lazy, fine).compute().isnull().any()
    with subtests.test("unsorted_lon"):
        shuffled = coarse.isel(lon=np.array([3, 0, 7, 1, 5, 2, 6, 4]))
        np.testing.assert_allclose(
            interpolate_coarse_to_fine_grid(shuffled, fine).values, result.values, atol=1e-12
        )
    with subtests.test("regional_edge_stays_nan"):
        lon = np.arange(16.25, 32.6, 1.25)
        lat = np.arange(-34.5, -22.0, 1.0)
        regional = _grid_da(
            np.outer(np.cos(np.deg2rad(lat)), np.sin(np.deg2rad(lon))), lat=lat, lon=lon
        )
        regional_fine = _grid_da(
            np.zeros((3, 5)), lat=[-30.0, -28.0, -26.0], lon=[16.0, 20.0, 25.0, 32.75, 33.0]
        )
        plain = regional.interp(lon=regional_fine["lon"], lat=regional_fine["lat"])
        assert plain.isnull().any()
        np.testing.assert_allclose(
            interpolate_coarse_to_fine_grid(regional, regional_fine).values,
            plain.values,
            atol=1e-12,
        )
        assert not bool(coarse_domain_mask(regional, regional_fine).all())


_GCM_GRIDS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "CESM2-WACCM6": (np.linspace(-90.0, 90.0, 192), np.arange(-180.0, 180.0, 1.25)),
    "gaussian-t85": (
        np.degrees(np.arcsin(np.polynomial.legendre.leggauss(128)[0])),
        np.arange(-180.0, 180.0, 1.40625),
    ),
    "UKESM1-1-LL": (
        np.arange(-89.375, 89.376, 1.25),
        np.arange(-179.0625, 179.07, 1.875),
    ),
}
_DTYPES = ("float32", "float64")


@pytest.fixture(scope="module")
def era5_fine_grid() -> xr.DataArray:
    lat = np.arange(-90.0, 90.001, 0.25)
    lon = np.arange(-180.0, 179.751, 0.25)
    return _grid_da(np.zeros((lat.size, lon.size)), lat=lat, lon=lon)


def _make_gcm_coarse(gcm: str) -> xr.DataArray:
    lat, lon = _GCM_GRIDS[gcm]
    data = np.cos(np.deg2rad(lat))[:, None] * np.sin(np.deg2rad(lon))[None, :] + 2.0
    return _grid_da(data, lat=lat, lon=lon)


def test_real_gcm_grid_covers_full_globe(era5_fine_grid, subtests):
    """#554: every real GCM grid regrids to a NaN-free global field in both dtypes."""
    for gcm in sorted(_GCM_GRIDS):
        for dtype in _DTYPES:
            with subtests.test(gcm=gcm, dtype=dtype):
                lat, lon = (a.astype(dtype) for a in _GCM_GRIDS[gcm])
                data = np.cos(np.deg2rad(lat.astype("float64")))[:, None] * np.ones(lon.size) + 2.0
                coarse = _grid_da(data.astype(dtype), lat=lat, lon=lon).rename("rsds")

                result = interpolate_coarse_to_fine_grid(coarse, era5_fine_grid)

                assert int(result.isnull().sum()) == 0
                assert bool(coarse_domain_mask(coarse, era5_fine_grid).all())
                assert result.dtype == np.dtype(dtype)


def test_interpolate_ukesm_west_seam_is_finite_and_periodic(era5_fine_grid):
    """#554: fine cells west of UKESM's first lon center are filled periodically."""
    coarse = _make_gcm_coarse("UKESM1-1-LL")

    result = interpolate_coarse_to_fine_grid(coarse, era5_fine_grid)

    seam = result.sel(lon=slice(-180.0, -179.25))
    assert seam.sizes["lon"] == 4
    assert not bool(seam.isnull().any())
    np.testing.assert_allclose(
        result.sel(lon=-180.0, lat=0.0).item(),
        np.interp(
            180.0,
            [179.0625, 180.9375],
            [
                coarse.sel(lat=0.0, lon=179.0625, method="nearest").item(),
                coarse.sel(lat=0.0, lon=-179.0625, method="nearest").item(),
            ],
        ),
        rtol=1e-6,
    )


def test_interpolate_pole_row_is_zonally_constant(era5_fine_grid, subtests):
    for gcm in ["gaussian-t85", "UKESM1-1-LL"]:
        result = interpolate_coarse_to_fine_grid(_make_gcm_coarse(gcm), era5_fine_grid)
        for pole in (-90.0, 90.0):
            with subtests.test(gcm=gcm, pole=pole):
                row = result.sel(lat=pole).values
                assert np.allclose(row, row[0])


def test_linear_and_slinear_agree_and_stay_nonnegative(era5_fine_grid, subtests):
    """Pins that linear adds no negative artifacts vs slinear (#111 to #554)."""
    lat, lon = _GCM_GRIDS["UKESM1-1-LL"]
    rng = np.random.default_rng(0)
    precip_like = rng.gamma(0.3, 2.0, size=(lat.size, lon.size))
    precip_like[rng.random(precip_like.shape) < 0.4] = 0.0
    fields = {"smooth": _make_gcm_coarse("UKESM1-1-LL").values, "precip_like": precip_like}
    target = xr.Dataset(coords={"lat": era5_fine_grid["lat"], "lon": era5_fine_grid["lon"]})

    for field, data in fields.items():
        with subtests.test(field):
            coarse = _grid_da(data, lat=lat, lon=lon).rename("pr")
            padded = format_for_regrid(coarse, target)
            linear = padded.interp(lat=target["lat"], lon=target["lon"], method="linear")
            slinear = padded.interp(lat=target["lat"], lon=target["lon"], method="slinear")

            np.testing.assert_allclose(linear.values, slinear.values, atol=1e-12)
            assert int((linear < 0).sum()) == int((slinear < 0).sum()) == 0
            np.testing.assert_allclose(
                interpolate_coarse_to_fine_grid(coarse, era5_fine_grid).values,
                slinear.values,
                atol=1e-12,
            )


def test_interpolate_coarse_to_fine_grid_handles_dayofyear_leading_dim(era5_fine_grid):
    lat, lon = _GCM_GRIDS["UKESM1-1-LL"]
    coarse = _grid_da(
        np.random.default_rng(0).random((5, lat.size, lon.size)),
        dayofyear=np.arange(1, 6),
        lat=lat,
        lon=lon,
    )

    result = interpolate_coarse_to_fine_grid(coarse, era5_fine_grid)

    assert result.dims == ("dayofyear", "lat", "lon")
    assert result.sizes["dayofyear"] == 5
    assert int(result.isnull().sum()) == 0


def test_upstream_pads_poles_and_both_seams_with_zero_margin(era5_fine_grid, subtests):
    """#554 canary: xarray-regrid wraps lon only while span >= 360 - dx, which GCMs hit exactly."""
    target = xr.Dataset(coords={"lat": era5_fine_grid.lat, "lon": era5_fine_grid.lon})
    for gcm in sorted(_GCM_GRIDS):
        for dtype in _DTYPES:
            with subtests.test(gcm=gcm, dtype=dtype):
                lat, lon = (a.astype(dtype) for a in _GCM_GRIDS[gcm])
                dx = float(np.diff(lon).max())
                assert float(lon.max() - lon.min()) == 360.0 - dx

                coarse = _grid_da(np.zeros((lat.size, lon.size), dtype=dtype), lat=lat, lon=lon)
                padded = format_for_regrid(coarse.rename("rsds"), target)

                assert padded["lat"].values[0] == -90.0
                assert padded["lat"].values[-1] == 90.0
                assert padded["lon"].values[0] < lon[0]
                assert padded["lon"].values[-1] > lon[-1]


def test_bypassing_regrid_linear_is_equivalent(era5_fine_grid, subtests):
    """Our padding + interp must equal `.regrid.linear`, and match it within 1 float32 ULP."""
    target = xr.Dataset(coords={"lat": era5_fine_grid["lat"], "lon": era5_fine_grid["lon"]})
    for gcm in sorted(_GCM_GRIDS):
        with subtests.test(gcm):
            coarse = _make_gcm_coarse(gcm)
            accessor = coarse.regrid.linear(target)
            order = list(set(target.coords).intersection(set(coarse.coords)))
            hand = format_for_regrid(coarse, target).interp(
                coords={k: target[k] for k in order}, method="linear"
            )
            np.testing.assert_array_equal(accessor.values, hand.transpose(*accessor.dims).values)

            coarse32 = coarse.astype("float32")
            ours = interpolate_coarse_to_fine_grid(coarse32, era5_fine_grid)
            reference = coarse32.regrid.linear(target).astype("float32").transpose(*ours.dims)
            np.testing.assert_array_equal(np.isnan(ours.values), np.isnan(reference.values))
            differing = ours.values != reference.values
            if differing.any():
                ulp = np.abs(
                    ours.values[differing].view(np.int32)
                    - reference.values[differing].view(np.int32)
                )
                assert ulp.max() <= 1


def test_get_historical_experiment_uses_unified_store(subtests):
    for gcm, member in [
        ("CESM2-WACCM6", "r1i1p1f1"),
        ("CESM2-WACCM6", "001"),
        ("UKESM1-1-LL", "r2i1p1f2"),
    ]:
        with subtests.test(gcm=gcm, member=member):
            mock_da = MagicMock(spec=xr.DataArray)
            mock_da.sel.return_value = mock_da
            mock_ds = MagicMock()
            mock_ds.proj.assign_crs.return_value = mock_ds
            mock_node = MagicMock()
            mock_node.to_dataset.return_value = mock_ds
            mock_dt = MagicMock()
            mock_dt.__getitem__ = MagicMock(return_value=mock_node)

            with (
                patch("saidownscale.downscaling_utils._gcm_datatree", return_value=mock_dt) as fn,
                patch("saidownscale.downscaling_utils.get_variable", return_value=mock_da),
            ):
                get_historical_experiment(gcm, member, "tas")

            fn.assert_called_once_with(gcm)
            mock_dt.__getitem__.assert_called_once_with("historical")


def test_swap_temperature_extremes(subtests):
    """#331: inverted tasmax/tasmin cells are swapped; monotone and NaN cells are left alone."""
    lat, lon = np.array([0.0, 1.0]), np.array([10.0, 11.0])
    tasmax = _grid_da([[300.0, 290.0], [np.nan, 305.0]], lat=lat, lon=lon).rename("tasmax")
    tasmin = _grid_da([[280.0, 295.0], [285.0, 300.0]], lat=lat, lon=lon).rename("tasmin")

    with subtests.test("2d"):
        new_max, new_min = swap_temperature_extremes(tasmax, tasmin)
        np.testing.assert_array_equal(new_max.values, [[300.0, 295.0], [np.nan, 305.0]])
        np.testing.assert_array_equal(new_min.values, [[280.0, 290.0], [285.0, 300.0]])
        assert (new_max.name, new_min.name) == ("tasmax", "tasmin")
    with subtests.test("unnamed_tasmin"):
        new_max, _ = swap_temperature_extremes(tasmax, tasmin.rename(None))
        assert new_max.name == "tasmax"
        assert new_max.values[0, 1] == 295.0
    with subtests.test("3d_dask_stays_lazy_and_monotone"):
        rng = np.random.default_rng(0)
        coords = {"time": pd.date_range("2020-01-01", periods=5), "lat": lat, "lon": lon}
        hi = _grid_da(rng.normal(300.0, 3.0, (5, 2, 2)), **coords).chunk({"time": 2})
        lo = _grid_da(rng.normal(300.0, 3.0, (5, 2, 2)), **coords).chunk({"time": 2})
        new_max, new_min = swap_temperature_extremes(hi, lo)
        assert new_max.chunks is not None and new_min.chunks is not None
        assert np.all(new_max.compute().values >= new_min.compute().values)
    with subtests.test("misaligned_coords_raise"), pytest.raises(ValueError):
        swap_temperature_extremes(tasmax, tasmin.assign_coords(lon=lon + 0.001))


def test_derive_tasmin(subtests):
    """#363: derive_tasmin fails loud when tasmax and dtr time axes differ."""

    def series(start, end, value):
        t = np.arange(start, end, dtype="datetime64[D]")
        return _grid_da(
            np.full((t.size, 1, 1), value, dtype="float32"), time=t, lat=[0.0], lon=[0.0]
        )

    with subtests.test("matching"):
        out = derive_tasmin(
            series("2015-01-01", "2015-01-05", 300.0), series("2015-01-01", "2015-01-05", 10.0)
        )
        assert out.name == "tasmin"
        assert float(out.isel(time=0, lat=0, lon=0)) == 290.0
    for match, dtr in [
        ("#363", series("2035-01-01", "2035-01-05", 10.0)),
        ("time ax", series("2015-01-01", "2015-01-03", 10.0)),
    ]:
        with subtests.test(match), pytest.raises(ValueError, match=match):
            derive_tasmin(series("2015-01-01", "2015-01-05", 300.0), dtr)


def _regional_grids(n_time: int = 40) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    time = pd.date_range("2015-01-01", periods=n_time, freq="D")
    coarse_lat = np.array([-30.0, -20.0, -10.0])
    coarse_lon = np.array([20.0, 30.0, 40.0])
    fine_lat = np.arange(-35.0, -4.9, 5.0)
    fine_lon = np.arange(15.0, 45.1, 5.0)

    def _da(lat, lon, offset):
        shape = (time.size, lat.size, lon.size)
        values = offset + np.arange(np.prod(shape), dtype="float64").reshape(shape) % 7
        return _grid_da(values, time=time, lat=lat, lon=lon).rename("tas")

    return (
        _da(coarse_lat, coarse_lon, 300.0),
        _da(coarse_lat, coarse_lon, 299.0),
        _da(fine_lat, fine_lon, 298.0),
    )


def _pole_gap_global_grids() -> tuple[xr.DataArray, xr.DataArray]:
    """UKESM1-1-LL geometry at test scale: coarse centers stop half a step short (#553, #554)."""
    coarse_lat = np.arange(-75.0, 90.0, 30.0)
    coarse_lon = np.arange(-157.5, 180.0, 45.0)
    data = (
        np.cos(np.deg2rad(coarse_lat))[:, np.newaxis]
        + 0.1 * np.sin(np.deg2rad(coarse_lon))[np.newaxis, :]
    )
    coarse = _grid_da(data, lat=coarse_lat, lon=coarse_lon)
    fine_lat = np.arange(-85.0, 90.0, 10.0)
    fine_lon = np.arange(-175.0, 180.0, 10.0)
    return coarse, _grid_da(np.zeros((fine_lat.size, fine_lon.size)), lat=fine_lat, lon=fine_lon)


def _global_grids() -> tuple[xr.DataArray, xr.DataArray]:
    coarse = _make_global_coarse_da().assign_coords(lat=[-90.0, -45.0, 0.0, 45.0, 90.0])
    fine_lat = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    fine_lon = np.arange(-180.0, 180.0, 10.0)
    return coarse, _grid_da(np.zeros((fine_lat.size, fine_lon.size)), lat=fine_lat, lon=fine_lon)


def test_coarse_domain_mask(subtests):
    coarse, fine = _global_grids()
    coarse_sim, _, obs_fine = _regional_grids()
    regional = coarse_sim.isel(time=0, drop=True)
    regional_fine = obs_fine.isel(time=0, drop=True)

    with subtests.test("global_covers_every_fine_cell"):
        assert bool(coarse_domain_mask(coarse, fine).all())
    for name, c, f in [("global", coarse, fine), ("regional", regional, regional_fine)]:
        with subtests.test(f"matches_interpolation_{name}"):
            np.testing.assert_array_equal(
                coarse_domain_mask(c, f).values,
                ~np.isnan(interpolate_coarse_to_fine_grid(c, f).values),
            )
    with subtests.test("regional_excludes_out_of_domain_frame"):
        mask = coarse_domain_mask(regional, regional_fine)
        assert not bool(mask.isel(lat=0).any())
        assert not bool(mask.isel(lon=0).any())
        assert bool(mask.isel(lat=3, lon=3))


def test_is_global_grid(subtests):
    """Sparse or truncated axes must not be mistaken for global ones (#553, #554)."""

    def box(lat, lon):
        return _grid_da(np.zeros((len(lat), len(lon))), lat=np.array(lat), lon=np.array(lon))

    global_coarse, _ = _global_grids()
    pole_gap, _ = _pole_gap_global_grids()
    regional, _, _ = _regional_grids()
    assert global_coarse.sizes["lat"] == 5
    for name, da, expected in [
        ("pole_inclusive_well_sampled", global_coarse, True),
        ("cell_center_short_of_poles", pole_gap, True),
        ("regional_subset", regional.isel(time=0, drop=True), False),
        ("truncated_lat", pole_gap.sel(lat=slice(-50.0, 50.0)), False),
        ("regional_lon", pole_gap.sel(lon=slice(-70.0, 70.0)), False),
        ("two_point_box", box([-45.0, 45.0], [-90.0, 90.0]), False),
        ("three_point_box", box([-45.0, 0.0, 45.0], [-120.0, 0.0, 120.0]), False),
        ("sparse_lat_global_lon", box([-45.0, 45.0], list(np.arange(-157.5, 180.0, 45.0))), False),
    ]:
        with subtests.test(name):
            assert is_global_grid(da) is expected


def test_downscale_from_coarse_nan_guards(subtests):
    """#517, #553, #554: NaN gates abort on real NaNs but tolerate a regional frame."""
    coarse_sim, obs_coarse, obs_fine = _regional_grids()

    with subtests.test("regional_edge_nans_do_not_abort"):
        result = downscale_from_coarse(da=coarse_sim, obs_coarse=obs_coarse, obs_fine=obs_fine)
        assert bool(np.isnan(result.isel(time=0, lat=0)).all())
        assert not bool(np.isnan(result.isel(time=0, lat=3, lon=3)))
    with subtests.test("nan_in_debiased_input_aborts"):
        bad = coarse_sim.copy()
        bad[5, :, :] = np.nan
        with pytest.raises(NaNCheckError, match="residuals"):
            downscale_from_coarse(da=bad, obs_coarse=obs_coarse, obs_fine=obs_fine)
    with subtests.test("nan_in_fine_observations_aborts"):
        bad = obs_fine.copy()
        bad[:, 3, 3] = np.nan
        with pytest.raises(NaNCheckError, match="doy_means"):
            downscale_from_coarse(da=coarse_sim, obs_coarse=obs_coarse, obs_fine=bad)
    with subtests.test("global_pole_and_dateline_gap_is_filled"):
        coarse, fine = _pole_gap_global_grids()

        def series(space, offset):
            time = pd.date_range("2015-01-01", periods=40, freq="D")
            values = offset + np.tile(space.values, (40, 1, 1))
            values += (np.arange(40, dtype="float64") % 7)[:, np.newaxis, np.newaxis]
            return _grid_da(values, time=time, lat=space["lat"], lon=space["lon"]).rename("tas")

        result = downscale_from_coarse(
            da=series(coarse, 300.0),
            obs_coarse=series(coarse, 299.0),
            obs_fine=series(xr.zeros_like(fine), 298.0),
        )
        assert int(result.isnull().sum()) == 0
        assert not bool(result.sel(lat=[-85.0, 85.0]).isnull().any())
        assert not bool(result.sel(lon=-175.0).isnull().any())


def test_downscaled_nan_gate_runs_with_and_without_conservation(subtests):
    coarse_sim, obs_coarse, obs_fine = _regional_grids()

    with subtests.test("gate_runs_without_conservation"):
        checked: list[str] = []
        real_assert = downscaling_utils.assert_no_nans

        def _spy(data, *, name, **kwargs):
            checked.append(name)
            return real_assert(data, name=name, **kwargs)

        with patch.object(downscaling_utils, "assert_no_nans", side_effect=_spy):
            downscale_from_coarse(da=coarse_sim, obs_coarse=obs_coarse, obs_fine=obs_fine)
        assert "downscaled" in checked
    with subtests.test("clean_conservation_run_passes"):
        result = downscale_from_coarse(
            da=coarse_sim, obs_coarse=obs_coarse, obs_fine=obs_fine, enforce_conservation=True
        )
        assert not bool(np.isnan(result.isel(time=0, lat=3, lon=3)))
    with subtests.test("nan_from_conservation_correction_aborts"):
        real_recoarsen = downscaling_utils.interpolate_fine_to_coarse_grid
        calls = []

        def _second_call_returns_nan(*args, **kwargs):
            out = real_recoarsen(*args, **kwargs)
            calls.append(1)
            return out.where(False) if len(calls) == 2 else out

        with (
            patch.object(
                downscaling_utils,
                "interpolate_fine_to_coarse_grid",
                side_effect=_second_call_returns_nan,
            ),
            pytest.raises(NaNCheckError, match="downscaled"),
        ):
            downscale_from_coarse(
                da=coarse_sim, obs_coarse=obs_coarse, obs_fine=obs_fine, enforce_conservation=True
            )
