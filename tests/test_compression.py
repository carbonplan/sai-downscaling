# """Unit tests for srm.compression codec configuration."""

# import numpy as np
# import pytest
# import xarray as xr

# from srm.compression import (
#     CHUNK_LAT,
#     CHUNK_LON,
#     CHUNK_TIME,
#     DEFAULT_COMPRESSOR,
#     SHARD_TIME,
#     UINT16_SCALE,
#     make_encoding,
# )


# def _da(n_lat=720, n_lon=1440):
#     """Minimal DataArray with lat/lon dims."""
#     return xr.DataArray(
#         np.zeros((1, 10, n_lat, n_lon)),
#         dims=["ensemble_member", "time", "lat", "lon"],
#     )


# class TestMakeEncoding:
#     def test_compressor_always_present(self):
#         assert make_encoding("tas", _da())["compressors"] == [DEFAULT_COMPRESSOR]

#     def test_chunk_shape(self):
#         enc = make_encoding("tas", _da())
#         assert enc["chunks"] == (1, CHUNK_TIME, CHUNK_LAT, CHUNK_LON)

#     def test_shard_shape_uses_da_spatial_dims(self):
#         enc = make_encoding("tas", _da(n_lat=180, n_lon=360))
#         assert enc["shards"] == (1, SHARD_TIME, 180, 360)

#     def test_float32_variable_no_quantization_fields(self):
#         enc = make_encoding("tas", _da())
#         assert "filters" not in enc
#         assert "dtype" not in enc
#         assert "fill_value" not in enc

#     @pytest.mark.parametrize("var", UINT16_SCALE)
#     def test_uint16_variable_dtype(self, var):
#         assert make_encoding(var, _da())["dtype"] == "uint16"

#     @pytest.mark.parametrize("var", UINT16_SCALE)
#     def test_uint16_fill_value_is_max_sentinel(self, var):
#         assert make_encoding(var, _da())["fill_value"] == 65535

#     @pytest.mark.parametrize("var", ["RSDS", "HURS", "PR"])
#     def test_case_insensitive_lookup(self, var):
#         assert "filters" in make_encoding(var, _da())
