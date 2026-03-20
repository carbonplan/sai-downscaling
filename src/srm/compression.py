"""Zarr codec configuration for store writes."""

from zarr.codecs import BloscCodec
from zarr.codecs.blosc import BloscShuffle
from zarr.codecs.numcodecs import FixedScaleOffset

DEFAULT_COMPRESSOR = BloscCodec(cname="zstd", clevel=9, shuffle=BloscShuffle.bitshuffle)

# Output chunk/shard layout — dims: (ensemble_member, time, lat, lon)
CHUNK_TIME = 1000
SHARD_TIME = 3000
CHUNK_LAT = 32
CHUNK_LON = 64

# Variables quantized to uint16. Encoded range 0–65534; 65535 = NaN fill.
# Edit scale to adjust precision; If you remove a var here, it will default to float32
UINT16_SCALE: dict[str, float] = {
    "pr": 65534 / 0.03,  # kg m-2 s-1  (~0.04 mm/day resolution)
    "hurs": 65534 / 100.0,  # %           (~0.0015 % resolution)
    "rsds": 65534 / 1500.0,  # W m-2       (~0.023 W m-2 resolution)
}


def make_encoding(var_name: str, da) -> dict:
    """Return a zarr encoding dict for one variable in the output store.

    Parameters
    ----------
    var_name:
        Variable name
    da:
        The DataArray being written
    """
    encoding = {
        "compressors": [DEFAULT_COMPRESSOR],
        "chunks": (1, CHUNK_TIME, CHUNK_LAT, CHUNK_LON),
        "shards": (1, SHARD_TIME, da.sizes["lat"], da.sizes["lon"]),
    }
    scale = UINT16_SCALE.get(var_name.lower())
    if scale is not None:
        encoding["filters"] = [
            FixedScaleOffset(offset=0, scale=scale, dtype="float32", astype="uint16")
        ]
        encoding["dtype"] = "uint16"
        encoding["fill_value"] = 65535
    return encoding
