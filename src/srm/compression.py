"""Zarr codec configuration for ERA5-resolution store writes."""

from zarr.codecs import BloscCodec
from zarr.codecs.blosc import BloscShuffle

DEFAULT_COMPRESSOR = BloscCodec(cname="zstd", clevel=9, shuffle=BloscShuffle.bitshuffle)


# ~4MB per chunk, time-series biased dumplings
# 8000 × 8 × 16 × 4 bytes is about 4MB
CHUNK_TIME = 8000
CHUNK_LAT = 8
CHUNK_LON = 16

# Shard layout
# 16000 × 72 × 144 × 4 bytes ~ 630MB;
SHARD_TIME = 16000
SHARD_LAT = 72  # 72  / CHUNK_LAT = 9 chunks
SHARD_LON = 144  # 144 / CHUNK_LON = 9 chunks


def make_encoding(var_name: str) -> dict:
    """Return a zarr encoding dict for a varaible

    Parameters
    ----------
    var_name
    """
    return {
        var_name: {
            "compressors": [DEFAULT_COMPRESSOR],
            "chunks": (CHUNK_TIME, CHUNK_LAT, CHUNK_LON),
            "shards": (SHARD_TIME, SHARD_LAT, SHARD_LON),
        }
    }
