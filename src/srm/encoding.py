"""
Zarr chunk, shard, and encoding settings for pipeline output arrays.

Shapes are balanced between time-series and map access. Use :func:`make_encoding`
for fine-resolution outputs and :func:`make_coarse_encoding` for coarse
GCM-resolution outputs.
"""

from zarr.codecs import BloscCodec

COMPRESSOR = BloscCodec()

# Chunk size ~3.6MB.
CHUNK_TIME = 365
CHUNK_LAT = 36
CHUNK_LON = 72

# Shard size ~271MB.
SHARD_TIME = 1095
SHARD_LAT = 180
SHARD_LON = 360

# Coarse chunk size ~2.5MB.
CHUNK_TIME_COARSE = 365
CHUNK_LAT_COARSE = 30
CHUNK_LON_COARSE = 60

# Coarse shard size ~68MB
SHARD_TIME_COARSE = 1095
SHARD_LAT_COARSE = 90
SHARD_LON_COARSE = 180


def make_encoding(var_name: str) -> dict:
    """Return a zarr encoding dict for a variable

    Parameters
    ----------
    var_name :  str
                Name of the variable to encode (used as the dict key).
    """
    return {
        var_name: {
            "chunks": (CHUNK_TIME, CHUNK_LAT, CHUNK_LON),
            "shards": (SHARD_TIME, SHARD_LAT, SHARD_LON),
            "compressors": [COMPRESSOR],
        }
    }


def make_coarse_encoding(var_name: str) -> dict:
    """Return a zarr encoding dict for a coarse GCM-resolution variable.

    Parameters
    ----------
    var_name : str
        Name of the variable to encode (used as the dict key).
    """
    return {
        var_name: {
            "chunks": (CHUNK_TIME_COARSE, CHUNK_LAT_COARSE, CHUNK_LON_COARSE),
            "shards": (SHARD_TIME_COARSE, SHARD_LAT_COARSE, SHARD_LON_COARSE),
            "compressors": [COMPRESSOR],
        }
    }
