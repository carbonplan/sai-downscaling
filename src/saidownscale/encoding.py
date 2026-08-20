"""
Zarr chunk, shard, and encoding settings for pipeline output arrays.

Chunk and shard sizes are tuned for time-series access patterns on icechunk stores.
Use :func:`make_encoding` for fine-resolution outputs and :func:`make_coarse_encoding`
for coarse GCM-resolution outputs.
"""

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

# Coarse GCM-resolution encoding (same chunk sizes; shards scaled to ~4× coarser grid)
CHUNK_TIME_COARSE = 8000
CHUNK_LAT_COARSE = 8
CHUNK_LON_COARSE = 16

# 16000 × 24 × 48 × 4 bytes ≈ 74 MB/shard (vs 630 MB for fine-res)
SHARD_TIME_COARSE = 16000
SHARD_LAT_COARSE = 24  # ~4° at 1° resolution
SHARD_LON_COARSE = 48  # ~4° at 1° resolution


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
        }
    }
