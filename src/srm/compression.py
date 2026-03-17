"""Zarr codec configuration for store writes.

All variables get Blosc/zstd at maximum effort with bitshuffle, which
reorders float bits by significance before compression and tends to outperform
the default codec for climate data.

Three non-negative bounded variables are additionally quantized to uint16 via
FixedScaleOffset, halving their on-disk footprint with negligible precision loss:

  rsds  (W m-2,        max 1500  → resolution 0.023 W m-2)
  hurs  (%,            max 100   → resolution 0.0015 %)
  pr    (kg m-2 s-1,   max 0.03  → resolution 4.6e-7 kg m-2 s-1 ≈ 0.04 mm/day)

FixedScaleOffset encodes as round((data - offset) * scale) and decodes as
encoded / scale + offset.  Value 65535 is reserved as the NaN sentinel.
"""

from numcodecs import Blosc, FixedScaleOffset

DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=9, shuffle=Blosc.BITSHUFFLE)

# Output store chunk/shard layout — dims: (ensemble_member, time, lat, lon)
OUTPUT_CHUNKS = (1, 1000, 32, 64)
OUTPUT_SHARD_TIME = 3000

# Max values include ~25% headroom over realistic GCM extremes so no clipping
# occurs in practice.  The encoded range is 0–65534; 65535 is fill_value.
UINT16_FILTERS: dict[str, FixedScaleOffset] = {
    "rsds": FixedScaleOffset(offset=0, scale=65534 / 1500.0, dtype="float32", astype="uint16"),
    "hurs": FixedScaleOffset(offset=0, scale=65534 / 100.0, dtype="float32", astype="uint16"),
    "pr": FixedScaleOffset(offset=0, scale=65534 / 0.03, dtype="float32", astype="uint16"),
}
UINT16_FILL_VALUE = 65535


def make_output_encoding(variables: list[str], n_lat: int, n_lon: int) -> dict:
    """Return encoding dict for all variables in the output consolidated store.

    Parameters
    ----------
    variables:
        Variable names to encode.
    n_lat, n_lon:
        Full spatial extent of the output grid (used as shard dimensions).
    """
    shards = (1, OUTPUT_SHARD_TIME, n_lat, n_lon)
    return {var: make_encoding_entry(var, chunks=OUTPUT_CHUNKS, shards=shards) for var in variables}


def make_encoding_entry(
    var_name: str,
    chunks: tuple | None = None,
    shards: tuple | None = None,
) -> dict:
    """Return a zarr encoding dict entry for one variable.

    Parameters
    ----------
    var_name:
        Variable name (case-insensitive lookup for uint16 candidates).
    chunks:
        Zarr chunk shape as a tuple aligned to the variable's dimensions.
    shards:
        Zarr shard shape (zarr v3 sharding).

    Returns
    -------
    dict
        Encoding entry suitable for passing to ``to_zarr`` / ``to_icechunk``
        via the ``encoding`` parameter.
    """
    entry: dict = {"compressor": DEFAULT_COMPRESSOR}
    if chunks is not None:
        entry["chunks"] = chunks
    if shards is not None:
        entry["shards"] = shards

    var_lower = var_name.lower()
    if var_lower in UINT16_FILTERS:
        entry["filters"] = [UINT16_FILTERS[var_lower]]
        entry["dtype"] = "uint16"
        entry["fill_value"] = UINT16_FILL_VALUE

    return entry
