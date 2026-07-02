"""Regression test: re-writing sharded output stores after dropping encoding.

Output stores use zarr v3 sharding. On reopen, xarray reports the shard shape as
``encoding['chunks']`` while dask holds the smaller inner chunks, so re-writing the
data (e.g. blessing a syrupy snapshot) fails unless the encoding is dropped. A
manual ``encoding.pop`` does not survive to the writer because the shard shape is
re-derived from the zarr backend at write time; only ``drop_encoding`` works.
"""

from __future__ import annotations

import dask.array as da
import pytest
import xarray as xr

from srm.validation import _drop_output_encoding


def _write_sharded_store(path) -> None:
    # zarr v3 sharding: shard (16, 8, 8) with inner chunk (8, 4, 4). One dask chunk
    # equals the shard on write so the initial store is valid.
    ds = xr.Dataset({"dtr": (("time", "lat", "lon"), da.zeros((16, 8, 8), chunks=(16, 8, 8)))})
    ds.to_zarr(
        path,
        mode="w",
        zarr_format=3,
        encoding={"dtr": {"shards": (16, 8, 8), "chunks": (8, 4, 4)}},
    )


def test_drop_output_encoding_enables_sharded_rewrite(tmp_path):
    _write_sharded_store(tmp_path / "src.zarr")
    tree = xr.open_datatree(tmp_path / "src.zarr", engine="zarr", chunks={}, consolidated=False)

    # Reproduces the blessing failure: the shard shape leaks in as encoding['chunks']
    # and no longer tiles the dask graph.
    with pytest.raises(ValueError, match="overlap multiple Dask chunks"):
        tree.to_dataset().to_zarr(tmp_path / "plain.zarr", mode="w")

    cleaned = _drop_output_encoding(tree)

    # The loader's transform makes the re-write valid again.
    cleaned.to_dataset().to_zarr(tmp_path / "fixed.zarr", mode="w")
