"""South Africa snapshot regression gate (issue #410).

Reads the SA output store produced by `bcsd run` on
configs/snapshot/cesm2-waccm/ and asserts each scenario group against the stored
snapshot. Requires:
  - the SA output store to already exist (run the produce step first), and
  - SNAPSHOT_STORAGE_PATH pointing at the snapshot icechunk store on carbonplan-srm.

Run: uv run pytest -m snapshot tests/test_snapshot_gate.py -v
First time / to bless: add --snapshot-update.
"""

from __future__ import annotations

import pytest

from srm.cache import ArtifactCache
from srm.cli import load_configs
from srm.config import SCENARIO_TO_GROUP
from srm.validation import _open_output_datatree

CONFIG_PATH = "configs/snapshot/cesm2-waccm/"


def _scenario_groups() -> list[str]:
    # All known output scenario groups (incl. historical produced by the fit
    # stage). The test skips any group absent from the produced store, so this
    # covers historical/ssp245/g6_1p5k without opening S3 at collection time.
    return sorted(set(SCENARIO_TO_GROUP.values()))


@pytest.fixture(scope="module")
def sa_output_tree():
    configs, options = load_configs(CONFIG_PATH)
    cache = ArtifactCache.from_config(configs[0], options)
    store_uri = cache._output_store
    return _open_output_datatree(store_uri, branch=cache.branch)


@pytest.mark.slow
@pytest.mark.snapshot
@pytest.mark.parametrize("group", _scenario_groups())
def test_south_africa_matches_snapshot(srm_snapshot, sa_output_tree, group):
    if group not in sa_output_tree.children:
        pytest.skip(f"scenario group '{group}' not present in SA output store")
    subtree = sa_output_tree[group]
    # Snapshot on the LEFT — required so xarray does not intercept the ==.
    assert srm_snapshot == subtree
