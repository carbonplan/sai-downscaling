"""The input ETL must be able to write to a non-``main`` icechunk branch.

Regenerating a store whose time axis changed cannot be done in place: ``r+`` requires matching
shapes, and ``determine_write_mode`` refuses ``"w"`` on a group that already exists. Writing to a
branch cut from the root snapshot sidesteps both, and promotion is then a pointer move rather than
a copy of the whole store (issue #521).

These run against a local filesystem repo, so they need no S3 and no cluster.
"""

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr

from srm.config import _ensure_root_group
from srm.input_data.etl_utils import determine_write_mode, write_variable_to_icechunk

BRANCH = "v521"


def _sample(offset: float) -> xr.Dataset:
    return xr.Dataset(
        {"tas": ("time", np.arange(offset, offset + 3, dtype="f4"))},
        coords={
            "time": np.array(["2015-01-01", "2015-01-02", "2015-01-03"], dtype="datetime64[ns]")
        },
    )


def _read(repo: icechunk.Repository, branch: str, group: str = "historical") -> np.ndarray:
    session = repo.readonly_session(branch)
    return xr.open_zarr(session.store, group=group, consolidated=False).tas.values


@pytest.fixture
def repo(tmp_path) -> icechunk.Repository:
    """A repo whose ``main`` already holds a group, plus an empty branch cut from the root."""
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(tmp_path / "test.icechunk"))
    )
    session = repo.writable_session("main")
    _sample(0).to_zarr(
        session.store, group="historical", mode="w", zarr_format=3, consolidated=False
    )
    session.commit("historical: original")

    root = list(repo.ancestry(branch="main"))[-1]
    repo.create_branch(BRANCH, snapshot_id=root.id)
    return repo


class TestBranchTargeting:
    def test_ensure_root_group_targets_the_requested_branch(self, repo):
        _ensure_root_group(repo, branch=BRANCH)

        zarr.open_group(repo.readonly_session(BRANCH).store, mode="r")  # raises if absent
        np.testing.assert_array_equal(_read(repo, "main"), [0.0, 1.0, 2.0])

    def test_write_variable_targets_the_requested_branch(self, repo):
        _ensure_root_group(repo, branch=BRANCH)

        write_variable_to_icechunk(
            _sample(100),
            repo,
            variable="tas",
            scenario="historical",
            chunks={"time": 3},
            shards={"time": 3},
            overwrite=False,
            var_in_store=False,
            group="historical",
            commit_message="historical: decoded from time_bnds",
            branch=BRANCH,
        )

        np.testing.assert_array_equal(_read(repo, BRANCH), [100.0, 101.0, 102.0])

    def test_writing_a_branch_leaves_main_untouched(self, repo):
        _ensure_root_group(repo, branch=BRANCH)
        write_variable_to_icechunk(
            _sample(100),
            repo,
            variable="tas",
            scenario="historical",
            chunks={"time": 3},
            shards={"time": 3},
            overwrite=False,
            var_in_store=False,
            group="historical",
            commit_message="historical: decoded from time_bnds",
            branch=BRANCH,
        )

        np.testing.assert_array_equal(_read(repo, "main"), [0.0, 1.0, 2.0])


class TestBranchCutFromRoot:
    def test_group_is_absent_so_the_write_mode_is_fresh(self, repo):
        """The reason this approach works at all.

        A branch cut from the root snapshot carries no groups, so determine_write_mode returns
        "w" and the changed time axis is written cleanly, rather than "a"/"r+" against the old one.
        """
        assert determine_write_mode(repo, group="historical") == "a"
        assert determine_write_mode(repo, branch=BRANCH, group="historical") == "w"

    def test_promotion_and_rollback_are_pointer_moves(self, repo):
        """reset_branch is the promotion step; nothing is copied and rollback is one call."""
        original_tip = repo.lookup_branch("main")
        _ensure_root_group(repo, branch=BRANCH)
        write_variable_to_icechunk(
            _sample(100),
            repo,
            variable="tas",
            scenario="historical",
            chunks={"time": 3},
            shards={"time": 3},
            overwrite=False,
            var_in_store=False,
            group="historical",
            commit_message="historical: decoded from time_bnds",
            branch=BRANCH,
        )

        repo.reset_branch("main", repo.lookup_branch(BRANCH))
        np.testing.assert_array_equal(_read(repo, "main"), [100.0, 101.0, 102.0])

        repo.reset_branch("main", original_tip)
        np.testing.assert_array_equal(_read(repo, "main"), [0.0, 1.0, 2.0])
