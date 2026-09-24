"""The input ETL writes to a non-main icechunk branch cut from the root snapshot (#521)."""

import icechunk
import numpy as np
import pytest
import xarray as xr
import zarr

from saidownscale.config import _ensure_root_group
from saidownscale.input_data.etl_utils import determine_write_mode, write_variable_to_icechunk

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


def test_group_is_absent_so_the_write_mode_is_fresh(repo):
    """A root-cut branch has no groups, so a changed time axis is written with "w", not "r+"."""
    assert determine_write_mode(repo, group="historical") == "a"
    assert determine_write_mode(repo, branch=BRANCH, group="historical") == "w"


def test_branch_write_leaves_main_untouched_and_promotes_by_pointer(repo, capsys):
    original_tip = repo.lookup_branch("main")
    _ensure_root_group(repo, branch=BRANCH)
    zarr.open_group(repo.readonly_session(BRANCH).store, mode="r")

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
        commit_message="written on the branch",
        branch=BRANCH,
    )

    np.testing.assert_array_equal(_read(repo, BRANCH), [100.0, 101.0, 102.0])
    np.testing.assert_array_equal(_read(repo, "main"), [0.0, 1.0, 2.0])
    printed = capsys.readouterr().out
    assert "written on the branch" in printed
    assert "historical: original" not in printed, "leaked main's history into a branch run"

    repo.reset_branch("main", repo.lookup_branch(BRANCH))
    np.testing.assert_array_equal(_read(repo, "main"), [100.0, 101.0, 102.0])
    repo.reset_branch("main", original_tip)
    np.testing.assert_array_equal(_read(repo, "main"), [0.0, 1.0, 2.0])
