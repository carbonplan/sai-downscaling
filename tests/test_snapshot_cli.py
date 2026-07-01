import numpy as np
import xarray as xr
from typer.testing import CliRunner

from srm.cli import app

runner = CliRunner()


def _tree(pr_values):
    return xr.DataTree.from_dict(
        {
            "g6_1p5k/pr": xr.Dataset(
                {"pr": xr.DataArray(np.asarray(pr_values, "float64"), dims=["x"])}
            ),
        }
    )


def test_compare_exits_zero_when_within_tolerance(monkeypatch):
    tree = _tree([0.0, 0.0, 0.0])
    monkeypatch.setattr("srm.cli._open_output_datatree", lambda uri, branch=None, tag=None: tree)
    result = runner.invoke(
        app, ["compare", "s3://a.icechunk", "s3://b.icechunk", "--branch", "main"]
    )
    assert result.exit_code == 0, result.stdout


def test_compare_exits_one_when_drifted(monkeypatch):
    trees = {"a": _tree([0.0, 0.0, 0.0]), "b": _tree([0.0, 0.0, 5.0])}
    calls = iter(["a", "b"])
    monkeypatch.setattr(
        "srm.cli._open_output_datatree",
        lambda uri, branch=None, tag=None: trees[next(calls)],
    )
    result = runner.invoke(
        app, ["compare", "s3://a.icechunk", "s3://b.icechunk", "--branch", "main"]
    )
    assert result.exit_code == 1, result.stdout


def test_snapshot_config_dir_loads_expected_scope():
    from srm.cli import load_configs

    configs, options = load_configs("configs/snapshot/cesm2-waccm/")
    all_vars = {"tas", "pr", "rsds", "hurs", "tasmax", "tasmin", "dtr"}
    by_scenario = {}
    for c in configs:
        by_scenario.setdefault(c.scenario, {"members": set(), "vars": set()})
        by_scenario[c.scenario]["members"].add(c.ensemble_member)
        by_scenario[c.scenario]["vars"].add(c.variable)

    assert set(by_scenario) == {"G6-1.5K", "SSP245"}
    assert by_scenario["G6-1.5K"]["members"] == {"003"}
    assert by_scenario["G6-1.5K"]["vars"] == all_vars
    assert by_scenario["SSP245"]["members"] == {"008"}
    assert by_scenario["SSP245"]["vars"] == all_vars
    assert all(tuple(c.subset_bounds) == (-35.0, -22.0, 16.0, 33.0) for c in configs)
