"""Unit tests for the AWS Batch task-config manifest."""

from __future__ import annotations

import json

import pytest

from saidownscale.batch_manifest import (
    MANIFEST_VERSION,
    build_manifest,
    read_manifest_entry,
    write_manifest,
)


def test_build_manifest_records_version_and_stage():
    manifest = build_manifest("transform_scenario", [{"variable": "tas"}])
    assert manifest["version"] == MANIFEST_VERSION
    assert manifest["stage"] == "transform_scenario"
    assert manifest["entries"] == [{"variable": "tas"}]


def test_round_trip_returns_entry_at_index(tmp_path):
    uri = str(tmp_path / "manifest.json")
    entries = [{"variable": "tas"}, {"variable": "pr"}, {"variable": "rsds"}]
    write_manifest(uri, "fit_historical", entries)
    assert read_manifest_entry(uri, 1) == {"variable": "pr"}


def test_write_manifest_returns_the_uri(tmp_path):
    uri = str(tmp_path / "manifest.json")
    assert write_manifest(uri, "fit_historical", [{"a": 1}]) == uri


def test_rejects_unknown_manifest_version(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 999, "stage": "x", "entries": [{}]}))
    with pytest.raises(ValueError, match="version"):
        read_manifest_entry(str(path), 0)


@pytest.mark.parametrize("index", [-1, 3])
def test_rejects_out_of_range_index(tmp_path, index):
    uri = str(tmp_path / "manifest.json")
    write_manifest(uri, "fit_historical", [{"a": 1}, {"a": 2}, {"a": 3}])
    with pytest.raises(IndexError, match="index"):
        read_manifest_entry(uri, index)
