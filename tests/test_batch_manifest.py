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


def test_manifest_round_trip_returns_entry_at_index(tmp_path):
    entries = [{"variable": "tas"}, {"variable": "pr"}, {"variable": "rsds"}]
    assert build_manifest("fit_historical", entries) == {
        "version": MANIFEST_VERSION,
        "stage": "fit_historical",
        "entries": entries,
    }
    uri = str(tmp_path / "manifest.json")
    assert write_manifest(uri, "fit_historical", entries) == uri
    assert read_manifest_entry(uri, 1) == {"variable": "pr"}


def test_rejects_unknown_manifest_version(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 999, "stage": "x", "entries": [{}]}))
    with pytest.raises(ValueError, match="version"):
        read_manifest_entry(str(path), 0)


def test_rejects_out_of_range_index(tmp_path, subtests):
    uri = str(tmp_path / "manifest.json")
    write_manifest(uri, "fit_historical", [{"a": 1}, {"a": 2}, {"a": 3}])
    for index in (-1, 3):
        with subtests.test(index=index):
            with pytest.raises(IndexError, match="index"):
                read_manifest_entry(uri, index)
