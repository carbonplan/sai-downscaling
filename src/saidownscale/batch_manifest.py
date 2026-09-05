"""
S3 manifest carrying per-task configuration for AWS Batch array jobs.

AWS Batch array jobs share one job definition and one set of container overrides across
every child; only ``AWS_BATCH_JOB_ARRAY_INDEX`` differs. That rules out the per-task
environment variable Coiled Batch supplies through ``map_over_task_var_dicts``. Instead,
each submission writes one manifest and every child reads its own entry by index.

Entries are byte-identical to the ``CONFIG_JSON`` payload the Coiled path uses, so
``batch_runner`` needs a new way to obtain the dict, not a new way to parse it.
"""

from __future__ import annotations

import json

import fsspec

#: Bumped whenever the manifest layout changes incompatibly.
MANIFEST_VERSION = 1


def build_manifest(stage: str, entries: list[dict]) -> dict:
    """
    Assemble a manifest document.

    Parameters
    ----------
    stage : str
        Pipeline stage the entries belong to.
    entries : list of dict
        One ``CONFIG_JSON``-shaped payload per array child, in array-index order.

    Returns
    -------
    dict
        Manifest document ready to serialize.
    """
    return {"version": MANIFEST_VERSION, "stage": stage, "entries": entries}


def write_manifest(uri: str, stage: str, entries: list[dict]) -> str:
    """
    Serialize a manifest to ``uri``.

    Parameters
    ----------
    uri : str
        Destination understood by ``fsspec`` (local path or ``s3://`` URI).
    stage : str
        Pipeline stage the entries belong to.
    entries : list of dict
        One payload per array child, in array-index order.

    Returns
    -------
    str
        The ``uri`` written, for convenient chaining into job submission.
    """
    with fsspec.open(uri, "w", auto_mkdir=True) as handle:
        json.dump(build_manifest(stage, entries), handle)
    return uri


def read_manifest_entry(uri: str, index: int) -> dict:
    """
    Read one entry out of a manifest.

    Parameters
    ----------
    uri : str
        Manifest location understood by ``fsspec``.
    index : int
        Array index of the entry to return.

    Returns
    -------
    dict
        The ``CONFIG_JSON``-shaped payload for this array child.

    Raises
    ------
    ValueError
        If the manifest version is not recognized.
    IndexError
        If ``index`` falls outside the manifest's entries.
    """
    with fsspec.open(uri, "r") as handle:
        manifest = json.load(handle)

    version = manifest.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported manifest version {version!r} at {uri}; expected {MANIFEST_VERSION}"
        )

    entries = manifest["entries"]
    if not 0 <= index < len(entries):
        raise IndexError(f"Manifest index {index} out of range; {uri} holds {len(entries)} entries")
    return entries[index]
