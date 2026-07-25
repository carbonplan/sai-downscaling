#!/usr/bin/env python
"""Pre-commit guard: reject Jupyter notebook cell *outputs* that contain cluster credentials.

Executed Coiled/dask QA notebooks embed a live cluster access token and scheduler URL in the
cluster-creation cell's output (the Coiled cluster card and the ``frisky.Client`` repr). Committing
those outputs leaks the token. This hook scans the outputs of every passed ``.ipynb`` and fails if a
known credential marker is present, so the leak is caught before it lands in a commit. QA-result
outputs (tables, maps, printed summaries) are never affected — only the connection markers below are.
"""

import json
import sys

# Substrings that appear only in Coiled/dask/frisky connection info, never in QA results.
SECRET_MARKERS = ("token=", "dask.host", "frisky-comm", "__frisky_dial_host")


def offending_cells(path: str) -> list[tuple[int, list[str]]]:
    """Find notebook cells whose outputs carry a credential marker.

    Parameters
    ----------
    path : str
        Path to a ``.ipynb`` file.

    Returns
    -------
    list of tuple of (int, list of str)
        One entry per offending cell: its position in the notebook and the markers found.
    """
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    hits: list[tuple[int, list[str]]] = []
    for index, cell in enumerate(nb.get("cells", [])):
        outputs = cell.get("outputs")
        if not outputs:
            continue
        blob = json.dumps(outputs)
        found = [marker for marker in SECRET_MARKERS if marker in blob]
        if found:
            hits.append((index, found))
    return hits


def main(argv: list[str]) -> int:
    """Scan each ``.ipynb`` argument; return 1 if any output holds a credential marker."""
    failed = False
    for path in argv:
        if not path.endswith(".ipynb"):
            continue
        try:
            hits = offending_cells(path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: could not scan {path}: {exc}", file=sys.stderr)
            continue
        for index, markers in hits:
            failed = True
            print(
                f"{path}: cell {index} output contains cluster credentials {markers}",
                file=sys.stderr,
            )
    if failed:
        print(
            "\nCluster tokens/URLs found in notebook outputs. Clear the cluster-creation cell's "
            "output before committing (keep the result cells) -- e.g. in Jupyter, or with nbformat "
            "set that cell's `outputs` to [] -- then re-stage.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
