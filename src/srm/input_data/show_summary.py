"""Write a processed icechunk store's xarray repr to the GitHub Actions job summary."""

import os
import sys

from srm.datasets import Dataset, catalog

_KEY_MAP: dict[tuple[str, str], str] = {
    ("CESM2-WACCM", "pangeo-historical"): "pangeo-CESM2-WACCM-historical-icechunk",
    ("CESM2-WACCM", "historical"): "CESM2-WACCM-historical-icechunk",
    ("CESM2-WACCM", "ssp245"): "CESM2-WACCM-SSP245-icechunk",
    ("CESM2-WACCM", "G6-1.5K"): "CESM2-WACCM-G6-1.5K-icechunk",
    ("MIROC-ES2H", "historical"): "MIROC-ES2H-historical-icechunk",
    ("MIROC-ES2H", "ssp245"): "MIROC-ES2H-SSP245-icechunk",
    ("MIROC-ES2H", "G6-1.5K"): "MIROC-ES2H-G6-1.5K-icechunk",
    ("MIROC-ES2H", "baseline"): "MIROC-ES2H-baseline-icechunk",
    ("UKESM", "historical"): "UKESM-historical-icechunk",
    ("UKESM", "SSP245"): "UKESM-SSP245-icechunk",
    ("UKESM", "SSP245-t-pr"): "UKESM-SSP245-t-pr-icechunk",
    ("UKESM", "G6-1.5K"): "UKESM-G6-1.5K-icechunk",
    ("UKESM", "G6-1.5K-t-pr"): "UKESM-G6-1.5K-t-pr-icechunk",
}


def main() -> None:
    gcm, scenario = sys.argv[1], sys.argv[2]

    key = _KEY_MAP.get((gcm, scenario))
    if key is None:
        print(f"No catalog key for ({gcm!r}, {scenario!r}) — skipping summary")
        return

    entry = catalog.datasets.get(key)
    if entry is None:
        print(f"Catalog key {key!r} not found — skipping summary")
        return

    ds = (
        entry.to_xarray(convert_calendar=False) if isinstance(entry, Dataset) else entry.to_xarray()
    )
    summary = f"## {gcm} / {scenario}\n\n```\n{repr(ds)}\n```\n"

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(summary)
    else:
        print(summary)


if __name__ == "__main__":
    main()
