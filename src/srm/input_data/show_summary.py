"""Write a processed icechunk store's xarray repr to the GitHub Actions job summary."""

import os
import sys

from srm.datasets import catalog

# Maps CLI scenario names to zarr group names within the unified per-GCM store.
_SCENARIO_TO_GROUP: dict[str, str] = {
    "historical": "historical",
    "ssp245": "ssp245",
    "SSP245": "ssp245",
    "G6-1.5K": "g6_1p5k",
    "g6_1p5k": "g6_1p5k",
}

_GCM_TO_KEY: dict[str, str] = {
    "CESM2-WACCM": "CESM2-WACCM-unified-icechunk",
    "MIROC-ES2H": "MIROC-ES2H-unified-icechunk",
    "UKESM": "UKESM-unified-icechunk",
}


def main() -> None:
    gcm, scenario = sys.argv[1], sys.argv[2]

    key = _GCM_TO_KEY.get(gcm)
    group = _SCENARIO_TO_GROUP.get(scenario)
    if key is None or group is None:
        print(f"No unified store mapping for ({gcm!r}, {scenario!r}) — skipping summary")
        return

    entry = catalog.datasets.get(key)
    if entry is None:
        print(f"Catalog key {key!r} not found — skipping summary")
        return

    ds = entry.to_xarray(group=group)
    summary = f"## {gcm} / {scenario}\n\n```\n{repr(ds)}\n```\n"

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(summary)
    else:
        print(summary)


if __name__ == "__main__":
    main()
