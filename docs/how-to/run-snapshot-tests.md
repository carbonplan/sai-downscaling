# How to Run the Snapshot Regression Gate

This guide shows you how to bless a snapshot baseline, run the South Africa regression gate against it, and compare two output stores directly. It assumes you already know what the pipeline does and why snapshot testing exists — for that background, see [Snapshot Regression Testing](../explanation/snapshot-testing.md). For a visual, cell-by-cell diff of two runs, use the [Snapshot comparison notebook](./snapshot-comparison.ipynb) instead.

The gate is marked `slow` and `snapshot`, so it is excluded from the default `pytest` run and only executes when you select it with `-m snapshot`. Every step below runs through `uv`, and the produce and blessing steps need access to S3 and Coiled.

## Step 1 — Produce the South Africa output

The gate reads an existing output store; it does not produce one. Run the South Africa snapshot configs first, which write the `qa` output for the G6-1.5K and SSP245 legs over the South Africa subset.

```bash
uv run bcsd run --config-path configs/snapshot/cesm2-waccm/
```

This runs on Coiled by default and can take a while. Add `--no-coiled` only if you have local source-data access and enough memory.

## Step 2 — Bless the first snapshot

Point `SNAPSHOT_STORAGE_PATH` at the snapshot store on `carbonplan-srm`, versioned to match the installed package. Then run the gate with `--snapshot-update`, which writes the current output as the baseline instead of comparing against it.

```bash
export SNAPSHOT_STORAGE_PATH="s3://carbonplan-srm/snapshots/$(uv run python -c 'import importlib.metadata as m; print(m.version("srm"))')"
uv run pytest -m snapshot tests/test_snapshot_gate.py --snapshot-update -v
```

Blessing overwrites the baseline for this package version, so only do it deliberately. Reserve it for the first snapshot, or for when you have confirmed that a change in the output is an intended improvement rather than a regression.

## Step 3 — Run the gate

With a baseline blessed, run the gate without `--snapshot-update`. It must be green: every scenario group present in the output store is compared against the stored snapshot under the per-variable tolerances.

```bash
uv run pytest -m snapshot tests/test_snapshot_gate.py -v
```

A failure prints a per-leaf table showing which `(scenario, variable)` leaves moved out of tolerance, along with their maximum absolute difference and the fraction of cells over tolerance. If the change is a regression, fix it; if it is intended, re-bless with Step 2.

## Step 4 — Compare two output stores directly

To compare any two output stores outside the pytest gate — for example a fresh run against a blessed baseline — use `bcsd compare`. It prints a difference table and exits `1` if any leaf is out of tolerance, so it works both interactively and in scripts.

```bash
uv run bcsd compare \
  s3://carbonplan-scratch/srm/output/qa/CESM2-WACCM-ERA5-lat-35to-22_lon16to33.icechunk \
  s3://carbonplan-srm/output/production/CESM2-WACCM-ERA5-global.icechunk \
  --branch v0.7.0
```

Restrict the comparison with repeatable `--scenario` and `--variable` flags when you only care about specific leaves. For a global-scale comparison, run the same command near the data with `uv run coiled batch run --region us-west-2 "bcsd compare A B --branch v0.7.0"`. See the [`bcsd compare` reference](../reference/cli.md) for the full option list.

## Step 5 — Verify a pull request

A pull request that touches modeling code is blocked by the `snapshot-required` check until it is verified. To satisfy it, dispatch the `snapshot` workflow with `scope: southafrica`, confirm the run is green, then add the `snapshot-verified` label to the pull request.

Pull requests that touch only files under `docs/` or `notebooks/`, or only Markdown, are exempt automatically and need no label. Set the workflow's `update` input to `true` only when you intend to re-bless the baseline as part of the run.

## See Also

- [Snapshot Regression Testing](../explanation/snapshot-testing.md) — why the gate exists and how the tolerance and two-snapshot models work.
- [Snapshot comparison notebook](./snapshot-comparison.ipynb) — visual difference maps, drift heatmap, and distributions for two runs.
- [CLI Reference](../reference/cli.md) — the full `bcsd compare` interface.
