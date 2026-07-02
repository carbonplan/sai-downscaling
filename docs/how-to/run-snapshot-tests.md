# How to Run the Snapshot Regression Gate

This guide shows you how to bless a snapshot baseline, run the South Africa regression gate against it, and compare two output stores directly. It assumes you already know what the pipeline does and why snapshot testing exists — for that background, see [Snapshot Regression Testing](../explanation/snapshot-testing.md). For a visual, cell-by-cell diff of two runs, use the [Snapshot comparison notebook](./snapshot-comparison.ipynb) instead.

The gate is marked `slow` and `snapshot`, so it is excluded from the default `pytest` run and only executes when you select it with `-m snapshot`. Every step below runs through `uv`, and the produce and blessing steps need access to S3 and Coiled.

## Step 0 — Set the shared environment

The produce step and the blessing step must run against the **same** icechunk branch, and `pytest` cannot be told the branch on the command line. `bcsd run --branch …` only configures that one `run` process; the gate resolves its branch from `PipelineOptions`, which reads the `BCSD_BRANCH` environment variable (falling back to the installed package version). Export both variables once so `run`, `--snapshot-update`, and the gate all agree.

```bash
# The snapshot branch is defined once in src/srm/snapshot/baselines.py; derive it
# so local runs, CI, and the global baseline all agree.
export BCSD_BRANCH="$(uv run python -c 'from srm.snapshot.baselines import CESM2_WACCM_GLOBAL as b; print(b.branch)')"
export SNAPSHOT_STORAGE_PATH="s3://carbonplan-srm/snapshots/${BCSD_BRANCH}"
```

`BCSD_BRANCH` must be a fixed branch, not the installed package version — the default version changes on every commit (`v0.7.0.post18` → `post19` → …), so the produced output and the gate would read different branches and the gate would fail with `ref not found`. Deriving it from `baselines.py` guarantees a stable value that matches CI. Do **not** pass `--branch` to `bcsd run` instead of exporting `BCSD_BRANCH`: the flag reaches `run` but not `pytest`, which reintroduces the mismatch.

## Step 1 — Produce the South Africa output

The gate reads an existing output store; it does not produce one. Run the South Africa snapshot configs first, which write the `qa` output for the G6-1.5K and SSP245 legs over the South Africa subset.

```bash
uv run bcsd run --config-path configs/snapshot/cesm2-waccm/
```

This runs on Coiled by default and can take a while. Add `--no-coiled` only if you have local source-data access and enough memory.

## Step 2 — Bless the first snapshot

With `BCSD_BRANCH` and `SNAPSHOT_STORAGE_PATH` exported in Step 0, run the gate with `--snapshot-update`, which writes the current output as the baseline instead of comparing against it. It reads the South Africa output on `BCSD_BRANCH` — the same branch Step 1 wrote to.

```bash
uv run pytest -m snapshot tests/test_snapshot_gate.py --snapshot-update -v
```

Blessing overwrites the baseline on the current snapshot branch, so only do it deliberately. Reserve it for the first snapshot, or for when you have confirmed that a change in the output is an intended improvement rather than a regression.

### Blessing a new baseline version

The snapshot branch lives in one place: `CESM2_WACCM_GLOBAL.branch` in `src/srm/snapshot/baselines.py`. Re-blessing on the *same* branch (above) is enough for an approved change that supersedes the current baseline. To start a *new* baseline lineage instead — for example a release, per issue #410 — bump `branch` in `baselines.py` in a PR. The `snapshot` workflow derives `BCSD_BRANCH` from that value, so the South Africa gate and the global comparison both follow automatically with no workflow edit. Re-export the Step 0 variables afterward so your local shell picks up the new branch, then bless.

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

The `--branch` value is the snapshot branch defined in `src/srm/snapshot/baselines.py` (here `v0.7.0`), and it applies to both stores. Restrict the comparison with repeatable `--scenario` and `--variable` flags when you only care about specific leaves. For a global-scale comparison, run the same command near the data with `uv run coiled batch run --region us-west-2 "bcsd compare A B --branch v0.7.0"`. See the [`bcsd compare` reference](../reference/cli.md) for the full option list.

## Step 5 — Verify a pull request

A pull request that touches modeling code is blocked by the `snapshot-required` check until it is verified. To satisfy it, dispatch the `snapshot` workflow with `scope: southafrica`, confirm the run is green, then add the `snapshot-verified` label to the pull request.

Pull requests that touch only files under `docs/` or `notebooks/`, or only Markdown, are exempt automatically and need no label. Set the workflow's `update` input to `true` only when you intend to re-bless the baseline as part of the run.

## See Also

- [Snapshot Regression Testing](../explanation/snapshot-testing.md) — why the gate exists and how the tolerance and two-snapshot models work.
- [Snapshot comparison notebook](./snapshot-comparison.ipynb) — visual difference maps, drift heatmap, and distributions for two runs.
- [CLI Reference](../reference/cli.md) — the full `bcsd compare` interface.
