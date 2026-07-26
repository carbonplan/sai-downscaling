# How to Compare a Run Against the Snapshot

This guide shows you how to check a candidate BCSD run against the canonical global snapshot before merging modeling changes. The whole check runs through the [snapshot comparison notebook](./snapshot-comparison.ipynb), which calls `srm.snapshot.compare_runs` for you, so you never write bespoke comparison code. For why the snapshot exists and how its verdict is built, see [Snapshot Regression Testing](../explanation/snapshot-testing.md).

## Before you start

- every command runs through `uv`
- Step 1 needs access to S3 and Coiled
- Step 2 needs read access to both output stores it opens

## Step 1: Produce the cheap South Africa run

The comparison reads existing output stores and does not produce them. Run the South Africa snapshot configs first:

```bash
uv run bcsd run --config-path configs/snapshot/cesm2-waccm/
```

This writes `qa` output for the G6-1.5K and SSP245 legs over the small South Africa subset. The configs cover a domain with a ~3° halo around the region of interest, which Step 2 trims away so that only cells with full neighborhoods in both the regional and the global run are compared.

The run goes to Coiled by default and finishes quickly because the subset is small. Add `--no-coiled` only if you have local source-data access and enough memory. Note which icechunk branch it writes to, the installed package version by default or `BCSD_BRANCH` if you set it, because Step 2 needs that as `candidate_branch`.

## Step 2: Run the comparison notebook

Open [`docs/how-to/snapshot-comparison.ipynb`](./snapshot-comparison.ipynb) in Jupyter. Set two values before running it:

- `candidate_branch`: the branch your Step 1 run wrote to. The default matches the snapshot branch from `baselines.py`, but the candidate and the snapshot are separate stores read on their own branches, so the two need not agree.
- `mode`: keep the default `"southafrica"`.

Run the notebook. It subsets the global snapshot to the candidate's extent with a plain per-leaf `.sel`, so a grid mismatch fails loudly rather than being quietly reconciled, then trims both runs to `roi_bounds` and compares what remains under the per-variable tolerances. The first cells print an overall PASS/FAIL and a per-leaf table, and the rest draw difference maps, a fraction-over-tolerance heatmap, and value distributions.

Treat `hurs`, `rsds`, and `dtr` as uninformative at this step. Their bias-correction fits amplify the tiny input differences that windowing the domain introduces, so they can move without any real change, and a wider halo does not help. Judge those three from the global run in Step 3; the South Africa check is a reliable signal for `tas`, `tasmax`, `tasmin`, and `pr`.

When the notebook has run, commit it **with its outputs** to your pull request. Those committed outputs are the evidence that the check ran and what it showed, which is what a reviewer reads and what the `snapshot-verified` label attests to.

## Step 3: Decide whether to merge

The comparison is a judgment aid, not an automatic pass/fail, so you read the result and decide. Handle the two cases as follows:

- **No change, and none expected.** Every leaf is within tolerance and you did not intend to change the outputs, so the change is safe to merge. The committed notebook records the clean comparison.
- **A change appears, or the change was intended.** The cheap South Africa check is not enough on its own. Produce a full global run, set `mode = "global"` in the notebook, and expand it with the global-versus-global comparison. Document what changed and why in the pull request, get sign-off from Claire or Ori, then merge. After merging, repoint `CESM2_WACCM_GLOBAL` in `src/srm/snapshot/baselines.py` at the new global run so it becomes the baseline for the next comparison.

## Step 4: Satisfy CI

The `snapshot-required` check blocks a pull request that touches modeling code until it carries the `snapshot-verified` label. Add that label once you have committed the comparison notebook and made the merge decision above, and the check turns green.

The gate fires only on changes under `src/srm/`, and even there it skips the snapshot harness (`snapshot/`), the ETL packages (`input_data/`), and the plumbing modules that do not affect model output (`analysis`, `batch_runner`, `cache`, `cli`, `encoding`, `plotting`, `validation`, `qaqc`). Changes to configs, dependencies, tests, docs, and notebooks never trigger it, so committing the notebook does not by itself require a label.

## See Also

- [Snapshot Regression Testing](../explanation/snapshot-testing.md): why the snapshot exists, the tolerance model, and what a passing report means.
- [Snapshot comparison notebook](./snapshot-comparison.ipynb): the difference maps, heatmap, and distributions you run in Step 2.
