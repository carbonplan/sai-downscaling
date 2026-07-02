# How to Compare a Run Against the Snapshot

This guide shows you how to check a candidate BCSD run against the canonical global snapshot before merging modeling changes. It is the concrete, per-pull-request procedure; for the background on why the snapshot exists and how the comparison works, see [Snapshot Regression Testing](../explanation/snapshot-testing.md). The whole check runs through the [snapshot comparison notebook](./snapshot-comparison.ipynb), which calls `srm.snapshot.compare_runs` under the hood, so you never write bespoke comparison code yourself.

Every command below runs through `uv`, and the produce step needs access to S3 and Coiled. The comparison notebook needs read access to the two output stores it opens.

## Step 1 — Produce the cheap South Africa run

The comparison reads existing output stores; it does not produce them. Run the South Africa snapshot configs first, which write the `qa` output for the G6-1.5K and SSP245 legs over the small South Africa subset, so the check is cheap to produce and cheap to diff.

```bash
uv run bcsd run --config-path configs/snapshot/cesm2-waccm/
```

This runs on Coiled by default and finishes quickly because the subset is small. Add `--no-coiled` only if you have local source-data access and enough memory.

## Step 2 — Run the comparison notebook

Open [`docs/how-to/snapshot-comparison.ipynb`](./snapshot-comparison.ipynb) and run all cells. Keep `mode = "southafrica"` (the default): the notebook subsets the global snapshot to the South Africa candidate's extent with a plain per-leaf `.sel` — a grid mismatch fails loudly rather than being reconciled — so the regional run is compared against the global baseline under the per-variable tolerances. The first cells print an overall PASS/FAIL and a per-leaf table; the remaining cells draw difference maps, a fraction-over-tolerance heatmap, and value distributions.

When the notebook has run, commit it **with its outputs** to your pull request. Those committed outputs are the evidence that the check ran and what it showed, which is what a reviewer reads and what the `snapshot-verified` label attests to.

## Step 3 — Decide whether to merge (issue #410)

The comparison is a judgment aid, not an automatic pass/fail, so you read the result and decide. Handle the two cases as follows:

- **No change, and none expected.** If every leaf is within tolerance and you did not intend to change the outputs, the change is safe: merge the pull request. The committed notebook records the clean comparison.
- **A change appears, or the change was intended.** If any leaf moves out of tolerance, or your work deliberately changes the outputs, the cheap South Africa check is not enough on its own. Produce a full **global** run, set `mode = "global"` in the notebook, and expand it with the global-vs-global comparison. Document what changed and why in the pull request, get sign-off from Claire or Ori, then merge. After merging, repoint `CESM2_WACCM_GLOBAL` in `src/srm/snapshot/baselines.py` at the new global run so it becomes the baseline for the next comparison.

## Step 4 — Satisfy CI

A pull request that touches modeling code is blocked by the `snapshot-required` check until it carries the `snapshot-verified` label. Add that label once you have committed the comparison notebook and made the merge decision above, and the check turns green.

Pull requests that touch only files under `docs/` or `notebooks/`, or only Markdown, are exempt automatically and need no label. The notebook itself lives under `docs/`, so committing it does not by itself trip the modeling-change detector.

## See Also

- [Snapshot Regression Testing](../explanation/snapshot-testing.md) — why the snapshot exists, the single-global-snapshot model, and the tolerance model.
- [Snapshot comparison notebook](./snapshot-comparison.ipynb) — the difference maps, heatmap, and distributions you run in Step 2.
