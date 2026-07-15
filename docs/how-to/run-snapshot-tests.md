# How to Compare a Run Against the Snapshot

This guide shows you how to check a candidate BCSD run against the canonical global snapshot before merging modeling changes. It is the concrete, per-pull-request procedure; for the background on why the snapshot exists and how the comparison works, see [Snapshot Regression Testing](../explanation/snapshot-testing.md). The whole check runs through the [snapshot comparison notebook](./snapshot-comparison.ipynb), which calls `srm.snapshot.compare_runs` under the hood, so you never write bespoke comparison code yourself.

Every command below runs through `uv`, and the produce step needs access to S3 and Coiled. The comparison notebook needs read access to the two output stores it opens.

## Step 1 — Produce the cheap South Africa run

The comparison reads existing output stores; it does not produce them. Run the South Africa snapshot configs first, which write the `qa` output for the G6-1.5K and SSP245 legs over the small South Africa subset, so the check is cheap to produce and cheap to diff. These configs run over a domain with a ~3° **halo** around the region of interest, because BCSD's regridding and spatial disaggregation have edge effects at a truncated domain boundary; the comparison trims that halo away (Step 2), so only interior cells — which had full neighborhoods in both the regional and the global run — are compared.

```bash
uv run bcsd run --config-path configs/snapshot/cesm2-waccm/
```

This runs on Coiled by default and finishes quickly because the subset is small. Add `--no-coiled` only if you have local source-data access and enough memory. The run writes to the icechunk branch `bcsd run` uses — the installed package version by default, or `BCSD_BRANCH` if you set it — which Step 2 needs as `candidate_branch`.

## Step 2 — Run the comparison notebook

Open [`docs/how-to/snapshot-comparison.ipynb`](./snapshot-comparison.ipynb) and set `candidate_branch` to the branch your Step 1 run wrote to (the default matches the snapshot branch from `baselines.py`). The candidate and the snapshot are separate stores read on their own branches, so these need not be the same. Keep `mode = "southafrica"` (the default): the notebook subsets the global snapshot to the South Africa candidate's extent with a plain per-leaf `.sel` — a grid mismatch fails loudly rather than being reconciled — then trims both runs to `roi_bounds` (the region of interest), dropping the halo so only interior cells are compared under the per-variable tolerances. The first cells print an overall PASS/FAIL and a per-leaf table; the remaining cells draw difference maps, a fraction-over-tolerance heatmap, and value distributions.

Note that `hurs`, `rsds`, and `dtr` use parametric bias-correction fits that amplify the tiny input differences introduced by windowing the domain, so they can diff spuriously in the South Africa proxy even with no real change (a wider halo does not help — the differences are interior, not at the boundary). Judge those three from the full global run in Step 3 (`mode = "global"`); the South Africa check is a reliable signal for the numerically stable variables (`tas`, `tasmax`, `tasmin`, `pr`). See [Snapshot Regression Testing](../explanation/snapshot-testing.md) for why.

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
