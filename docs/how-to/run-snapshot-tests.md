# How to Compare a Run Against the Snapshot

This guide shows you how to check a candidate BCSD run against the snapshot baseline before merging modeling changes. It is the concrete, per-pull-request procedure; for the background on why the snapshot exists and how the comparison works, see [Snapshot Regression Testing](../explanation/snapshot-testing.md). The whole check runs through the [snapshot comparison notebook](./snapshot-comparison.ipynb), which calls `srm.snapshot.compare_runs` under the hood, so you never write bespoke comparison code yourself.

Every command below runs through `uv`, and the produce step needs access to S3 and Coiled. The comparison notebook needs read access to the two output stores it opens.

## Step 1 — Produce the cheap South Africa run

The comparison reads existing output stores; it does not produce them. Run the South Africa snapshot configs first, which write the `qa` output for the G6-1.5K and SSP245 legs over the small South Africa subset, so the check is cheap to produce and cheap to diff. These configs run over a domain with a ~3° **halo** around the region of interest, because BCSD's regridding and spatial disaggregation have edge effects at a truncated domain boundary; the comparison trims that halo away (Step 2). The baseline shares the halo, so trimming is a carry-over from the global-baseline era and only narrows coverage; see issue #592 review notes.

```bash
uv run bcsd run --config-path configs/snapshot/cesm2-waccm6/ \
  --executor "$(uv run python -c 'from srm.snapshot.baselines import CESM2_WACCM_SOUTH_AFRICA as b; print(b.executor)')"
```

**Match the baseline's executor.** `PipelineOptions.executor` defaults to `coiled`, and the snapshot configs do not override it, so running without the flag produces a Coiled candidate whatever the baseline is. That matters: the executor is visible in the answers. A Coiled run and an AWS Batch run of identical code differ on `dtr`, `pr` and derived `tasmin` by up to 0.00003 K, the smallest gap the stored format can represent at that temperature. Under the default exact-equality verdict those 22 leaves fail on their own, with no code change involved, and a gate that fails for a reason unrelated to your work is one people learn to skim past. `baselines.py` records the executor for each baseline so you can read it rather than guess; the command above reads it directly.

The run finishes quickly because the subset is small. Add `--executor local` only if you have local source-data access and enough memory, and expect the same class of difference against either remote baseline. The run writes to the icechunk branch `bcsd run` uses — the installed package version by default, or `BCSD_BRANCH` if you set it — which Step 2 needs as `candidate_branch`.

## Step 2 — Run the comparison notebook

Open [`docs/how-to/snapshot-comparison.ipynb`](./snapshot-comparison.ipynb) and set `candidate_branch` to the branch your Step 1 run wrote to (the default matches the snapshot branch from `baselines.py`). The candidate and the snapshot are separate stores read on their own branches, so these need not be the same. Keep `mode = "southafrica"` (the default): the notebook reads the regional baseline from `baselines.py`, then trims both runs to `roi_bounds` (the region of interest). The default verdict is exact equality: the baseline is a regional run over the same `subset_bounds`, so any difference is a code change. The first cells print an overall PASS/FAIL and a per-leaf table; the remaining cells draw difference maps, a fraction-over-tolerance heatmap, and value distributions.

Against the regional baseline all seven variables are reliable, because both runs share an extent and feed identical inputs to the bias-correction fits. That was not true of the old global baseline: `hurs`, `rsds`, and `dtr` use iterative maximum-likelihood fits that the domain window perturbs, so they diffed spuriously and had to be judged from `mode = "global"`. See [Snapshot Regression Testing](../explanation/snapshot-testing.md) for why.

When the notebook has run, commit it **with its outputs** to your pull request. Those committed outputs are the evidence that the check ran and what it showed, which is what a reviewer reads and what the `snapshot-verified` label attests to.

## Step 3 — Decide whether to merge (issue #410)

The comparison is a judgment aid, not an automatic pass/fail, so you read the result and decide. Handle the two cases as follows:

- **No change, and none expected.** If every leaf matches exactly and you did not intend to change the outputs, the change is safe: merge the pull request. The committed notebook records the clean comparison.
- **A change appears, or the change was intended.** If any leaf differs, or your work deliberately changes the outputs, the cheap South Africa check is not enough on its own. Produce a full **global** run, set `mode = "global"` in the notebook, and pass `tolerances=TOLERANCES` to `_compare_datatrees`; a global candidate and the global baseline come from different regrid extents, so exact equality would report float32 noise as a finding. Document what changed and why in the pull request, get sign-off from Claire or Ori, then merge.

Before deciding, check that the **Run-to-run determinism** pass in the notebook is green if you set `replicate_branch`. Exact equality only means something if the pipeline is reproducible; a red determinism pass makes every other verdict unreadable.

## Step 4 — Rebaseline (automatic, one manual edit)

Both baselines are rebuilt at release time. Publishing a GitHub release triggers `.github/workflows/deploy.yml`:

| job | produces | pointer |
|---|---|---|
| `snapshot` | the regional baseline over `configs/snapshot/cesm2-waccm6/`, then freezes it as icechunk tag `snapshot-<release tag>` | `CESM2_WACCM_SOUTH_AFRICA` |
| `production` | the global run | `CESM2_WACCM_GLOBAL` |

The one manual step is repointing `src/srm/snapshot/baselines.py` at the new release. The `snapshot` job prints both fields in its workflow summary, the store URI as well as the branch. Paste the whole block: a release can move the URI too, and a pointer with a new branch on an old store resolves to a branch that does not exist. Do this in the release pull request, otherwise every subsequent comparison diffs against the previous release and inherits its already-approved changes as failures.

To rebuild a baseline outside a release, run the same two commands the job runs:

```bash
uv run bcsd run --config-path configs/snapshot/cesm2-waccm6/
uv run bcsd release --config-path configs/snapshot/cesm2-waccm6/ --tag snapshot-<name>
```

`bcsd release` creates an icechunk tag. A branch stays writable, so without the tag a later `bcsd run` carrying a matching `BCSD_BRANCH` can overwrite the baseline in place, and the next comparison would then diff a candidate against itself and report a clean pass.

## Step 5 — Satisfy CI

A pull request that touches modeling code is blocked by the `snapshot-required` check until it carries the `snapshot-verified` label. Add that label once you have committed the comparison notebook and made the merge decision above, and the check turns green.

Pull requests that touch only files under `docs/` or `notebooks/`, or only Markdown, are exempt automatically and need no label. The notebook itself lives under `docs/`, so committing it does not by itself trip the modeling-change detector.

## See Also

- [Snapshot Regression Testing](../explanation/snapshot-testing.md) — why the snapshot exists, the two baselines, and why exact equality holds only at a fixed extent.
- [Snapshot comparison notebook](./snapshot-comparison.ipynb) — the difference maps, heatmap, and distributions you run in Step 2.
