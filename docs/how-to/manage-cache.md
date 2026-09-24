---
orphan: true
---

# Manage the cache

If you want to resume an interrupted run, force a recompute, or find out what a run already wrote,
this page covers it. We keep 2 icechunk stores for each combination of global climate model (GCM),
observation dataset, and spatial subset: one for intermediate artifacts and one for final outputs.
Every artifact within a store is written as a zarr group on a named branch, which defaults to the
installed package version.

## Cache store locations

```
# Scratch store: obs regridded plus optional intermediates
s3://carbonplan-srm/scratch/cache/{environment}/{gcm}-{obs_dataset}-{subset_id}.icechunk
  branch: v1.2.3   ← defaults to installed package version

# Output store: fine-res historical, scenario results, and debiased coarse data
s3://carbonplan-srm/scratch/output/{environment}/{gcm}-{obs_dataset}-{subset_id}.icechunk
  branch: v1.2.3

# Examples for CESM2-WACCM6, ERA5, global run:
s3://carbonplan-srm/scratch/cache/qa/CESM2-WACCM6-ERA5-global.icechunk
s3://carbonplan-srm/scratch/output/qa/CESM2-WACCM6-ERA5-global.icechunk

# Regional subset (South Africa):
s3://carbonplan-srm/scratch/cache/qa/CESM2-WACCM6-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk
s3://carbonplan-srm/scratch/output/qa/CESM2-WACCM6-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk
```

The paths above are the defaults, which every non-production config uses. Production configs
override `output_dir` to CarbonPlan's Source Cooperative repository and keep only their intermediate
artifacts under `scratch/cache/production/`, so nothing on this page deletes published data. Another
2 config families override both directories to avoid resolving to a shared store:
`configs/snapshot/` writes under `scratch/snapshot/`, and `configs/qa/obs-comparison/` writes under
`scratch/obs-comparison/`.

Within each store, artifacts live as zarr groups. Every group below is written on a normal run,
with one exception: `dtr` stops after its `debiased_coarse` group and never gets a fine one, as
[`dtr` is bias-corrected but not published](../methods/pipeline-architecture.md#dtr-is-bias-corrected-but-not-published)
explains.

| Store | Group pattern | Stage | Always written |
|-------|---------------|-------|----------------|
| scratch | `obs/{variable}` | Stage 1 | yes |
| output | `{method}/historical/{variable}/{hist_member}` | Stage 2 | yes |
| output | `{method}/{scenario_group}/{variable}/{member}` | Stage 3 | yes |
| output | `{method}/debiased_coarse/historical/{variable}/{hist_member}` | Stage 2 | yes |
| output | `{method}/debiased_coarse/{scenario_group}/{variable}/{member}` | Stage 3 | yes |

`{method}` is `bcsd` or `qdmsd`, the lowercase `downscaling_method`. Every group above except
`obs/{variable}` is namespaced under it, so both methods can write to the same store and share
one observation regrid. Stores written before this change have no such segment; their groups
begin directly with `historical/`, the scenario group, or `debiased_coarse/`.

Those 5 groups are the primary artifacts. The `debiased_coarse` groups expose the GCM data after
bias correction but before spatial disaggregation, at the native coarse GCM resolution of roughly
1° to 2°. `hist_member` is the resolved historical parent ensemble member (see
[ensemble member lineage](ensemble-member-lineage.ipynb)); for most variables it equals `member`,
but for stratospheric aerosol injection (SAI) scenarios the 2 can differ.

## Intermediate artifacts

The scratch store holds 3 more groups, which appear only when you set `save_intermediate: true` in
your config. They capture the pipeline state between computation steps, which lets you debug
detrending behavior without re-running the full stage.

| Group pattern | Written by | Contents |
|---------------|------------|----------|
| `{method}/detrended_scenario/{scenario_group}/{variable}/{member}` | `transform_scenario` | Scenario data after detrending (9-year running mean removed) |
| `{method}/trend_scenario/{scenario_group}/{variable}/{member}` | `transform_scenario` | The trend signal extracted during detrending (added back after bias correction) |
| `{method}/debiased_scenario/{scenario_group}/{variable}/{member}` | `transform_scenario` | Scenario after bias correction, before re-trending |

Not every variable writes every intermediate, because only the detrended variables have a trend to
save. The table below gives each one, and all of it still requires `save_intermediate: true`.

| Variable | `detrended_scenario` | `trend_scenario` | `debiased_scenario` |
| --- | --- | --- | --- |
| `tas` | yes | yes | yes |
| `tasmax` | yes | yes | yes |
| `tasmin` | no | no | no |
| `pr` | no | no | yes |
| `rsds` | no | no | yes |
| `dtr` | no | no | yes |
| `hurs` | no | no | yes |

:::{admonition} `tasmin` dependency on `dtr` and `tasmax` coarse outputs
:class: note

`fit_historical_tasmin` and `transform_scenario_tasmin` derive `tasmin` as `tasmax − dtr`, where
`dtr` is the diurnal temperature range. `fit_historical` and `transform_scenario` dispatch to these
variants automatically whenever `variable == "tasmin"`, so you need no special flag or config. After
spatial disaggregation, we reconcile `tasmin` against the fine `tasmax` so that `tasmax >= tasmin`
everywhere (issue #331). Both variants read the groups below as **hard dependencies** from the
output store, and the `dtr` and `tasmax` stages write them unconditionally.

| Stage | Reads from output store |
| --- | --- |
| `fit_historical_tasmin` | `{method}/debiased_coarse/historical/dtr/{hist_member}`, `{method}/debiased_coarse/historical/tasmax/{hist_member}` |
| `transform_scenario_tasmin` | `{method}/debiased_coarse/{group}/dtr/{member}`, `{method}/debiased_coarse/{group}/tasmax/{member}` |

Run `dtr` and `tasmax` before `tasmin`, with no `save_intermediate` flag required:

```bash
# 1. Run dtr and tasmax (coarse outputs written automatically)
uv run saidownscale run --config-path configs/dtr.yaml
uv run saidownscale run --config-path configs/tasmax.yaml

# 2. Now run tasmin (reads debiased_coarse groups written above)
uv run saidownscale run --config-path configs/tasmin.yaml
```

:::

## Resumability

If you interrupt a run and restart it with the same config, the pipeline skips the completed stages.
It decides by checking whether the corresponding zarr group already exists in the branch ancestry:

```bash
# Start run
uv run saidownscale run --config-path configs/example.yaml
^C  # Interrupt after stage 1 completes

# Check what's cached
uv run saidownscale status --config-path configs/example.yaml

# Resume (automatically skips completed stages)
uv run saidownscale run --config-path configs/example.yaml
# Only runs stages 2 and 3
```

## Force recompute

Pass `--force` to recompute an artifact that is already cached. Narrow it with `--stage` when you
want to rebuild one stage and keep the rest:

```bash
# Force all stages
uv run saidownscale run --config-path configs/example.yaml --force

# Force only the scenario stage (reuses existing obs and historical artifacts)
uv run saidownscale run --config-path configs/example.yaml --stage transform_scenario --force
```

## Check cache status

Use `saidownscale status` to see which artifacts are complete for your configs. It reads the same
cache the orchestrator does, so what it reports is what a run would skip:

```bash
uv run saidownscale status --config-path configs/example.yaml --verbose
```

The output shows a per-stage progress table and, with `--verbose`, the resolved store paths. See
[`saidownscale status`](../reference/cli.md#saidownscale-status-check-cache-status) in the CLI
reference for the full format.

## List cached artifacts

`cache-list` prints the groups committed on the branch a config resolves to. The filters narrow the
listing without changing which branch it reads:

```bash
# List all groups on the current branch for all stores
uv run saidownscale cache-list --config-path configs/example.yaml

# Filter by stage
uv run saidownscale cache-list --config-path configs/example.yaml --stage obs

# Filter by GCM and variable
uv run saidownscale cache-list --config-path configs/example.yaml --gcm CESM2-WACCM6 --variable tas
```

The filters combine, so you can narrow to one stage and one variable at once. See
[`saidownscale cache-list`](../reference/cli.md#saidownscale-cache-list-list-cached-artifacts) in
the CLI reference for every option.

## Clear cache

`cache-clear` deletes groups rather than listing them, so check with `cache-list` first. The command
prompts for confirmation unless you pass `--yes`:

```bash
# Clear all cache for the environment/branch in the config (prompts for confirmation)
uv run saidownscale cache-clear --config-path configs/example.yaml

# Clear a specific stage without prompting
uv run saidownscale cache-clear --config-path configs/example.yaml --stage scenarios --yes

# Clear only a specific GCM
uv run saidownscale cache-clear --config-path configs/example.yaml --gcm CESM2-WACCM6 --yes
```

:::{admonition} Environment-scoped clearing
:class: warning

Cache clearing respects the `environment` setting in your config. A config with
`environment: "production"` clears only the production cache, never qa.
:::

The same filters apply here as to `cache-list`, which is what makes a dry `cache-list` run worth
doing first. See
[`saidownscale cache-clear`](../reference/cli.md#saidownscale-cache-clear-clear-cache) in the CLI
reference for every option.

## Programmatic cache inspection

If you want to inspect cached artifacts from Python rather than the CLI, use `ArtifactCache`
directly. It is the same class the orchestrator and the pipeline use:

```python
import yaml
from saidownscale.downscaling_config import DownscalingConfig, PipelineOptions
from saidownscale.cache import ArtifactCache

raw = yaml.safe_load(open("configs/example.yaml"))
config = DownscalingConfig(**raw)
options = PipelineOptions(**raw)
cache = ArtifactCache.from_config(config, options)

# Check whether the obs artifact exists for this config
obs_loc = cache.obs_loc
print(f"Obs cached: {cache.exists(obs_loc)}")
print(f"Store: {obs_loc.store_path}")
print(f"Group: {obs_loc.group}")

# List all groups committed on the current branch of this store
groups = cache.list_groups_on_branch(obs_loc.store_path)
print(f"All cached groups: {groups}")

# Inspect both scratch and output stores at once
intermediates = cache.list_intermediate_groups()
for store_path, group_list in intermediates.items():
    print(store_path)
    for g in group_list:
        print(f"  {g}")
```

`list_intermediate_groups` classifies each group by stripping its leading `{method}` segment before
matching against the intermediate-prefix list. It recognizes both the namespaced
(`{method}/detrended_scenario/...`) and pre-namespace (`detrended_scenario/...`) forms, so the
summary above works unchanged against branches written before and after that change.

## See also

These 2 pages cover the ground next to this one:

| Page | What it covers |
| --- | --- |
| [Pipeline architecture](../methods/pipeline-architecture.md) | How we designed the cache, and why |
| [CLI reference](../reference/cli.md) | Full option listings for `status`, `cache-list`, and `cache-clear` |
