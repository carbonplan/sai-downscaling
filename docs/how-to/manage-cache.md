# How to Manage the Cache

The pipeline uses two icechunk stores per `(GCM, obs-dataset, spatial-subset)` combination — one
for intermediate artifacts and one for final outputs. All artifacts within a store are written as
zarr groups on a named branch (by default the installed package version).

## Cache Store Locations

```
# Scratch store — obs regridded + optional intermediates
s3://carbonplan-srm/scratch/cache/{environment}/{gcm}-{obs_dataset}-{subset_id}.icechunk
  branch: v1.2.3   ← defaults to installed package version

# Output store — fine-res historical + scenario results + debiased coarse data
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
override `output_dir` to CarbonPlan's Source Cooperative repository and keep only their
intermediate artifacts under `scratch/cache/production/`, so nothing in this how-to deletes
published data. Two config families also override both directories to avoid resolving to a shared
store: `configs/snapshot/` writes under `scratch/snapshot/` and `configs/qa/obs-comparison/` writes
under `scratch/obs-comparison/`.

Within each store the zarr groups are:

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

These five are the primary artifacts — written unconditionally on every run. The `debiased_coarse`
groups expose the GCM data after bias correction but before spatial disaggregation, at the native
coarse GCM resolution (~1–2°). `hist_member` is the resolved historical parent member (see
[ensemble member lineage](ensemble-member-lineage.ipynb)); for most
variables it equals `member`, but for SAI scenarios they can differ.

## Intermediate Artifacts

Three additional groups appear in the scratch store only when `save_intermediate: true` is set in
your config. They capture the pipeline state between computation steps and are useful for
debugging detrending behavior without re-running the full stage.

| Group pattern | Written by | Contents |
|---------------|------------|----------|
| `{method}/detrended_scenario/{scenario_group}/{variable}/{member}` | `transform_scenario` | Scenario data after detrending (9-year running mean removed) |
| `{method}/trend_scenario/{scenario_group}/{variable}/{member}` | `transform_scenario` | The trend signal extracted during detrending (added back after bias correction) |
| `{method}/debiased_scenario/{scenario_group}/{variable}/{member}` | `transform_scenario` | Scenario after bias correction, before re-trending |

**Which intermediates are written per variable** (all require `save_intermediate: true`):

| Variable | `detrended_scenario` | `trend_scenario` | `debiased_scenario` |
|----------|----------------------|------------------|---------------------|
| `tas`    | ✓                    | ✓                | ✓                   |
| `tasmax` | ✓                    | ✓                | ✓                   |
| `tasmin` | —                    | —                | —                   |
| `pr`     | —                    | —                | ✓                   |
| `rsds`   | —                    | —                | ✓                   |
| `dtr`    | —                    | —                | ✓                   |
| `hurs`   | —                    | —                | ✓                   |

:::{admonition} `tasmin` dependency on `dtr` and `tasmax` coarse outputs
:class: note

`fit_historical_tasmin` and `transform_scenario_tasmin` derive `tasmin` as `tasmax − dtr`
(diurnal temperature range). `fit_historical` and `transform_scenario` dispatch to these variants
automatically whenever `variable == "tasmin"`, so no special flag or config is required; after
spatial disaggregation, `tasmin` is reconciled against the fine `tasmax` so that `tasmax >= tasmin`
everywhere (issue #331). They read the following groups as **hard dependencies** from the output
store (written unconditionally by the `dtr` and `tasmax` stages):

| Stage | Reads from output store |
|-------|-------------------------|
| `fit_historical_tasmin` | `{method}/debiased_coarse/historical/dtr/{hist_member}`, `{method}/debiased_coarse/historical/tasmax/{hist_member}` |
| `transform_scenario_tasmin` | `{method}/debiased_coarse/{group}/dtr/{member}`, `{method}/debiased_coarse/{group}/tasmax/{member}` |

Run `dtr` and `tasmax` before `tasmin` — no `save_intermediate` flag required:

```bash
# 1. Run dtr and tasmax (coarse outputs written automatically)
uv run saidownscale run --config-path configs/dtr.yaml
uv run saidownscale run --config-path configs/tasmax.yaml

# 2. Now run tasmin (reads debiased_coarse groups written above)
uv run saidownscale run --config-path configs/tasmin.yaml
```

:::

## Resumability

If you interrupt a run and restart with the same config, the pipeline automatically skips completed
stages by checking whether the corresponding zarr group already exists in the branch ancestry:

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

## Force Recompute

To force recomputation (ignoring the cache):

```bash
# Force all stages
uv run saidownscale run --config-path configs/example.yaml --force

# Force only the scenario stage (reuses existing obs and historical artifacts)
uv run saidownscale run --config-path configs/example.yaml --stage transform_scenario --force
```

## Check Cache Status

Use `saidownscale status` to see which artifacts are complete for your configs:

```bash
uv run saidownscale status --config-path configs/example.yaml --verbose
```

See [CLI reference — saidownscale status](../reference/cli.md#saidownscale-status--check-cache-status) for the
full output format.

## List Cached Artifacts

```bash
# List all groups on the current branch for all stores
uv run saidownscale cache-list --config-path configs/example.yaml

# Filter by stage
uv run saidownscale cache-list --config-path configs/example.yaml --stage obs

# Filter by GCM and variable
uv run saidownscale cache-list --config-path configs/example.yaml --gcm CESM2-WACCM6 --variable tas
```

See [CLI reference — saidownscale cache-list](../reference/cli.md#saidownscale-cache-list--list-cached-artifacts)
for all options.

## Clear Cache

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
`environment: "production"` will only clear production cache, not qa.
:::

See [CLI reference — saidownscale cache-clear](../reference/cli.md#saidownscale-cache-clear--clear-cache) for all
options.

## Programmatic Cache Inspection

You can inspect cached artifacts programmatically using `ArtifactCache`:

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

`list_intermediate_groups` classifies each group by stripping its leading `{method}` segment
before matching against the intermediate-prefix list. It recognizes both the namespaced
(`{method}/detrended_scenario/...`) and pre-namespace (`detrended_scenario/...`) forms, so the
summary above works unchanged against branches written before and after this change.

## See Also

- [Pipeline architecture](../methods/pipeline-architecture.md) — how the cache system is
  designed and why
- [CLI reference](../reference/cli.md) — full option listings for status, cache-list, cache-clear
