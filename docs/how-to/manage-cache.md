# How to Manage the Cache

The pipeline uses two icechunk stores per `(GCM, obs-dataset, spatial-subset)` combination — one
for intermediate artifacts and one for final outputs. All artifacts within a store are written as
zarr groups on a named branch (by default the installed package version).

## Cache Store Locations

```
# Scratch store — obs regridded + historical + optional intermediates
s3://carbonplan-scratch/srm/cache/{environment}/{gcm}-{obs_dataset}-{subset_id}.icechunk
  branch: v1.2.3   ← defaults to installed package version

# Output store — final downscaled scenario results
s3://carbonplan-scratch/srm/outputs/{environment}/{gcm}-{obs_dataset}-{subset_id}.icechunk
  branch: v1.2.3

# Examples for CESM2-WACCM, ERA5, global run:
s3://carbonplan-scratch/srm/cache/qa/CESM2-WACCM-ERA5-global.icechunk
s3://carbonplan-scratch/srm/outputs/qa/CESM2-WACCM-ERA5-global.icechunk

# Regional subset (South Africa):
s3://carbonplan-scratch/srm/cache/qa/CESM2-WACCM-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk
s3://carbonplan-scratch/srm/outputs/qa/CESM2-WACCM-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk
```

Within each store the zarr groups are:

| Store | Group pattern | Stage |
|-------|---------------|-------|
| scratch | `obs/{variable}` | Stage 1 |
| scratch | `historical/{variable}/{member}` | Stage 2 |
| output | `{scenario_group}/{variable}/{member}` | Stage 3 |

Intermediate groups (`debiased_historical/`, `detrended_scenario/`, etc.) appear in the scratch
store only when `save_intermediate: true` is set in your config.

## Resumability

If you interrupt a run and restart with the same config, the pipeline automatically skips completed
stages by checking whether the corresponding zarr group already exists in the branch ancestry:

```bash
# Start run
uv run bcsd run --config-path configs/example.yaml
^C  # Interrupt after stage 1 completes

# Check what's cached
uv run bcsd status --config-path configs/example.yaml

# Resume (automatically skips completed stages)
uv run bcsd run --config-path configs/example.yaml
# Only runs stages 2 and 3
```

## Force Recompute

To force recomputation (ignoring the cache):

```bash
# Force all stages
uv run bcsd run --config-path configs/example.yaml --force

# Force only the scenario stage (keeps obs and historical in cache)
uv run bcsd run --config-path configs/example.yaml --stage transform_scenario --force
```

## Check Cache Status

Use `bcsd status` to see which artifacts are complete for your configs:

```bash
uv run bcsd status --config-path configs/example.yaml --verbose
```

See [CLI reference — bcsd status](../reference/cli.md#bcsd-status--check-cache-status) for the
full output format.

## List Cached Artifacts

```bash
# List all groups on the current branch for all stores
uv run bcsd cache-list --config-path configs/example.yaml

# Filter by stage
uv run bcsd cache-list --config-path configs/example.yaml --stage obs

# Filter by GCM and variable
uv run bcsd cache-list --config-path configs/example.yaml --gcm CESM2-WACCM --variable tas
```

See [CLI reference — bcsd cache-list](../reference/cli.md#bcsd-cache-list--list-cached-artifacts)
for all options.

## Clear Cache

```bash
# Clear all cache for the environment/branch in the config (prompts for confirmation)
uv run bcsd cache-clear --config-path configs/example.yaml

# Clear a specific stage without prompting
uv run bcsd cache-clear --config-path configs/example.yaml --stage scenarios --yes

# Clear only a specific GCM
uv run bcsd cache-clear --config-path configs/example.yaml --gcm CESM2-WACCM --yes
```

:::{admonition} Environment-scoped clearing
:class: warning

Cache clearing respects the `environment` setting in your config. A config with
`environment: "production"` will only clear production cache, not qa.
:::

See [CLI reference — bcsd cache-clear](../reference/cli.md#bcsd-cache-clear--clear-cache) for all
options.

## Programmatic Cache Inspection

You can inspect cached artifacts programmatically using `ArtifactCache`:

```python
import yaml
from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.cache import ArtifactCache

raw = yaml.safe_load(open("configs/example.yaml"))
config = BCSDConfig(**raw)
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

## See Also

- [Pipeline architecture](../explanation/pipeline-architecture.md) — how the cache system is
  designed and why
- [Compare outputs across code versions](compare-outputs-across-versions.md) — using `--branch` to
  track multiple datasets
- [CLI reference](../reference/cli.md) — full option listings for status, cache-list, cache-clear
