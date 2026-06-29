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

| Store | Group pattern | Stage | Always written |
|-------|---------------|-------|----------------|
| scratch | `obs/{variable}` | Stage 1 | yes |
| scratch | `historical/{variable}/{member}` | Stage 2 | yes |
| output | `{scenario_group}/{variable}/{member}` | Stage 3 | yes |

These three are the primary artifacts — written unconditionally on every run.

## Intermediate Artifacts

Five additional groups appear in the scratch store only when `save_intermediate: true` is set in
your config. They capture the pipeline state between computation steps and are useful for
debugging bias-correction or detrending behaviour without re-running the full stage.

| Group pattern | Written by | Contents |
|---------------|------------|----------|
| `debiased_historical/{variable}/{member}` | `fit_historical` | GCM historical after quantile-mapping bias correction, before spatial disaggregation |
| `detrended_scenario/{scenario_group}/{variable}/{member}` | `transform_scenario` | Scenario data after detrending (9-year running mean removed) |
| `trend_scenario/{scenario_group}/{variable}/{member}` | `transform_scenario` | The trend signal extracted during detrending (added back after bias correction) |
| `debiased_scenario/{scenario_group}/{variable}/{member}` | `transform_scenario` | Scenario after bias correction, before re-trending |
| `debiased_retrended_scenario/{scenario_group}/{variable}/{member}` | `transform_scenario` | Scenario after bias correction and re-trending, before spatial disaggregation |

**Which intermediates are written per variable** (all require `save_intermediate: true`):

| Variable | `debiased_historical` | `detrended_scenario` | `trend_scenario` | `debiased_scenario` | `debiased_retrended_scenario` |
|----------|-----------------------|----------------------|------------------|---------------------|-------------------------------|
| `tas`    | ✓                     | ✓                    | ✓                | ✓                   | ✓                             |
| `tasmax` | ✓                     | ✓                    | ✓                | ✓                   | ✓                             |
| `tasmin` | ✓ ‡                   | —                    | —                | —                   | ✓ ‡                           |
| `pr`     | ✓                     | —                    | —                | ✓                   | ✓ †                           |
| `rsds`   | ✓                     | —                    | —                | ✓                   | ✓ †                           |
| `dtr`    | ✓                     | —                    | —                | ✓                   | ✓ †                           |
| `hurs`   | ✓                     | —                    | —                | ✓                   | ✓ †                           |

† No retrend applied; `debiased_retrended_scenario` contains identical data to `debiased_scenario`.

‡ `tasmin` uses a derived pathway (`fit_historical_tasmin` / `transform_scenario_tasmin`): it is
computed as `tasmax − dtr` from those variables' cached intermediates. Only `debiased_historical`
and `debiased_retrended_scenario` are written; the detrend and standalone debiased groups are not.

:::{admonition} `tasmin` stages require `save_intermediate: true` for `dtr` and `tasmax`
:class: important

`fit_historical_tasmin` and `transform_scenario_tasmin` derive `tasmin` as `tasmax − dtr`
(diurnal temperature range). They read the following groups as **hard dependencies** — if either
is missing the stage fails with a `ValueError`:

| Stage | Reads |
|-------|-------|
| `fit_historical_tasmin` | `debiased_historical/dtr/{member}`, `debiased_historical/tasmax/{member}` |
| `transform_scenario_tasmin` | `debiased_retrended_scenario/{group}/dtr/{member}`, `debiased_retrended_scenario/{group}/tasmax/{member}` |

Always run the prerequisite stages first with `save_intermediate: true`:

```bash
# 1. Run dtr and tasmax with save_intermediate enabled
BCSD_SAVE_INTERMEDIATE=true uv run bcsd run --config-path configs/dtr.yaml
BCSD_SAVE_INTERMEDIATE=true uv run bcsd run --config-path configs/tasmax.yaml

# 2. Now run tasmin (reads the debiased intermediate groups written above)
uv run bcsd run --config-path configs/tasmin.yaml
```

:::

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
