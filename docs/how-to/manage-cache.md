# How to Manage the Cache

The pipeline provides intelligent caching at multiple levels to enable efficient reuse and resumability. This guide covers how to inspect, resume, force-recompute, and clear cached artifacts.

## Cache Strategy

**Two-Tier Storage:**

1. **scratch_dir**: intermediate artifacts that are reused across multiple runs
   - observations regridded to GCM grid (shared across all ensembles/scenarios)
   - historical downscaling (shared across all scenarios for an ensemble)

2. **output_dir**: final scenario outputs
   - downscaled scenario data with full metadata
   - organized by environment for clear separation

## Cache Locations

```
s3://carbonplan-scratch/srm/bcsd-cache/
├── qa/                                    # QA environment (testing)
│   ├── v1/                                # Version 1 artifacts
│   │   └── obs/
│   │       └── CESM2-WACCM/
│   │           └── tas/
│   │               ├── global/
│   │               │   └── obs_regridded.icechunk
│   │               └── lat-35.0to-22.0_lon16.0to33.0/
│   │                   └── obs_regridded.icechunk
│   └── v2/                                # Version 2 (after methodological changes)
│       └── ...
└── production/
    └── ...

s3://carbonplan-scratch/srm/outputs/
├── qa/
│   ├── v1/
│   │   ├── historical/
│   │   │   └── CESM2-WACCM/
│   │   │       └── tas/
│   │   │           └── r1i1p1f1/
│   │   │               └── global/
│   │   │                   └── {varconfig_hash}/
│   │   │                       └── historical.icechunk
│   │   ├── ssp245/
│   │   │   └── CESM2-WACCM/
│   │   │       └── tas/
│   │   │           └── r1i1p1f1/
│   │   │               ├── global/
│   │   │               │   └── {varconfig_hash}/
│   │   │               │       └── ssp245.icechunk
│   │   │               └── lat-35.0to-22.0_lon16.0to33.0/
│   │   │                   └── {varconfig_hash}/
│   │   │                       └── ssp245.icechunk
│   │   └── g6-1.5k/
│   │       └── CESM2-WACCM/
│   │           └── tas/
│   │               └── r1i1p1f1/
│   │                   └── global/
│   │                       └── {varconfig_hash}/
│   │                           └── g6-1.5k.icechunk
│   └── v2/
│       └── ...
└── production/
    └── ...
```

## Resumability

If you interrupt a run and restart with the same config:

```bash
# Start run
uv run bcsd run --config-path configs/example.yaml
^C  # Interrupt after stage 1 completes

# Check what's cached
uv run bcsd status --config-path configs/example.yaml
# Shows: prepare_observations ✓, fit_historical ✗, transform_scenario ✗

# Resume (automatically skips completed stages)
uv run bcsd run --config-path configs/example.yaml
# Only runs stages 2 and 3
```

## Cache Dependencies

The cache system validates dependencies before each stage:

```python
Stage 2 (fit_historical):
  - requires: obs_regridded
  - if missing: Raises ValueError with clear message
  
Stage 3 (transform_scenario):
  - requires: obs_regridded AND historical
  - if either missing: Raises ValueError
```

## Force Recompute

To force recomputation (ignoring cache):

```bash
# Force all stages
uv run bcsd run --config-path configs/example.yaml --force

# Force only scenario stage (keeps obs and historical cache)
uv run bcsd run --config-path configs/example.yaml --stage transform_scenario --force
```

## Check Cache Status

Use `bcsd status` to see which artifacts are complete:

```bash
uv run bcsd status --config-path configs/example.yaml --verbose
```

See [CLI reference — bcsd status](../reference/cli.md#bcsd-status--check-cache-status) for the full output format.

## List Cached Artifacts

```bash
# List all cached artifacts
uv run bcsd cache-list --config-path configs/example.yaml

# Filter by stage
uv run bcsd cache-list --config-path configs/example.yaml --stage obs

# Filter by GCM and variable
uv run bcsd cache-list --config-path configs/example.yaml --gcm CESM2-WACCM --variable tas
```

See [CLI reference — bcsd cache-list](../reference/cli.md#bcsd-cache-list--list-cached-artifacts) for all options.

## Clear Cache

```bash
# Clear all cache for the environment/version in the config (prompts for confirmation)
uv run bcsd cache-clear --config-path configs/example.yaml

# Clear a specific stage without prompting
uv run bcsd cache-clear --config-path configs/example.yaml --stage scenarios --yes

# Clear only a specific GCM
uv run bcsd cache-clear --config-path configs/example.yaml --gcm CESM2-WACCM --yes
```

:::{admonition} Environment-scoped clearing
:class: warning

Cache clearing respects the `environment` setting in your config. If you have `environment: "production"`, it will only clear production cache, not qa.
:::

See [CLI reference — bcsd cache-clear](../reference/cli.md#bcsd-cache-clear--clear-cache) for all options.

## Programmatic Cache Inspection

You can inspect cached artifacts programmatically:

```python
from srm.bcsd_config import BCSDConfig
from srm.cache import ArtifactCache

config = BCSDConfig(**yaml.safe_load(open("configs/example.yaml")))
cache = ArtifactCache(
    scratch_dir=config.scratch_dir,
    environment=config.environment,
    version=config.version,
    output_dir=config.output_dir,
)

# Check if specific artifact exists
obs_path = cache.get_obs_path(config)
print(f"Observations cached: {cache.exists(obs_path)}")

# List all artifacts
artifacts = cache.list_artifacts(stage="obs")
print(f"Cached observation artifacts: {len(artifacts)}")
```

## See Also

- [Pipeline architecture](../explanation/pipeline-architecture.md) — how the cache system is designed and why
- [Compare outputs across code versions](compare-outputs-across-versions.md) — using `version` to track multiple datasets
- [CLI reference](../reference/cli.md) — full option listings for status, cache-list, cache-clear
