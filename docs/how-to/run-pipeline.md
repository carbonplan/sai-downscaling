# How to Run the BCSD Pipeline

## Installation

```bash
uv sync --all-groups
```

The CLI is installed as `bcsd` command via the package entry point.

## Demo Notebook

For a complete walkthrough with visualizations, see [demo-new-pipeline.ipynb](./demo-new-pipeline.ipynb). The notebook demonstrates:

- configuration setup with regional subsetting (South Africa)
- running the three-stage pipeline
- loading and visualizing results at Cape Town
- cache inspection and status checking

## Quick Start

The recommended way to run the pipeline — especially across multiple GCMs, variables, ensemble members, or scenarios — is `bcsd run-matrix`. It generates and runs every combination from the command line without needing any config files:

```bash
# 2 GCMs × 2 variables × 3 members × 2 scenarios
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --gcm MIROC-ES2H \
  --variable tas --variable pr \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \
  --scenario ssp245 --scenario G6-1pt5k \
  --predict-period-start 2015 --predict-period-end 2100 \
  --scratch-dir "s3://carbonplan-scratch/srm/bcsd-cache" \
  --output-dir "s3://carbonplan-scratch/srm/outputs/"
```

Use `--dry-run` to preview the generated matrix before executing:

```bash
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --variable tas --member r1i1p1f1 \
  --scenario ssp245 --predict-period-start 2015 --predict-period-end 2100 \
  --dry-run
```

For a **single run** or when you already have a config file, use `bcsd run`:

```bash
# Run from a config file
uv run bcsd run --config-path configs/example.yaml

# Override branch without editing the file
uv run bcsd run --config-path configs/example.yaml --branch v2

# Override environment via environment variable
BCSD_ENVIRONMENT=production uv run bcsd run --config-path configs/example.yaml
```

Check pipeline status at any time:

```bash
uv run bcsd status --config-path configs/example.yaml --verbose
```

## Batch Processing

The recommended approach for all multi-run workflows is `bcsd run-matrix`. It takes the cartesian product of the dimensions you specify and handles everything — no config files to write or manage.

```bash
# 3 members × 2 scenarios for CESM2-WACCM tas, with deduplication
uv run bcsd run-matrix \
  --gcm CESM2-WACCM \
  --variable tas \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \
  --scenario SSP245 --scenario G6-1.5K \
  --predict-period-start 2015 --predict-period-end 2100 \
  --scratch-dir "s3://carbonplan-scratch/srm/bcsd-cache" \
  --output-dir "s3://carbonplan-scratch/srm/outputs/" \
  --environment qa
```

The orchestrator automatically deduplicates shared work across the matrix:

- **1** `prepare_observations` task (one per GCM/variable, shared across all members and scenarios)
- **3** `fit_historical` tasks (one per ensemble member, shared across scenarios)
- **6** `transform_scenario` tasks (one per member/scenario combination)

### Multi-Scenario Example

```bash
uv run bcsd run-matrix \
  --gcm CESM2-WACCM \
  --variable tas \
  --member r1i1p1f1 \
  --scenario SSP245 --scenario G6-1.5K \
  --predict-period-start 2015 --predict-period-end 2100 \
  --scratch-dir "s3://carbonplan-scratch/srm/bcsd-cache" \
  --output-dir "s3://carbonplan-scratch/srm/outputs/"
# obs and historical artifacts are computed once and reused for both scenarios
```

### Using Config Files (Alternative)

For workflows that are driven by version-controlled YAML config files, `bcsd run` accepts a directory of configs and applies the same deduplication logic:

<details>
<summary>Example: generating and running a directory of config files</summary>

```bash
for member in r1i1p1f1 r2i1p1f1 r3i1p1f1; do
  cat > configs/batch/cesm2-tas-ssp245-e${member}.yaml <<EOF
gcm: "CESM2-WACCM"
variable: "tas"
ensemble_member: "${member}"
scenario: "SSP245"
train_period_start: 1978
train_period_end: 2014
predict_period_start: 2015
predict_period_end: 2100
scratch_dir: "s3://carbonplan-scratch/srm/bcsd-cache"
output_dir: "s3://carbonplan-scratch/srm/outputs"
environment: "qa"
# branch defaults to installed package version; omit unless pinning a specific cache namespace
EOF
done

uv run bcsd run --config-path configs/batch/
```

</details>

## Local Execution

For testing or small regions, disable Coiled and run locally:

```bash
# Single config (sequential execution of stages)
uv run bcsd run --config-path configs/example.yaml --no-coiled

# Matrix run locally (useful for testing)
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --variable tas --member r1i1p1f1 \
  --scenario ssp245 --predict-period-start 2015 --predict-period-end 2100 \
  --subset-bounds '-35,-22,16,33' \
  --no-coiled
```

## See Also

- [CLI reference](../reference/cli.md) — full option listings for every command
- [Configuration reference](../reference/configuration.md) — all config fields and environment variable overrides
- [Manage the cache](manage-cache.md) — resumability, force recompute, cache inspection and clearing
- [Compare outputs across code versions](compare-outputs-across-versions.md) — validate pipeline changes on a test region
- [Pipeline architecture](../explanation/pipeline-architecture.md) — how the three stages and caching work
