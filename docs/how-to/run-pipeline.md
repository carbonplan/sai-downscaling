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

The recommended way to run the pipeline is `bcsd run` with a config file. Config files are version-controlled and reproducible, and they are what the QA and production deploys consume, so they are the right choice for any run you want to repeat or review:

```bash
# Run from a config file
uv run bcsd run --config-path configs/example.yaml

# Override branch without editing the file
uv run bcsd run --config-path configs/example.yaml --branch v2

# Override environment via environment variable
BCSD_ENVIRONMENT=production uv run bcsd run --config-path configs/example.yaml
```

A single config file can also expand into many runs. List values for `gcm`, `variables`, `ensemble_members`, `scenarios`, and `downscaling_methods` produce one run per cartesian-product combination — see the [matrix config format](../reference/configuration.md#matrix-config-format) reference for the syntax.

Check pipeline status at any time:

```bash
uv run bcsd status --config-path configs/example.yaml --verbose
```

For quick ad-hoc runs from the command line without writing a config file, `bcsd run-matrix` takes each dimension as a repeatable option and generates every combination for you:

```bash
# 2 GCMs × 2 variables × 3 members × 2 scenarios
uv run bcsd run-matrix \
  --downscaling-method BCSD \
  --gcm CESM2-WACCM6 --gcm UKESM1-1-LL \
  --variable tas --variable pr \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \
  --scenario ssp245 --scenario G6-1.5K \
  --predict-period-start 2015 --predict-period-end 2100 \
  --scratch-dir "s3://carbonplan-srm/scratch/cache" \
  --output-dir "s3://carbonplan-srm/scratch/output/"
```

Use `--dry-run` to preview the generated matrix before executing:

```bash
uv run bcsd run-matrix \
  --downscaling-method BCSD \
  --gcm CESM2-WACCM6 --variable tas --member r1i1p1f1 \
  --scenario ssp245 --predict-period-start 2015 --predict-period-end 2100 \
  --dry-run
```

## Batch Processing

For all multi-run workflows, the recommended approach is a directory of config files run through `bcsd run`. Version-controlled configs are what the QA and production deploys use, and the orchestrator applies the same deduplication logic regardless of how the runs are specified:

<details>
<summary>Example: generating and running a directory of config files</summary>

```bash
for member in r1i1p1f1 r2i1p1f1 r3i1p1f1; do
  cat > configs/batch/cesm2-tas-ssp245-e${member}.yaml <<EOF
gcm: "CESM2-WACCM6"
variable: "tas"
ensemble_member: "${member}"
scenario: "SSP245"
train_period_start: 1978
train_period_end: 2014
predict_period_start: 2015
predict_period_end: 2100
scratch_dir: "s3://carbonplan-srm/scratch/cache"
output_dir: "s3://carbonplan-srm/scratch/output"
environment: "qa"
# branch defaults to installed package version; omit unless pinning a specific cache namespace
EOF
done

uv run bcsd run --config-path configs/batch/
```

</details>

A single matrix config file expresses the same set of runs more compactly, with list values for `gcm`/`variables`/`ensemble_members`/`scenarios`/`downscaling_methods`. See the [matrix config format](../reference/configuration.md#matrix-config-format) reference for the syntax and its restrictions.

For a quick ad-hoc batch without config files, `bcsd run-matrix` takes the cartesian product of the dimensions you pass on the command line and handles everything itself:

```bash
# 3 members × 2 scenarios for CESM2-WACCM6 tas, with deduplication
uv run bcsd run-matrix \
  --downscaling-method BCSD \
  --gcm CESM2-WACCM6 \
  --variable tas \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \
  --scenario SSP245 --scenario G6-1.5K \
  --predict-period-start 2015 --predict-period-end 2100 \
  --scratch-dir "s3://carbonplan-srm/scratch/cache" \
  --output-dir "s3://carbonplan-srm/scratch/output/" \
  --environment qa
```

The orchestrator automatically deduplicates shared work across the matrix:

- **1** `prepare_observations` task (one per GCM/variable, shared across all members and scenarios)
- **3** `fit_historical` tasks (one per ensemble member, shared across scenarios)
- **6** `transform_scenario` tasks (one per member/scenario combination)

### Multi-Scenario Example

```bash
uv run bcsd run-matrix \
  --downscaling-method BCSD \
  --gcm CESM2-WACCM6 \
  --variable tas \
  --member r1i1p1f1 \
  --scenario SSP245 --scenario G6-1.5K \
  --predict-period-start 2015 --predict-period-end 2100 \
  --scratch-dir "s3://carbonplan-srm/scratch/cache" \
  --output-dir "s3://carbonplan-srm/scratch/output/"
# obs and historical artifacts are computed once and reused for both scenarios
```

## Ensembles with mixed data extents

Ensemble members rarely all cover the same period, and `bcsd run` rejects any config whose `predict_period_end` runs past a member's real data extent (see the [per-member data extents](../reference/configuration.md#prediction-period-and-per-member-data-extents) reference for the full table). Because a config carries a single `predict_period`, members with different extents have to be split into separate files, each with a `predict_period_end` matched to its group.

The production CESM2-WACCM6 SSP245 configs are organized exactly this way. Members 006–010 run to `predict_period_end: 2069` (`cesm2-waccm-ssp245-tas-global-trunc-2069.yaml`), and the full-length members 001–005 run to 2099. Drop the per-extent files in one directory and run them together — deduplication still applies across the whole set:

```bash
uv run bcsd run --config-path configs/production/cesm2-waccm/
```

## Local Execution

For testing or small regions, run the stages in-process instead of dispatching them:

```bash
# Single config (sequential execution of stages)
uv run bcsd run --config-path configs/example.yaml --executor local

# Matrix run locally (useful for testing)
uv run bcsd run-matrix \
  --downscaling-method BCSD \
  --gcm CESM2-WACCM6 --variable tas --member r1i1p1f1 \
  --scenario ssp245 --predict-period-start 2015 --predict-period-end 2100 \
  --subset-bounds '-35,-22,16,33' \
  --executor local
```

## See Also

- [CLI reference](../reference/cli.md) — full option listings for every command
- [Configuration reference](../reference/configuration.md) — all config fields and environment variable overrides
- [Manage the cache](manage-cache.md) — resumability, force recompute, cache inspection and clearing
- [Pipeline architecture](../explanation/pipeline-architecture.md) — how the three stages and caching work
