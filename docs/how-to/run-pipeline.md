---
orphan: true
---

# Run the pipeline

If you want to run the downscaling pipeline yourself, start here. We support 2 ways to launch a run,
and which one fits depends on whether you need to repeat the run later.

| If you want to | Use | Because |
| --- | --- | --- |
| Repeat or review a run later | `saidownscale run --config-path` | Config files are version controlled, and they are what our QA and production deploys consume |
| Try a one-off combination | `saidownscale run-matrix` | Takes each axis as a repeatable option and expands every combination, with no file to write |

## Install

```bash
uv sync --all-groups
```

The package entry point installs the CLI as the `saidownscale` command. Run it through `uv run` so
that it resolves against the project environment.

## Demo notebook

For a complete walkthrough with figures, see
[demo-new-pipeline.ipynb](./demo-new-pipeline.ipynb). The notebook covers:

- configuration setup with regional subsetting (South Africa)
- running the 3-stage pipeline
- loading and plotting results at Cape Town
- cache inspection and status checking

## Run from a config file

Run the pipeline with `saidownscale run` and a config file whenever you want the run to be
repeatable. The examples below show a plain run and the 2 overrides you reach for most often:

```bash
# Run from a config file
uv run saidownscale run --config-path configs/example.yaml

# Override branch without editing the file
uv run saidownscale run --config-path configs/example.yaml --branch v2

# Override environment via environment variable
SAIDOWNSCALE_ENVIRONMENT=production uv run saidownscale run --config-path configs/example.yaml
```

A single config file can also expand into many runs. List values for `gcm`, `variables`,
`ensemble_members`, `scenarios`, and `downscaling_methods` produce one run per cartesian-product
combination, as the [matrix config format](../reference/configuration.md#matrix-config-format)
reference describes.

Check pipeline status at any time:

```bash
uv run saidownscale status --config-path configs/example.yaml --verbose
```

For a quick ad-hoc run with no config file to write, `saidownscale run-matrix` takes each dimension
as a repeatable option and generates every combination for you:

```bash
# 2 GCMs × 2 variables × 3 members × 2 scenarios
uv run saidownscale run-matrix \
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
uv run saidownscale run-matrix \
  --downscaling-method BCSD \
  --gcm CESM2-WACCM6 --variable tas --member r1i1p1f1 \
  --scenario ssp245 --predict-period-start 2015 --predict-period-end 2100 \
  --dry-run
```

## Batch processing

For any multi-run workflow, prefer a directory of config files run through `saidownscale run`.
Version-controlled configs are what our QA and production deploys use, and the orchestrator applies
the same deduplication however you specify the runs:

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

uv run saidownscale run --config-path configs/batch/
```

</details>

A single matrix config file expresses the same set of runs more compactly, with list values for
`gcm`/`variables`/`ensemble_members`/`scenarios`/`downscaling_methods`. See the
[matrix config format](../reference/configuration.md#matrix-config-format) reference for the syntax
and its restrictions.

For a quick ad-hoc batch with no config files, `saidownscale run-matrix` takes the cartesian product
of the dimensions you pass on the command line and handles everything itself:

```bash
# 3 members × 2 scenarios for CESM2-WACCM6 tas, with deduplication
uv run saidownscale run-matrix \
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

The orchestrator deduplicates shared work across the matrix, so that run submits 10 tasks rather
than 18:

- **1** `prepare_observations` task, one per GCM and variable, shared across all ensemble
  members and scenarios
- **3** `fit_historical` tasks, one per ensemble member, shared across scenarios
- **6** `transform_scenario` tasks, one per ensemble member and scenario combination

### Multi-scenario example

```bash
uv run saidownscale run-matrix \
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

Ensemble members rarely all cover the same period, and `saidownscale run` rejects any config whose
`predict_period_end` runs past a member's real data extent. The
[per-member data extents](../reference/configuration.md#prediction-period-and-per-member-data-extents)
reference carries the full table. Because a config holds a single `predict_period`, you split
members with different extents into separate files, each with a `predict_period_end` matched to its
group.

Our production CESM2-WACCM6 SSP245 configs are organized exactly this way. Members 006 to 010 run to
`predict_period_end: 2069` in `cesm2-waccm6-ssp245-tas-global-trunc-2069.yaml`, and the full-length
members 001 to 005 run to 2099. Drop the per-extent files in one directory and run them together,
because deduplication still applies across the whole set:

```bash
uv run saidownscale run --config-path configs/production/cesm2-waccm6/
```

## Local execution

For a test or a small region, run the stages in-process instead of dispatching them. A local run
needs source-data access and enough memory to hold the subset:

```bash
# Single config (sequential execution of stages)
uv run saidownscale run --config-path configs/example.yaml --executor local

# Matrix run locally (useful for testing)
uv run saidownscale run-matrix \
  --downscaling-method BCSD \
  --gcm CESM2-WACCM6 --variable tas --member r1i1p1f1 \
  --scenario ssp245 --predict-period-start 2015 --predict-period-end 2100 \
  --subset-bounds '-35,-22,16,33' \
  --executor local
```

## See also

These 4 pages cover the ground next to this one:

| Page | What it covers |
| --- | --- |
| [CLI reference](../reference/cli.md) | Full option listings for every command |
| [Configuration reference](../reference/configuration.md) | Every config field and its environment variable override |
| [Manage the cache](manage-cache.md) | Resumability, force recompute, cache inspection and clearing |
| [Pipeline architecture](../methods/pipeline-architecture.md) | How the 3 stages and the caching work |
