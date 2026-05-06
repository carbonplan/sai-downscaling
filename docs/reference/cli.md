# CLI Reference

This page is the exhaustive reference for all `bcsd` commands, their options, and usage examples.

## `bcsd run-matrix` — Run Pipeline Over a Matrix (Recommended)

> **Recommended for multi-run workflows.** Specify each dimension as a repeatable option and the CLI runs every combination — no config files needed. The orchestrator automatically deduplicates shared work across stages.

```bash
uv run bcsd run-matrix [OPTIONS]
```

**Options:**

- `--gcm TEXT` (required, repeatable): GCM name
- `--variable TEXT` (required, repeatable): variable to downscale
- `--member TEXT` (required, repeatable): ensemble member label (e.g. `r1i1p1f1`, `01`)
- `--scenario TEXT` (repeatable): scenario name. Omit for historical-only runs.
- `--predict-period-start INTEGER`: start year of prediction period (required when `--scenario` is given)
- `--predict-period-end INTEGER`: end year of prediction period (required when `--scenario` is given)
- `--train-period-start INTEGER`: start year of training period (default: `1978`)
- `--train-period-end INTEGER`: end year of training period (default: `2014`)
- `--scratch-dir TEXT`: base directory for cached artifacts
- `--output-dir TEXT`: directory for final outputs
- `--environment TEXT`: environment (default: `qa`)
- `--version TEXT`: version identifier (default: `v1`)
- `--subset-bounds TEXT`: spatial bounds as `'lat_min,lat_max,lon_min,lon_max'`
- `--stage TEXT`: run specific stage (`obs`/`historical`/`scenario`/`all`, default: `all`)
- `--force`: force recompute even if cached
- `--coiled/--no-coiled`: use Coiled for distributed execution (default: `--coiled`)
- `--dry-run`: print the generated configs in a table without executing

**Examples:**

```bash
# 2 GCMs x 2 variables x 3 members x 2 scenarios
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --gcm MIROC-ES2H \
  --variable tas --variable pr \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \
  --scenario ssp245 --scenario G6-1pt5k \
  --predict-period-start 2015 --predict-period-end 2100 \
  --scratch-dir "s3://carbonplan-scratch/srm/bcsd-cache" \
  --output-dir "s3://carbonplan-scratch/srm/outputs/"

# Preview what would run without executing
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --gcm MIROC-ES2H \
  --variable tas \
  --member r1i1p1f1 --member r2i1p1f1 \
  --scenario ssp245 \
  --predict-period-start 2015 --predict-period-end 2100 \
  --dry-run

# Historical-only (omit --scenario)
uv run bcsd run-matrix \
  --gcm CESM2-WACCM \
  --variable tas --variable pr \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1

# Regional subset
uv run bcsd run-matrix \
  --gcm CESM2-WACCM \
  --variable tas \
  --member r1i1p1f1 \
  --scenario ssp245 \
  --predict-period-start 2015 --predict-period-end 2100 \
  --subset-bounds '-35,-22,16,33'

# Run only a specific stage
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --variable tas --member r1i1p1f1 \
  --scenario ssp245 --predict-period-start 2015 --predict-period-end 2100 \
  --stage scenario

# Force recompute of all runs
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --variable tas --member r1i1p1f1 \
  --scenario ssp245 --predict-period-start 2015 --predict-period-end 2100 \
  --force
```

The matrix is the cartesian product `GCMs × variables × members × scenarios`. The orchestrator automatically deduplicates shared work: `prepare_observations` runs once per (GCM, variable) combination and `fit_historical` runs once per (GCM, variable, ensemble) combination, regardless of how many scenarios are in the matrix.

**How Deduplication Works:**

```
Example: CESM2-WACCM, tas, ensembles [r1i1p1f1, r2i1p1f1, r3i1p1f1], ssp245

stage 1 (prepare_observations):
  - 1 task runs (shared across all ensembles and scenarios)
  - key: (CESM2-WACCM, tas)
  - output: obs_regridded cached once, reused 3 times

stage 2 (fit_historical):
  - 3 tasks run (one per ensemble member)
  - keys: (CESM2-WACCM, tas, r1i1p1f1), (CESM2-WACCM, tas, r2i1p1f1), (CESM2-WACCM, tas, r3i1p1f1)
  - outputs: historical cached for each ensemble, reused across scenarios

stage 3 (transform_scenario):
  - 3 tasks run (one per ensemble/scenario combination)
  - all run in parallel since dependencies are already cached
```

---

## `bcsd run` — Execute Pipeline from Config File

Run the BCSD downscaling pipeline for a **single config** or a **directory of pre-existing config files**.

> For new multi-run workflows, prefer `bcsd run-matrix` instead.

```bash
uv run bcsd run --config-path PATH [OPTIONS]
```

**Options:**

- `--config-path TEXT` (required, repeatable): path to YAML config file or directory of configs (can be specified multiple times)
- `--stage TEXT`: run specific stage (`prepare_observations`/`fit_historical`/`transform_scenario`/`all`, default: `all`)
- `--force`: force recompute even if cached
- `--coiled/--no-coiled`: use Coiled for distributed execution (default: `--coiled`)
- `--version TEXT`: override the `version` field from the config (e.g. `v2`)

**Examples:**

```bash
# Run full pipeline for a single config
uv run bcsd run --config-path configs/example.yaml

# Run only observation regridding stage locally
uv run bcsd run --config-path configs/example.yaml --stage prepare_observations --no-coiled

# Force recompute of historical stage (ignores cache)
uv run bcsd run --config-path configs/example.yaml --stage fit_historical --force

# Override version (write outputs under v2/ path without editing config files)
uv run bcsd run --config-path configs/example.yaml --version v2

# Override environment for production run
BCSD_ENVIRONMENT=production uv run bcsd run --config-path configs/example.yaml

# Process all configs in a directory
uv run bcsd run --config-path configs/cesm2-ensemble/
```

**Stage details:**

- `prepare_observations`: regrid ERA5 to GCM grid (shared across ensembles)
- `fit_historical`: debias and downscale historical period (shared across scenarios)
- `transform_scenario`: debias and downscale future scenario (final output)
- `all`: run all three stages in sequence (default)

---

## `bcsd status` — Check Cache Status

Check which artifacts are cached and view pipeline progress.

```bash
uv run bcsd status --config-path PATH [--verbose]
```

**Options:**

- `--config-path TEXT` (required): path to config file or directory
- `--verbose`: show detailed cache and output paths
- `--version TEXT`: override the `version` field from the config

**Example:**

```bash
uv run bcsd status --config-path configs/example.yaml --verbose

# Output:
# Cache Configuration:
#   Cache Path: s3://carbonplan-scratch/srm/bcsd-cache
#   Output Path: s3://carbonplan-scratch/srm/outputs
#   Environment: qa
#   Version: v1
#
# Example Paths:
#   Obs: s3://.../bcsd-cache/qa/v1/obs/CESM2-WACCM/tas/lat-35.0to-22.0_lon16.0to33.0/obs_regridded.icechunk
#   Historical: s3://.../outputs/qa/v1/historical/CESM2-WACCM/tas/r1i1p1f1/lat-35.0to-22.0_lon16.0to33.0/{varconfig_hash}/historical.icechunk
#   Scenario: s3://.../outputs/qa/v1/ssp245/CESM2-WACCM/tas/r1i1p1f1/lat-35.0to-22.0_lon16.0to33.0/{varconfig_hash}/ssp245.icechunk
#
# Stage Progress:
# ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Stage                ┃ Total ┃ Cached ┃ Missing ┃ Progress        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ Prepare Observations │     1 │      1 │       0 │ 1/1 (100%)      │
# │ Fit Historical       │     1 │      1 │       0 │ 1/1 (100%)      │
# │ Transform Scenario   │     1 │      0 │       1 │ 0/1 (0%)        │
# └──────────────────────┴───────┴────────┴─────────┴─────────────────┘
```

---

## `bcsd cache-list` — List Cached Artifacts

List all cached artifacts with optional filtering.

```bash
uv run bcsd cache-list --config-path PATH [OPTIONS]
```

**Options:**

- `--config-path TEXT` (required): path to config file or directory (uses scratch_dir from config)
- `--stage TEXT`: filter by stage (`obs`/`historical`/`scenarios`)
- `--gcm TEXT`: filter by GCM model
- `--variable TEXT`: filter by variable

**Examples:**

```bash
# List all cached artifacts from config's scratch_dir
uv run bcsd cache-list --config-path configs/example.yaml

# List only observation artifacts
uv run bcsd cache-list --config-path configs/example.yaml --stage obs

# List specific GCM/variable combination
uv run bcsd cache-list --config-path configs/example.yaml --gcm CESM2-WACCM --variable tas
```

---

## `bcsd cache-clear` — Clear Cache

Delete cached artifacts with optional filtering.

```bash
uv run bcsd cache-clear --config-path PATH [OPTIONS]
```

**Options:**

- `--config-path TEXT` (required): path to config file or directory
- `--stage TEXT`: delete specific stage (`obs`/`historical`/`scenarios`)
- `--gcm TEXT`: delete only specific GCM
- `--variable TEXT`: delete only specific variable
- `--yes/-y`: skip confirmation prompt

**Examples:**

```bash
# Clear all cache (with confirmation)
uv run bcsd cache-clear --config-path configs/example.yaml

# Clear specific stage without confirmation
uv run bcsd cache-clear --config-path configs/example.yaml --stage scenarios --yes

# Clear specific GCM
uv run bcsd cache-clear --config-path configs/example.yaml --gcm CESM2-WACCM --yes
```

> [!WARNING]
> Cache clearing respects the `environment` setting in your config. If you have `environment: "production"`, it will only clear production cache, not qa.
