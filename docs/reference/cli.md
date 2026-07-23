# CLI Reference

This page is the exhaustive reference for all `bcsd` commands, their options, and usage examples.

## `bcsd run` — Execute Pipeline from Config File (Recommended)

Run the BCSD downscaling pipeline for a **single config** or a **directory of config files**. Config files support the [matrix format](../reference/configuration.md#matrix-config-format) — list values for `gcm`/`variables`/`ensemble_members`/`scenarios` are expanded into one run per cartesian-product combination.

> **Recommended for most workflows.** Config files are version-controlled and reproducible, and they are what the QA and production deploys consume. For quick ad-hoc runs from the command line without config files, use `bcsd run-matrix` instead.

```bash
uv run bcsd run --config-path PATH [OPTIONS]
```

**Options:**

- `--config-path TEXT` (required, repeatable): path to YAML config file or directory of configs (can be specified multiple times)
- `--stage TEXT`: run specific stage. Accepts either short (`obs`/`historical`/`scenario`) or long (`prepare_observations`/`fit_historical`/`transform_scenario`) names, or `all` (default: `all`)
- `--force`: force recompute even if cached
- `--coiled/--no-coiled`: use Coiled for distributed execution (default: `--coiled`)
- `--branch TEXT`: override the output icechunk branch (e.g. `v2`). Defaults to the branch resolved from the config (the installed package version).
- `--save-intermediate`: save and display intermediate artifacts

**Examples:**

```bash
# Run full pipeline for a single config
uv run bcsd run --config-path configs/example.yaml

# Run only observation regridding stage locally
uv run bcsd run --config-path configs/example.yaml --stage prepare_observations --no-coiled

# Force recompute of historical stage (ignores cache)
uv run bcsd run --config-path configs/example.yaml --stage fit_historical --force

# Override branch (write outputs to the v2 icechunk branch without editing config files)
uv run bcsd run --config-path configs/example.yaml --branch v2

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

## `bcsd run-matrix` — Run Pipeline Over a Matrix

> Specify each dimension as a repeatable option and the CLI runs every combination — no config files needed. This is convenient for quick, ad-hoc runs; for repeatable or reviewable runs, prefer `bcsd run` with a config file. The orchestrator automatically deduplicates shared work across stages.

```bash
uv run bcsd run-matrix [OPTIONS]
```

**Options:**

- `--gcm TEXT` (required, repeatable): GCM name
- `--variable TEXT` (required, repeatable): variable to downscale
- `--member TEXT` (required, repeatable): ensemble member label (e.g. `r1i1p1f1`, `01`)
- `--scenario TEXT` (repeatable): scenario name (e.g. `SSP245`, `G6-1.5K`). Omit for historical-only runs.
- `--predict-period-start INTEGER`: start year of prediction period (required when `--scenario` is given)
- `--predict-period-end INTEGER`: end year of prediction period (required when `--scenario` is given)
- `--train-period-start INTEGER`: start year of training period (default: `1978`)
- `--train-period-end INTEGER`: end year of training period (default: `2014`)
- `--scratch-dir TEXT`: base directory for cached artifacts (default: `s3://carbonplan-scratch/srm/cache/`)
- `--output-dir TEXT`: directory for final outputs (default: `s3://carbonplan-scratch/srm/outputs/`)
- `--environment TEXT`: environment (default: `qa`)
- `--branch TEXT`: icechunk output branch (default: `main`)
- `--subset-bounds TEXT`: spatial bounds as `'lat_min,lat_max,lon_min,lon_max'`
- `--debias-approach TEXT`: bias-correction approach — `parametric`, `nonparametric`, `nonparametric_hybrid`, `nonparametric_hybrid_2sided` (default: `nonparametric_hybrid_2sided`)
- `--stage TEXT`: run specific stage (`obs`/`historical`/`scenario`/`all`, default: `all`)
- `--force`: force recompute even if cached
- `--coiled/--no-coiled`: use Coiled for distributed execution (default: `--coiled`)
- `--dry-run`: print the generated configs in a table without executing
- `--save-intermediate`: save intermediate artifacts (detrended, debiased, etc.) to cache
- `--verbose / -v`: enable verbose logging

**Per-variable config overrides** (applied to every variable in the matrix; normally auto-set from the variable — see [Variable-Specific Auto-Configuration](../reference/configuration.md#variable-specific-auto-configuration)):

- `--detrend-data / --no-detrend-data`: override `detrend_data`
- `--do-windowing / --no-do-windowing`: override `do_windowing`
- `--running-window-length INTEGER`: override `running_window_length`
- `--downscaling-method TEXT`: override `downscaling_method` (`additive`, `multiplicative`)
- `--downscaling-clim-method TEXT`: override `downscaling_clim_method` (`simple`, `fft`)
- `--detrend-method TEXT`: override `detrend_method` (`additive`, `multiplicative`)

**Examples:**

```bash
# 2 GCMs x 2 variables x 3 members x 2 scenarios
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --gcm MIROC-ES2H \
  --variable tas --variable pr \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \
  --scenario SSP245 --scenario G6-1.5K \
  --predict-period-start 2015 --predict-period-end 2100 \
  --scratch-dir "s3://carbonplan-scratch/srm/cache/" \
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

## `bcsd validate` — Validate Input Datasets

Validate input datasets against the validation matrix before running the pipeline. Exits with code 1 if any blocking check fails.

```bash
uv run bcsd validate [OPTIONS]
```

**Options:**

- `--config-path TEXT / -c TEXT` (repeatable): path to YAML config or directory. Derives the GCMs and scenarios to validate from the loaded configs. Supports the matrix format (list fields).
- `--gcm TEXT` (repeatable): GCM(s) to validate explicitly. Defaults to all known GCMs when neither `--config-path` nor `--gcm` is given.
- `--scenario TEXT` (repeatable): scenario(s) to validate explicitly. Defaults to all known scenarios when neither `--config-path` nor `--scenario` is given.
- `--coiled/--no-coiled`: run the lazy dask reductions on a short-lived Coiled Dask cluster instead of in-process (default: `--coiled`)
- `--n-workers INTEGER`: Coiled worker count, or adaptive minimum when `--adaptive-max` is set (default: `4`)
- `--worker-vm-type TEXT`: Coiled worker VM type (default: `r8g.2xlarge`)
- `--adaptive-max INTEGER`: enable adaptive scaling up to this many workers

**Examples:**

```bash
# Validate only the datasets referenced by a config directory (recommended)
uv run bcsd validate --config-path configs/qa/
uv run bcsd validate --config-path configs/production/

# Validate a specific GCM/scenario combination
uv run bcsd validate --gcm CESM2-WACCM --scenario SSP245

# Validate all known datasets locally (no Coiled cluster)
uv run bcsd validate --no-coiled
```

When `--config-path` is given, `bcsd validate` extracts the unique GCMs and scenarios from those configs and validates only those combinations. This matches exactly what `bcsd run` will consume.

---

## `bcsd validate-output` — Validate Output Stores

Validate downscaled **output** datatree store(s), one leaf (scenario / variable / member) at a time, and render a table per store. Exits with code 1 if any blocking check fails in any store. When `$GITHUB_STEP_SUMMARY` is set, a markdown report is appended there in addition to the console tables.

```bash
uv run bcsd validate-output [STORE_URIS...] [OPTIONS]
```

Exactly one of positional `STORE_URIS` **or** `--config-path` is required.

**Arguments:**

- `STORE_URIS` (repeatable): one or more output datatree icechunk store URIs.

**Options:**

- `--config-path TEXT / -c TEXT` (repeatable): derive the output store URIs from config(s) instead of passing them directly.
- `--branch TEXT`: icechunk branch to read. When derived via `--config-path`, defaults to the branch those configs resolve to (the same branch `run` writes).
- `--tag TEXT`: icechunk tag to read.
- `--scenario TEXT` (repeatable): restrict validation to matching scenario subtrees. Defaults to all.
- `--variable TEXT` (repeatable): restrict validation to matching variable subtrees. Defaults to all.
- `--coiled/--no-coiled`: run the lazy dask reductions on a short-lived Coiled Dask cluster (default: `--coiled`)
- `--n-workers INTEGER`: Coiled worker count (default: `4`)
- `--worker-vm-type TEXT`: Coiled worker VM type (default: `r8g.2xlarge`)
- `--adaptive-max INTEGER`: enable adaptive scaling up to this many workers

**Examples:**

```bash
# Validate explicit output store(s)
uv run bcsd validate-output s3://carbonplan-scratch/srm/outputs/qa/main/... --no-coiled

# Derive the store URIs from the same configs `bcsd run` consumes
uv run bcsd validate-output --config-path configs/qa/

# Validate only a specific scenario/variable subtree
uv run bcsd validate-output --config-path configs/qa/ --scenario SSP245 --variable tas
```

---

## `bcsd status` — Check Cache Status

Check which artifacts are cached and view pipeline progress.

```bash
uv run bcsd status --config-path PATH [--verbose]
```

**Options:**

- `--config-path TEXT` (required): path to config file or directory
- `--verbose / -v`: show detailed cache and output store paths
- `--branch TEXT`: override the output icechunk branch (e.g. `v2`)

**Example:**

```bash
uv run bcsd status --config-path configs/example.yaml --verbose

# Output:
# Cache Configuration:
#   Cache Path: s3://carbonplan-scratch/srm/cache/
#   Output Path: s3://carbonplan-scratch/srm/outputs/
#   Environment: qa
#   Branch: main
#
# Example Paths:
#   Obs: s3://carbonplan-scratch/srm/cache/qa/CESM2-WACCM-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk
#   Historical: s3://carbonplan-scratch/srm/outputs/qa/CESM2-WACCM-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk
#   Scenario: s3://carbonplan-scratch/srm/outputs/qa/CESM2-WACCM-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk
#
# (one icechunk store per GCM-obs-subset tuple; obs/historical/scenario are
#  groups inside it, and versions are icechunk branches, not path segments)
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

- `--config-path TEXT / -c TEXT`: path to config file or directory (default: `configs/example.yaml`; uses `scratch_dir` from the config)
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

- `--config-path TEXT / -c TEXT`: path to config file or directory (default: `configs/example.yaml`)
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

:::{admonition} Environment-scoped clearing
:class: warning

Cache clearing respects the `environment` setting in your config. If you have `environment: "production"`, it will only clear production cache, not qa.
:::
