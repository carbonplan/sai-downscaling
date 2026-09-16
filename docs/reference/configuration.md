---
orphan: true
---

# Configuration Reference

Configuration files use YAML format with Pydantic validation. All fields are validated before execution to catch errors early.

## Two-class model

Configuration is split across two Pydantic classes loaded from the same flat YAML:

- **`DownscalingConfig`** — run identity: the parameters that uniquely identify a downscaling run and affect computation results (model, variable, time periods, bias-correction method). Changes here bust the cache.
- **`PipelineOptions`** — operational settings: storage paths, environment, branch, and runtime flags. Changes here do not affect computation results.

Both classes use `extra="ignore"`, so a single flat YAML file is accepted by both — no nested sections required.

## Matrix config format

Any of the five dimension fields can be a list. `load_configs` expands them into one `DownscalingConfig` per cartesian-product combination:

```yaml
gcm: "CESM2-WACCM6"                              # singular, still works
variables: ["tas", "pr"]                         # list, expands
ensemble_members: ["001", "002", "003"]
scenarios: ["SSP245"]
downscaling_methods: ["BCSD", "QDMSD"]           # see Downscaling method below
predict_period_start: 2015
predict_period_end: 2099                         # must fit every member's data extent (see below)
```

Both singular (`variable`) and plural (`variables`) key names are accepted, but not both at once for the same axis. All other fields are shared across every combination.

`downscaling_methods` differs in kind from the other four axes. Those select a slice of input data, while this one selects an algorithm, so each entry re-derives `variable_config` from its own defaults table rather than reusing one. The two methods share a single regridded observation artifact and write under separate group prefixes, which is what lets a comparison run live in one config file.

**Restrictions:**

| Combination | Why it is rejected |
| --- | --- |
| `variable_config` with more than one entry in `variables` | It would silently apply to every variable, including those with incompatible settings, such as additive `tas` settings applied to `pr`. |
| `variable_config` with more than one entry in `downscaling_methods` | It is passed through verbatim, and its `debias_approach` can only agree with one method. |
| `debias_approach` (top level or in `variable_overrides`) with more than one entry in `downscaling_methods` | `qdm` requires `QDMSD` and `QDMSD` requires `qdm`, so one arm of the product always contradicts the value. |
| Both `variables` and `variable`, or any other plural/singular pair for one axis | The two spellings name the same axis, and keeping both is ambiguous. |

For the first case, remove `variable_config` and rely on per-variable defaults (see [Variable-Specific Auto-Configuration](#variable-specific-auto-configuration)), or split into separate files. For the `debias_approach` cases, drop the override and let each method's defaults table supply it, or list one method at a time.

## Prediction period and per-member data extents

`predict_period_start` and `predict_period_end` must fall within the valid data extent of every ensemble member the config expands to. Those extents are not uniform: some members are truncated years before the nominal scenario end and the unified store NaN-pads them to that end, so a predict period that overshoots would silently downscale padding. `saidownscale run` and `saidownscale run-matrix` guard against this with `check_config_time_domain` before submitting any work, raising a blocking error that lists every config whose predict period falls outside its member's bounds.

The extent for a `(gcm, scenario, ensemble_member)` triple is resolved from a per-member override table first, then the scenario's nominal bounds, and is left unchecked when neither is registered. The authoritative table is `_MEMBER_TIME_BOUNDS` in `src/saidownscale/validation.py`; the CESM2-WACCM6 SSP245 spread is representative:

| Members | Valid end year |
|---|---|
| 001–005 | 2099 |
| 006–010 | 2069 |

007–010 each have a single stray non-NaN day on 2070-01-01 in the raw GCM input, with the rest of 2070 NaN; that one day does not extend their valid extent past 2069.

UKESM1-1-LL SSP245 ends 2099 while its G6-1.5K ends 2084. Because a single config carries one `predict_period`, members with different extents cannot share a config — each extent group needs its own file with a matching `predict_period_end`. SAI/G6 scenarios can technically start before their own data, because the pipeline bridges the gap: for `G6-1.5K` that bridge is SSP245, and for the `G6-1.5K-END` termination run, whose store begins in 2085, it is SSP245 through 2034 followed by the parent `G6-1.5K` member 002 for 2035–2084. Those bridge years are another scenario's data, so publishing them under this scenario's label is what issue #448 hit, where pre-2035 `g6_1p5k` output drew `tas` and `tasmax` from different SSP245 realizations and produced `tas > tasmax`.

`config_time_domain` therefore enforces the start bound for every scenario, SAI included, and set `predict_period_start` to the scenario's own data start: 2035 for `G6-1.5K` and 2085 for `G6-1.5K-END`. For the workflow of splitting a run across extent groups, see [Ensembles with mixed data extents](../how-to/run-pipeline.md#ensembles-with-mixed-data-extents).

## DownscalingConfig Fields (run identity)

These fields identify a downscaling run and affect computation results. Changing any of these busts the cache (`config_hash` covers all of them).

```yaml
# Model identifiers (singular or list)
gcm: "CESM2-WACCM6"                    # GCM model name
variable: "tas"                        # Variable: tas, tasmax, tasmin, pr, rsds, dtr, hurs
ensemble_member: "r1i1p1f1"            # Ensemble member label (e.g. "r1i1p1f1", "01")
scenario: "SSP245"                     # Scenario: SSP245, G6-1.5K, etc. (null for historical-only)

# Observation dataset (catalog key)
obs_dataset: "ERA5"                    # Observation dataset key (default: "ERA5")

# Time periods
train_period_start: 1978               # Training period start year (default: 1978)
train_period_end: 2014                 # Training period end year (default: 2014)
predict_period_start: 2015             # Prediction period start year (required if scenario set)
predict_period_end: 2100               # Prediction period end year (required if scenario set)

# Spatial subsetting (null for global)
subset_bounds: [-35, -22, 16, 33]     # [lat_min, lat_max, lon_min, lon_max]

# Variable-specific settings (auto-loaded from per-variable defaults if not specified)
variable_config:
  detrend_data: true                   # Whether to detrend (auto-set based on variable)
  detrend_method: "additive"           # "additive" or "multiplicative" trend model
  do_windowing: true                   # Use a running window for quantile mapping
  running_window_length: 31            # Running-window length in days
  running_window_step_length: 1        # Days the running window advances per step
  disaggregation_method: "additive"    # "additive" for temperature-like vars, "multiplicative" for pr/rsds
  disaggregation_clim_method: "fft"    # "fft" or "simple" climatology smoothing
  debias_approach: "nonparametric_hybrid_2sided"  # parametric, nonparametric, nonparametric_hybrid, nonparametric_hybrid_2sided, qdm

# Per-variable overrides, keyed by variable name (matrix configs only)
variable_overrides:
  dtr:
    debias_approach: "nonparametric"
```

**Required:** `gcm`, `variable`, `ensemble_member`. All others have defaults or are conditionally required (e.g. `predict_period_*` when `scenario` is set).

:::{note}
`debias_approach` is our own field and is a superset of ibicus's `mapping_type` argument. ibicus only accepts `parametric` or `nonparametric`; the `nonparametric_hybrid` and `nonparametric_hybrid_2sided` values are hybrid strategies the pipeline composes on top of ibicus.
:::

## Per-variable overrides

`debias_approach` and every other `VariableConfig` field resolve per variable through three tiers, last writer wins:

| Tier | Source | Scope |
| --- | --- | --- |
| 1 | `VariableConfig.for_variable()` table | Built-in default for that variable |
| 2 | `--debias-approach` and the other `VariableConfig` CLI flags | Every variable in the run (CLI only) |
| 3 | `variable_overrides` (YAML) or `--variable-override` (CLI) | One named variable |

There is no run-wide tier in YAML, deliberately. A top-level `debias_approach` would be silently discarded by `extra = "ignore"`, so it is rejected outright. Set `variable_config` directly in a single-variable config, or name each variable under `variable_overrides` in a matrix config.

`variable_overrides` is keyed by variable name, so it is order-independent. A key naming a variable outside the run is an error, not a silent no-op. It is only valid in matrix configs; a single-variable config should use `variable_config` directly.

```bash
saidownscale run-matrix --gcm CESM2-WACCM6 \
  --variable tasmax --variable dtr \
  --member 007 --scenario ssp245 \
  --predict-period-start 2015 --predict-period-end 2069 \
  --variable-override dtr:debias_approach=nonparametric
```

:::{note}
`dtr` overrides propagate into `tasmin`, which the pipeline reconstructs as `tasmax - dtr`. The `tasmin` output's `srm_downscaling:bias_correction_method` attribute reports only `tasmin`'s own approach.
:::

## Overrides and the artifact cache

Store paths key on `(gcm, obs_dataset, subset)` and group paths on `(stage, variable, ensemble_member)`. Neither encodes `VariableConfig`, so two runs that differ only in a `variable_overrides` entry resolve to exactly the same location on the same branch.

The pipeline detects this rather than preventing it. On a cache hit, it compares the artifact's `srm_downscaling:config_json` provenance attribute against the current run's `variable_config` and raises `CacheConfigMismatchError` when they differ, naming both values. Without the check, the second run would report a hit, skip the stage, and feed artifacts built under different bias-correction settings to every downstream stage.

To run two configurations side by side, give each its own branch:

```bash
SAIDOWNSCALE_BRANCH=v0.13.0-dtr-nonparam saidownscale run-matrix ... --variable-override dtr:debias_approach=nonparametric
```

| Case | Behavior |
| --- | --- |
| Stored `variable_config` matches | Normal cache hit |
| Stored `variable_config` differs | `CacheConfigMismatchError`, naming each differing field |
| No `config_json` attribute (artifact predates config provenance) | Hit allowed, logged at debug; unverifiable is not the same as mismatched |
| Regridded observations (`obs/...`) | Never verified, since regridding reads no `VariableConfig` field |
| Sibling-variable lookup, e.g. `tasmin` reading `dtr` | Never verified, since the sibling's intended config is not knowable from this run |

## PipelineOptions Fields (operational)

These fields control storage paths and runtime behavior. They do not affect computation results and are not included in `config_hash`.

```yaml
# Storage paths
scratch_dir: "s3://bucket/path"        # Base directory for intermediate artifacts (default: s3://carbonplan-srm/scratch/cache/)
output_dir: "s3://bucket/path"         # Directory for final downscaled outputs — historical + scenario (default: s3://carbonplan-srm/scratch/output/)

# Environment and versioning
environment: "qa"                      # Environment: qa, production (default: "qa")
# branch: "v1.0.post12"              # Override to pin a specific cache namespace (default: installed package version)

# Execution
executor: "coiled"                     # Where stage tasks run: aws-batch, coiled, local (default: "coiled")
batch_job_queue: "srm-production"      # AWS Batch job queue (default: "srm-production")
batch_job_definition: "srm-downscaling"  # AWS Batch job definition, optionally name:revision (default: "srm-downscaling")
batch_region: "us-west-2"              # Region for the AWS Batch control plane (default: "us-west-2")

# Runtime flags
verbose: true                          # Enable verbose logging (default: true)
rechunk_workflow: true                 # Enable strategic rechunking between stages (default: true)
apply_ocean_mask: false                # Mask ocean pixels to NaN in final output (default: false)
save_intermediate: false               # Save intermediate artifacts for debugging (default: false)

# Post-bias-correction clipping
clip_values: true                      # Apply per-variable clipping after bias correction (default: true)
clip_bounds:                           # Per-variable [min, max] bounds applied when clip_values=true
  pr: {min: 0.0}                        #   defaults: pr >= 0
  rsds: {min: 0.0}                      #            rsds >= 0
  hurs: {min: 0.0, max: 105.0}          #            0 <= hurs <= 105
```

All `PipelineOptions` fields are optional — defaults are suitable for most runs. Override `scratch_dir` and `output_dir` to point at your own storage.

### Choosing an executor

`executor` selects where a stage's tasks run. The choice affects cost and nothing else: all three produce identical output, because each task runs the same `saidownscale.batch_runner` entry point.

| Value | Where tasks run | Cost per vCPU-hour |
| --- | --- | --- |
| `aws-batch` | AWS Batch array jobs on Graviton instances | $0.0589 (EC2 only) |
| `coiled` | Coiled Batch VMs | $0.1089 (EC2 plus Coiled's $0.05 platform fee) |
| `local` | Sequentially, in the current process | none |

The `batch_*` fields apply only to `aws-batch` and are usually left at their defaults. Deploy jobs override `batch_job_queue` to separate qa from production traffic, and set `batch_job_definition` to a pinned `name:revision` so a run cannot execute an image built from a different commit.

The removed `use_coiled` boolean is rejected rather than ignored. Because `PipelineOptions` sets `extra = "ignore"`, a stale `use_coiled` key would otherwise be silently dropped and quietly change which executor runs.

## Branch Defaulting

The `branch` field defaults to the **public version of the installed `saidownscale` package** (e.g. `v1.0.post12`),
derived via:

```python
from packaging.version import Version
from importlib.metadata import version as pkg_version
"v" + Version(pkg_version("saidownscale")).public  # e.g. "v1.0.post12", strips local/dirty markers
```

The branch is an icechunk branch created inside each unified per-GCM store. This means:

- Each commit merged to `main` automatically gets its own cache namespace (via the `post-release` counter).
- Team members on the same commit share the same cache namespace even if one has a dirty working tree.
- Cache busts only when you intentionally advance the version (i.e. a new release or new commits on `main`).

To **pin** a specific namespace (e.g. to reuse artifacts across a version bump), override explicitly:

```yaml
branch: "v1.0.post5"   # pin to an earlier commit's cache branch
```

## Environment Variable Override

You can override the `environment` field using the `SAIDOWNSCALE_ENVIRONMENT` environment variable, and
`branch` using `SAIDOWNSCALE_BRANCH`:

```bash
# Override environment for this run
SAIDOWNSCALE_ENVIRONMENT=production saidownscale run --config-path configs/example.yaml

# Override branch for this run
SAIDOWNSCALE_BRANCH=v1.0.post5 saidownscale run --config-path configs/example.yaml
```

You can also override `branch` directly on the CLI without editing the config file:

```bash
# Pin to a specific branch's cache
uv run saidownscale run --config-path configs/example.yaml --branch v1.0.post5
```

This is useful for:

- testing configs locally with `qa` before running in `production`
- pinning `branch` to reuse cached artifacts from a known-good commit
- running the same config in different environments without editing the file
- CI/CD pipelines that deploy to different environments

## Variable-Specific Auto-Configuration

The pipeline automatically sets variable-specific parameters from the per-variable defaults in `VariableConfig.for_variable` (`src/saidownscale/downscaling_config.py`). Which table it reads is set by the required top-level `downscaling_method` key, described in [Downscaling method](#downscaling-method) below.

### BCSD defaults

All variables use a `running_window_length` of `31` days and a `running_window_step_length` of `1` day, with `debias_approach: nonparametric_hybrid_2sided`.

| Variable | detrend_data | detrend_method | do_windowing | disaggregation_method | disaggregation_clim_method |
| --- | --- | --- | --- | --- | --- |
| `tas` | `true` | `additive` | `true` | `additive` | `fft` |
| `tasmax` | `true` | `additive` | `true` | `additive` | `fft` |
| `tasmin` | `true` | `additive` | `true` | `additive` | `fft` |
| `pr` | `false` | `multiplicative` | `true` | `multiplicative` | `fft` |
| `rsds` | `false` | `multiplicative` | `true` | `multiplicative` | `fft` |
| `dtr` | `false` | `multiplicative` | `true` | `multiplicative` | `fft` |
| `hurs` | `false` | `additive` | `true` | `multiplicative` | `fft` |

### QDMSD defaults

All variables use a `running_window_length` of `91` days and a `running_window_step_length` of `31` days, with `debias_approach: qdm`. Quantile delta mapping carries the climate trend through its own quantile mapping, so `detrend_data` is `false` for every variable. The `detrend_method`, `disaggregation_method`, and `disaggregation_clim_method` columns match the BCSD table above.

You can override these per run through the nested `variable_config` block in the config file, or with the `saidownscale run-matrix` override flags (`--disaggregation-method`, `--detrend-data/--no-detrend-data`, etc.).

## Downscaling method

Every config must set a top-level `downscaling_method`. There is deliberately no default, so each run records which method produced its output.

| Value | Meaning |
| --- | --- |
| `BCSD` | Detrend, quantile-map, retrend, then spatially disaggregate. |
| `QDMSD` | Quantile delta mapping, then spatially disaggregate. No separate detrend/retrend step. |

```yaml
downscaling_method: "BCSD"
```

The key selects which per-variable defaults table `VariableConfig.for_variable` reads. It
is recorded in the store metadata as `srm_downscaling:downscaling_method`, and it also
namespaces the store layout: every group except the shared `obs/{variable}` lives under a
`bcsd/` or `qdmsd/` segment. Both methods can therefore write to one store and share a
single observation regrid.

To run both methods over identical inputs, use the plural key as a
[matrix axis](#matrix-config-format):

```yaml
downscaling_methods: ["BCSD", "QDMSD"]
```

The pair regrids observations once and fits each method separately, which is cheaper than
two runs because obs deduplication is deliberately method-blind. On the command line the
equivalent is a repeated flag:

```bash
uv run saidownscale run-matrix --gcm CESM2-WACCM6 --variable pr --member 003 --scenario SSP245 \
  --downscaling-method BCSD --downscaling-method QDMSD \
  --predict-period-start 2015 --predict-period-end 2099
```

## Validation Examples

The configuration system catches common errors:

```yaml
# ❌ Missing predict periods for scenario
scenario: "SSP245"
# Error: predict_period_start and predict_period_end must be specified when scenario is set

# ❌ Invalid bounds
subset_bounds: [-35, -40, 16, 33]  # lat_min > lat_max
# Error: lat_min (-35) must be < lat_max (-40)

# ❌ Invalid time periods
train_period_start: 1990
train_period_end: 1980
# Error: train_period_end (1980) must be >= train_period_start (1990)
```
