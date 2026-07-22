# Configuration Reference

Configuration files use YAML format with Pydantic validation. All fields are validated before execution to catch errors early.

## Two-class model

Configuration is split across two Pydantic classes loaded from the same flat YAML:

- **`BCSDConfig`** — run identity: the parameters that uniquely identify a BCSD run and affect computation results (model, variable, time periods, bias-correction method). Changes here bust the cache.
- **`PipelineOptions`** — operational settings: storage paths, environment, branch, and runtime flags. Changes here do not affect computation results.

Both classes use `extra="ignore"`, so a single flat YAML file is accepted by both — no nested sections required.

## Matrix config format

Any of the four dimension fields can be a list. `load_configs` expands them into one `BCSDConfig` per cartesian-product combination:

```yaml
gcm: "CESM2-WACCM"                              # singular — still works
variables: ["tas", "pr"]                         # list — expands
ensemble_members: ["001", "002", "003"]
scenarios: ["SSP245"]
predict_period_start: 2015
predict_period_end: 2099                         # must fit every member's data extent (see below)
```

Both singular (`variable`) and plural (`variables`) key names are accepted. All other fields are shared across every combination.

**Restriction:** `variable_config` may not be set when `variables` contains more than one entry — it would silently apply to every variable, including those with incompatible settings (e.g. additive `tas` settings applied to `pr`). Remove it and rely on per-variable defaults (see [Variable-Specific Auto-Configuration](#variable-specific-auto-configuration)), or split into separate files.

## Prediction period and per-member data extents

`predict_period_start` and `predict_period_end` must fall within the valid data extent of every ensemble member the config expands to. Those extents are not uniform: some members are truncated years before the nominal scenario end and the unified store NaN-pads them to that end, so a predict period that overshoots would silently downscale padding. `bcsd run` and `bcsd run-matrix` guard against this with `check_config_time_domain` before submitting any work, raising a blocking error that lists every config whose predict period falls outside its member's bounds.

The extent for a `(gcm, scenario, ensemble_member)` triple is resolved from a per-member override table first, then the scenario's nominal bounds, and is left unchecked when neither is registered. The authoritative table is `_MEMBER_TIME_BOUNDS` in `src/srm/validation.py`; the CESM2-WACCM SSP245 spread is representative:

| Members | Valid end year |
|---|---|
| 001–005 | 2099 |
| 006 | 2069 |
| 007–010 | 2070 |

MIROC-ES2H SSP245 and G6-1.5K members all end 2084, and UKESM SSP245 ends 2099 while its G6-1.5K ends 2084. Because a single config carries one `predict_period`, members with different extents cannot share a config — each extent group needs its own file with a matching `predict_period_end`. SAI/G6 scenarios are the sole start-side asymmetry: `predict_period_start` may precede the scenario's data start (the pipeline bridges the gap with SSP245), so only the end bound is enforced for them. For the workflow of splitting a run across extent groups, see [Ensembles with mixed data extents](../how-to/run-pipeline.md#ensembles-with-mixed-data-extents).

## BCSDConfig Fields (run identity)

These fields identify a BCSD run and affect computation results. Changing any of these busts the cache (`config_hash` covers all of them).

```yaml
# Model identifiers (singular or list)
gcm: "CESM2-WACCM"                    # GCM model name
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

# Bias-correction method
debias_approach: "nonparametric_hybrid_2sided"  # parametric, nonparametric, nonparametric_hybrid, nonparametric_hybrid_2sided (default: "nonparametric_hybrid_2sided")

# Variable-specific settings (auto-loaded from per-variable defaults if not specified)
variable_config:
  detrend_data: true                   # Whether to detrend (auto-set based on variable)
  detrend_method: "additive"           # "additive" or "multiplicative" trend model
  do_windowing: true                   # Use a running window for quantile mapping
  running_window_length: 31            # Running-window length in days (default: 31)
  downscaling_method: "additive"       # "additive" for temperature-like vars, "multiplicative" for pr/rsds
  downscaling_clim_method: "fft"       # "fft" or "simple" climatology smoothing
```

**Required:** `gcm`, `variable`, `ensemble_member`. All others have defaults or are conditionally required (e.g. `predict_period_*` when `scenario` is set).

> **Note:** `debias_approach` is our own field and is a superset of ibicus's `mapping_type` argument. ibicus only accepts `parametric` or `nonparametric`; the `nonparametric_hybrid` and `nonparametric_hybrid_2sided` values are hybrid strategies the pipeline composes on top of ibicus. The former name `mapping_type` was renamed to `debias_approach` — a config still using `mapping_type` now raises an error.

## PipelineOptions Fields (operational)

These fields control storage paths and runtime behavior. They do not affect computation results and are not included in `config_hash`.

```yaml
# Storage paths
scratch_dir: "s3://bucket/path"        # Base directory for intermediate artifacts (default: s3://carbonplan-scratch/srm/cache/)
output_dir: "s3://bucket/path"         # Directory for final downscaled outputs — historical + scenario (default: s3://carbonplan-scratch/srm/outputs/)

# Environment and versioning
environment: "qa"                      # Environment: qa, production (default: "qa")
# branch: "v1.0.post12"              # Override to pin a specific cache namespace (default: installed package version)

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

## Branch Defaulting

The `branch` field defaults to the **public version of the installed `srm` package** (e.g. `v1.0.post12`),
derived via:

```python
from packaging.version import Version
from importlib.metadata import version as pkg_version
"v" + Version(pkg_version("srm")).public  # e.g. "v1.0.post12", strips local/dirty markers
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

You can override the `environment` field using the `BCSD_ENVIRONMENT` environment variable, and
`branch` using `BCSD_BRANCH`:

```bash
# Override environment for this run
BCSD_ENVIRONMENT=production bcsd run --config-path configs/example.yaml

# Override branch for this run
BCSD_BRANCH=v1.0.post5 bcsd run --config-path configs/example.yaml
```

You can also override `branch` directly on the CLI without editing the config file:

```bash
# Pin to a specific branch's cache
uv run bcsd run --config-path configs/example.yaml --branch v1.0.post5
```

This is useful for:

- testing configs locally with `qa` before running in `production`
- pinning `branch` to reuse cached artifacts from a known-good commit
- running the same config in different environments without editing the file
- CI/CD pipelines that deploy to different environments

## Variable-Specific Auto-Configuration

The pipeline automatically sets variable-specific parameters from the per-variable defaults in `VariableConfig.for_variable` (`src/srm/bcsd_config.py`). All variables use a `running_window_length` of `31` days.

| Variable | detrend_data | detrend_method | do_windowing | downscaling_method | downscaling_clim_method |
| --- | --- | --- | --- | --- | --- |
| `tas` | `true` | `additive` | `true` | `additive` | `fft` |
| `tasmax` | `true` | `additive` | `true` | `additive` | `fft` |
| `tasmin` | `true` | `additive` | `true` | `additive` | `fft` |
| `pr` | `false` | `multiplicative` | `true` | `multiplicative` | `fft` |
| `rsds` | `false` | `multiplicative` | `true` | `multiplicative` | `fft` |
| `dtr` | `false` | `multiplicative` | `true` | `multiplicative` | `fft` |
| `hurs` | `false` | `additive` | `true` | `multiplicative` | `fft` |

You can override these per run through the nested `variable_config` block in the config file, or with the `bcsd run-matrix` override flags (`--downscaling-method`, `--detrend-data/--no-detrend-data`, etc.).

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
