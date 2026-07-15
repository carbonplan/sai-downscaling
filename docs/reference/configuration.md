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
ensemble_members: ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]
scenarios: ["SSP245", "G6-1.5K"]
predict_period_start: 2015
predict_period_end: 2100
```

Both singular (`variable`) and plural (`variables`) key names are accepted. All other fields are shared across every combination.

**Restriction:** `variable_config` may not be set when `variables` contains more than one entry — it would silently apply to every variable, including those with incompatible settings (e.g. additive `tas` settings applied to `pr`). Remove it and rely on per-variable defaults (see [Variable-Specific Auto-Configuration](#variable-specific-auto-configuration)), or split into separate files.

## BCSDConfig Fields (run identity)

These fields identify a BCSD run and affect computation results. Changing any of these busts the cache (`config_hash` covers all of them).

```yaml
# Model identifiers (singular or list)
gcm: "CESM2-WACCM"                    # GCM model name
variable: "tas"                        # Variable: tas, tasmax, pr, rsds
ensemble_member: "r1i1p1f1"            # Ensemble member label (e.g. "r1i1p1f1", "01")
scenario: "SSP245"                     # Scenario: SSP245, G6-1.5K, etc. (null for historical-only)

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
  do_windowing: true                   # Use 31-day running window for QM
  downscaling_method: "additive"       # "additive" for temp, "multiplicative" for precip
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

The pipeline automatically sets variable-specific parameters based on `BCSD_CONFIG` defaults:

| Variable | detrend_data | do_windowing | downscaling_method | downscaling_clim_method |
| --- | --- | --- | --- | --- |
| `tas` | `true` | `true` | `additive` | `fft` |
| `tasmax` | `true` | `true` | `additive` | `fft` |
| `pr` | `false` | `true` | `multiplicative` | `simple` |
| `rsds` | `true` | `true` | `multiplicative` | `simple` |

You can override these in the config file if needed.

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
