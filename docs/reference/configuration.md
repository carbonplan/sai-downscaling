# Configuration Reference

Configuration files use YAML format with Pydantic validation. All fields are validated before execution to catch errors early.

## Required Fields

```yaml
# Model identifiers
gcm: "CESM2-WACCM"                    # GCM model name
variable: "tas"                        # Variable: tas, tasmax, pr, rsds
ensemble_member: "r1i1p1f1"            # Ensemble member label (e.g. "r1i1p1f1", "01")
scenario: "SSP245"                     # Scenario: SSP245, G6-1.5K, etc.

# Time periods
train_period_start: 1978               # Training period start year
train_period_end: 2014                 # Training period end year
predict_period_start: 2015             # Prediction period start year (required if scenario set)
predict_period_end: 2100               # Prediction period end year (required if scenario set)

# Storage
cache_dir: "s3://bucket/path"         # Base directory for intermediate artifacts
output_dir: "s3://bucket/path"        # Directory for final scenario outputs
```

## Optional Fields

```yaml
# Spatial subsetting (null for global)
subset_bounds: [-35, -22, 16, 33]     # [lat_min, lat_max, lon_min, lon_max]

# Environment isolation (default: "qa")
environment: "qa"                      # Environment: qa, staging, production
# Version identifier (default: "v1")
version: "v1"                          # Bump to invalidate all cached artifacts without changing environment

# Variable-specific settings (auto-loaded if not specified)
variable_config:
  detrend_data: true                   # Whether to detrend (auto-set based on variable)
  do_windowing: true                   # Use 31-day running window for QM
  downscaling_method: "additive"       # "additive" for temp, "multiplicative" for precip
  downscaling_clim_method: "fft"       # "fft" or "simple" climatology smoothing

# Runtime options
verbose: true                          # Enable verbose logging (default: true)
rechunk_workflow: true                 # Enable strategic rechunking (default: true)
mapping_type: "parametric"             # QM method: "parametric" or "nonparametric"
```

## Environment Variable Override

You can override the `environment` field using the `BCSD_ENVIRONMENT` environment variable, and `version` using `BCSD_VERSION`:

```bash
# Override environment for this run
BCSD_ENVIRONMENT=production bcsd run --config-path configs/example.yaml

# Override version for this run
BCSD_VERSION=v2 bcsd run --config-path configs/example.yaml
```

You can also override `version` directly on the CLI without editing the config file:

```bash
# Write outputs under v2/ paths
uv run bcsd run --config-path configs/example.yaml --version v2
```

This is useful for:

- testing configs locally with `qa` before running in `production`
- bumping `version` to invalidate all cached artifacts (e.g. after a methodological change)
- running the same config in different environments without editing the file
- CI/CD pipelines that deploy to different environments

## Variable-Specific Auto-Configuration

The pipeline automatically sets variable-specific parameters based on `BCSD_CONFIG` defaults:

| Variable | detrend_data | do_windowing | downscaling_method | downscaling_clim_method |
|----------|--------------|--------------|-------------------|------------------------|
| `tas`    | `true`       | `true`       | `additive`        | `fft`                  |
| `tasmax` | `true`       | `true`       | `additive`        | `fft`                  |
| `pr`     | `false`      | `true`       | `multiplicative`  | `simple`               |
| `rsds`   | `true`       | `true`       | `multiplicative`  | `simple`               |

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
