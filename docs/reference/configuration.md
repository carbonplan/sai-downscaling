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
scratch_dir: "s3://bucket/path"         # Base directory for intermediate artifacts
output_dir: "s3://bucket/path"        # Directory for final scenario outputs
```

## Optional Fields

```yaml
# Spatial subsetting (null for global)
subset_bounds: [-35, -22, 16, 33]     # [lat_min, lat_max, lon_min, lon_max]

# Environment isolation (default: "qa")
environment: "qa"                      # Environment: qa, staging, production
# Version identifier (default: installed package version, e.g. "v1.0.post12")
# version: "v1.0.post12"              # Override to pin a specific cache namespace

# Variable-specific settings (auto-loaded if not specified)
variable_config:
  detrend_data: true                   # Whether to detrend (auto-set based on variable)
  do_windowing: true                   # Use 31-day running window for QM
  downscaling_method: "additive"       # "additive" for temp, "multiplicative" for precip
  downscaling_clim_method: "fft"       # "fft" or "simple" climatology smoothing

# Runtime options
verbose: true                          # Enable verbose logging (default: true)
rechunk_workflow: true                 # Enable strategic rechunking (default: true)
mapping_type: "parametric"             # QM method: see MappingType in bcsd_config.py
```

## Version Defaulting

The `version` field defaults to the **public version of the installed `srm` package** (e.g. `1.0.post12`), derived via:

```python
from packaging.version import Version
from importlib.metadata import version as pkg_version
"v" + Version(pkg_version("srm")).public  # e.g. "v1.0.post12", strips local/dirty markers
```

This means:
- Each commit merged to `main` automatically gets its own cache namespace (via the `post-release` counter).
- Team members on the same commit share the same cache namespace even if one has a dirty working tree.
- Cache busts only when you intentionally advance the version (i.e. a new release or new commits on `main`).

To **pin** a specific namespace (e.g. to reuse artifacts across a version bump), override explicitly:

```yaml
version: "v1.0.post5"   # pin to an earlier commit's cache
```

## Environment Variable Override

You can override the `environment` field using the `BCSD_ENVIRONMENT` environment variable, and `version` using `BCSD_VERSION`:

```bash
# Override environment for this run
BCSD_ENVIRONMENT=production bcsd run --config-path configs/example.yaml

# Override version for this run
BCSD_VERSION=v1.0.post5 bcsd run --config-path configs/example.yaml
```

You can also override `version` directly on the CLI without editing the config file:

```bash
# Pin to a specific version's cache paths
uv run bcsd run --config-path configs/example.yaml --version v1.0.post5
```

This is useful for:

- testing configs locally with `qa` before running in `production`
- pinning `version` to reuse cached artifacts from a known-good commit
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
