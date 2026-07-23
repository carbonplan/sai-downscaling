---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# How to Access Downscaled Output Data

The pipeline writes downscaled output into icechunk stores on S3. Each store holds all scenarios,
variables, and ensemble members for a single GCM/obs-dataset/spatial-subset combination, organised
as zarr groups. This guide shows how to construct the correct store path, open a session on the
right branch, and load data.

## Anatomy of an output store

Output stores live under `s3://carbonplan-srm/output/` and follow this path pattern:

```text
s3://carbonplan-srm/output/{environment}/{gcm}-{obs_dataset}-{subset_id}.icechunk
```

| Component | Values | Example |
| --- | --- | --- |
| `environment` | `qa`, `production` | `production` |
| `gcm` | `CESM2-WACCM`, `MIROC-ES2H`, `UKESM` | `CESM2-WACCM` |
| `obs_dataset` | `ERA5`, `GDEX-GMF` | `ERA5` |
| `subset_id` | `global` or `lat{min}to{max}_lon{min}to{max}` | `global` |

Within each store, data is organised in zarr groups:

```text
historical/{variable}/{hist_member}
{scenario_group}/{variable}/{ensemble_member}
debiased_coarse/historical/{variable}/{hist_member}
debiased_coarse/{scenario_group}/{variable}/{ensemble_member}
```

| Component | Values | Example |
| --- | --- | --- |
| `scenario_group` | `ssp245`, `g6_1p5k`, `esgf_ssp245` | `ssp245` |
| `variable` | `tas`, `tasmax`, `tasmin`, `pr`, `rsds`, `dtr`, `hurs` | `tas` |
| `ensemble_member` | e.g. `001`, `002`, `r1i1p1f1` | `001` |
| `hist_member` | resolved historical parent member | `r1i1p1f1` |

A fully-populated global CESM2-WACCM store would contain groups like `historical/tas/r1i1p1f1`,
`ssp245/tas/003`, `g6_1p5k/pr/003`, `debiased_coarse/historical/tas/r1i1p1f1`, and
`debiased_coarse/g6_1p5k/tas/003`.

The top-level `historical/` group holds the fully downscaled historical data at fine ERA5
resolution — the historical-period analog of the scenario outputs.

The `debiased_coarse` groups hold GCM data after quantile-mapping bias correction but **before**
spatial disaggregation to ERA5 resolution — they remain at the native coarse GCM grid (~1–2°).
These are useful for research that needs to isolate the bias-correction step from the spatial
downscaling step.

## Choosing the right branch

Each pipeline run writes to an icechunk branch whose name matches the release tag that triggered
it. The branch defaults to the installed `srm` package version (e.g. `v0.10.0`), so release tags
follow semantic versioning rather than a date stamp. To read a specific run's output, use the
corresponding release tag as the branch name. Production releases are listed at
[github.com/carbonplan/srm-downscaling/releases](https://github.com/carbonplan/srm-downscaling/releases).

The **current production release is branch `v0.10.0`** of the global CESM2-WACCM store at
`s3://carbonplan-srm/output/production/CESM2-WACCM-ERA5-global.icechunk`. The examples below read
from it anonymously — no AWS credentials are needed for the public production bucket. For
authenticated access (for example to QA stores under `carbonplan-scratch`), replace
`anonymous=True, region="us-west-2"` with `from_env=True`.

## Opening a single variable/member/scenario

```{code-cell} python

import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="carbonplan-srm",
    prefix="output/production/CESM2-WACCM-ERA5-global.icechunk",
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v0.10.0")  # current production release

ds = xr.open_zarr(
    session.store,
    group="g6_1p5k/tasmax/003",
    consolidated=False,
    zarr_format=3,
    chunks="auto",
)
print(ds)
```

## Opening all scenarios as a DataTree

If you want a unified view across scenario groups, open the full store as an `xr.DataTree`. Each
scenario group becomes a node in the tree, and you can navigate or select across them.

```{code-cell} python

import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="carbonplan-srm",
    prefix="output/production/CESM2-WACCM-ERA5-global.icechunk",
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v0.10.0")  # current production release

dt = xr.open_datatree(
    session.store,
    engine="zarr",
    consolidated=False,
    zarr_format=3,
)
# Access a specific subtree or dataset
g6_tasmax = dt["g6_1p5k/tasmax/003"].to_dataset()
```

## Discovering what is in a store

If you are unsure which groups have been written, open the store as a `DataTree` and inspect it.
Printing the tree shows all available scenario groups, variables, and ensemble members without
loading any data.

```{code-cell} python

import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="carbonplan-srm",
    prefix="output/production/CESM2-WACCM-ERA5-global.icechunk",
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v0.10.0")

dt = xr.open_datatree(session.store, engine="zarr", consolidated=False, zarr_format=3)
print(dt)
```

## Inspecting dataset provenance

Every dataset written by the pipeline carries a set of `srm_downscaling:*` attributes that record
the exact configuration used to produce it. These live on `ds.attrs` and are written at the time
the icechunk commit is made, so they travel with the data regardless of how or where it is
accessed.

| Attribute | Contents |
| --- | --- |
| `srm_downscaling:config_json` | Full `BCSDConfig` serialised as a JSON string |
| `srm_downscaling:config_hash` | 12-character SHA-256 of computation-affecting fields only |
| `srm_downscaling:version` | `srm` package version that produced the data |
| `srm_downscaling:gcm` | GCM name |
| `srm_downscaling:scenario` | Scenario (or `"historical"`) |
| `srm_downscaling:variable` | Variable name |
| `srm_downscaling:ensemble_member` | Ensemble member label |
| `srm_downscaling:historical_ensemble_member` | Resolved historical lineage member |
| `srm_downscaling:ssp245_ensemble_member` | Resolved SSP2-4.5 bridge member |
| `srm_downscaling:observation_dataset` | Observation dataset used (e.g. `ERA5`) |
| `srm_downscaling:bias_correction_method` | Quantile-mapping method |
| `srm_downscaling:downscaling_method` | Spatial disaggregation method |
| `srm_downscaling:train_period` | Training period as `"{start}-{end}"` |
| `srm_downscaling:creation_date` | UTC date the artifact was written |

### Comparing a YAML config to a stored dataset

`srm_downscaling:config_json` lets you round-trip a YAML config file directly against the attrs
stored in a dataset — no field-by-field comparison needed.

```python
import yaml
from srm.bcsd_config import BCSDConfig

# Reconstruct BCSDConfig from the YAML you intend to run
with open("configs/production/cesm2-waccm/cesm2-waccm-ssp245-std.yaml") as f:
    yaml_config = BCSDConfig(**yaml.safe_load(f))

# Reconstruct BCSDConfig from what was actually written
stored_config = BCSDConfig.model_validate_json(ds.attrs["srm_downscaling:config_json"])

# Quick equality check (computation-affecting fields only)
yaml_config.config_hash == stored_config.config_hash

# Full field-by-field diff if you need to know what changed
yaml_config.model_dump() == stored_config.model_dump()
```

`config_hash` is the fastest check: it covers only the fields that affect the computed output
(GCM, variable, scenario, periods, subset bounds, mapping type, variable config), so it returns
`True` even if operational fields like `scratch_dir` or `verbose` differ between the two configs.
Use `model_dump()` equality when you need an exact match across all fields.

## Accessing debiased coarse data

The `debiased_coarse` groups use the same store and branch as the fine-res outputs but live under
a `debiased_coarse/` prefix. Historical coarse data is stored under
`debiased_coarse/historical/{variable}/{hist_member}`; scenario coarse data under
`debiased_coarse/{scenario_group}/{variable}/{ensemble_member}`.

```{code-cell} python

import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="carbonplan-srm",
    prefix="output/production/CESM2-WACCM-ERA5-global.icechunk",
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v0.10.0")

# Debiased coarse historical (coarse GCM grid, ~1°)
ds_hist_coarse = xr.open_zarr(
    session.store,
    group="debiased_coarse/historical/tas/r1i1p1f1",
    consolidated=False,
    zarr_format=3,
    chunks="auto",
)

# Debiased coarse scenario (coarse GCM grid, bias-corrected + re-trended)
ds_scen_coarse = xr.open_zarr(
    session.store,
    group="debiased_coarse/g6_1p5k/tas/003",
    consolidated=False,
    zarr_format=3,
    chunks="auto",
)
```

## See Also

- [Pipeline architecture](explanation/pipeline-architecture.md) — how output stores are structured and written
- [Run the pipeline](how-to/run-pipeline.md) — producing output data from scratch
- [Input data catalog](input-data.md) — raw GCM, ERA5, and NASA-NEX input datasets
