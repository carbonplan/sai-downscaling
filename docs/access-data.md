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

```
s3://carbonplan-srm/output/{environment}/{gcm}-{obs_dataset}-{subset_id}.icechunk
```

| Component | Values | Example |
| --- | --- | --- |
| `environment` | `qa`, `production` | `production` |
| `gcm` | `CESM2-WACCM`, `MIROC-ES2H`, `UKESM` | `CESM2-WACCM` |
| `obs_dataset` | `ERA5`, `GDEX-GMF` | `ERA5` |
| `subset_id` | `global` or `lat{min}to{max}_lon{min}to{max}` | `global` |

Within each store, data is organised in zarr groups:

```
{scenario_group}/{variable}/{ensemble_member}
debiased_coarse/historical/{variable}/{hist_member}
debiased_coarse/{scenario_group}/{variable}/{ensemble_member}
```

| Component | Values | Example |
| --- | --- | --- |
| `scenario_group` | `ssp245`, `g6_1p5k`, `esgf_ssp245` | `ssp245` |
| `variable` | `tas`, `tasmax`, `pr`, `rsds`, … | `tas` |
| `ensemble_member` | e.g. `001`, `002`, `r1i1p1f1` | `001` |
| `hist_member` | resolved historical parent member | `r1i1p1f1` |

A fully-populated global CESM2-WACCM store would contain groups like `ssp245/tas/001`,
`g6_1p5k/pr/003`, `debiased_coarse/historical/tas/r1i1p1f1`, and
`debiased_coarse/g6_1p5k/tas/001`.

The `debiased_coarse` groups hold GCM data after quantile-mapping bias correction but **before**
spatial disaggregation to ERA5 resolution — they remain at the native coarse GCM grid (~1–2°).
These are useful for research that needs to isolate the bias-correction step from the spatial
downscaling step.

## Choosing the right branch

Each pipeline run writes to an icechunk branch whose name matches the GitHub release tag that
triggered it (e.g. `v2026.6.25.0`). To read a specific run's output, use the corresponding
release tag as the branch name. Production releases are listed at
[github.com/carbonplan/srm-downscaling/releases](https://github.com/carbonplan/srm-downscaling/releases).

## Opening a single variable/member/scenario

```{code-cell} python

import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="carbonplan-scratch",
    prefix="srm/output/qa/CESM2-WACCM-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk",
    from_env=True,
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v2026.6.25.0")  # replace with the actual release branch

ds = xr.open_zarr(
    session.store,
    group="g6_1p5k/tasmax/002",
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
    bucket="carbonplan-scratch",
    prefix="srm/output/qa/CESM2-WACCM-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk",
    from_env=True,
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v2026.6.25.0")  # replace with the actual release branch

dt = xr.open_datatree(
    session.store,
    engine="zarr",
    consolidated=False,
    zarr_format=3,
)
# Access a specific subtree or dataset
g6_tasmax = dt["g6_1p5k/tasmax/002"].to_dataset()
```

## Discovering what is in a store

If you are unsure which groups have been written, open the store as a `DataTree` and inspect it.
Printing the tree shows all available scenario groups, variables, and ensemble members without
loading any data.

```{code-cell} python

import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="carbonplan-scratch",
    prefix="srm/output/qa/CESM2-WACCM-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk",
    from_env=True,
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v2026.6.25.0")

dt = xr.open_datatree(session.store, engine="zarr", consolidated=False, zarr_format=3)
print(dt)
```

## Accessing debiased coarse data

The `debiased_coarse` groups use the same store and branch as the fine-res outputs but live under
a `debiased_coarse/` prefix. Historical coarse data is stored under
`debiased_coarse/historical/{variable}/{hist_member}`; scenario coarse data under
`debiased_coarse/{scenario_group}/{variable}/{ensemble_member}`.

```{code-cell} python

import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="carbonplan-scratch",
    prefix="srm/output/qa/CESM2-WACCM-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk",
    from_env=True,
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v2026.6.30.0")

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
    group="debiased_coarse/g6_1p5k/tas/001",
    consolidated=False,
    zarr_format=3,
    chunks="auto",
)
```

## See Also

- [Pipeline architecture](explanation/pipeline-architecture.md) — how output stores are structured and written
- [Compare outputs across versions](how-to/compare-outputs-across-versions.md) — using branches to validate pipeline changes
- [Run the pipeline](how-to/run-pipeline.md) — producing output data from scratch
- [Input data catalog](input-data.md) — raw GCM, ERA5, and NASA-NEX input datasets
