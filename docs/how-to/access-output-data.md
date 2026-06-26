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
```

| Component | Values | Example |
| --- | --- | --- |
| `scenario_group` | `ssp245`, `g6_1p5k`, `esgf_ssp245` | `ssp245` |
| `variable` | `tas`, `tasmax`, `pr`, `rsds`, … | `tas` |
| `ensemble_member` | e.g. `001`, `002`, `r1i1p1f1` | `001` |

A fully-populated global CESM2-WACCM store would contain groups like
`ssp245/tas/001`, `ssp245/tas/002`, `g6_1p5k/pr/003`, and so on.

## Choosing the right branch

Each pipeline run writes to an icechunk branch named after the installed package version (e.g.
`v1.2.3`). To read data produced by a specific run, you need the branch name that was active when
the run completed. The branch defaults to the `srm` package version at deploy time; production
runs use the version tied to the GitHub release tag.

Use `bcsd status` to check which branches and groups are populated for a given config:

```bash
uv run bcsd status --config-path configs/production/cesm2-waccm.yaml --verbose
```

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

If you are unsure which groups have been written, construct the store path and use
`ArtifactCache.list_groups_on_branch()`:

```{code-cell} python

from srm.cache import ArtifactCache

store_path = "s3://carbonplan-scratch/srm/output/qa/CESM2-WACCM-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk"
cache = ArtifactCache(
    scratch_dir="s3://carbonplan-scratch/srm/cache/",
    environment="qa",
    branch="v2026.6.25.0",
    output_dir="s3://carbonplan-scratch/srm/output/",
)
groups = cache.list_groups_on_branch(store_path)
print(groups)
```

## See Also

- [Pipeline architecture](../explanation/pipeline-architecture.md) — how output stores are structured and written
- [Compare outputs across versions](compare-outputs-across-versions.md) — using branches to validate pipeline changes
- [Run the pipeline](run-pipeline.md) — producing output data from scratch
