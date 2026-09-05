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
variables, and ensemble members for a single GCM/obs-dataset/spatial-subset combination, organized
as zarr groups. This guide shows how to construct the correct store path, open a session on the
right branch, and load data.

## Anatomy of an output store

Published production output lives in CarbonPlan's
[Source Cooperative repository](https://source.coop/carbonplan/srm-downscaling), which is
world-readable and needs no AWS credentials. Store paths follow this pattern:

```text
s3://us-west-2.opendata.source.coop/carbonplan/srm-downscaling/output/production/{gcm}-{obs_dataset}-{subset_id}.icechunk
```

| Component | Values | Example |
| --- | --- | --- |
| `gcm` | `CESM2-WACCM6`, `UKESM1-1-LL` | `CESM2-WACCM6` |
| `obs_dataset` | `ERA5`, `GDEX-GMF` | `ERA5` |
| `subset_id` | `global` or `lat{min}to{max}_lon{min}to{max}` | `global` |

```{note}
Releases up to and including `v0.13.0` predate the model-name correction in issue #598, so
they live under `CESM2-WACCM-ERA5-global.icechunk` and record `srm_downscaling:gcm` as
`CESM2-WACCM`. Later releases live under `CESM2-WACCM6-ERA5-global.icechunk`. The examples on
this page open whichever store holds the current release.
```

Within each store, data is organized in zarr groups:

```text
{method}/historical/{variable}/{hist_member}
{method}/{scenario_group}/{variable}/{ensemble_member}
{method}/debiased_coarse/historical/{variable}/{hist_member}
{method}/debiased_coarse/{scenario_group}/{variable}/{ensemble_member}
```

| Component | Values | Example |
| --- | --- | --- |
| `method` | `bcsd`, `qdmsd` | `bcsd` |
| `scenario_group` | `ssp245`, `g6_1p5k`, `g6_1p5k_end`, `esgf_ssp245` | `ssp245` |
| `variable` | `tas`, `tasmax`, `tasmin`, `pr`, `rsds`, `hurs`; `dtr` (under `debiased_coarse/` only) | — |
| `ensemble_member` | e.g. `003`, `008`, `r3i1p1f1` | `003` |
| `hist_member` | resolved historical parent member | `r3i1p1f1` |

The `{method}` segment names the downscaling method that produced the group. Releases up
to and including `v0.13.0` predate it and have no such segment, so a group path there
begins directly with `historical/`, the scenario group, or `debiased_coarse/`. The code
examples below read `v0.13.0` and use that older layout. Regridded observations live in
the scratch store at `obs/{variable}` with no method segment, because a single regrid is
shared by both methods.

Which variables and members are actually present depends on the release, and members differ between
variables within a single release. See [What the current release
contains](#what-the-current-release-contains) below for the exact inventory before you construct a
group path.

The top-level `historical/` group holds the fully downscaled historical data at fine ERA5
resolution — the historical-period analog of the scenario outputs.

The `debiased_coarse` groups hold GCM data after quantile-mapping bias correction but **before**
spatial disaggregation to ERA5 resolution — they remain at the native coarse GCM grid (~1–2°).
These are useful for research that needs to isolate the bias-correction step from the spatial
downscaling step.

`dtr` is the one variable that appears under `debiased_coarse/` but not under the fine-resolution
groups, because the pipeline bias-corrects it only so that `tasmin = tasmax − dtr` can be
reconstructed. Use `tasmax − tasmin` when you need the diurnal range at fine resolution, and see
[`dtr` is bias-corrected but not published](explanation/pipeline-architecture.md#dtr-is-bias-corrected-but-not-published)
for the full reasoning.

## Choosing the right branch

Each pipeline run writes to an icechunk branch whose name matches the release tag that triggered
it. The branch defaults to the installed `srm` package version (e.g. `v0.13.0`), so release tags
follow semantic versioning rather than a date stamp. To read a specific run's output, use the
corresponding release tag as the branch name. Production releases are listed at
[github.com/carbonplan/srm-downscaling/releases](https://github.com/carbonplan/srm-downscaling/releases).

The **current production release is branch `v0.13.0`** of the global CESM2-WACCM store at
`s3://us-west-2.opendata.source.coop/carbonplan/srm-downscaling/output/production/CESM2-WACCM-ERA5-global.icechunk`.
The examples below read it anonymously, since a Source Cooperative repository needs no AWS
credentials.

The store also carries a `main` branch, but it is an empty anchor commit rather than a run. Reading
it returns no data, so always name a release branch explicitly.

## What the current release contains

Branch `v0.13.0` of the global CESM2-WACCM store holds the groups below. Member labels are not
uniform across variables within a release, so check this table rather than assuming one member
covers every variable.

| Scenario group | Variables | Members |
| --- | --- | --- |
| `historical` | `tas`, `pr`, `rsds`, `hurs` | `r2i1p1f1`, `r3i1p1f1` |
| `historical` | `tasmax`, `tasmin` | `001` |
| `ssp245` | `tas`, `pr`, `rsds`, `hurs` | `003`, `008` |
| `ssp245` | `tasmax`, `tasmin` | `008` |
| `g6_1p5k` | `tas`, `pr`, `rsds`, `hurs`, `tasmax`, `tasmin` | `002`, `003` |
| `g6_1p5k_end` | `tas`, `pr`, `rsds`, `hurs`, `tasmax`, `tasmin` | `002` |

The `debiased_coarse/` subtree holds one coarse-grid group for every fine-grid group listed
above, plus a `dtr` group under each scenario group in the table. Those `dtr` groups are the
only ones with no fine-grid counterpart. This release contains no `esgf_ssp245` group, so
nothing beyond those two sets is present.

## Opening a single variable/member/scenario

```{code-cell} python

import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="carbonplan/srm-downscaling/output/production/CESM2-WACCM-ERA5-global.icechunk",
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v0.13.0")  # current production release

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
    bucket="us-west-2.opendata.source.coop",
    prefix="carbonplan/srm-downscaling/output/production/CESM2-WACCM-ERA5-global.icechunk",
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v0.13.0")  # current production release

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
    bucket="us-west-2.opendata.source.coop",
    prefix="carbonplan/srm-downscaling/output/production/CESM2-WACCM-ERA5-global.icechunk",
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v0.13.0")

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
| `srm_downscaling:config_json` | Full `DownscalingConfig` serialized as a JSON string |
| `srm_downscaling:config_hash` | 12-character SHA-256 of computation-affecting fields only |
| `srm_downscaling:version` | `srm` package version that produced the data |
| `srm_downscaling:gcm` | GCM name |
| `srm_downscaling:gcm_description` | Model description from the catalog, e.g. `CESM2.1.5-WACCM6(TSMLT)`. Absent on releases before the rename. |
| `srm_downscaling:scenario` | Scenario (or `"historical"`) |
| `srm_downscaling:variable` | Variable name |
| `srm_downscaling:ensemble_member` | Ensemble member label |
| `srm_downscaling:historical_ensemble_member` | Resolved historical lineage member |
| `srm_downscaling:ssp245_ensemble_member` | Resolved SSP2-4.5 bridge member |
| `srm_downscaling:observation_dataset` | Observation dataset used (e.g. `ERA5`) |
| `srm_downscaling:bias_correction_method` | Quantile-mapping method |
| `srm_downscaling:downscaling_method` | Downscaling method (`BCSD` or `QDMSD`) |
| `srm_downscaling:disaggregation_method` | Spatial disaggregation method (`additive` or `multiplicative`) |
| `srm_downscaling:train_period` | Training period as `"{start}-{end}"` |
| `srm_downscaling:creation_date` | UTC date the artifact was written |

Regridded observations carry a reduced set. That artifact is shared across every ensemble
member, scenario, and downscaling method for a given GCM, so it records only `version`,
`gcm`, `gcm_description`, `variable`, `observation_dataset`, and `creation_date`. The presence of
`srm_downscaling:downscaling_method` on a group therefore means that group depends on the
downscaling method.

### Comparing a YAML config to a stored dataset

`srm_downscaling:config_json` lets you round-trip a YAML config file directly against the attrs
stored in a dataset — no field-by-field comparison needed.

```python
import yaml
from saidownscale.downscaling_config import DownscalingConfig

# Reconstruct DownscalingConfig from the YAML you intend to run
with open("configs/production/cesm2-waccm6/cesm2-waccm6-ssp245-std.yaml") as f:
    yaml_config = DownscalingConfig(**yaml.safe_load(f))

# Reconstruct DownscalingConfig from what was actually written
stored_config = DownscalingConfig.model_validate_json(ds.attrs["srm_downscaling:config_json"])

# Quick equality check (computation-affecting fields only)
yaml_config.config_hash == stored_config.config_hash

# Full field-by-field diff if you need to know what changed
yaml_config.model_dump() == stored_config.model_dump()
```

On releases up to `v0.13.0`, `srm_downscaling:config_json` records the retired name
`CESM2-WACCM`, and `DownscalingConfig` rejects it with a message pointing at issue #598. To compare
against one of those stores, load the attribute with `json.loads` and compare the fields you
need directly, or map the name to `CESM2-WACCM6` before validating.

`config_hash` is the fastest check: it covers only the fields that affect the computed output
(GCM, variable, scenario, periods, subset bounds, mapping type, variable config), so it returns
`True` even if operational fields like `scratch_dir` or `verbose` differ between the two configs.
Use `model_dump()` equality when you need an exact match across all fields.

## Accessing debiased coarse data

The `debiased_coarse` groups use the same store and branch as the fine-res outputs but live
under a `debiased_coarse/` prefix, nested inside the method segment. Historical coarse data
is stored under `{method}/debiased_coarse/historical/{variable}/{hist_member}`; scenario
coarse data under `{method}/debiased_coarse/{scenario_group}/{variable}/{ensemble_member}`.
On `v0.13.0` and earlier, both paths omit the leading `{method}/`, which is why the example
below opens `debiased_coarse/historical/tas/r3i1p1f1`.

```{code-cell} python

import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="carbonplan/srm-downscaling/output/production/CESM2-WACCM-ERA5-global.icechunk",
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v0.13.0")

# Debiased coarse historical (coarse GCM grid, ~1°)
ds_hist_coarse = xr.open_zarr(
    session.store,
    group="debiased_coarse/historical/tas/r3i1p1f1",
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
