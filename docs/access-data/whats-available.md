# What's available

We are publishing downscaled, daily climate model output for stratospheric aerosol injection (SAI) scenarios. The outputs cover 4 scenarios, 2 global climate models (GCMs), all available ensemble members, and 5 variables. GCM output is bias-corrected against ERA5 and downscaled to a global 0.25° grid with two methods.

| Property | Value |
| --- | --- |
| Release | `v1.0.0` |
| GCMs | `CESM2-WACCM6`, `UKESM1-1-LL` |
| Downscaling methods | `bcsd` (bias correction and spatial disaggregation), `qdmsd` (quantile delta mapping and spatial disaggregation) |
| Observations | ERA5 |
| Variables | `tas`, `tasmax`, `tasmin`, `pr`, `rsds` |
| Temporal resolution | Daily |
| Spatial extent | Global |

In addition to the historical period (`historical`) and the baseline climate scenario (`ssp245`), we downscale two stratospheric aerosol injection scenarios. In the scenario `g6_1p5k`, greenhouse gas emissions continue at `ssp245` levels while sulfate aerosols are injected into the stratosphere to hold warming to 1.5 °C. The termination shock scenario (`g6_1p5k_end`) extends `g6_1p5k` ensemble member `002` to 2100, simulated an abrupt end of aerosol injection at the end of 2084.

| Scenario | Group name | Years | GCMs |
| --- | --- | --- | --- |
| Historical | `historical` | 1978 to 2014 | Both |
| SSP2-4.5 | `ssp245` | 2015 to 2099 | Both |
| G6-1.5K | `g6_1p5k` | 2035 to 2084 | Both |
| G6-1.5K termination | `g6_1p5k_end` | 2085 to 2100 | `CESM2-WACCM6` only |

`g6_1p5k_end` continues `g6_1p5k` member `002` after SAI stops at the end of 2084. Some
`CESM2-WACCM6` members in `ssp245` end before 2099, as the member tables under
[Data shape](#data-shape) show.

| Product | Description | Grid | Variables |
| --- | --- | --- | --- |
| Downscaled | Bias-corrected and spatially disaggregated. This is the main product. | 0.25° | `tas`, `tasmax`, `tasmin`, `pr`, `rsds` |
| Bias-corrected coarse (`debiased_coarse`) | Bias-corrected, but not spatially disaggregated. Use it to separate the effect of bias correction from the effect of downscaling. | Native GCM grid, about 1° to 2° | The same five, plus `dtr` |

The processed GCM input data that the pipeline started from is also available. It holds daily model
output before bias correction, and it adds `hurs` (near-surface relative humidity) everywhere
except `UKESM1-1-LL` `ssp245`.

## Data location

All data lives in CarbonPlan's
[Source Cooperative repository](https://source.coop/carbonplan/srm-downscaling). The bucket is
public in AWS `us-west-2`, so you don't need AWS credentials to read it. Each {term}`store` is an
{term}`Icechunk` {term}`repository` under
`s3://us-west-2.opendata.source.coop/carbonplan/srm-downscaling/`, at the paths below.

| Data | Path | {term}`Branch <branch>` |
| --- | --- | --- |
| `CESM2-WACCM6` output | `output/production/CESM2-WACCM6-ERA5-global.icechunk` | `v1.0.0` |
| `UKESM1-1-LL` output | `output/production/UKESM1-1-LL-ERA5-global.icechunk` | `v1.0.0` |
| `CESM2-WACCM6` input | `input/processed/CESM2-WACCM6.icechunk` | `main` |
| `UKESM1-1-LL` input | `input/processed/UKESM1-1-LL.icechunk` | `main` |

### Data access

There are two options for accessing the data from the Source Cooperative repository.

If you want to access small quantities of data without writing code yourself, we built a set of [access utilities](./access-utilities.md) that you can use to subset, transform, and export the downscaled data from the cloud to your local environment.

Alternatively, you can use tools like [zarr-python](https://zarr.readthedocs.io/en/latest/), [icechunk](https://icechunk.io/en/stable/getting-started/quickstart/) or [xarray](https://xarray.dev) to stream data from the cloud. The following example opens one downscaled group/variable

```python
import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="carbonplan/srm-downscaling/output/production/CESM2-WACCM6-ERA5-global.icechunk",
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v1.0.0")

ds = xr.open_zarr(session.store, group="bcsd/g6_1p5k/tas/001", consolidated=False)
```

## Data shape

### Group layout

Output stores hold one {term}`Zarr` {term}`group` per method, scenario, variable, and ensemble
member. Both methods publish the same set of groups.

```text
{method}/{scenario}/{variable}/{member}                    # downscaled
{method}/debiased_coarse/{scenario}/{variable}/{member}    # bias-corrected coarse
```

Input stores are organized differently. They hold one group per scenario, with every variable on
Input data stores are organized differently. Groups are defined by per scenario. Opening a single group therefore gives you an array for each variable with the dimensions `(ensemble_member, time, lat, lon)`.

### Ensemble members

Available ensemble members depend on the GCM, scenario, and variable. Both output data products and both methods publish
the same members, and under `debiased_coarse`, `dtr` has the same members as `tasmax` and `tasmin`.

**`CESM2-WACCM6`**

| Scenario | Variables | Members | Years |
| --- | --- | --- | --- |
| `historical` | `pr`, `rsds`, `tas` | `r1i1p1f1`, `r2i1p1f1`, `r3i1p1f1` | 1978 to 2014 |
| `historical` | `tasmax`, `tasmin` | `001` | 1978 to 2014 |
| `ssp245` | `pr`, `rsds`, `tas` | `001` to `005` | 2015 to 2099 |
| `ssp245` | All five | `006` | 2015 to 2068 |
| `ssp245` | All five | `007` to `010` | 2015 to 2069 |
| `g6_1p5k` | All five | `001`, `002`, `003` | 2035 to 2084 |
| `g6_1p5k_end` | All five | `002` | 2085 to 2100 |

For the model `CESM2-WACCM6`, no historical ensemble member carries all five variables, and in `ssp245` only members
`006` to `010` do. To avoid mixing realizations, we advise you pick a member that carries every variable you
need.

**`UKESM1-1-LL`**

| Scenario | Variables | Members | Years |
| --- | --- | --- | --- |
| `historical` | All five | `u-by791` | 1978 to 2014 |
| `ssp245` | All five | `r2i1p1f2`, `r3i1p1f2`, `r12i1p1f2` | 2015 to 2099 |
| `g6_1p5k` | All five | `r2i1p1f2`, `r3i1p1f2`, `r12i1p1f2` | 2035 to 2084 |

### Grid, time, and chunks

| Property | Downscaled | Bias-corrected coarse |
| --- | --- | --- |
| Dimensions | `time`, `lat`, `lon` | `time`, `lat`, `lon` |
| Grid | 0.25°, 721 × 1440 cells | `CESM2-WACCM6`: 192 × 288 cells (about 0.94° × 1.25°); `UKESM1-1-LL`: 144 × 192 cells (1.25° × 1.875°) |
| Longitude convention | -180 to 180 | -180 to 180 |
| Calendar | Proleptic Gregorian | Proleptic Gregorian |
| Data type | `float32` | `float32` |
| {term}`Chunk <chunk>` size (`time`, `lat`, `lon`) | 365 × 36 × 72 (1 year × 9° × 18°) | 365 × 30 × 60 |
| {term}`Shard <shard>` size (`time`, `lat`, `lon`) | 1095 × 180 × 360 (3 years × 45° × 90°) | 1095 × 90 × 180 |

A read fetches whole chunks, so the cost of a request depends on how many chunks it touches rather
than how many values it returns. A time series at a single point reads about one chunk per year,
while a wide region reads many chunks for every year.

### Quality flags

Groups carry quality flags alongside their variable, in both products. Each flag is a `uint8`
{term}`array` where `0` means no known issue and `1` means a known issue. The `dtr` groups under
`debiased_coarse` carry no flags.

| Flag | Dimensions | Present on | Marks |
| --- | --- | --- | --- |
| `qa_flag_time_varying` | `time`, `lat`, `lon` | Every group except `dtr` | Pixel-days that fail a quality check: outlier screening, a variable-specific plausible range, or a physical relationship such as `tasmax` not falling below `tas`. |
| `trend_distortion_flag` | `lat`, `lon` | Every group except `dtr` | Pixels where debiasing or downscaling distorts how scenarios compare with each other or with historical, relative to raw GCM output. It's evaluated on the ensemble mean. |
| `qa_flag_time_invariant` | `lat`, `lon` | Some members only, listed below | Pixels where debiasing or downscaling distorts the same scenario comparisons for a single member, beyond a 5% threshold. |

`qa_flag_time_invariant` is present on these members only. On `CESM2-WACCM6`, they're
`r2i1p1f1`, `r3i1p1f1`, and `001` in `historical`, `003` and `008` in `ssp245`, `002` and `003` in
`g6_1p5k`, and `002` in `g6_1p5k_end`. On `UKESM1-1-LL`, they're `u-by791` and `r2i1p1f2`.

The flags summarize checks run after downscaling. For details, see the
[output integrity checks](../explanation/qa-qc/output-integrity-checks.ipynb) and
[trend distortion check](../explanation/qa-qc/trend-distortion-check.ipynb) pages.

## Glossary

The access data pages use these storage terms in the same sense as the Icechunk, Zarr, and
xarray documentation. Where this dataset uses a term more narrowly, the definition says how.

:::{glossary}
array
  A multidimensional grid of values of a single type, such as `tas` on `(time, lat, lon)`. Zarr
  splits each array into {term}`chunks <chunk>`, and xarray reads it as a variable.

branch
  A named reference to one version of a {term}`repository` that moves forward as new versions are
  written. Every repository has a `main` branch, and each release of this dataset is a branch
  named after the release, such as `v1.0.0`.

chunk
  The piece of an {term}`array` that is stored and read as a unit. A read always fetches whole
  chunks, so the number of chunks a request touches sets its cost.

group
  A container in a {term}`tree` that holds {term}`arrays <array>` and other groups, much like a
  folder holds files. In the output stores, each innermost group, such as `bcsd/ssp245/tas/003`,
  holds one variable and its coordinates, plus any quality flags. xarray opens it as a `Dataset`.

Icechunk
  An open-source storage engine for {term}`Zarr` data that adds version control. Every change is
  saved as an immutable snapshot, and {term}`branches <branch>` name the versions you can open.
  See the [Icechunk documentation](https://icechunk.io/) for details.

repository
  Icechunk's name for a versioned {term}`store`: a Zarr store that holds groups and arrays along
  with their history. You open a repository and then read from one of its
  {term}`branches <branch>`.

shard
  A single stored object that bundles several {term}`chunks <chunk>` of an array. A downscaled
  shard in this dataset bundles 75 chunks, covering 3 years × 45° × 90°. Sharding keeps the number
  of stored objects manageable, while reads still work chunk by chunk.

store
  The place that holds the metadata and data of a {term}`tree` of groups and arrays. In these
  docs, a store is one Icechunk {term}`repository`, and there's one store per GCM for the output
  and one per GCM for the input. In code, a session on a branch gives you the Zarr store to pass
  to xarray, as `session.store`.

tree
  The nested structure of {term}`groups <group>` and {term}`arrays <array>` in a {term}`store`,
  which Zarr calls a hierarchy. A group's path, such as `bcsd/ssp245/tas/003`, spells out its
  place in the tree, and xarray can open a whole tree as a `DataTree`.

Zarr
  An open format for large multidimensional {term}`arrays <array>` that are split into
  {term}`chunks <chunk>` and organized into a {term}`tree` of {term}`groups <group>`. xarray,
  zarr-python, and many other tools can read it.
:::
