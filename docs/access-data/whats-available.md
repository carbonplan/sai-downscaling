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

In addition to the historical period (`historical`) and the baseline climate scenario (`ssp245`), we downscale two stratospheric aerosol injection scenarios. In the scenario `g6_1p5k`, greenhouse gas emissions continue at `ssp245` levels while sulfate aerosols are injected into the stratosphere to hold warming to 1.5 °C. The termination shock scenario (`g6_1p5k_end`) extends `g6_1p5k` ensemble member `002` to 2100, simulating an abrupt end of aerosol injection at the end of 2084.

| Scenario | Group name | Years | GCMs |
| --- | --- | --- | --- |
| Historical | `historical` | 1978 to 2014 | `CESM2-WACCM6` and `UKESM1-1-LL` |
| SSP2-4.5 | `ssp245` | 2015 to 2099 (some `CESM2-WACCM6` ensemble members end in 2068 or 2069) | `CESM2-WACCM6` and `UKESM1-1-LL` |
| G6-1.5K | `g6_1p5k` | 2035 to 2084 | `CESM2-WACCM6` and `UKESM1-1-LL` |
| G6-1.5K termination | `g6_1p5k_end` | 2085 to 2100 | `CESM2-WACCM6` only |

At a high level, the release includes 2 output data products, along with the processed GCM input
data they were built from. The table below summarizes all 3.

| Product | Description | Grid | Variables |
| --- | --- | --- | --- |
| Downscaled | Bias-corrected and spatially disaggregated. This is the main product. | 0.25° | `tas`, `tasmax`, `tasmin`, `pr`, `rsds` |
| Coarse bias-corrected | Bias-corrected, but not spatially disaggregated. Use it to evaluate the effect of bias correction without the spatial disaggregation step. | Native GCM grid, about 1° to 2° | The same five, plus `dtr` (diurnal temperature range) |
| Processed input | Daily GCM output that the pipeline started from, before bias correction. | Native GCM grid, about 1° to 2° | The same five |

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

A {term}`branch` is a version of a store. We publish one branch per release, and so far that is
only `v1.0.0`, which is the branch the example below opens. The input stores keep their data on
`main`.

### Data access

We offer two ways to access the data from the Source Cooperative repository. Which one fits best
depends on how much data you need and whether you want a local copy.

- **Download a local copy.** If you want to work with a small amount of data on your own machine,
  or you prefer netCDF files, we built a set of [access utilities](./access-utilities.md). You can
  use them to subset, transform, and export the downscaled data from the cloud to your local
  environment without writing code yourself.
- **Stream data from the cloud.** If you're comfortable working with data in the cloud without
  keeping a local copy, you can use tools like
  [zarr-python](https://zarr.readthedocs.io/en/latest/),
  [Icechunk](https://icechunk.io/en/stable/getting-started/quickstart/), or
  [xarray](https://xarray.dev). The following example opens one downscaled group from the current
  release.

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

Zarr organizes data in a nested structure that you can think of as a file system. The key building
block is the {term}`group`, a bundle of arrays and metadata at a particular path that you can open
as a single labeled dataset.

Each output {term}`store` holds both data products for one GCM, and each release of that store is
a {term}`branch`. Within a branch, a group's path is built from the method, scenario, variable, and
ensemble member, with an extra `debiased_coarse` level for the coarse bias-corrected product.
Opening one group gives you a dataset with one climate variable on `(time, lat, lon)`, plus any
quality flags. Both methods publish the same set of groups.

```text
{method}/{scenario}/{variable}/{member}                    # downscaled
{method}/debiased_coarse/{scenario}/{variable}/{member}    # coarse bias-corrected
```

Input data stores are organized differently. Each group holds one scenario, so opening a single
group gives you an array for each variable with the dimensions `(ensemble_member, time, lat, lon)`.

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

The `UKESM1-1-LL` historical run is a single model suite rather than one realization of an
ensemble, so it is named by its suite ID, `u-by791`, instead of a label like `r2i1p1f2`. Every
`UKESM1-1-LL` scenario member branches from that one run.

### Grid, time, and chunks

The groups described above tell you which part of the data you're reading. Within a group, each
variable is an {term}`array` that is physically split into {term}`chunks <chunk>`: fixed-size blocks
that are compressed and read as a single unit. A read fetches whole chunks, so the cost of a request
depends on how many chunks it touches rather than how many values it returns. For this dataset, a
time series at a single point reads about one chunk per year, while a wide region reads many chunks
for every year.

Chunks are bundled into larger files called {term}`shards <shard>`. Shards don't change which chunks
a request reads, so you can mostly ignore them when estimating what a request costs. The table below
summarizes the grid, time axis, and chunk layout of both output products.

| Property | Downscaled | Coarse bias-corrected |
| --- | --- | --- |
| Dimensions | `time`, `lat`, `lon` | `time`, `lat`, `lon` |
| Grid | 0.25°, 721 × 1440 cells | `CESM2-WACCM6`: 192 × 288 cells (about 0.94° × 1.25°); `UKESM1-1-LL`: 144 × 192 cells (1.25° × 1.875°) |
| Longitude convention | -180 to 180 | -180 to 180 |
| Calendar | Proleptic Gregorian | Proleptic Gregorian |
| Data type | `float32` | `float32` |
| {term}`Chunk <chunk>` size (`time`, `lat`, `lon`) | 365 × 36 × 72 (1 year × 9° × 18°) | 365 × 30 × 60 |
| {term}`Shard <shard>` size (`time`, `lat`, `lon`) | 1095 × 180 × 360 (3 years × 45° × 90°) | 1095 × 90 × 180 |

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
xarray documentation. Where this dataset uses a term more narrowly, the definition says how. For
terms that come up in the access utilities, such as lazy loading and data read, see the utilities'
[glossary](https://github.com/carbonplan/sai-downscaling-data-utils/blob/main/GLOSSARY.md).

:::{glossary}
array
  A block of values laid out along named dimensions, like a table extended to more than two
  dimensions. In this dataset, `tas` is an array on `(time, lat, lon)`, and each quality flag is an
  array too. Arrays hold the actual numbers, and xarray shows each one as a variable when you open a
  {term}`group`. Zarr splits every array into {term}`chunks <chunk>` for storage.

branch
  A named version of a {term}`repository`. We publish each release of this dataset as a branch named
  after the release, as of now only `v1.0.0`, so you can keep reading the same release even after a
  newer one comes out. When you open a repository, always choose a branch by name. The `main`
  branch of an output store exists but holds no data.

chunk
  A fixed-size block of an {term}`array`, compressed and stored on its own. Splitting arrays into
  chunks means you can read just the part of the data you need, instead of the whole dataset.
  Because a chunk is always read in full, the number of chunks a request touches sets its cost. In
  the downscaled product, one chunk covers 1 year over a 9° × 18° tile, about 3.8 MB before
  compression.

group
  A named container for {term}`arrays <array>` and other groups, much like a folder that holds files
  and subfolders. Groups let a single {term}`store` hold many datasets, so you can open just the one
  you need. In the output stores, the path `bcsd/ssp245/tas/003` points to one group, and opening it
  with xarray gives you a dataset with that variable, its coordinates, and any quality flags.

Icechunk
  An open-source storage engine for {term}`Zarr` data that adds version control, similar to how Git
  tracks changes to code. Every change is saved as a snapshot, and {term}`branches <branch>` give
  names to the versions you can open. We use Icechunk so that you can open any release by name. See
  the [Icechunk documentation](https://icechunk.io/) to learn more.

repository
  Icechunk's word for a {term}`store` that also keeps a history of its versions. In these docs,
  "store" and "repository" refer to the same thing: each GCM's output is one repository, and so is
  each GCM's input. To read data, you open the repository and then choose one of its
  {term}`branches <branch>`, as the example under [Data location](#data-location) shows.

shard
  A bundle of {term}`chunks <chunk>` saved together as a single file in cloud storage, so the store
  holds fewer, larger files. You can mostly ignore shards when you read data. A request still reads
  only the chunks it needs, so shards don't change what it costs. A downscaled shard holds 75
  chunks, covering 3 years over a 45° × 90° tile.

store
  The container that holds a whole {term}`tree` of groups and arrays, along with their metadata. A
  store takes the place that a single file has in formats like netCDF, but its contents are spread
  across many smaller files in cloud storage. We publish one store per GCM for the output and one
  per GCM for the input. In code, `session.store` is the store you pass to xarray or zarr-python.

tree
  The way {term}`groups <group>` nest inside a {term}`store`, like folders within folders. Zarr
  calls this a hierarchy. Each part of a group's path is one level of the tree: in
  `bcsd/ssp245/tas/003`, the levels are the method, scenario, variable, and ensemble member. xarray
  can open a whole tree as a `DataTree`, but when you only need one dataset, opening its group
  directly is quicker.

Zarr
  An open, cloud-friendly format for large multidimensional {term}`arrays <array>`. Instead of
  saving everything in one big file, Zarr stores data as many {term}`chunks <chunk>` organized into
  a {term}`tree` of {term}`groups <group>`, so tools can read just the pieces they need over the
  internet. The stores in this dataset use Zarr version 3, which xarray and zarr-python can read.
:::
