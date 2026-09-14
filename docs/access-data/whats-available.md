# What's available

SRM Downscaling publishes daily output from two global climate models (GCMs) for stratospheric
aerosol injection (SAI) scenarios. The output is bias-corrected against ERA5 and downscaled to a
global 0.25° grid with two methods. The current release is `v1.0.0`.

| Property | Value |
| --- | --- |
| Release | `v1.0.0` |
| GCMs | `CESM2-WACCM6`, `UKESM1-1-LL` |
| Downscaling methods | `bcsd` (bias correction and spatial disaggregation), `qdmsd` (quantile delta mapping and spatial disaggregation) |
| Observations | ERA5 |
| Variables | `tas`, `tasmax`, `tasmin`, `pr`, `rsds` |
| Temporal resolution | Daily |
| Spatial extent | Global |

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
public in AWS `us-west-2`, so you don't need AWS credentials to read it. Each store is an
[Icechunk](https://icechunk.io/) repository under
`s3://us-west-2.opendata.source.coop/carbonplan/srm-downscaling/`, at the paths below.

| Data | Path | Branch |
| --- | --- | --- |
| `CESM2-WACCM6` output | `output/production/CESM2-WACCM6-ERA5-global.icechunk` | `v1.0.0` |
| `UKESM1-1-LL` output | `output/production/UKESM1-1-LL-ERA5-global.icechunk` | `v1.0.0` |
| `CESM2-WACCM6` input | `input/processed/CESM2-WACCM6.icechunk` | `main` |
| `UKESM1-1-LL` input | `input/processed/UKESM1-1-LL.icechunk` | `main` |

Always open a branch by name. Output store branches are named after releases, and the `main`
branch of an output store holds no data.

The following example opens one downscaled group from the current release. To subset or download
data without writing this code yourself, use the [access utilities](access-utilities.md).

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

Output stores hold one Zarr group per method, scenario, variable, and ensemble member. Both
methods publish the same set of groups.

```text
{method}/{scenario}/{variable}/{member}                    # downscaled
{method}/debiased_coarse/{scenario}/{variable}/{member}    # bias-corrected coarse
```

`dtr` appears only under `debiased_coarse`, because the pipeline bias-corrects it only to
reconstruct `tasmin`. At 0.25°, compute the diurnal temperature range as `tasmax - tasmin`.

Input stores are organized differently. They hold one group per scenario, with every variable on
`(ensemble_member, time, lat, lon)`.

### Ensemble members

Available members depend on the GCM, scenario, and variable. Both products and both methods publish
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

On `CESM2-WACCM6`, no historical member carries all five variables, and in `ssp245` only members
`006` to `010` do. To avoid mixing realizations, pick a member that carries every variable you
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
| Chunk size (`time`, `lat`, `lon`) | 365 × 36 × 72 (1 year × 9° × 18°) | 365 × 30 × 60 |
| Shard size (`time`, `lat`, `lon`) | 1095 × 180 × 360 (3 years × 45° × 90°) | 1095 × 90 × 180 |

A read fetches whole chunks, so the cost of a request depends on how many chunks it touches rather
than how many values it returns. A time series at a single point reads about one chunk per year,
while a wide region reads many chunks for every year.

### Quality flags

Groups carry quality flags alongside their variable, in both products. Each flag is a `uint8`
array where `0` means no known issue and `1` means a known issue. The `dtr` groups under
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
