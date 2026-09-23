<p align="left">
<a href='https://carbonplan.org'>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://carbonplan-assets.s3.amazonaws.com/monogram/light-small.png">
  <img alt="CarbonPlan monogram." height="48" src="https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png">
</picture>
</a>
</p>

## Stratospheric Aerosol Injection (SAI) model outputs

This repository contains downscaled climate model output for Stratospheric Aerosol Injection (SAI) scenarios covering 2 GCMs, 2 downscaling methods, 4 scenarios and multiple ensemble members and variables, totaling ~21TB.

NetCDF files and harmonized Icechunk datacubes for `CESM2-WACCM6` and `UKESM1-1-LL` are stored in this repository under `/input`.

- **GCMs:** `CESM2-WACCM6`, `UKESM1-1-LL`
- **Scenarios:** `historical`, `SSP2-4.5`, `G6-1.5K` and `G6-1.5K-end` (termination-shock run for CESM2-WACCM6)
- **Downscaling methods:** BCSD (Bias-Correction and Spatial-Disaggregation) and QDMSD (Quantile Delta-Mapped Spatial Disaggregation)
- **Observations:** daily ERA5
 
See our [documentation](https://sai-downscaling.readthedocs.io/methods/scientific-approach.html) for more details on our scientific approach.

## Data access

> [!TIP]
> For data access utilities and example notebooks for working with the data check out the [sai-downscaling-data-utils repository](https://github.com/carbonplan/sai-downscaling-data-utils).

The data is stored in the [Icechunk](https://icechunk.io/en/stable/) format, which can be read by tools like `zarr-python`, `Xarray` and others.
You can open a single group directly, which is faster than traversing the entire DataTree.

```python
import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="carbonplan/srm-downscaling/output/production/CESM2-WACCM6-ERA5-global.icechunk",  # or UKESM1-1-LL-ERA5-global.icechunk
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v1.0.0")  # current production release

ds = xr.open_zarr(session.store, group="bcsd/g6_1p5k/tas/001")
```

Or open the full store as an `xr.DataTree` to browse all scenario groups, variables, and ensemble members at once — this walks the entire tree and can take a couple minutes:

```python
dt = xr.open_datatree(session.store, engine="zarr")
print(dt)
```

See the store structure below for the full group layout without opening anything. Each GCM is a separate store, and group layouts differ between them.  See the [Data access](https://sai-downscaling.readthedocs.io/access-data/whats-available.html) documentation page for details.

<details open>
<summary><h3 style="display:inline;">CESM2-WACCM6</h3></summary>

<div style="max-height: 420px; overflow-y: auto;">

| Downscaling Methods | Scenario | Variables | Ensemble Members | Example group path |
|---|---|---|---|---|
| bcsd, qdmsd | g6_1p5k | pr, rsds, tas, tasmax, tasmin | 001, 002, 003 | `bcsd/g6_1p5k/pr/001` |
| bcsd, qdmsd | g6_1p5k_end | pr, rsds, tas, tasmax, tasmin | 002 | `bcsd/g6_1p5k_end/pr/002` |
| bcsd, qdmsd | historical | pr, rsds, tas | r1i1p1f1, r2i1p1f1, r3i1p1f1 | `bcsd/historical/pr/r1i1p1f1` |
| bcsd, qdmsd | historical | tasmax, tasmin | 001 | `bcsd/historical/tasmax/001` |
| bcsd, qdmsd | ssp245 | pr, rsds, tas | 001, 002, 003, 004, 005, 006, 007, 008, 009, 010 | `bcsd/ssp245/pr/001` |
| bcsd, qdmsd | ssp245 | tasmax, tasmin | 006, 007, 008, 009, 010 | `bcsd/ssp245/tasmax/006` |
| bcsd, qdmsd | debiased_coarse/g6_1p5k | dtr, pr, rsds, tas, tasmax, tasmin | 001, 002, 003 | `bcsd/debiased_coarse/g6_1p5k/dtr/001` |
| bcsd, qdmsd | debiased_coarse/g6_1p5k_end | dtr, pr, rsds, tas, tasmax, tasmin | 002 | `bcsd/debiased_coarse/g6_1p5k_end/dtr/002` |
| bcsd, qdmsd | debiased_coarse/historical | dtr, tasmax, tasmin | 001 | `bcsd/debiased_coarse/historical/dtr/001` |
| bcsd, qdmsd | debiased_coarse/historical | pr, rsds, tas | r1i1p1f1, r2i1p1f1, r3i1p1f1 | `bcsd/debiased_coarse/historical/pr/r1i1p1f1` |
| bcsd, qdmsd | debiased_coarse/ssp245 | dtr, tasmax, tasmin | 006, 007, 008, 009, 010 | `bcsd/debiased_coarse/ssp245/dtr/006` |
| bcsd, qdmsd | debiased_coarse/ssp245 | pr, rsds, tas | 001, 002, 003, 004, 005, 006, 007, 008, 009, 010 | `bcsd/debiased_coarse/ssp245/pr/001` |

</div>

</details>

<br>

<details open>
<summary><h3 style="display:inline;">UKESM1-1-LL</h3></summary>

<div style="max-height: 420px; overflow-y: auto;">

| Downscaling Methods | Scenario | Variables | Ensemble Members | Example group path |
|---|---|---|---|---|
| bcsd, qdmsd | g6_1p5k | pr, rsds, tas, tasmax, tasmin | r12i1p1f2, r2i1p1f2, r3i1p1f2 | `bcsd/g6_1p5k/pr/r12i1p1f2` |
| bcsd, qdmsd | historical | pr, rsds, tas, tasmax, tasmin | u-by791 | `bcsd/historical/pr/u-by791` |
| bcsd, qdmsd | ssp245 | pr, rsds, tas, tasmax, tasmin | r12i1p1f2, r2i1p1f2, r3i1p1f2 | `bcsd/ssp245/pr/r12i1p1f2` |
| bcsd, qdmsd | debiased_coarse/g6_1p5k | dtr, pr, rsds, tas, tasmax, tasmin | r12i1p1f2, r2i1p1f2, r3i1p1f2 | `bcsd/debiased_coarse/g6_1p5k/dtr/r12i1p1f2` |
| bcsd, qdmsd | debiased_coarse/historical | dtr, pr, rsds, tas, tasmax, tasmin | u-by791 | `bcsd/debiased_coarse/historical/dtr/u-by791` |
| bcsd, qdmsd | debiased_coarse/ssp245 | dtr, pr, rsds, tas, tasmax, tasmin | r12i1p1f2, r2i1p1f2, r3i1p1f2 | `bcsd/debiased_coarse/ssp245/dtr/r12i1p1f2` |

</div>

</details>

## Documentation

Project documentation: https://sai-downscaling.readthedocs.io

- [Data access](https://sai-downscaling.readthedocs.io/access-data/whats-available.html) — how to list and open input datasets
- [Scientific approach](https://sai-downscaling.readthedocs.io/methods/scientific-approach.html) — BCSD and QDMSD downscaling approaches
- [Explainer article](https://carbonplan.org/research/sai-downscaling-explainer)

## Terms of use

By viewing this data, you agree to CarbonPlan’s [Terms of Use](https://carbonplan.org/terms) and [Privacy Policy](https://carbonplan.org/privacy).
License and attribution information are stored along side the data in colocated LICENSE.txt files.

### Output data
Output data is licensed CC-BY-4.0. Details are in [output/LICENSE.txt](https://data.source.coop/carbonplan/srm-downscaling/output/LICENSE.txt).

### Input data
Input data is covered by multiple licenses. Details can be found in the [licenses](https://sai-downscaling.readthedocs.org/access-data/licenses.html) section of our documentation.
