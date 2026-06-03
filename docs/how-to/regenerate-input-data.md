# Regenerate Input Data

The `process input data` workflow re-processes raw GCM NetCDF files into the icechunk stores that the BCSD pipeline reads from. You'd run this when:

- New ensemble members or variables have been added to an existing scenario
- A raw source file was corrected upstream and needs to be re-ingested
- The icechunk store got corrupted or accidentally deleted

The workflow is a thin wrapper around the per-GCM processing scripts in `src/srm/input_data/tensor/CMIP6/`. It supports CESM2-WACCM, MIROC-ES2H, UKESM, and NASA-NEX.

## Triggering the workflow

1. Go to **Actions → process input data → Run workflow**
2. Select a **GCM** from the dropdown
3. Enter a **scenario** (see valid values below)
4. Optionally provide an **S3 cleanup path** if you want to wipe the existing store before reprocessing
5. Click **Run workflow**

## Inputs

| Input | Required | Description |
|-------|----------|-------------|
| `gcm` | yes | GCM to process. One of `CESM2-WACCM`, `MIROC-ES2H`, `UKESM`, `NASA-NEX` |
| `scenario` | yes | Scenario name. Must match exactly — see the table below |
| `s3_cleanup_path` | no | S3 prefix to delete before processing. If left empty, the existing store is left in place and the script will overwrite it. See [When to use S3 cleanup](#when-to-use-s3-cleanup) |
| `extra_flags` | no | Additional flags passed through to the processing script (e.g. `--subset`) |

### Valid scenarios

| GCM | Valid scenarios |
|-----|----------------|
| `CESM2-WACCM` | `pangeo-historical`, `historical`, `ssp245`, `G6-1.5K` |
| `MIROC-ES2H` | `historical`, `ssp245`, `G6-1.5K`, `baseline` |
| `UKESM` | `historical`, `SSP245`, `SSP245-t-pr`, `G6-1.5K`, `G6-1.5K-t-pr` |
| `NASA-NEX` | `historical`, `SSP245` |

Scenario names are case-sensitive and must match the values in the table above exactly.

## When to use S3 cleanup

The `s3_cleanup_path` field accepts an S3 prefix. When provided, the workflow runs `aws s3 rm <path> --recursive` before processing. This is useful when the existing icechunk store is in a bad state and you want a clean start rather than an overwrite.

Common paths:

| GCM | Scenario | S3 cleanup path |
|-----|----------|-----------------|
| CESM2-WACCM | `pangeo-historical` | `s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-Historical/icechunk/pangeo-CESM2-WACCM-historical.icechunk/` |
| CESM2-WACCM | `historical` | `s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-Historical/icechunk/CESM2-WACCM-historical.icechunk/` |
| CESM2-WACCM | `ssp245` | `s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-SSP245/icechunk/CESM2_WACCM_SSP245.icechunk/` |
| CESM2-WACCM | `G6-1.5K` | `s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-G6-1.5K/icechunk/CESM2-WACCM-G6-1.5k.icechunk/` |

:::{admonition} S3 cleanup is irreversible
:class: warning

The workflow does not create a backup before deleting. Double-check the path before triggering — a trailing `/` is required to avoid accidentally deleting adjacent stores.
:::

## Job summary

Once processing completes, the workflow opens the written icechunk store via the `srm.datasets` catalog and appends the xarray `repr` to the [job summary](https://github.blog/news-insights/product-news/supercharging-github-actions-with-job-summaries/). It looks something like this:

```
<xarray.Dataset> Size: 625GB
Dimensions:          (ensemble_member: 10, time: 31391, lat: 192, lon: 288)
Coordinates:
  * ensemble_member  (ensemble_member) object '001' '002' ... '009' '010'
  * time             (time) datetime64[ns] 2015-01-01 ... 2101-01-01
  * lat              (lat) float64 -90.0 -89.06 ... 89.06 90.0
  * lon              (lon) float64 -180.0 -178.8 ... 178.8
Data variables:
    hurs             (ensemble_member, time, lat, lon) float32 dask.array
    pr               (ensemble_member, time, lat, lon) float32 dask.array
    rsds             (ensemble_member, time, lat, lon) float32 dask.array
    tas              (ensemble_member, time, lat, lon) float32 dask.array
    ...
```

This gives you a quick sanity check on dimensions, ensemble members, and variables without having to open the store manually. NASA-NEX does not produce a summary (its output is a virtual store only).
