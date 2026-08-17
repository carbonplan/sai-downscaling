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

# Input Data Catalog

Input datasets are stored as [icechunk](https://icechunk.io) stores on S3 and exposed through the
built-in catalog. The catalog holds three dataset types: `Datatree` (unified per-GCM stores
organized as zarr group trees), `Dataset` (flat icechunk stores), and `VirtualDataset` (virtual
icechunk stores that reference external chunks).

## Bucket layout

```
s3://carbonplan-srm/
├── input/
│   ├── raw/                  # source NetCDF exactly as fetched from the modeling centers
│   │   ├── CESM2-WACCM/netcdf/{historical,ssp245,g6-1p5k,g6-1p5k-end}/
│   │   ├── MIROC-ES2H/netcdf/{historical,esgf-ssp245,baseline,g6-1p5k}/
│   │   └── UKESM/netcdf/{historical,ssp245,g6-1p5k,ssp245-t-pr,g6-1p5k-t-pr}/
│   ├── processed/            # unified per-GCM icechunk stores with scenario zarr groups
│   └── vector/               # vector assets (ocean mask)
└── output/                   # downscaled results, one icechunk store per (gcm, obs, subset)
```

### Raw source archive

Raw drops are named for the source run, which is not always the ETL scenario key. The table below
maps each drop directory to the key the ETL modules use to request it:

| GCM | Source drop | ETL scenario key | Contents |
| --- | --- | --- | --- |
| `CESM2-WACCM` | `historical` | `historical` | CMIP6 historical, 1850-2014 |
| `CESM2-WACCM` | `ssp245` | `SSP245` | SSP2-4.5, 2015-2099 |
| `CESM2-WACCM` | `g6-1p5k` | `G6-1.5K` | G6-1.5K SAI, 2035-2084 |
| `CESM2-WACCM` | `g6-1p5k-end` | `G6-1.5K-END` | G6-1.5K termination run, 2085-2100 |
| `MIROC-ES2H` | `historical` | `historical` | CMIP6 historical, 1850-2014 |
| `MIROC-ES2H` | `esgf-ssp245` | `esgf-ssp245` | ESGF SSP2-4.5, 2015-2100 |
| `MIROC-ES2H` | `baseline` | `ssp245` | GeoMIP baseline (SSP245 continuation), from 2020 |
| `MIROC-ES2H` | `g6-1p5k` | `G6-1.5K` | G6-1.5K SAI |
| `UKESM` | `historical` | `historical` | CMIP6 historical |
| `UKESM` | `ssp245` | `SSP245` | SSP2-4.5, primary source |
| `UKESM` | `g6-1p5k` | `G6-1.5K` | G6-1.5K SAI, primary source |
| `UKESM` | `ssp245-t-pr` | `SSP245` | private T/PR archive, source for `pr`/`tas`/`tasmin`/`tasmax` |
| `UKESM` | `g6-1p5k-t-pr` | `G6-1.5K` | private T/PR archive, source for `pr`/`tas`/`tasmin`/`tasmax` |

ERA5, GDEX-GMF and NASA-NEX have no raw copy in this bucket. Their ETLs stream directly from
ARCO-ERA5 on GCS, from OSDF/DTN over HTTPS, and by virtual reference into `s3://nex-gddp-cmip6`
respectively.

## Listing available datasets

```{code-cell} python
from srm import catalog

catalog.list()
```

## Dataset types and paths

| Name | Type | S3 path |
| --- | --- | --- |
| `CESM2-WACCM` | `Datatree` | `s3://carbonplan-srm/input/processed/cesm2-waccm.icechunk` |
| `MIROC-ES2H` | `Datatree` | `s3://carbonplan-srm/input/processed/miroc-es2h.icechunk` |
| `UKESM` | `Datatree` | `s3://carbonplan-srm/input/processed/ukesm.icechunk` |
| `ERA5` | `Dataset` | `s3://carbonplan-srm/input/processed/era5.icechunk` |
| `NASA-NEX-SSP245` | `VirtualDataset` | `s3://carbonplan-srm/input/processed/nasa-nex/ssp245/virtual.icechunk` |
| `NASA-NEX-historical` | `VirtualDataset` | `s3://carbonplan-srm/input/processed/nasa-nex/historical/virtual.icechunk` |
| `GDEX-GMF` | `Dataset` | `s3://carbonplan-srm/input/processed/gdex-gmf.icechunk` |
| `ocean-mask` | `VectorDataset` | `s3://carbonplan-srm/input/vector/GSHHS/GSHHS.parquet` |

## Opening a Datatree dataset (CESM2-WACCM, MIROC-ES2H, UKESM)

`Datatree` entries hold multiple scenarios as zarr group nodes within a single icechunk store.
Calling `.to_xarray()` with no arguments returns the full `xr.DataTree`; passing a `group`
returns a flat `xr.Dataset` for that node only.

```{code-cell} python
from srm import catalog

cesm2_waccm = catalog.get("CESM2-WACCM").to_xarray()
cesm2_waccm
```

```{code-cell} python
cesm2_waccm_historical = catalog.get("CESM2-WACCM").to_xarray(group="historical")
cesm2_waccm_historical
```

```{code-cell} python
cesm2_waccm_ssp245 = catalog.get("CESM2-WACCM").to_xarray(group="ssp245")
cesm2_waccm_ssp245
```

```{code-cell} python
cesm2_waccm_g6 = catalog.get("CESM2-WACCM").to_xarray(group="g6_1p5k")
cesm2_waccm_g6
```

`g6_1p5k_end` is the termination-shock continuation of `g6_1p5k` member `002`: SAI stops at the end of year 2084 and then the termination shock run continues to the end of 2100. The group holds only those years — the 2035–2084 SAI years it continues stay in `g6_1p5k`.`

```{code-cell} python
cesm2_waccm_g6_end = catalog.get("CESM2-WACCM").to_xarray(group="g6_1p5k_end")
cesm2_waccm_g6_end
```

## Opening a flat Dataset (ERA5, GDEX-GMF)

```{code-cell} python
era5 = catalog.get("ERA5").to_xarray()
era5
```

```{code-cell} python
gdex = catalog.get("GDEX-GMF").to_xarray()
gdex
```

## Opening a VirtualDataset (NASA-NEX)

NASA-NEX stores are virtual: the icechunk store holds chunk references that point at the public
`s3://nex-gddp-cmip6/` bucket. No credentials are needed to read NASA-NEX data; the virtual
chunk container is configured for anonymous access automatically.

```{code-cell} python
nex_ssp245 = catalog.get("NASA-NEX-SSP245").to_xarray()
nex_ssp245
```

```{code-cell} python
nex_historical = catalog.get("NASA-NEX-historical").to_xarray()
nex_historical
```

## Opening a dataset with lower-level icechunk control

If you need direct control over the icechunk session (e.g. to pin a specific snapshot or branch):

```{code-cell} python
import icechunk
import xarray as xr

entry = catalog.get("CESM2-WACCM")
storage = icechunk.s3_storage(bucket=entry.bucket, prefix=entry.prefix, from_env=True)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")

dt = xr.open_datatree(session.store, engine="zarr", consolidated=False, zarr_format=3, chunks="auto")
dt
```

See the [subsetting and exporting notebook](data-access-notebooks/subsetting-and-exporting.ipynb)
for examples of loading spatial subsets and exporting to NetCDF.
