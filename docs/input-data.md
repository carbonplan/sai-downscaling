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

Input datasets are stored as [Icechunk](https://icechunk.io) stores on AWS S3 and exposed through the
built-in catalog. The catalog holds three dataset types: `Datatree` (unified per-GCM stores
organised as zarr group trees), `Dataset` (flat icechunk stores), and `VirtualDataset` (virtual
icechunk stores that reference external chunks).

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
NASA-NEX stores are virtual: the Icechunk store holds chunk references that point at the public
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
