# Data Access

Input datasets are stored as [icechunk](https://icechunk.io) stores on S3 and exposed through the built-in catalog.

## Listing available datasets

```python
from srm import catalog
print(catalog)
```

```
Dataset Catalog (6 datasets)
+---------------------------------+----------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| Name                            | Format   | Path                                                                                                   | Expected Chunks                                            |
+=================================+==========+========================================================================================================+============================================================+
| CESM2-WACCM-Historical-icechunk | icechunk | s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-Historical/icechunk/CESM2-WACCM-Historical.icechunk | {'time': 13521, 'lat': 8, 'lon': 16}                       |
+---------------------------------+----------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| CESM2-WACCM-G6-1.5K-icechunk    | icechunk | s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-G6-1.5K/icechunk/CESM2-WACCM-G6-1.5k.icechunk       | {'ensemble_member': 1, 'time': 18251, 'lat': 8, 'lon': 16} |
+---------------------------------+----------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| CESM2-WACCM-SSP245-icechunk     | icechunk | s3://carbonplan-srm/input/tensor/CESM2/CESM2-WACCM-SSP245/icechunk/CESM2-WACCM-SSP245.icechunk         | {'ensemble_member': 1, 'time': 20076, 'lat': 8, 'lon': 16} |
+---------------------------------+----------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| MIROC-ES2H-G6-1.5K-icechunk     | icechunk | s3://carbonplan-srm/input/tensor/MIROC-ES2H/MIROC-ES2H-G6-1.5K/updated_MIROC-ES2H-G6-1.5K.icechunk     | {'ensemble_member': 1, 'time': 18263, 'lat': 8, 'lon': 16} |
+---------------------------------+----------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| MIROC-ES2H-baseline-icechunk    | icechunk | s3://carbonplan-srm/input/tensor/MIROC-ES2H/MIROC-ES2H-baseline/updated_MIROC-ES2H-baseline.icechunk   | {'ensemble_member': 1, 'time': 23742, 'lat': 8, 'lon': 16} |
+---------------------------------+----------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| ERA5                            | icechunk  | s3://carbonplan-srm/input/tensor/ERA5/ERA5.icechunk                                                    | {'time': 23741, 'lat': 7, 'lon': 14}                       |
+---------------------------------+----------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
```

## Opening a dataset via the catalog

```python
from srm import catalog

cesm_historical = catalog.get("CESM2-WACCM-Historical-icechunk").to_xarray()
cesm_historical
```

## Opening a dataset manually

If you need lower-level access (e.g. to control the icechunk session):

```python
import icechunk
import xarray as xr
from srm import catalog

ds_meta = catalog.get("CESM2-WACCM-Historical-icechunk")

storage = icechunk.s3_storage(bucket=ds_meta.bucket, prefix=ds_meta.prefix, from_env=True)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, consolidated=False)
print(ds)
```

See also the [subsetting and exporting notebook](../data-access-notebooks/subsetting-and-exporting.ipynb) for examples of working with spatial subsets.
