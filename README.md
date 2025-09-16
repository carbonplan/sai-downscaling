# SRM


## Data usage

You can list the available datasets using the built-in catalog.

```python

from srm import catalog
print(catalog)

```

```bash 

Dataset Catalog (4 datasets)
--------------------------------------------------------------------------------
CESM-G6-1.5K-icechunk | icechunk   | s3://carbonplan-srm/input/tensor/CESM-G6-1.5K/icechunk/icechunk
CESM-G6-1.5K-virtual | icechunk   | s3://carbonplan-srm/input/tensor/CESM-G6-1.5K/icechunk/virtual_icechunk
CESM2-WACCM-SSP245-icechunk | icechunk   | s3://carbonplan-srm/input/tensor/CESM2-WACCM-SSP245/icechunk/icechunk
CESM2-WACCM-SSP245-virtual | icechunk   | s3://carbonplan-srm/input/tensor/CESM2-WACCM-SSP245/icechunk/virtual_icechunk
```

### Open Icechunk stores with Xarray

#### CESM2-WACCM-SSP245
```python

#!pip install icechunk xarray
import icechunk
import xarray as xr

# Load CESM-WACCM-SSP245
storage = icechunk.s3_storage(
    bucket='carbonplan-srm', prefix='input/tensor/CESM2-WACCM-SSP245/icechunk/icechunk', from_env=True
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, consolidated=False)
print(ds)
```

#### CESM-G6-1.5K
```python

#!pip install icechunk xarray
import icechunk
import xarray as xr

# Load CESM-G6-1.5K
storage = icechunk.s3_storage(
    bucket='carbonplan-srm', prefix='input/tensor/CESM-G6-1.5K/icechunk/icechunk', from_env=True
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, consolidated=False)
print(ds)
```

#### ERA5

```python

#!pip install icechunk xarray
import icechunk
import xarray as xr

# Load CESM-G6-1.5K
storage = icechunk.s3_storage(
    bucket='carbonplan-srm', prefix='input/tensor/era5_rechunked_resampled.icechunk', from_env=True
)
repo = icechunk.Repository.open(storage)
session = repo.writable_session("main")
ds = xr.open_zarr(session.store, consolidated=False)
print(ds)
```


## Development

**Clone the repo**:

`git clone https://github.com/carbonplan/carbonplan-srm.git`

**Install uv**:

`curl -LsSf https://astral.sh/uv/install.sh | sh` 

[Installation info](https://docs.astral.sh/uv/getting-started/installation/)

**Install the dependencies**:

`uv sync`
