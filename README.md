# SRM

## Development

**Clone the repo**:

`git clone https://github.com/carbonplan/carbonplan-srm.git`

**Install uv**:

`curl -LsSf https://astral.sh/uv/install.sh | sh` 

[Installation info](https://docs.astral.sh/uv/getting-started/installation/)

**Install the dependencies**:

`uv sync --all-groups`

** Linting**

We use `pre-commit` to run linting checks. You can run it against your branch with `uv run pre-commit run --all-files`.

### Using Coiled

[Coiled](https://docs.coiled.io/index.html) is a SaaS we use for cloud computing. It has a lot of options, but the main one for our use case here will be to create JupyterLab sessions on a cloud VM. This allows us to use compute resources that are in the same physical location as where our data lives, which helps in performance and avoids data egress charges.

#### Initial setup
You should install `uv` and install the project dependencies with the commands listed above. This will install coiled as a cli tool in this repo.


#### Start a coiled Jupyterlab
With coiled you can choose what size of VM you want to run your JupyterLab session. You can specify which type of VM you wish in the coiled cli commands. A list of some commonly used VM's are [available here](https://aws.amazon.com/ec2/instance-types/m8g/). A good starting point is an `m8g.large`. 

In this repository run:
`uv run coiled notebook start --vm-type m8g.large --region 'us-west-2` 

That should sync the software environment and start a JupyterLab session with that environment. 


## Data usage

You can list the available datasets using the built-in catalog.

```python

from srm import catalog
print(catalog)

```

```bash 

| CESM-WACCM-Historical-icechunk | icechunk | s3://carbonplan-srm/input/tensor/CESM2-WACCM-Historical/icechunk/icechunk         |
| CESM-WACCM-G6-1.5K-icechunk    | icechunk | s3://carbonplan-srm/input/tensor/CESM-WACCM-G6-1.5K/icechunk/icechunk             |
| CESM2-WACCM-SSP245-icechunk    | icechunk | s3://carbonplan-srm/input/tensor/CESM2-WACCM-SSP245/icechunk/icechunk             |
| ERA5                           | icechunk | s3://carbonplan-srm/input/tensor/era5_rechunked_resampled.icechunk                |

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




