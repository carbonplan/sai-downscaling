# SRM

## Development

**Clone the repo**:

`git clone https://github.com/carbonplan/carbonplan-srm.git`

**Install uv**:

`curl -LsSf https://astral.sh/uv/install.sh | sh` 

[Installation info](https://docs.astral.sh/uv/getting-started/installation/)

**Install the dependencies**:

`uv sync --all-groups`


### Using Coiled

[Coiled](https://docs.coiled.io/index.html) is a SaaS we use for cloud computing. It has a lot of options, but the main one for our use case here will be to create JupyterLab sessions on a cloud VM. This allows us to use compute resources that are in the same physical location as where our data lives, which helps in performance and avoids data egress charges.

#### Initial setup
You should install `uv` and install the project dependencies with the commands listed above. This will install coiled as a cli tool in this repo.


#### Start a coiled JupyterLab
With coiled you can choose what size of VM you want to run your JupyterLab session. You can specify which type of VM you wish in the coiled cli commands. A list of some commonly used VM's are [available here](https://aws.amazon.com/ec2/instance-types/m8g/). A good starting point is an `m8g.large`. 

In this repository run:
`uv run coiled notebook start --vm-type m8g.large --region 'us-west-2' --tag Project=SRM`

That should sync the software environment and start a JupyterLab session with that environment. 


## Data usage

You can list the available datasets using the built-in catalog.

```python

from srm import catalog
print(catalog)

```

```bash 

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
| ERA5                            | icechunk | s3://carbonplan-srm/input/tensor/ERA5/ERA5.icechunk                                                    | {'time': 23741, 'lat': 7, 'lon': 14}                       |
+---------------------------------+----------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+

```

### Open datasets directly from the catalog

```python

from srm import catalog 

# Load CESM-WACCM-Historical
cesm_historical = catalog.get("CESM2-WACCM-Historical-icechunk").to_xarray()
cesm_historical
```

### Open datasets manually

```python

import icechunk
import xarray as xr
from srm import catalog 

# Load CESM-WACCM-Historical
ds_meta = catalog.get("CESM2-WACCM-Historical-icechunk")

storage = icechunk.s3_storage(bucket=ds_meta.bucket, prefix=ds_meta.prefix, from_env=True)

repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, consolidated=False)
print(ds)
```


## Development

Some general guidelines:
- Only contribute via Pull Requests (PRs)
- If you add core utilities, please add relevant pytest unit tests. 
- Use pre-commit to lint your code 
- If you add non-core dependencies, add them to a dependency group

### Environment 
This package uses `uv` for environment management. Non-core libraries should be added a dependency group.

`uv sync` will install the core `dependencies` listed in `pyproject.toml`

`uv sync --group dev` will install the core dependencies plus those listed in the group `[dev]`

`uv sync --all-groups` will install dependencies from every listed group.


### Linting

We use `pre-commit` + `ruff` to run linting checks. You can run it against your branch with `uv run pre-commit run --all-files`.

### Testing
`pytest` is used for our unit testing. Make sure you have the test dependencies installed with `uv sync --group dev`.

**Running the test suite:**: `uv run pytest tests/ -n auto -vv`
This will run all the tests using multiple cpu cores and print verbose logs.

**Running a single test**: `uv run pytest 'tests/test_input_data.py::TestCatalogDatasets::test_variable_units[ERA5]'`
This will run a single test with a single input.

**Running a test marked "slow"** `uv run pytest 'tests/test_input_data.py::TestCatalogDatasets::test_negative_precip[CESM2-WACCM-Historical-icechunk]' -vv -m slow`
This will force a test that has been decorated with `@pytest.mark.slow` to run. These tests are usually computationally intensive and should only be run when source data is modified.