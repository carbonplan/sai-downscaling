---
orphan: true
---

# Contributing

If you want to change the pipeline, start by setting up a development environment, then run the
linters and the tests before you open a pull request. This page covers all 3.

## Environment setup

We manage the environment with [uv](https://docs.astral.sh/uv/), so `uv` is the only thing you
install by hand:

```bash
# Clone the repo
git clone https://github.com/carbonplan/sai-downscaling.git
cd sai-downscaling

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (core + dev + extras)
uv sync --all-groups
```

The 3 sync commands differ in how much they install. Pick the narrowest one that covers what you
are about to do:

| Command | Installs |
| --- | --- |
| `uv sync` | Core dependencies only |
| `uv sync --group dev` | Core dependencies plus the development tooling |
| `uv sync --all-groups` | Everything, including the documentation and QA groups |

## Guidelines

- Contribute via Pull Requests only.
- Add `pytest` unit tests for any new core utilities.
- Run pre-commit checks before pushing (see [Linting](#linting)).
- Add non-core dependencies to an appropriate dependency group in `pyproject.toml`.

## Linting

We lint and format with [`prek`](https://github.com/j178/prek), a drop-in `pre-commit` replacement
that reads the same `.pre-commit-config.yaml`. Run the command below to check every file:

```bash
uv run prek run --all-files
```

## Testing

```bash
# Run the full test suite (parallel, verbose)
uv run pytest tests/ -n auto -vv

# Run a single test
uv run pytest 'tests/test_input_data.py::TestCatalogDatasets::test_variable_units[ERA5]'

# Run a slow test (requires source data access)
uv run pytest 'tests/test_input_data.py::TestCatalogDatasets::test_negative_precip[CESM2-WACCM6-historical-icechunk]' -vv -m slow
```

We decorate slow tests with `@pytest.mark.slow` and exclude them from the default run, because they
read source data from S3. Run them only when the source data itself changes.

## Cloud compute with Coiled

We use [Coiled](https://docs.coiled.io/index.html) for cloud compute, which keeps compute
co-located with the S3 data in `us-west-2`. That avoids egress charges and is much faster than
pulling the data to a laptop.

Start a JupyterLab session on a cloud virtual machine (VM):

```bash
uv run coiled notebook start --vm-type m8g.large --region 'us-west-2' --tag Project=SRM
```

See the [AWS instance types page](https://aws.amazon.com/ec2/instance-types/m8g/) for the available
VM sizes. `m8g.large` is a good starting point.
