---
orphan: true
---

# Contributing

## Environment setup

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
# Clone the repo
git clone https://github.com/carbonplan/sai-downscaling.git
cd sai-downscaling

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (core + dev + extras)
uv sync --all-groups
```

`uv sync` installs only core dependencies. `uv sync --group dev` adds dev tooling. `uv sync --all-groups` installs everything.

## Guidelines

- Contribute via Pull Requests only.
- Add `pytest` unit tests for any new core utilities.
- Run pre-commit checks before pushing (see [Linting](#linting)).
- Add non-core dependencies to an appropriate dependency group in `pyproject.toml`.

## Linting

We use [`prek`](https://github.com/j178/prek) (a drop-in `pre-commit` replacement that reads the same `.pre-commit-config.yaml`) for linting and code formatting. Run the following command to check all files:

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

Slow tests are decorated with `@pytest.mark.slow` and are intentionally excluded from the default run — only run them when source data is modified.

## Cloud compute with Coiled

[Coiled](https://docs.coiled.io/index.html) is used for cloud compute. It keeps compute co-located with S3 data (us-west-2) to avoid egress and improve performance.

Start a JupyterLab session on a cloud VM:

```bash
uv run coiled notebook start --vm-type m8g.large --region 'us-west-2' --tag Project=SRM
```

See the [AWS instance types page](https://aws.amazon.com/ec2/instance-types/m8g/) for available VM sizes. `m8g.large` is a good starting point.
