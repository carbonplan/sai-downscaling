<p align="left">
<a href='https://carbonplan.org'>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://carbonplan-assets.s3.amazonaws.com/monogram/light-small.png">
  <img alt="CarbonPlan monogram." height="48" src="https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png">
</picture>
</a>
</p>

# SAI Downscale

## Scalable downscaling pipeline for Stratospheric Aerosol Injection (SAI) model outputs

This repository implements a scalable, cloud-native pipeline for downscaling Stratospheric Aerosol Injection (SAI) climate model outputs. It uses the [BCSD](docs/explanation/scientific-approach.md) (Bias-Correction and Spatial-Disaggregation)
method to spatially downscale daily `CESM2-WACCM` GCM output from historical, SSP2-4.5, G6-1.5K, and G6-1.5K termination-shock SAI scenarios using daily ERA5 observation data.

> **Note:** This repository reflects the code and infrastructure used for this specific project. It is not maintained as a general-purpose, plug-and-play downscaling tool. Treat it as a reference — a place to borrow patterns, adapt components, or learn from rather than something to run as-is.

## Data access
Pipeline data inputs and outputs are stored on [Source-Coop](https://source.coop/) in a public `us-west-2` bucket. The data is stored in the [Icechunk](https://icechunk.io/en/stable/) format, which can be read by tools like `zarr-python`, `Xarray` and others. See [Data access](docs/access-data.md) for opening single groups, subsetting, and more.

Open the full store as an `xr.DataTree` to browse scenario groups, variables, and ensemble members:

```python
import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="carbonplan/srm-downscaling/output/production/CESM2-WACCM-ERA5-global.icechunk",
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v0.12.0")  # current production release

dt = xr.open_datatree(session.store, engine="zarr", consolidated=False, zarr_format=3)
print(dt)
```

## Documentation
Project documentation: https://carbonplan.github.io/sai-downscale/

- [Data access](docs/access-data.md) — how to list and open input datasets
- [CLI usage](docs/reference/cli.md) — running the downscaling pipeline from the command line
- [Scientific approach](docs/explanation/scientific-approach.md) — BCSD downscaling approach
- [Pipeline architecture](docs/explanation/pipeline-architecture.md) — how the pipeline is structured

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/carbonplan/sai-downscale.git
cd sai-downscale
uv sync --all-groups
```

## License

MIT — see the LICENSE file for details.

## About Us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/sai-downscale/issues/new) or [sending us an email](mailto:hello@carbonplan.org)

[github-ci-badge]: https://github.com/carbonplan/sai-downscale/actions/workflows/test.yml/badge.svg
[github-ci-link]: https://github.com/carbonplan/sai-downscale/actions/workflows/test.yml
[deploy-badge]: https://github.com/carbonplan/sai-downscale/actions/workflows/deploy.yml/badge.svg
[deploy-link]: https://github.com/carbonplan/sai-downscale/actions/workflows/deploy.yml
[codecov-badge]: https://img.shields.io/codecov/c/github/carbonplan/sai-downscale.svg?logo=codecov
[codecov-link]: https://codecov.io/gh/carbonplan/sai-downscale
[license-badge]: https://img.shields.io/github/license/carbonplan/sai-downscale
[repo-link]: https://github.com/carbonplan/sai-downscale
[pre-commit.ci-badge]: https://results.pre-commit.ci/badge/github/carbonplan/sai-downscale/main.svg
[pre-commit.ci-link]: https://results.pre-commit.ci/latest/github/carbonplan/sai-downscale/main
[rtd-badge]: https://readthedocs.org/projects/sai-downscale/badge/?version=latest
[rtd-link]: https://sai-downscale.readthedocs.io/en/latest/?badge=latest
