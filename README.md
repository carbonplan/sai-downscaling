<p align="left" >
<a href='https://carbonplan.org'>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://carbonplan-assets.s3.amazonaws.com/monogram/light-small.png">
  <img alt="CarbonPlan monogram." height="48" src="https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png">
</picture>
</a>
</p>

# SRM

| CI          | [![GitHub Workflow Status][github-ci-badge]][github-ci-link] [![Deploy Status][deploy-badge]][deploy-link] [![Code Coverage Status][codecov-badge]][codecov-link] [![pre-commit.ci status][pre-commit.ci-badge]][pre-commit.ci-link] |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **License** |                                                                                                       [![License][license-badge]][repo-link]                                                                                                       |
| **Docs**    |                                                                                                   [![Documentation Status][rtd-badge]][rtd-link]                                                                                                   |

Scalable downscaling pipeline for Solar Radiation Management (SRM) model outputs.

## Installation

```bash
git clone https://github.com/carbonplan/srm-downscaling.git
cd srm-downscaling
uv sync --all-groups
```

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

## Documentation

- [Data access](docs/data-access.md) — how to list and open input datasets
- [CLI usage](docs/cli-usage.md) — running the downscaling pipeline from the command line
- [Contributing](docs/contributing.md) — environment setup, linting, testing, and cloud compute

## License

> [!IMPORTANT]
> SRM-Downscaling code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About Us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/{repo-name}/issues/new) or [sending us an email](mailto:hello@carbonplan.org)

[github-ci-badge]: https://github.com/carbonplan/srm-downscaling/actions/workflows/test.yml/badge.svg
[github-ci-link]: https://github.com/carbonplan/srm-downscaling/actions/workflows/test.yml
[deploy-badge]: https://github.com/carbonplan/srm-downscaling/actions/workflows/deploy.yml/badge.svg
[deploy-link]: https://github.com/carbonplan/srm-downscaling/actions/workflows/deploy.yml
[codecov-badge]: https://img.shields.io/codecov/c/github/carbonplan/srm-downscaling.svg?logo=codecov
[codecov-link]: https://codecov.io/gh/carbonplan/srm-downscaling
[license-badge]: https://img.shields.io/github/license/carbonplan/srm-downscaling
[repo-link]: https://github.com/carbonplan/srm-downscaling
[pre-commit.ci-badge]: https://results.pre-commit.ci/badge/github/carbonplan/srm-downscaling/main.svg
[pre-commit.ci-link]: https://results.pre-commit.ci/latest/github/carbonplan/srm-downscaling/main
[rtd-badge]: https://readthedocs.org/projects/srm-downscaling/badge/?version=latest
[rtd-link]: https://srm-downscaling.readthedocs.io/en/latest/?badge=latest
