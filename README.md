# SRM

| CI          | [![GitHub Workflow Status][github-ci-badge]][github-ci-link] [![Deploy Status][github-ci-slow-badge]][github-ci-slow-link] [![Code Coverage Status][codecov-badge]][codecov-link] [![pre-commit.ci status][pre-commit.ci-badge]][pre-commit.ci-link] |
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

[github-ci-badge]: https://github.com/carbonplan/srm-downscaling/actions/workflows/test.yml/badge.svg
[github-ci-link]: https://github.com/carbonplan/srm-downscaling/actions/workflows/test.yml
[github-ci-slow-badge]: https://github.com/carbonplan/srm-downscaling/actions/workflows/test_slow.yml/badge.svg
[github-ci-slow-link]: https://github.com/carbonplan/srm-downscaling/actions/workflows/test_slow.yml
[codecov-badge]: https://img.shields.io/codecov/c/github/carbonplan/srm-downscaling.svg?logo=codecov
[codecov-link]: https://codecov.io/gh/carbonplan/srm-downscaling
[license-badge]: https://img.shields.io/github/license/carbonplan/srm-downscaling
[repo-link]: https://github.com/carbonplan/srm-downscaling
[pre-commit.ci-badge]: https://results.pre-commit.ci/badge/github/carbonplan/srm-downscaling/main.svg
[pre-commit.ci-link]: https://results.pre-commit.ci/latest/github/carbonplan/srm-downscaling/main
[rtd-badge]: https://readthedocs.org/projects/srm-downscaling/badge/?version=latest
[rtd-link]: https://srm-downscaling.readthedocs.io/en/latest/?badge=latest
