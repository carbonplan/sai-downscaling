# SRM Downscaling

## Quick links

- [**GitHub repository**](https://github.com/carbonplan/srm-downscaling)
- [**Dataset releases**](https://github.com/carbonplan/srm-downscaling/releases)

## Getting Started

::::{tab-set}
:::{tab-item} Using SRM Downscaled Data
If you want to **access and analyze the downscaled data**:

1. Visit [Access data](./access-data.md) for an overview of all available datasets and how to access them.
2. Check out the [Subsetting and exporting data](./data-access-notebooks/subsetting-and-exporting.ipynb) guide to learn how to work with the data in Python, including loading, exploring, and exporting subsets of the datasets.
:::
:::{tab-item} Running the Pipeline
If you want to **run the downscaling pipeline**:

3. Start with the [Interactive pipeline demo](./how-to/demo-new-pipeline.ipynb) notebook for a hands-on walkthrough with visualizations.
4. Follow the [Run the pipeline](./how-to/run-pipeline.md) guide for installation, quick start, and batch processing.
5. See the [CLI reference](./reference/cli.md) for all commands and options.
6. See the [Configuration reference](./reference/configuration.md) for all config fields and validation rules.
:::
::::

## Support

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/carbonplan/srm-downscaling/issues)
- **General Inquiries**: [hello@carbonplan.org](mailto:hello@carbonplan.org)

## License

SRM Downscaling code is released under the MIT License. See [LICENSE](https://github.com/carbonplan/srm-downscaling/blob/main/LICENSE) for details. See [Access data](./access-data.md) for information about data licensing.

```{toctree}
:hidden:
:maxdepth: 2
access-data
terms-of-data-access
data-access-notebooks/subsetting-and-exporting
contributing
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Methods

explanation/pipeline-architecture
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/cli
reference/configuration
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: How-to guides

how-to/demo-new-pipeline
how-to/run-pipeline
how-to/manage-cache
how-to/compare-outputs-across-versions
how-to/run-multi-model-ensemble
```
