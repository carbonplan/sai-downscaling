# SRM Downscaling

## Quick links

- [**GitHub repository**](https://github.com/carbonplan/sai-downscale)
- [**Dataset releases**](https://github.com/carbonplan/sai-downscale/releases)

## Getting Started

::::{tab-set}
:::{tab-item} Using SRM Downscaled Data
If you want to **access and analyze the downscaled data**:

1. Visit [Access data](./access-data.md) for how to open downscaled output stores, navigate scenarios, and discover available groups.
2. Visit [Input data catalog](./input-data.md) for an overview of the raw GCM, ERA5, and NASA-NEX datasets fed into the pipeline.
3. Check out the [Subsetting and exporting data](./data-access-notebooks/subsetting-and-exporting.ipynb) guide to learn how to work with the data in Python, including loading, exploring, and exporting subsets of the datasets.
:::
:::{tab-item} Running the Pipeline
If you want to **run the downscaling pipeline**:

4. Start with the [Interactive pipeline demo](./how-to/demo-new-pipeline.ipynb) notebook for a hands-on walkthrough with visualizations.
5. Follow the [Run the pipeline](./how-to/run-pipeline.md) guide for installation, quick start, and batch processing.
6. See the [CLI reference](./reference/cli.md) for all commands and options.
7. See the [Configuration reference](./reference/configuration.md) for all config fields and validation rules.
8. See [Deploy the pipeline](./how-to/deploy.md) for QA and production deployment via GitHub Actions.
:::
::::

## Support

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/carbonplan/sai-downscale/issues)
- **General Inquiries**: [hello@carbonplan.org](mailto:hello@carbonplan.org)

## License

SRM Downscaling code is released under the MIT License. See [LICENSE](https://github.com/carbonplan/sai-downscale/blob/main/LICENSE) for details. See [Input data catalog](./input-data.md) for information about data licensing.

```{toctree}
:hidden:
:maxdepth: 2
access-data
input-data
terms-of-data-access
data-access-notebooks/subsetting-and-exporting
contributing
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Methods
explanation/scientific-approach
explanation/pipeline-architecture
explanation/snapshot-testing
explanation/qa-qc/input-data-global-timeseries
explanation/qa-qc/plausible-value-check
explanation/qa-qc/output-integrity-checks
explanation/qa-qc/trend-distortion-check
explanation/qa-qc/regional-run-small-multiples
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/cli
reference/configuration
reference/api/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: How-to guides

how-to/pipeline-stage-debugger
how-to/demo-new-pipeline
how-to/run-pipeline
how-to/manage-cache
how-to/run-snapshot-tests
how-to/snapshot-comparison
how-to/deploy
how-to/regenerate-input-data
how-to/validate-input-data-stores
how-to/ensemble-member-lineage
```
