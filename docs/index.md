# SAI Downscaling

We downscaled global stratospheric aerosol injection (SAI) climate model output to a 0.25° grid, so
that researchers can study the potential regional impacts of SAI on domains such as public health,
agriculture, and ecosystems, particularly in the Global South. The published dataset covers 2 global
climate models (GCMs), 4 scenarios, and 5 variables at daily resolution, each bias-corrected against
daily ERA5 reanalysis.

We downscaled with 2 methods rather than one, because the choice of method is itself a source of
uncertainty that users should be able to see in the data. Bias correction and spatial disaggregation
(BCSD) is the established starting point, and quantile delta mapping with spatial disaggregation
(QDMSD) carries the climate trend through its own quantile mapping instead of a separate detrending
step. We publish both for every scenario, variable, and ensemble member.

If you want the data, start with [What's available](./access-data/whats-available.md). If you want
to know how we produced it, start with [Scientific approach](./methods/scientific-approach.md), and
[Developer resources](./developer-resources.md) covers running the pipeline yourself.

## Links

If you want to read the pipeline code, get the data, read about the science, or understand how the
project was shaped, start with the resources below. Each one lives in a different place.

| Resource | What you'll find |
| --- | --- |
| [GitHub repository](https://github.com/carbonplan/sai-downscaling) | Pipeline code, run configs, and issue tracker |
| [Source Cooperative](https://source.coop/carbonplan/srm-downscaling) | Published input and output data |
| [Convenings summary (PDF)](assets/convenings-summary.pdf) | What we heard in the project convenings, and how each recommendation maps onto what we built |
| Explainers | **TODO:** Link to the explainer once it's published. |

## Terms

The code and the data are released under different terms. The table below links to each.

| What | Terms |
| --- | --- |
| Code | [MIT License](https://github.com/carbonplan/sai-downscaling/blob/main/LICENSE) |
| Data | [Access terms](./access-data/access-terms.md) |

```{toctree}
:hidden:
:maxdepth: 2
:caption: Access the data
access-data/whats-available
access-data/access-utilities
access-data/access-terms
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Methods
methods/scientific-approach
methods/pipeline-architecture
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Developer resources
developer-resources
```
