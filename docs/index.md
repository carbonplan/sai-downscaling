# SAI Downscaling

We downscaled global stratospheric aerosol injection (SAI) climate model output data to a 0.25° grid, so
that researchers can study the potential regional impacts of SAI on domains such as public health,
agriculture, and ecosystems, particularly in the Global South. You can read more about this
project in the accompanying
[explainer](https://carbonplan.org/research/sai-downscaling-explainer). The published dataset
covers two global climate models (GCMs), four scenarios, and five variables at daily resolution, each
bias-corrected against daily ERA5 reanalysis.

We downscaled with two methods rather than one to support the exploration of uncertainty associated
with the choice of method itself. Bias correction and spatial disaggregation (BCSD) is the
established starting point, and quantile delta mapping with spatial disaggregation (QDMSD) carries
the climate trend through its own quantile mapping instead of a separate detrending step. We
publish both for every scenario, variable, and ensemble member.

If you want the data, start with [What's available](./access-data/whats-available.md). If you want
to know how we produced it, start with [Scientific methods](./methods/scientific-approach.md).

We also encourage developers to explore the extensive explanatory information in our
[GitHub repo](https://github.com/carbonplan/sai-downscaling). These resources will help anyone
interested in modifying or extending the pipeline for other use cases. We welcome feedback or
contributions by [opening an issue](https://github.com/carbonplan/sai-downscaling/issues/new).

:::{important}
By downloading or accessing SAI Downscaling data, you agree to the [Terms of Data Access](./access-data/terms-of-data-access.md).
See the [licenses and citation](./access-data/licenses.md) section for guidance specific to each of the products in this data release.
:::

```{toctree}
:hidden:
:maxdepth: 2
:caption: Access the data
access-data/whats-available
access-data/access-utilities
access-data/licenses
access-data/terms-of-data-access
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Methods
methods/scientific-approach
Convenings summary <https://github.com/carbonplan/sai-downscaling/blob/main/docs/assets/convenings-summary.pdf>
```
