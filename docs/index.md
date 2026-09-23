# SAI Downscaling

We downscaled global stratospheric aerosol injection (SAI) climate model output to a 0.25° grid, so
that researchers can study the potential regional impacts of SAI on domains such as public health,
agriculture, and ecosystems, particularly in the Global South. You can read more about this
project in the accompanying
[explainer](https://carbonplan.org/research/sai-downscaling-explainer). The published dataset
covers 2 global climate models (GCMs), 4 scenarios, and 5 variables at daily resolution, each
bias-corrected against daily ERA5 reanalysis.

We downscaled with 2 methods rather than one to support the exploration of uncertainty associated
with the choice of method itself. Bias correction and spatial disaggregation (BCSD) is the
established starting point, and quantile delta mapping with spatial disaggregation (QDMSD) carries
the climate trend through its own quantile mapping instead of a separate detrending step. We
publish both for every scenario, variable, and ensemble member.

If you want the data, start with [What's available](./access-data/whats-available.md). If you want
to know how we produced it, start with [Scientific methods](./methods/scientific-approach.md).

:::{important}
By downloading or accessing SAI Downscaling data, you agree to the
[Terms of Data Access](./access-data/terms-of-data-access.md). Those terms cover every product,
while the [licenses and citation](./access-data/licenses.md) differ per GCM and scenario.
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
methods/pipeline-architecture
Convenings summary <https://github.com/carbonplan/sai-downscaling/blob/main/docs/assets/convenings-summary.pdf>
```
