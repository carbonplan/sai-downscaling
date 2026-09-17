<p align="left">
<a href='https://carbonplan.org'>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://carbonplan-assets.s3.amazonaws.com/monogram/light-small.png">
  <img alt="CarbonPlan monogram." height="48" src="https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png">
</picture>
</a>
</p>

# SAI-Downscale

## Scalable downscaling pipeline for Stratospheric Aerosol Injection (SAI) model outputs

This repository implements a scalable, cloud-native pipeline for downscaling SAI climate model outputs. It uses the [BCSD](docs/explanation/scientific-approach.md) (Bias-Correction and Spatial-Disaggregation) and QDMSD (Quantile Delta Mapping - Spatial Disaggregation) methods to spatially downscale daily `CESM2-WACCM6` and `UKESM1-1-LL` GCM output from historical, SSP2-4.5, and G6-1.5K scenarios, plus the CESM G6-1.5K termination-shock run, using daily ERA5 observation data.

> **Note:** This repository reflects the code and infrastructure used for this specific project. It is not maintained as a general-purpose, plug-and-play downscaling tool. Treat it as a reference — a place to borrow patterns, adapt components, or learn from rather than something to run as-is.

## Data access
Pipeline data inputs and outputs are stored on [Source-Coop](https://source.coop/carbonplan/srm-downscaling) in a public AWS `us-west-2` bucket. The data is in the [Icechunk](https://icechunk.io/en/stable/) format, which can be read by tools like [zarr-python](https://icechunk.io/en/stable/getting-started/howto/#reading-writing-and-modifying-data-with-zarr), [Xarray](https://icechunk.io/en/stable/getting-started/howto/#reading-and-writing-data-with-xarray), and others. See [Data access](https://carbonplan.github.io/srm-downscaling/access-data.html) for opening single groups, subsetting, and more.

### Example

Open a single group directly — this is faster than traversing the entire Datatree.

```python
import icechunk
import xarray as xr

storage = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="carbonplan/srm-downscaling/output/production/CESM2-WACCM6-ERA5-global.icechunk",  # or UKESM1-1-LL-ERA5-global.icechunk
    anonymous=True,
    region="us-west-2",
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="v1.0.0")  # current production release

ds = xr.open_zarr(session.store, group="bcsd/g6_1p5k/tas/001")
```

Or open the full store as an `xr.DataTree` to browse all scenario groups, variables, and ensemble members at once — this walks the entire tree and can take a couple minutes.

```python
dt = xr.open_datatree(session.store, engine="zarr")
print(dt)
```

Each GCM is a separate store, and group layouts differ between them — see [Data access](https://carbonplan.github.io/srm-downscaling/access-data.html) for details.

## Documentation
Project documentation: https://carbonplan.github.io/srm-downscaling/

- [Data access](https://carbonplan.github.io/srm-downscaling/access-data.html) — how to list and open input datasets
- [CLI usage](https://carbonplan.github.io/srm-downscaling/reference/cli.html) — running the downscaling pipeline from the command line
- [Scientific approach](https://carbonplan.github.io/srm-downscaling/explanation/scientific-approach.html) — BCSD downscaling approach
- [Pipeline architecture](https://carbonplan.github.io/srm-downscaling/explanation/pipeline-architecture.html) — how the pipeline is structured

## Installation

> [!NOTE]
> Installation is not needed for accessing data.

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/carbonplan/srm-downscaling.git
cd srm-downscaling
uv sync --all-groups
```

## License

MIT — see the LICENSE file for details.

## About Us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/srm-downscaling/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
