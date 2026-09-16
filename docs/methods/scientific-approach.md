# Scientific approach

TK high level overview.

## Input data used

We downscaled three main model scenarios: historical, ssp245, and G6-1.5K. TK what are these scenarios, how do they relate to each other. Big caveat: scenarios =/= reality of how SRM would go, these are idealized, see Sanderson paper.

We also added a new model run.

Figure TK. Super rough schematic as example.

We downscaled a bunch of variables and ensemble members. See below.

Table TK. Number of ensemble members for each model, experiment, and variable.

For observations, we used ERA5.

## Downscaling algorithm

We debiased and downscaled coarse model output using a daily variant of the bias correction / spatial disaggregation (BCSD) method, largely following the implementation from the NASA Earth eXchange Downscaled CMIP6 Climate Projections (NEX-GDDP). This implementation is a daily variant of the monthly method in @Wood2002 and @Wood2004, and consists of four high-level steps: pre-processing data, detrending, bias correction, and spatial disaggregation. We used a [Github repo](TK link specific to tag/commit) and a [technical document](https://TK) as points of reference for understanding the NEX-GDDP implementation. Below, we describe the high-level steps as we implemented them for mean temperature, and then describe variable-specific differences in the implementation.

### Pre-processing data pt 1 (what to call this?)

The raw inputs for this product are model output from GCM simulations at the daily resolution and atmospheric reanalysis from ERA5 at the hourly(?) resolution. The raw GCM inputs are available at TK-link and the ERA5 data is available at TK-link. Because these datasets come from different modeling centers for general use cases, they are not immediately ready for cloud-based statistical downscaling. We pre-processed these files to prepare them for our workflow. Specifically, we processed data so that all variables have the same name and units, GCM ensemble members were labeled in a consistent way, non-physical negative values were removed (TK this is possible because of rounding errors, link to example), TK other preprocessing, and datasets were chunked in a way efficient for our workflow. These preprocessed inputs are available in our data catalog.

After pre-processing model inputs, we went through the BCSD workflow below for each model, experiment, and ensemble member.

### Coarsening training dataset

In BCSD the bias correction requires training a different independent model for each GCM pixel based on observational data at that same resolution. Because the ERA5 training data’s native scale is on a different (i.e. finer) grid from the GCM data, the ERA5 training data must be regridded to match the resolution of each GCM simulation. Because GCMs often have different resolutions, this regridding must be done separately for each GCM. We used the conservative, rectilinear regridding as implemented in the [xarray-regrid](https://xarray-regrid.readthedocs.io/en/latest/index.html) package.

### Detrending

Next, we detrended the GCM model output from future scenarios (ssp245 and G6-1.5K). We did this by first calculating the modeled historical mean monthly climatology from 1978 to 2014 (historical). Then, we calculated the GCM scenario trend as the 9-year running average for each month (e.g. running mean of all Februaries) minus the historical mean monthly climatology. We then removed the trend (either additively for temperature, or multiplicatively for precipitation and solar radiation) from the GCM scenario data and used this detrended GCM data for the subsequent bias correction step. We saved the trend, which we reapplied to the data at the end of the bias correction step.

### Bias correction

Next, we bias corrected the GCM data on each GCM’s coarse, native grid. Then, we used the historical 1978-2014 period to define cumulative distribution functions (CDFs) for quantile mapping from historical detrended GCM to historical coarsened observations. We constructed the CDF separately for each calendar day, based on values in the 31-day window centered on the calendar day being adjusted. For example, the bias correction for February 10 mean temperature in a given grid cell is based on the distribution of all mean temperatures in that grid cell from January 26 to February 25, during the 1978-2014 period. We used the ibicus python package (@spuler2024) to debias the detrended GCM data using nonparametric quantile mapping. After completing the bias correction, we reapplied the GCM trend that we had saved during the detrending step above. The output of this step is debiased coarse GCM data.

### Spatial disaggregation

We then spatially disaggregated the debiased coarse GCM data to the high-resolution observational grid. First, we calculated the daily climatology of the high-resolution observations, which we then smoothed the raw daily climatology using a Fast Fourier Transform, keeping three harmonics. Next, we aggregated the high-resolution smoothed daily climatology to the coarse resolution GCM grid. We then removed the coarse daily climatology from the coarse bias-corrected GCM data. This step creates an anomaly layer that describes, for example, how different, at the coarse scale, a particular February 10 is from the average February 10 in the observations. Next, we bilinearly interpolated the coarse residuals to the high-resolution observational grid. Finally, we returned the high-resolution climatology from step one to the high-resolution residuals. We removed/returned the climatology using subtraction/addition for temperature, and using division/multiplication for other variables (see section TK).

### Variable-specific implementation

The implementation above applies to mean temperature (`tas`) and maximum temperature (`tasmax`). Minimum temperature (`tasmin`) is not bias-corrected directly: following the NASA-NEX approach, we bias-correct `tasmax` and the diurnal temperature range (`dtr = tasmax − tasmin`), reconstruct `tasmin = tasmax − dtr` on the debiased coarse grid, and then — because `tasmax` and `tasmin` are spatially disaggregated independently — swap any fine cells left with `tasmax < tasmin` so that the physical constraint `tasmax >= tasmin` holds everywhere in the published output. Precip: . Special constraints for relative humidity.

## Quality checks

The notebooks below show the quality checks we ran on the inputs and outputs. They live in the
`notebooks/QA_QC/` folder of the GitHub repository.

- [Input data global mean time series](https://github.com/carbonplan/sai-downscaling/blob/main/notebooks/QA_QC/input-data-global-timeseries.ipynb)
- [Plausible value check](https://github.com/carbonplan/sai-downscaling/blob/main/notebooks/QA_QC/plausible-value-check.ipynb)
- [Output integrity checks](https://github.com/carbonplan/sai-downscaling/blob/main/notebooks/QA_QC/output-integrity-checks.ipynb)
- [Trend distortion check](https://github.com/carbonplan/sai-downscaling/blob/main/notebooks/QA_QC/trend-distortion-check.ipynb)
- [Regional run small multiples](https://github.com/carbonplan/sai-downscaling/blob/main/notebooks/QA_QC/regional-run-small-multiples.ipynb)

## References

TK
