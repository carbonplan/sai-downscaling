"""
Reproduces the "KEY FIGURE TO REPRODUCE" from check_doy_reasonable_bounds.ipynb
(cell 27), stripped to the bare minimum needed for that one figure.

At a single point (ilat, ilon), plots the debiased SSP245 rsds time series against
the day-of-year plausibility threshold (max_value_strict: the max of observed and
raw-GCM historical values ever seen at that latitude/day-of-year), circling points
that exceed the threshold. Three panels: full year, Jan-Feb zoom, Nov-Dec zoom.

max_value_strict only needs to be known at ilat, so (unlike the source notebook,
which computes it globally for reuse in other figures) this subsets to ilat before
reducing over lon/time -- same result, far less data to load. Data volume at a
single latitude is small enough that no dask/frisky cluster is needed (cf.
point_debias_diagnostic.py); add one back if S3 reads are slow on your connection.

Run with: .venv/bin/python notebooks/QA_QC/rsds/reproduce_doy_bound_violation_figure.py
"""

import matplotlib.pyplot as plt

from srm import catalog
from srm.utils import open_icechunk

# --------------------------------------------------------------------- config
path_output = "s3://carbonplan-srm/output/production/CESM2-WACCM-ERA5-global.icechunk"
branch = "v0.11.1"
obs_cache_path = "s3://carbonplan-scratch/srm/bcsd-cache/production/CESM2-WACCM-ERA5-global.icechunk"
obs_branch = "v0.11.1"  # scratch/intermediate artifacts always live on "main", per ArtifactCache

var = "rsds"
ssp_ens = "008"
hist_ens = "r3i1p1f1"

ilat, ilon = 80, 0

# --------------------------------------------------------------------- load
coarse_debiased_ssp = open_icechunk(
    path=path_output,
    branch=branch,
    group=f"debiased_coarse/ssp245/{var}/{ssp_ens}",
)[var]

coarse_obs = open_icechunk(
    path=obs_cache_path,
    branch=obs_branch,
    group=f"obs/{var}",
)[var]

raw_historical = (
    catalog.get("CESM2-WACCM").to_xarray()["historical"][var].sel(ensemble_member=hist_ens)
)

# --------------------------------------------------------------------- calc
# The maximum solar radiation on a given day of year for any gridcell is the same
# across longitudes, so max(dim="lon") is the plausibility ceiling at that lat/doy.
coarse_ssp_pt = coarse_debiased_ssp.sel(lon=ilon, lat=ilat, method="nearest")

max_value_obs = (
    coarse_obs.sel(lat=ilat, method="nearest")
    .max(dim="lon")
    .groupby("time.dayofyear")
    .max(dim="time")
    .load()
)
max_value_gcm = (
    raw_historical.isel(time=slice(1, None))
    .sel(lat=ilat, method="nearest")
    .max(dim="lon")
    .groupby("time.dayofyear")
    .max(dim="time")
    .load()
)
max_value_strict = max_value_obs.where(max_value_obs > max_value_gcm, max_value_gcm)

threshold = max_value_strict.sel(dayofyear=coarse_ssp_pt["time.dayofyear"])

# --------------------------------------------------------------------- KEY FIGURE
fig, axes = plt.subplots(ncols=3, figsize=(20, 6))

axes[0].plot(coarse_ssp_pt["time.dayofyear"], coarse_ssp_pt, ".")
axes[0].plot(
    coarse_ssp_pt["time.dayofyear"],
    coarse_ssp_pt.where(coarse_ssp_pt > threshold),
    "o",
    fillstyle="none",
)
max_value_strict.plot(ax=axes[0])

axes[1].plot(coarse_ssp_pt["time.dayofyear"], coarse_ssp_pt, ".")
axes[1].plot(
    coarse_ssp_pt["time.dayofyear"],
    coarse_ssp_pt.where(coarse_ssp_pt > threshold),
    "o",
    fillstyle="none",
)
axes[1].set_xlim([0, 60])
axes[1].set_ylim([0, 1])
max_value_strict.plot(ax=axes[1])

axes[2].plot(coarse_ssp_pt["time.dayofyear"], coarse_ssp_pt, ".")
axes[2].plot(
    coarse_ssp_pt["time.dayofyear"],
    coarse_ssp_pt.where(coarse_ssp_pt > threshold),
    "o",
    fillstyle="none",
)
axes[2].set_xlim([270, 367])
axes[2].set_ylim([0, 0.2])
max_value_strict.plot(ax=axes[2])

plt.show()
