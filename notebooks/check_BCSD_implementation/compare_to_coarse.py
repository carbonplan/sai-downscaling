import boto3
import icechunk
import matplotlib.pyplot as plt
import xarray as xr

from srm import catalog


def open_icechunk(path):
    bucket, prefix = path.replace("s3://", "").split("/", 1)
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    ds = xr.open_dataset(session.store, engine="zarr", chunks={})
    return ds


def get_fname(var, scenario, version, ens, gcm, spatial_domain):
    fname = (
        "s3://carbonplan-scratch/srm/outputs/qa/"
        + version
        + "/"
        + scenario
        + "/"
        + gcm
        + "/"
        + var
        + "/"
        + ens
        + "/"
        + spatial_domain
        + "/*/"
        + scenario
        + ".icechunk/"
    )

    return fname


def resolve_s3_glob(path):
    """Resolve a single * wildcard in an S3 path to a real path."""
    bucket, prefix = path.replace("s3://", "").split("/", 1)

    before, after = prefix.split("*/", 1)

    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=before, Delimiter="/")

    matches = [f"s3://{bucket}/{cp['Prefix']}{after}" for cp in response.get("CommonPrefixes", [])]

    if not matches:
        raise ValueError(f"No S3 paths matched: {path}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches: {matches}")

    return matches[0]


def get_coarse_data(scenario="ssp245", ens="001", gcm="CESM2-WACCM", future_time_slice=False):
    if scenario == "ssp245":
        catalog_key = gcm + "-SSP245-icechunk"
        if future_time_slice:
            time_slice = slice("2015", "2070")
        else:
            time_slice = slice("2015", "2100")
    elif scenario == "g6-1.5k":
        catalog_key = gcm + "-G6-1.5K-icechunk"
        time_slice = slice("2015", "2084")
    elif scenario == "historical":
        if gcm == "CESM2-WACCM":
            if ens == "001":
                catalog_key = gcm + "-historical-icechunk"
            else:
                catalog_key = "pangeo-" + gcm + "-historical-icechunk"
        else:
            catalog_key = gcm + "-historical-icechunk"
        time_slice = slice("1978", "2014")

    ds = catalog.get(catalog_key).to_xarray()
    ds_subset = (ds.sel(ensemble_member=ens).sel(lon=slice(25, 32), lat=slice(-33, -25))).sel(
        time=time_slice
    )

    return ds_subset


def make_comparison(
    var,
    ens_hist,
    ens_fut,
    version,
    spatial_domain,
    gcm,
    vmax=296,
    vmin=284,
    vmax_delta=2.8,
    vmin_delta=-2.8,
):
    #### Get data

    fname = get_fname(
        var=var,
        scenario="historical",
        ens=ens_hist,
        gcm=gcm,
        version=version,
        spatial_domain=spatial_domain,
    )
    fname = resolve_s3_glob(fname)
    historical_downscaled_var = open_icechunk(path=fname)[var]

    fname = get_fname(
        var=var,
        scenario="ssp245",
        ens=ens_fut,
        gcm=gcm,
        version=version,
        spatial_domain=spatial_domain,
    )
    fname = resolve_s3_glob(fname)
    scenario_downscaled_var = open_icechunk(path=fname)[var]
    scenario_downscaled_var = scenario_downscaled_var.sel(time=slice("2015", "2069"))

    hist_ds = get_coarse_data(scenario="historical", ens=ens_hist, gcm=gcm)
    scenario_ds = get_coarse_data(scenario="ssp245", ens=ens_fut, gcm=gcm)

    #### Make figures
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    ax = axs[0, 0]
    hist_coarse = hist_ds[var].mean(dim="time").load()
    hist_coarse.plot(ax=ax, vmax=vmax, vmin=vmin)

    ax = axs[0, 1]
    scenario_coarse = scenario_ds[var].mean(dim="time").load()
    scenario_coarse.plot(ax=ax, vmax=vmax, vmin=vmin)

    ax = axs[0, 2]
    delta_coarse = scenario_coarse - hist_coarse
    delta_coarse.plot(ax=ax, vmax=vmax_delta, vmin=vmin_delta, cmap=plt.cm.RdBu_r)

    ax = axs[1, 0]
    historical_downscaled = historical_downscaled_var.mean(dim="time").load()
    historical_downscaled.plot(ax=ax, vmax=vmax, vmin=vmin)

    ax = axs[1, 1]
    scenario_downscaled = scenario_downscaled_var.mean(dim="time").load()
    scenario_downscaled.plot(ax=ax, vmax=vmax, vmin=vmin)

    ax = axs[1, 2]
    delta_downscaled = scenario_downscaled - historical_downscaled
    delta_downscaled.plot(ax=ax, vmax=vmax_delta, vmin=vmin_delta, cmap=plt.cm.RdBu_r)

    plt.tight_layout()
    plt.savefig(gcm + "_" + var + "_enshist" + ens_hist + "_ensfut" + ens_fut + ".png", dpi=500)
