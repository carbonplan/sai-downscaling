import duckdb
import geopandas as gpd
import icechunk
import pandas as pd
import rasterix
import xarray as xr
from rasterix.rasterize.rasterio import geometry_clip

from srm import catalog

VARMAP = {
    "tas": {
        "CESM": "TREFHT",
        "ERA5": "2m_temperature",
    },
    "tasmax": {
        "CESM": "TREFHTMX",
        "ERA5": "maximum_2m_temperature_since_previous_post_processing",
    },
    "tasmin": {
        "CESM": "TREFHTMN",
        "ERA5": "minimum_2m_temperature_since_previous_post_processing",
    },
    "pr": {
        "CESM": "PRECT",
        "ERA5": "mean_total_precipitation_rate",
    },
    "rsds": {
        "CESM": "FSDS",
        "ERA5": "mean_surface_downward_short_wave_radiation_flux",
    },
}

UNITCONV = {
    "tas": {
        "CESM": 1,
        "ERA5": 1,
    },
    "tasmax": {
        "CESM": 1,
        "ERA5": 1,
    },
    "tasmin": {
        "CESM": 1,
        "ERA5": 1,
    },
    "pr": {
        "CESM": 997,
        "ERA5": 1,
    },
    "rsds": {
        "CESM": 1,
        "ERA5": 1,
    },
}


def get_data(varname: str = "tas"):
    var_cesm = VARMAP[varname]["CESM"]
    var_era5 = VARMAP[varname]["ERA5"]
    unit_conv_cesm_to_era5 = UNITCONV[varname]["CESM"]

    def fix_coords(ds: xr.Dataset):
        ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
        ds = ds.sortby(ssp245.lon)
        return ds

    ssp245_cat = catalog.get("CESM2-WACCM-SSP245-icechunk")

    ssp245_storage = icechunk.s3_storage(
        bucket=ssp245_cat.bucket,
        prefix=ssp245_cat.prefix,
        from_env=True,
    )
    ssp245_repo = icechunk.Repository.open(ssp245_storage)
    ssp245_session = ssp245_repo.readonly_session("main")
    ssp245 = xr.open_zarr(ssp245_session.store, consolidated=False)
    ssp245 = fix_coords(ssp245)
    ssp245 = ssp245.proj.assign_crs(spatial_ref="epsg:4326")
    ssp245 = ssp245.isel(ensemble_member_inferred=0)

    model_historical_cat = catalog.get("CESM-WACCM-Historical-icechunk")

    model_historical_storage = icechunk.s3_storage(
        bucket=model_historical_cat.bucket,
        prefix=model_historical_cat.prefix,
        from_env=True,
    )
    model_historical_repo = icechunk.Repository.open(model_historical_storage)
    model_historical_session = model_historical_repo.readonly_session("main")
    model_historical = xr.open_zarr(model_historical_session.store, consolidated=False)
    model_historical = fix_coords(model_historical)

    g61pt5k_cat = catalog.get("CESM-WACCM-G6-1.5K-icechunk")

    g61pt5k_storage = icechunk.s3_storage(
        bucket=g61pt5k_cat.bucket,
        prefix=g61pt5k_cat.prefix,
        from_env=True,
    )
    g61pt5k_repo = icechunk.Repository.open(g61pt5k_storage)
    g61pt5k_session = g61pt5k_repo.readonly_session("main")
    g61pt5k = xr.open_zarr(g61pt5k_session.store, consolidated=False)
    g61pt5k = fix_coords(g61pt5k)
    g61pt5k = g61pt5k.isel(ensemble_member_inferred=0)

    era5_cat = catalog.get("ERA5")
    era5_storage = icechunk.s3_storage(
        bucket=era5_cat.bucket,
        prefix=era5_cat.prefix,
        from_env=True,
    )
    era5_repo = icechunk.Repository.open(era5_storage)
    era5_session = era5_repo.readonly_session("main")
    era5 = xr.open_zarr(era5_session.store).pipe(rasterix.assign_index)
    era5 = era5.proj.assign_crs(spatial_ref="epsg:4326")

    df = duckdb.sql(
        """install httpfs; load httpfs; install spatial; load spatial; SELECT name, ST_AsText(geom) as geometry FROM ST_Read('https://carbonplan-data.s3.us-west-2.amazonaws.com/countries-50m.json') WHERE NAME = 'South Africa'"""
    ).df()
    df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])
    south_africa_geom = gpd.GeoDataFrame(df, geometry="geometry")

    lon_min, lat_min, lon_max, lat_max = south_africa_geom.total_bounds
    lon_min = lon_min - 2
    lon_max = lon_max + 2
    lat_min = lat_min - 2
    lat_max = lat_max + 2

    era5_south_africa = geometry_clip(
        era5, south_africa_geom[["geometry"]], xdim="longitude", ydim="latitude"
    )

    era5_south_africa_bounds = era5.sel(
        longitude=slice(lon_min, lon_max),
        latitude=slice(lat_max, lat_min),  # Note: descending order for latitude
    )

    model_historical_south_africa_bounds = (
        model_historical.sortby("lat", ascending=False)
        .sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_max, lat_min),  # Note: descending order for latitude
        )
        .load()
    )

    model_historical_south_africa = geometry_clip(
        model_historical.sortby("lat", ascending=False),
        south_africa_geom[["geometry"]],
        xdim="lon",
        ydim="lat",
    ).load()

    ssp245_south_africa_bounds = ssp245.sortby("lat", ascending=False).sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_max, lat_min),  # Note: descending order for latitude
    )

    ssp245_south_africa = geometry_clip(
        ssp245.sortby("lat", ascending=False),
        south_africa_geom[["geometry"]],
        xdim="lon",
        ydim="lat",
    )

    g61pt5k_south_africa_bounds = g61pt5k.sortby("lat", ascending=False).sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_max, lat_min),  # Note: descending order for latitude
    )

    g61pt5k_south_africa = geometry_clip(
        g61pt5k.sortby("lat", ascending=False),
        south_africa_geom[["geometry"]],
        xdim="lon",
        ydim="lat",
    )

    ds_future_dict = {
        "ssp245": ssp245_south_africa_bounds,
        "ssp245_clipped": ssp245_south_africa,
        "g61pt5k": g61pt5k_south_africa_bounds,
        "g61pt5k_clipped": g61pt5k_south_africa,
    }

    ds_hist_dict = {
        "model_hist_clipped": model_historical_south_africa,
        "model_hist": model_historical_south_africa_bounds,
        "era5_clipped": era5_south_africa,
        "era5": era5_south_africa_bounds,
    }

    for key, ds in ds_future_dict.items():
        ds = ds.where(ds["time.year"] >= 2050, drop=True)
        ds = ds.where(ds["time.year"] < 2070, drop=True)
        ds_future_dict[key] = ds[var_cesm] * unit_conv_cesm_to_era5

    for key, ds in ds_hist_dict.items():
        ds = ds.where(ds["time.year"] >= 1978, drop=True)
        ds = ds.where(ds["time.year"] < 2015, drop=True)
        if key in ["model_hist", "model_hist_clipped"]:
            ds_hist_dict[key] = ds[var_cesm] * unit_conv_cesm_to_era5
        else:
            ds_hist_dict[key] = ds[var_era5]

    dict_all = ds_future_dict | ds_hist_dict

    dict_all["era5_coarse"] = dict_all["era5"].interp(
        longitude=dict_all["ssp245"].lon,
        latitude=dict_all["ssp245"].lat,
        method="linear",
    )

    return dict_all


def debias_simulations(debiaser, dict_all: dict):
    tas_cm_hist_debiased = debiaser.apply(
        obs=dict_all["era5_coarse"].values,
        cm_hist=dict_all["model_hist"].values,
        cm_future=dict_all["model_hist"].values,
        time_obs=dict_all["era5_coarse"]["time"].values,
        time_cm_hist=dict_all["model_hist"]["time"].values,
    )

    g61pt5k_fut_debiased = debiaser.apply(
        obs=dict_all["era5_coarse"].values,
        cm_hist=dict_all["model_hist"].values,
        cm_future=dict_all["g61pt5k"].values,
        time_obs=dict_all["era5_coarse"]["time"].values,
        time_cm_hist=dict_all["model_hist"]["time"].values,
        time_cm_future=dict_all["g61pt5k"]["time"].values,
    )

    ssp245_fut_debiased = debiaser.apply(
        obs=dict_all["era5_coarse"].values,
        cm_hist=dict_all["model_hist"].values,
        cm_future=dict_all["ssp245"].values,
        time_obs=dict_all["era5_coarse"]["time"].values,
        time_cm_hist=dict_all["model_hist"]["time"].values,
        time_cm_future=dict_all["ssp245"]["time"].values,
    )

    dict_all["model_hist_debiased"] = xr.DataArray(
        data=tas_cm_hist_debiased,
        coords={
            "lat": dict_all["model_hist"]["lat"],
            "lon": dict_all["model_hist"]["lon"],
            "time": dict_all["model_hist"]["time"],
        },
        dims=["time", "lat", "lon"],
    )

    dict_all["g61pt5k_debiased"] = xr.DataArray(
        data=g61pt5k_fut_debiased,
        coords={
            "lat": dict_all["g61pt5k"]["lat"],
            "lon": dict_all["g61pt5k"]["lon"],
            "time": dict_all["g61pt5k"]["time"],
        },
        dims=["time", "lat", "lon"],
    )

    dict_all["ssp245_debiased"] = xr.DataArray(
        data=ssp245_fut_debiased,
        coords={
            "lat": dict_all["ssp245"]["lat"],
            "lon": dict_all["ssp245"]["lon"],
            "time": dict_all["ssp245"]["time"],
        },
        dims=["time", "lat", "lon"],
    )

    return dict_all


def calculate_error_map(obs_coarse: xr.DataArray, obs_fine: xr.DataArray, dict_all: dict):
    def calculate_doy_means(ds):
        ds_xr = xr.DataArray(
            ds.data,
            dims=ds.dims,
            coords={k: v for k, v in ds.coords.items() if k != "spatial_ref"},
        )

        ds_xr = ds_xr.assign_coords(time=("time", pd.to_datetime(ds["time"].values)))

        ds_xr_doy_mean = ds_xr.groupby("time.dayofyear").mean("time")

        return ds_xr_doy_mean

    obs_coarse_reset = obs_coarse.reset_coords(["longitude", "latitude"], drop=True)
    obs_coarse_on_fine_grid = obs_coarse_reset.interp(
        lon=dict_all["era5"]["longitude"],
        lat=dict_all["era5"]["latitude"],
        method="linear",
    )
    error_map = obs_fine.mean(dim="time") - obs_coarse_on_fine_grid.mean(dim="time")

    obs_fine_doy_means = calculate_doy_means(obs_fine)

    obs_coarse_on_fine_grid_doy_means = calculate_doy_means(obs_coarse_on_fine_grid)

    error_map = obs_fine_doy_means - obs_coarse_on_fine_grid_doy_means

    return error_map


def downscale_from_coarse(da: xr.DataArray, error_map: xr.DataArray, fine_grid: xr.DataArray):
    da_fine_grid = da.interp(
        lon=fine_grid["longitude"],
        lat=fine_grid["latitude"],
        method="linear",
    )

    return da_fine_grid.groupby("time.dayofyear") + error_map
