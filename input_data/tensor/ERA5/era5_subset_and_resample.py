# adapted from ocr/input_data/tensor/conus404/subset_conus404.py


from typing import Literal, get_args

import coiled
import icechunk
import xarray as xr
import zarr
from distributed import Client
from icechunk.xarray import to_icechunk

# from dataclasses import dataclass

# @dataclass
# class VAR:
#     resampling_method: Literal['mean','max','min']


zarr.config.set({"async.concurrency": 128})


WIND_VARS = Literal["10m_u_component_of_wind", "10m_v_component_of_wind"]
MAX_RESAMPLING = Literal["maximum_2m_temperature_since_previous_post_processing"]
MIN_RESAMPLING = Literal["minimum_2m_temperature_since_previous_post_processing"]
MEAN_RESAMPLING = Literal[
    "mean_total_precipitation_rate",
    "2m_temperature",
    # "surface_pressure"  # TBD
    # "2m_dewpoint_temperature", # TBD
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_surface_downward_long_wave_radiation_flux",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
ERA5_VARS = MAX_RESAMPLING | MIN_RESAMPLING | MEAN_RESAMPLING


INPUT_ZARR_STORE_CONFIG = {
    "url": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
}

DEFAULT_CLUSTER_ARGS = {
    "name": "SRM-resample-era5",
    "region": "us-west-2",
    "n_workers": [10, 100],
    "tags": {"Project": "OCR"},
    "worker_vm_types": ["r8g.medium", "r8g.large"],
    "scheduler_vm_types": "c8g.8xlarge",
    "spot_policy": "spot_with_fallback",
    "use_best_zone": True,
}

DEFAULT_STORAGE_CONFIG = {
    "bucket": "carbonplan-srm",
    "prefix": "input/tensor/era5_rechunked_resampled.icechunk",
    "region": "us-west-2",
}


def setup_cluster(cluster_args: dict) -> Client:
    """Set up a Coiled cluster with the specified arguments.

    Parameters
    ----------
    cluster_args: dict
        Arguments to pass to the Coiled cluster setup.

    Returns
    -------
    Client: A Dask client connected to the Coiled cluster.
    """
    args = cluster_args
    try:
        cluster = coiled.Cluster(**args)
        client = cluster.get_client()

        return client
    except Exception as exc:
        raise (f"Error setting up cluster: {exc}")


def setup_repository(storage_config: dict) -> tuple:
    import icechunk

    """Set up and return an icechunk repository.

    Parameters
    ----------
    storage_config: dict
        Configuration for the storage backend, including bucket, prefix, and region.

    Returns
    -------
    tuple: A tuple containing the icechunk repository and a writable session.
        Tuple of (repository, session)
    """
    config = storage_config

    storage = icechunk.s3_storage(
        bucket=config["bucket"],
        prefix=config["prefix"],
        region=config["region"],
    )

    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session("main")

    return repo, session


def resample_time(
    ds: xr.Dataset, start_year: int, end_year: int, variable: ERA5_VARS
) -> xr.Dataset:
    try:
        pass
    except Exception as exc:
        raise ImportWarning(
            f"flox is not installed. Add it to greatly speedup resampling operations. {exc}"
        )

    if variable in get_args(MAX_RESAMPLING):
        return ds.sel(time=slice(f"{start_year}", f"{end_year}")).resample(time="d").max()
    elif variable in get_args(MIN_RESAMPLING):
        return ds.sel(time=slice(f"{start_year}", f"{end_year}")).resample(time="d").min()
    elif variable in get_args(MEAN_RESAMPLING):
        return ds.sel(time=slice(f"{start_year}", f"{end_year}")).resample(time="d").mean()
    else:
        raise ValueError(f"variable: {variable} is not in {ERA5_VARS}")


def convert_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Convert 0-360 longitude to -180-180"""
    ds.coords["longitude"] = (ds.coords["longitude"] + 180) % 360 - 180
    return ds.sortby(ds.longitude)


def load_dataset(variables: ERA5_VARS, start_year: int = 1950, end_year: int = 2014):
    """Loads variable[s] from the public gcs ERA5 store.

    Parameters
    ----------
    variable: ERA5_VARS
        The variable[s] to load from the dataset.


    Returns
    -------
    xr.Dataset: An Xarray Dataset with the selected variables
    """

    ds = xr.open_zarr(INPUT_ZARR_STORE_CONFIG["url"])[[variables]]

    # subset time
    ds = ds.sel(time=slice(f"{start_year}", f"{end_year}"))

    ds = convert_longitude(ds)
    # TODO: derive vars (ex wind speed) 2->1

    return ds


def process_dataset(
    ds: xr.Dataset, variable: ERA5_VARS, start_year: int = 1950, end_year: int = 2014
) -> xr.Dataset:
    """Load a specific variable from the public gcs store

    Parameters
    ----------
    variable: str
        The variable to load from the dataset.


    Returns
    -------
    xr.Dataset: An Xarray Dataset with the selected variables
    """
    # ~100Mb chunks, but not split spatially. Getting larger chunks to reduce scheduler task pressure
    ds = ds.chunk({"time": 48, "latitude": 721, "longitude": 1440})

    ds = resample_time(ds=ds, start_year=start_year, end_year=end_year, variable=variable)

    # ~115MB, some spatial chunking
    ds = ds.chunk({"time": 730, "latitude": 144, "longitude": 288})

    # wait(ds)

    # Clean up encoding to avoid conflicts
    for variable_ in ds.variables:
        ds[variable_].encoding.pop("chunks", None)
        ds[variable_].encoding.pop("preferred_chunks", None)
        ds[variable_].encoding.pop("compressors", None)

    ds.attrs.update(
        {
            "valid_time_start": "1950-01-01",
            "valid_time_stop": "2014-12-31",
            "valid_time_stop_era5t": "2014-12-31",
        }
    )
    return ds


def write_dataset(ds: xr.Dataset, variable: ERA5_VARS, session: icechunk.Session):
    to_icechunk(ds, session, mode="a")
    session.commit(f"{variable}")


def main(ds: xr.Dataset, variable: ERA5_VARS, start_year: int, end_year: int):
    # TODO add step to derive vars (wind_speed, rh etc.)
    ds = process_dataset(ds, variable=variable, start_year=start_year, end_year=end_year)
    write_dataset(ds, variable=variable, session=session)


if __name__ == "__main__":
    client = setup_cluster(DEFAULT_CLUSTER_ARGS)
    _, session = setup_repository(DEFAULT_STORAGE_CONFIG)

    # for variable in get_args(ERA5_VARS):
    #     ds = load_dataset(variables=variable, start_year=start_year, end_year=end_year)
    #     main(variable=variable, start_year=1950, end_year=2014)

    # import xclim

    # # winds = xclim.indicators.convert.wind_speed_from_vector(
    # #     uas=ds_u["10m_u_component_of_wind"], vas=ds_v["10m_v_component_of_wind"]
    # # )
    # wind_ds = xr.merge(winds)["sfcWind"]
    # client.cluster.close()
