import xarray as xr
import cubed
from typing import Literal, get_args
import zarr

zarr.config.set({"async.concurrency": 128})


MAX_RESAMPLING = Literal["maximum_2m_temperature_since_previous_post_processing"]
MIN_RESAMPLING = Literal["minimum_2m_temperature_since_previous_post_processing"]
MEAN_RESAMPLING = Literal[
    "mean_total_precipitation_rate",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_surface_downward_short_wave_radiation_flux",
    "mean_surface_downward_long_wave_radiation_flux",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]
ERA5_VARS = Literal[MAX_RESAMPLING, MIN_RESAMPLING, MEAN_RESAMPLING]

INPUT_ZARR_STORE_CONFIG = {
    "url": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
}


# spec = cubed.Spec(
#     work_dir="s3://carbonplan-scratch/cubed/scratch",
#     allowed_mem="4GB",
#     executor_name="coiled",
#     executor_options={'name':'cubed_rechunk','vm_type':'m8g.xlarge','spot_policy':'spot_with_fallback','n_workers':[100,300],'tags':{'Project':'OCR'}},

# )

spec = cubed.Spec(
    work_dir="s3://carbonplan-srm/cubed/scratch",
    allowed_mem="12GB",  # r8g.large is 16gb ram, so 3/4?
    executor_name="coiled",
    executor_options={
        "name": "cubed-era5-test",
        "vm_type": ["r8g.medium", "r8g.large"],
        "n_workers": [1, 100],
        "spot_policy": "spot_with_fallback",
        "use_best_zone": True,
        "region": "us-west-2",
        "tags": {"Project": "SRM"},
    },
)

# def setup_repository(storage_config: dict) -> tuple:
#     """Set up and return an icechunk repository.

#     Parameters
#     ----------
#     storage_config: dict
#         Configuration for the storage backend, including bucket, prefix, and region.

#     Returns
#     -------
#     tuple: A tuple containing the icechunk repository and a writable session.
#         Tuple of (repository, session)
#     """
#     config = storage_config

#     storage = icechunk.s3_storage(
#         bucket=config['bucket'],
#         prefix=config['prefix'],
#         region=config['region'],
#     )

#     repo = icechunk.Repository.open_or_create(storage)
#     session = repo.writable_session('main')

#     return repo, session


def resample_time(ds: xr.Dataset, variable: ERA5_VARS) -> xr.Dataset:
    """Resample time dimension based on variable type."""
    try:
        import flox  # noqa: F401
    except ImportError:
        raise ImportWarning(
            "flox is not installed. Add it to greatly speedup resampling operations."
        )

    if variable in get_args(MAX_RESAMPLING):
        return ds.resample(time="d").max()
    elif variable in get_args(MIN_RESAMPLING):
        return ds.resample(time="d").min()
    elif variable in get_args(MEAN_RESAMPLING):
        return ds.resample(time="d").mean()
    else:
        raise ValueError(f"variable: {variable} is not in {ERA5_VARS}")


def convert_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Convert 0-360 longitude to -180-180"""
    ds.coords["longitude"] = (ds.coords["longitude"] + 180) % 360 - 180
    return ds.sortby(ds.longitude)


def load_and_process_dataset(
    variable: ERA5_VARS,
    start_year: int = 1950,
    end_year: int = 1950,
    spec: cubed.Spec = spec,
):
    """Load and process ERA5 data using Cubed arrays.

    Parameters
    ----------
    variable: ERA5_VARS
        The variable to load from the dataset.
    start_year: int
        Starting year for data selection
    end_year: int
        Ending year for data selection
    spec: cubed.Spec
        Cubed specification for computation

    Returns
    -------
    xr.Dataset: An Xarray Dataset with processed data using Cubed arrays
    """
    # Load dataset with Cubed backend
    ds = xr.open_zarr(
        INPUT_ZARR_STORE_CONFIG["url"],
        chunked_array_type="cubed",
        from_array_kwargs={"spec": spec},
    )[[variable]]

    # Subset time
    ds = ds.sel(time=slice(f"{start_year}", f"{end_year}"))

    # Convert longitude coordinates
    ds = convert_longitude(ds)

    # Initial chunking - larger chunks to reduce task count
    # ~100MB chunks but keep spatial dimensions intact initially
    ds = ds.chunk(
        {"time": 25, "latitude": 721, "longitude": 1440},
        chunked_array_type="cubed",
        from_array_kwargs={"spec": spec},
    )

    # Resample from hourly to daily
    ds = resample_time(ds, start_year, end_year, variable)

    # Rechunk after resampling for better I/O patterns
    # Smaller spatial chunks since we now have daily data
    ds = ds.chunk(
        {"time": 8000, "latitude": 48, "longitude": 72},
        chunked_array_type="cubed",
        from_array_kwargs={"spec": spec},
    )

    # Clean up encoding to avoid conflicts with Icechunk
    for var_name in ds.variables:
        ds[var_name].encoding.pop("chunks", None)
        ds[var_name].encoding.pop("preferred_chunks", None)
        ds[var_name].encoding.pop("compressors", None)

    return ds


def main_cubed(variable: ERA5_VARS, start_year: int = 1950, end_year: int = 1950):
    """Main function to process ERA5 data with Cubed and write to Icechunk.

    Parameters
    ----------
    variable: ERA5_VARS
        Variable to process
    start_year: int
        Starting year for processing
    end_year: int
        Ending year for processing
    """
    print(f"Processing {variable} from {start_year} to {end_year}")


if __name__ == "__main__":
    # Process single variable
    ds = load_and_process_dataset(
        variable="2m_temperature", start_year=1950, end_year=1950, spec=spec
    )
    ds.to_zarr(
        "s3://carbonplan-scratch/cubed_srm_test/subset_era5.zarr",
        zarr_format=3,
        consolidated=False,
        mode="w",
    )
