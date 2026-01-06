import time
from dataclasses import dataclass, field
from typing import Literal, get_args

import click
import icechunk
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk

from srm import catalog
from srm.config import ClusterConfig, init_repo, setup_cluster, setup_local_client
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 128})


@dataclass
class ERA5Config:
    WIND_VARS = Literal["10m_u_component_of_wind", "10m_v_component_of_wind"]
    MAX_RESAMPLING = Literal["maximum_2m_temperature_since_previous_post_processing"]
    MIN_RESAMPLING = Literal["minimum_2m_temperature_since_previous_post_processing"]
    MEAN_RESAMPLING = Literal[
        "mean_total_precipitation_rate",
        "2m_temperature",
        "mean_surface_downward_short_wave_radiation_flux",
        "mean_surface_downward_long_wave_radiation_flux",
        "sfcWind",
        "surface_pressure",
    ]
    ALL_VARS = MAX_RESAMPLING | MIN_RESAMPLING | MEAN_RESAMPLING

    ERA5_TO_CMIP6_VARIABLE_MAPPING = {
        "mean_total_precipitation_rate": "pr",
        "2m_temperature": "tas",
        "minimum_2m_temperature_since_previous_post_processing": "tasmin",
        "maximum_2m_temperature_since_previous_post_processing": "tasmax",
        # 'ADDME_HURS': 'hurs',# we only have specific_humidity at levels in hpa, so can we back out 2m?
        "mean_surface_downward_short_wave_radiation_flux": "rsds",
        "sfcWind": "sfcWind",
        # 'ADDME_HUSS': 'huss',
        "mean_surface_downward_long_wave_radiation_flux": "rlds",
        "surface_pressure": "ps",
    }

    input_url: str = (
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    )
    start_year: int = 1950
    end_year: int = 2014

    input_chunking = {"time": 24, "latitude": 721, "longitude": 1440}

    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    ALL_VARS_LIST = (
        list(get_args(MAX_RESAMPLING))
        + list(get_args(MIN_RESAMPLING))
        + list(get_args(MEAN_RESAMPLING))
    )

    def __post_init__(self):
        """Fetch output location from catalog and unpack"""
        from srm import catalog

        era5_dataset = catalog.get("ERA5")
        self.bucket = era5_dataset.bucket
        self.prefix = era5_dataset.prefix
        self.output_path = era5_dataset.path

        self.encoding = {
            "chunks": era5_dataset.expected_chunks,
            "shards": era5_dataset.expected_shards,
        }


def _resample_time(ds, variable):
    config = ERA5Config()

    if variable in get_args(config.MAX_RESAMPLING):
        return ds.resample(time="d").max()
    elif variable in get_args(config.MIN_RESAMPLING):
        return ds.resample(time="d").min()
    elif variable in get_args(config.MEAN_RESAMPLING):
        return ds.resample(time="d").mean()
    else:
        raise ValueError(f"Unknown variable: {variable}")


def _compute_wind_speed(ds_u: xr.Dataset, ds_v: xr.Dataset) -> xr.Dataset:
    import xclim

    winds = xclim.indicators.convert.wind_speed_from_vector(
        uas=ds_u["10m_u_component_of_wind"], vas=ds_v["10m_v_component_of_wind"]
    )
    return xr.merge(winds)[["sfcWind"]]


def _load_era5(variable, config: ERA5Config):
    # chunks=None skips using dask.
    # This uses xarray’s internally private lazy indexing classes, but data is eagerly loaded into memory as numpy arrays when accessed.
    # This can be more efficient ... when large arrays are sliced before computation.

    from obstore.store import from_url
    from zarr.storage import ObjectStore

    store = from_url(config.input_url, skip_signature=True)
    zstore = ObjectStore(store)

    ds = (
        xr.open_dataset(zstore, engine="zarr", chunks=None)[[variable]]
        .sel(time=slice(f"{config.start_year}", f"{config.end_year}"))
        .drop_encoding()
    )
    return ds.chunk(config.input_chunking).drop_encoding()


def _trim_negative(
    ds: xr.Dataset,
) -> xr.Dataset:
    return ds.clip(min=0)


def _standardize_vars(ds: xr.Dataset, config: ERA5Config) -> xr.Dataset:
    for era5_var, cmip6_var in config.ERA5_TO_CMIP6_VARIABLE_MAPPING.items():
        if era5_var in ds.data_vars:
            ds = ds.rename({era5_var: cmip6_var})
    return ds


def _preprocess_era5(ds: xr.Dataset, config: ERA5Config):
    # slice by time, update longitude and sortby. Update dim names, standardize var names
    ds = lon_to_180(ds, lon_name="longitude")
    ds = ds.sortby(["latitude", "longitude"])
    ds = ds.rename({"longitude": "lon", "latitude": "lat"})
    ds = _standardize_vars(ds, config)
    return ds


def _encoding(ds: xr.Dataset, config: ERA5Config):
    encoding = {}
    for var_name in ds.data_vars:
        var = ds[var_name]
        # make a tuple for the chunks from the dict
        var_chunks = tuple(config.encoding["chunks"][d] for d in var.dims)
        var_shards = tuple(config.encoding["shards"][d] for d in var.dims)
        encoding[var_name] = {
            "chunks": var_chunks,
            "shards": var_shards,
        }
    return encoding


def _update_attrs(ds: xr.Dataset, var: str, config: ERA5Config) -> xr.Dataset:
    import cf_xarray  # noqa ignore

    if var == "pr":
        ds["pr"].attrs["units"] = "kg m-2 s-1"

    if var in ["rlds", "rsds"]:
        ds[var].attrs["units"] = "W m-2"

    ds = ds.cf.add_bounds("time")
    ds = ds.cf.add_bounds("lat")
    ds = ds.cf.add_bounds("lon")

    ds.attrs.update(
        {
            "valid_time_start": f"{config.start_year}-01-01",
            "valid_time_stop": f"{config.end_year}-12-31",
        }
    )

    return ds


def write_to_icechunk(
    ds,
    session,
    encoding: dict,
    commit_message: str,
    write_mode: str,
    config: ERA5Config,
):
    # rechunk to shard shape
    ds = ds.chunk(config.encoding["shards"])

    to_icechunk(ds, session, encoding=encoding, mode=write_mode)
    session.commit(commit_message)


def _determine_mode_based_on_ancestry(
    repo: icechunk.Repository, branch: str = "main"
) -> str:
    # check the icechunk ancestry to see if data already exists. Change mode to append if so.
    history = list(repo.ancestry(branch=branch))
    if len(history) <= 1:
        return "w"
    else:
        return "a"


def process_era5_pipeline(
    variables: list[str],
    start_year: int = 1950,
    end_year: int = 2014,
    use_coiled: bool = False,
    verbose: bool = True,
):
    era5_cat = catalog.get("ERA5")
    config = ERA5Config(start_year=start_year, end_year=end_year)

    if use_coiled:
        client = setup_cluster(config.cluster)
    else:
        client = setup_local_client()

    repo, session = init_repo(era5_cat.bucket, era5_cat.prefix, readonly=False)

    try:
        for var in variables:
            repo, session = init_repo(era5_cat.bucket, era5_cat.prefix, readonly=False)
            write_mode = _determine_mode_based_on_ancestry(repo)

            if verbose:
                print(f"Processing {var}...")

            if var == "sfcWind":
                ds_u = _load_era5(variable="10m_u_component_of_wind", config=config)
                ds_v = _load_era5(variable="10m_v_component_of_wind", config=config)
                ds = _compute_wind_speed(ds_u, ds_v)
            else:
                ds = _load_era5(variable=var, config=config)

            ds = _preprocess_era5(ds, config)
            if var == "mean_total_precipitation_rate":
                ds = _trim_negative(ds)
            ds = _resample_time(ds, var)
            ds = _update_attrs(ds, var, config)
            encoding = _encoding(ds, config)
            print(ds)
            print(encoding)
            # try explicity chunking to shard size
            write_to_icechunk(
                ds,
                session,
                encoding=encoding,
                commit_message=f"{var}",
                write_mode=write_mode,
                config=config,
            )

            # if we're doing multiple vars, switch to append after the first group is written
            write_mode = "a"
            if verbose:
                print(f"Committed {var}")

            time.sleep(
                10
            )  # Try to slow down a bit for GCS: obstore.exceptions.GenericError: Generic GCS error: Error performing GET

    finally:
        client.shutdown()


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--variable",
    multiple=True,
    type=click.Choice(ERA5Config.ALL_VARS_LIST),
    required=True,
)
@click.option("--start-year", type=int, default=1950)
@click.option("--end-year", type=int, default=2014)
@click.option("--coiled/--local", default=False)
def era5(variable, start_year, end_year, coiled):
    process_era5_pipeline(
        variables=list(variable),
        start_year=start_year,
        end_year=end_year,
        use_coiled=coiled,
        verbose=True,
    )


if __name__ == "__main__":
    cli()
