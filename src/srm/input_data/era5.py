import dataclasses
import logging

import typer
import xarray as xr
import zarr
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from srm.config import (
    ClusterConfig,
    VarSpec,
    VarStandards,
    init_repo,
    setup_cluster,
    setup_local_client,
)
from srm.input_data.etl_utils import (
    add_cf_bounds,
    build_encoding_dict,
    determine_write_mode,
    trim_negative_precipitation,
    update_variable_attrs,
    write_dataset_to_icechunk,
)
from srm.utils import lon_to_180

zarr.config.set({"async.concurrency": 128})

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

INPUT_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
INPUT_CHUNKING = {"time": 24, "latitude": 721, "longitude": 1440}

ERA5_TO_CMIP6 = {
    "mean_total_precipitation_rate": "pr",
    "2m_temperature": "tas",
    "minimum_2m_temperature_since_previous_post_processing": "tasmin",
    "maximum_2m_temperature_since_previous_post_processing": "tasmax",
    "mean_surface_downward_short_wave_radiation_flux": "rsds",
    "mean_surface_downward_long_wave_radiation_flux": "rlds",
    "surface_pressure": "ps",
    "hurs": "hurs",
}

# Maps each ERA5 source variable to its daily resampling operation
RESAMPLE_OPS = {
    "maximum_2m_temperature_since_previous_post_processing": "max",
    "minimum_2m_temperature_since_previous_post_processing": "min",
    "mean_total_precipitation_rate": "mean",
    "2m_temperature": "mean",
    "mean_surface_downward_short_wave_radiation_flux": "mean",
    "mean_surface_downward_long_wave_radiation_flux": "mean",
    "surface_pressure": "mean",
}

ALL_VARS = list(ERA5_TO_CMIP6.keys())
CMIP6_TO_ERA5 = {v: k for k, v in ERA5_TO_CMIP6.items()}
DERIVED_VARS = ["hurs"]

OUTPUT_BUCKET = "carbonplan-srm"
OUTPUT_PREFIX = "input/tensor/era5.icechunk"
OUTPUT_CHUNKS: dict[str, int] = {"time": 1, "lat": 721, "lon": 1440}
OUTPUT_SHARDS: dict[str, int] = {"time": 30, "lat": 721, "lon": 1440}

VAR_SPECS: dict[str, VarSpec] = {
    f.default.name: f.default for f in dataclasses.fields(VarStandards)
}

# 48 hourly steps = 2 calendar days; enough for resample + CF bounds to run cleanly
_DRY_RUN_HOURLY_STEPS = 48


def load_era5(variable: str, start_year: int, end_year: int) -> xr.Dataset:
    """Load a single ERA5 variable from the ARCO GCS archive.

    Parameters
    ----------
    variable : str
        ERA5 variable name (e.g. ``"2m_temperature"``).
    start_year : int
        First year of the time slice (inclusive).
    end_year : int
        Last year of the time slice (inclusive).

    Returns
    -------
    xr.Dataset
        Hourly ERA5 data chunked with ``INPUT_CHUNKING``, encoding stripped.
    """
    from obstore.store import from_url
    from zarr.storage import ObjectStore

    store = from_url(INPUT_URL, skip_signature=True)
    zstore = ObjectStore(store)
    ds = (
        xr.open_dataset(zstore, engine="zarr", chunks=None)[[variable]]
        .sel(time=slice(f"{start_year}", f"{end_year}"))
        .drop_encoding()
    )
    return ds.chunk(INPUT_CHUNKING).drop_encoding()


def load_hurs_from_era5(start_year: int, end_year: int, nsteps: int | None = None) -> xr.Dataset:
    """Derive daily relative humidity from ERA5 temperature and dewpoint.

    Loads ``2m_temperature`` and ``2m_dewpoint_temperature``, computes relative
    humidity via ``xclim``, preprocesses coordinates, and resamples to daily mean.

    Parameters
    ----------
    start_year : int
        First year of the time slice (inclusive).
    end_year : int
        Last year of the time slice (inclusive).
    nsteps : int or None, optional
        If set, truncate the hourly time axis to this many steps before computing.
        Intended for dry-run sampling only.

    Returns
    -------
    xr.Dataset
        Daily ``hurs`` dataset with CMIP6 coordinate names (``lat``, ``lon``).
    """
    import xclim

    era5_tas = load_era5("2m_temperature", start_year, end_year)
    era5_tdps = load_era5("2m_dewpoint_temperature", start_year, end_year)
    if nsteps is not None:
        era5_tas = era5_tas.isel(time=slice(0, nsteps))
        era5_tdps = era5_tdps.isel(time=slice(0, nsteps))
    hurs = xclim.convert.relative_humidity_from_dewpoint(
        tas=era5_tas["2m_temperature"], tdps=era5_tdps["2m_dewpoint_temperature"]
    )
    return hurs.to_dataset(name="hurs").pipe(preprocess_era5).resample(time="d").mean()


def preprocess_era5(ds: xr.Dataset) -> xr.Dataset:
    """Standardize ERA5 coordinate names and variable names to CMIP6 conventions.

    Shifts longitudes to [-180, 180], sorts by latitude and longitude, renames
    ``longitude``/``latitude`` to ``lon``/``lat``, and renames data variables
    using ``ERA5_TO_CMIP6``.

    Parameters
    ----------
    ds : xr.Dataset
        Raw ERA5 dataset with ``longitude`` and ``latitude`` dimensions.

    Returns
    -------
    xr.Dataset
        Dataset with CMIP6-style coordinate and variable names.
    """
    return (
        ds.pipe(lon_to_180, lon_name="longitude")
        .sortby(["latitude", "longitude"])
        .rename({"longitude": "lon", "latitude": "lat"})
        .rename({k: v for k, v in ERA5_TO_CMIP6.items() if k in ds.data_vars})
    )


def resample_to_daily(ds: xr.Dataset, variable: str) -> xr.Dataset:
    """Resample an ERA5 dataset to daily frequency using the variable's canonical operation.

    The operation (``max``, ``min``, or ``mean``) is looked up from ``RESAMPLE_OPS``.

    Parameters
    ----------
    ds : xr.Dataset
        Hourly ERA5 dataset.
    variable : str
        ERA5 source variable name used as the key into ``RESAMPLE_OPS``.

    Returns
    -------
    xr.Dataset
        Daily dataset.
    """
    op = RESAMPLE_OPS[variable]
    return getattr(ds.resample(time="d"), op)()


def _display_dry_run_result(ds: xr.Dataset, variable: str) -> None:
    logger.info("Computing sample result for %s...", variable)
    ds = ds.compute()

    table = Table(show_header=True, header_style="bold green")
    table.add_column("Variable", style="cyan")
    table.add_column("Shape")
    table.add_column("Dims")
    table.add_column("Units", style="yellow")
    table.add_column("Min", justify="right", style="blue")
    table.add_column("Max", justify="right", style="blue")

    for var_name in ds.data_vars:
        da = ds[var_name]
        table.add_row(
            var_name,
            str(da.shape),
            " × ".join(da.dims),
            da.attrs.get("units", "—"),
            f"{float(da.min()):.4g}",
            f"{float(da.max()):.4g}",
        )

    time_start = str(ds.time.values[0])[:10]
    time_end = str(ds.time.values[-1])[:10]
    store = f"s3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}"
    console.print(
        Panel(
            table,
            title=f"[bold green]✓ {variable}[/] | {time_start} → {time_end}",
            subtitle=f"[dim]→ {store}[/dim]",
            border_style="green",
        )
    )


def process_era5_var(variable: str, start_year: int, end_year: int, dry_run: bool = False) -> None:
    """Process a single ERA5 variable and write it to the icechunk store.

    Dispatches to the appropriate loader, applies the standard pipe chain of
    transforms, and writes the result to the icechunk store at ``OUTPUT_PREFIX``.

    Parameters
    ----------
    variable : str
        Variable to process as a CMIP6 name (e.g. ``"tas"``, ``"pr"``, ``"hurs"``).
    start_year : int
        First year to include.
    end_year : int
        Last year to include.
    dry_run : bool, optional
        If True, process a small sample (``_DRY_RUN_HOURLY_STEPS`` steps) and
        display the result instead of writing to the store. Default is False.
    """
    # Resolve CMIP6 name → ERA5 source name for loading and resampling lookups.
    era5_var = CMIP6_TO_ERA5.get(variable, variable)

    if variable == "hurs":
        logger.info(
            "Deriving hurs from 2m_temperature and 2m_dewpoint_temperature (%d–%d)",
            start_year,
            end_year,
        )
        ds = (
            load_hurs_from_era5(
                start_year,
                start_year if dry_run else end_year,
                nsteps=_DRY_RUN_HOURLY_STEPS if dry_run else None,
            )
            .pipe(add_cf_bounds)
            .pipe(update_variable_attrs, VAR_SPECS)
        )
    else:
        op = RESAMPLE_OPS[era5_var]
        logger.info(
            "Loading %s → %s (daily %s, %d–%d)", era5_var, variable, op, start_year, end_year
        )
        raw = load_era5(era5_var, start_year, start_year if dry_run else end_year)
        if dry_run:
            raw = raw.isel(time=slice(0, _DRY_RUN_HOURLY_STEPS))
        ds = (
            raw.pipe(preprocess_era5)
            .pipe(trim_negative_precipitation)
            .pipe(resample_to_daily, era5_var)
            .pipe(add_cf_bounds)
            .pipe(update_variable_attrs, VAR_SPECS)
        )

    if dry_run:
        _display_dry_run_result(ds, variable)
        return

    repo, session = init_repo(OUTPUT_BUCKET, OUTPUT_PREFIX, readonly=False)
    write_mode = determine_write_mode(repo)
    logger.info(
        "Writing %s to s3://%s/%s (mode=%s)", variable, OUTPUT_BUCKET, OUTPUT_PREFIX, write_mode
    )
    encoding = build_encoding_dict(ds, OUTPUT_CHUNKS, OUTPUT_SHARDS)
    write_dataset_to_icechunk(
        ds,
        session,
        encoding=encoding,
        shards=OUTPUT_SHARDS,
        commit_message=variable,
        write_mode=write_mode,
    )
    logger.info("✓ Done: %s", variable)


def process_era5_pipeline(
    variables: list[str],
    start_year: int = 1950,
    end_year: int = 2014,
    use_coiled: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the ERA5 ETL pipeline for one or more variables.

    Sets up a Dask cluster (local or Coiled), iterates over ``variables``,
    and calls ``process_era5_var`` for each.

    Parameters
    ----------
    variables : list[str]
        Variables to process.
    start_year : int, optional
        First year to include. Default is 1950.
    end_year : int, optional
        Last year to include. Default is 2014.
    use_coiled : bool, optional
        Use a Coiled cluster instead of a local Dask cluster. Default is False.
    dry_run : bool, optional
        If True, run transforms on a small sample and display results without
        writing or starting a cluster. Default is False.
    """
    if dry_run:
        logger.info("Dry run: %d hourly steps per variable, no writes", _DRY_RUN_HOURLY_STEPS)
        for var in variables:
            process_era5_var(var, start_year, end_year, dry_run=True)
        return

    cluster_config = ClusterConfig()
    client = setup_cluster(cluster_config) if use_coiled else setup_local_client()
    try:
        for var in variables:
            logger.info("Processing %s", var)
            process_era5_var(var, start_year, end_year)
    finally:
        client.shutdown()


app = typer.Typer()


@app.command()
def era5(
    variable: list[str] = typer.Option(
        ...,
        help=f"Variable(s) to process (CMIP6 names). Choices: {list(CMIP6_TO_ERA5) + DERIVED_VARS}",
    ),
    start_year: int = typer.Option(1950, help="First year to include."),
    end_year: int = typer.Option(2014, help="Last year to include."),
    coiled: bool = typer.Option(False, "--coiled/--local", help="Use Coiled cluster."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=f"Run transforms on a {_DRY_RUN_HOURLY_STEPS}-step sample and display results; no writes.",
    ),
) -> None:
    """Process ERA5 variables and write them to the icechunk store."""
    process_era5_pipeline(
        variables=variable,
        start_year=start_year,
        end_year=end_year,
        use_coiled=coiled,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
