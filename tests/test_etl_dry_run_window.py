"""``--dry-run`` must sample from inside the scenario's time window.

The UKESM T/PR source files begin before the G6-1.5K window opens, so taking the first
``_DRY_RUN_STEPS`` steps of the file and clipping to ``TIME_RANGE`` afterwards leaves an empty
time axis. The dry-run then dies in the summary table with ``ValueError: zero-size array to
reduction operation fmin which has no identity``. ``cesm2_waccm`` and ``miroc`` already clip
before they sample; these tests pin that ordering for every module.
"""

import numpy as np
import pytest
import xarray as xr

from srm.input_data import cesm2_waccm, miroc, ukesm


def _sample_dataset(start: str, n_steps: int) -> xr.Dataset:
    """Daily dataset whose time axis starts before the scenario window opens."""
    time = np.arange(np.datetime64(start), np.datetime64(start) + n_steps, dtype="datetime64[D]")
    return xr.Dataset(
        {
            "tas": (
                ("time", "lat", "lon"),
                np.full((n_steps, 2, 2), 280.0, dtype="f4"),
            )
        },
        coords={
            "time": time.astype("datetime64[ns]"),
            # deliberately unsorted / 0-360 so sortby and lon_to_180 both do work
            "lat": np.array([10.0, -10.0]),
            "lon": np.array([0.0, 200.0]),
        },
    )


@pytest.mark.parametrize(
    "module, preprocess, scenario",
    [
        (ukesm, ukesm._preprocess_ukesm, "G6-1.5K"),
        # MIROC clips only its two CMIP6 scenarios; G6-1.5K has no TIME_RANGE entry.
        (miroc, miroc._preprocess_miroc, "esgf-ssp245"),
    ],
)
def test_dry_run_sample_falls_inside_scenario_window(module, preprocess, scenario):
    start_year, end_year = module.TIME_RANGE[scenario].split("-")
    # Start a year before the window so a head-of-file sample would be entirely clipped away.
    ds = _sample_dataset(f"{int(start_year) - 1}-01-01", 1000)

    result = preprocess(ds, scenario, subset=True)

    assert result.sizes["time"] == module._DRY_RUN_STEPS
    assert result.time.min() >= np.datetime64(f"{start_year}-01-01")
    assert result.time.max() <= np.datetime64(f"{end_year}-12-31")


def test_ukesm_dry_run_sample_is_full_length_on_a_360_day_calendar():
    """UKESM source files are 360_day, so the sample must survive calendar conversion.

    Converting 360_day to proleptic_gregorian inserts roughly six NaN days per year, so a
    head-sample taken before conversion must still leave at least ``_DRY_RUN_STEPS`` steps
    afterwards.
    """
    scenario = "G6-1.5K"
    start_year, end_year = ukesm.TIME_RANGE[scenario].split("-")
    time = xr.date_range(
        f"{int(start_year) - 1}-01-01", periods=1000, freq="D", calendar="360_day", use_cftime=True
    )
    ds = xr.Dataset(
        {"tas": (("time", "lat", "lon"), np.full((1000, 2, 2), 280.0, dtype="f4"))},
        coords={"time": time, "lat": np.array([10.0, -10.0]), "lon": np.array([0.0, 200.0])},
    )

    result = ukesm._preprocess_ukesm(ds, scenario, subset=True)

    assert result.sizes["time"] == ukesm._DRY_RUN_STEPS
    assert result.time.min() >= np.datetime64(f"{start_year}-01-01")
    assert result.time.max() <= np.datetime64(f"{end_year}-12-31")


def test_ukesm_historical_dry_run_sample_is_full_length():
    """``historical`` has no TIME_RANGE entry, so it is sampled from the head of the file."""
    ds = _sample_dataset("1850-01-01", 1000)

    result = ukesm._preprocess_ukesm(ds, "historical", subset=True)

    assert result.sizes["time"] == ukesm._DRY_RUN_STEPS


def test_cesm_dry_run_sample_falls_inside_scenario_window():
    scenario = "G6-1.5K"
    start_year, end_year = cesm2_waccm.TIME_RANGE[scenario].split("-")
    ds = _sample_dataset(f"{int(start_year) - 1}-01-01", 1000)

    result = cesm2_waccm._preprocess_cesm(ds, scenario, "tas", subset=True)

    assert result.sizes["time"] == cesm2_waccm._DRY_RUN_STEPS
    assert result.time.min() >= np.datetime64(f"{start_year}-01-01")
    assert result.time.max() <= np.datetime64(f"{end_year}-12-31")
