"""``--dry-run`` must sample from inside the scenario's time window, after clipping."""

import numpy as np
import xarray as xr

from saidownscale.input_data import cesm2_waccm, ukesm


def _sample_dataset(start: str, n_steps: int = 1000, calendar: str | None = None) -> xr.Dataset:
    if calendar is None:
        time = np.arange(
            np.datetime64(start), np.datetime64(start) + n_steps, dtype="datetime64[D]"
        ).astype("datetime64[ns]")
    else:
        time = xr.date_range(start, periods=n_steps, freq="D", calendar=calendar, use_cftime=True)
    return xr.Dataset(
        {"tas": (("time", "lat", "lon"), np.full((n_steps, 2, 2), 280.0, dtype="f4"))},
        coords={"time": time, "lat": np.array([10.0, -10.0]), "lon": np.array([0.0, 200.0])},
    )


def _year_before(module, scenario: str) -> str:
    return f"{int(module.TIME_RANGE[scenario].split('-')[0]) - 1}-01-01"


def test_dry_run_sample_is_full_length_and_inside_the_window(subtests):
    g6 = "G6-1.5K"
    cases = {
        "ukesm_g6": (
            ukesm,
            lambda: ukesm._preprocess_ukesm(
                _sample_dataset(_year_before(ukesm, g6)), g6, subset=True
            ),
            g6,
        ),
        "ukesm_g6_360_day": (
            ukesm,
            lambda: ukesm._preprocess_ukesm(
                _sample_dataset(_year_before(ukesm, g6), calendar="360_day"), g6, subset=True
            ),
            g6,
        ),
        "ukesm_historical_head": (
            ukesm,
            lambda: ukesm._preprocess_ukesm(
                _sample_dataset("1850-01-01"), "historical", subset=True
            ),
            None,
        ),
        "cesm_g6": (
            cesm2_waccm,
            lambda: cesm2_waccm._preprocess_cesm(
                _sample_dataset(_year_before(cesm2_waccm, g6)), g6, "tas", subset=True
            ),
            g6,
        ),
    }
    for case, (module, run, scenario) in cases.items():
        with subtests.test(case=case):
            result = run()
            assert result.sizes["time"] == module._DRY_RUN_STEPS
            if scenario is not None:
                start_year, end_year = module.TIME_RANGE[scenario].split("-")
                assert result.time.min() >= np.datetime64(f"{start_year}-01-01")
                assert result.time.max() <= np.datetime64(f"{end_year}-12-31")
