"""
Tests for calculate_out_of_range_mask() in pipeline.py
"""

import numpy as np
import pandas as pd
import xarray as xr

from srm.pipeline import calculate_out_of_range_mask


def make_time_series(values: float, start_year=1980, end_year=1982):
    """Helper function to create a small model_hist DataArray with a daily time index."""
    times = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    data = np.full(len(times), values)
    return xr.DataArray(data, coords={"time": times}, dims=["time"])


def test_in_range_returns_false():
    """Values within the historical range should not be flagged as out of range."""

    # Historical is always 10.0, so range is [10, 10]
    model_hist = make_time_series(10.0)

    # Scenario is also 10.0 — exactly in range
    scenario = make_time_series(10.0, start_year=2050, end_year=2052)

    result = calculate_out_of_range_mask(
        model_hist=model_hist, scenario_detrended=scenario, center_window=31
    )

    assert not result.any(), "Expected all False (in range), but got some True"


def test_above_range_returns_true():
    """Values above the historical max should be flagged as out of range."""
    model_hist = make_time_series(10.0)
    # Scenario is 20.0 — above historical max of 10
    scenario = make_time_series(20.0, start_year=2050, end_year=2052)

    result = calculate_out_of_range_mask(
        model_hist=model_hist, scenario_detrended=scenario, center_window=31
    )

    assert result.all(), "Expected all True (out of range), but got some False"


def test_below_range_returns_true():
    """Values below the historical min should be flagged as out of range."""
    model_hist = make_time_series(10.0)
    # Scenario is 0 — below historical min of 10
    scenario = make_time_series(0.0, start_year=2050, end_year=2052)

    result = calculate_out_of_range_mask(
        model_hist=model_hist, scenario_detrended=scenario, center_window=31
    )

    assert result.all(), "Expected all True (out of range), but got some False"
