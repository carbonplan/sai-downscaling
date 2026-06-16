from __future__ import annotations

import xarray as xr

from srm import catalog

# Spatial range bounds for a single day (isel(time=1)), computed across full spatial extent.
# Goal: catch obvious unit mismatches (e.g. Celsius instead of Kelvin, fraction instead of %).
# Ranges are wide intentionally — based on ERA5 observed range +/- large margins.
VAR_SPATIAL_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "tas": {"min": (100, 400), "max": (100, 400)},
    "tasmin": {"min": (100, 400), "max": (100, 400)},
    "tasmax": {"min": (100, 400), "max": (100, 400)},
    "pr": {"min": (0, 1e-7), "max": (0.0001, 0.03)},
    "rsds": {"min": (-1, 100), "max": (100, 1000)},
    "hurs": {"min": (0, 40), "max": (40, 900)},
    "dtr": {"min": (0, 10), "max": (10, 150)},
}


class ValidationResult:
    def __init__(self, is_valid: bool, issues: list[str]):
        self.is_valid = is_valid
        self.issues = issues

    def __bool__(self):
        return self.is_valid

    def __repr__(self):
        status = "valid" if self.is_valid else "invalid"
        if self.issues:
            return f"{status}: {', '.join(self.issues)}"
        return status


def check_nans(ds):
    return ds.isnull().sum().values


def outlandishly_high_precip(ds):
    # highest ever recorded daily precipitation value was 1825 mm/day
    # on Reunion in 1966 according to https://www.weather.gov/owp/hdsc_world_record
    # so we'll say anything over 2000 is outlandishly high
    outlandishly_high_threshold = 2000
    return (ds["pr"] > outlandishly_high_threshold).sum().values


def outlandishly_high_temp(da):
    # highest ever recorded temperature value was 56.7 at Furnace Creek, CA in 1913
    # according to https://wmo.int/sites/default/files/2025-07/Table_Records_25Jul2025.pdf
    # so we'll say anything over 65 degC is outlandishly high
    outlandishly_high_threshold = 65 + 273.15  # convert to kelvin
    return (da > outlandishly_high_threshold).sum().values


def outlandishly_low_temp(da):
    # highest ever recorded temperature value was -89.2 at Vostok, Antarctica in 1983
    # according to https://wmo.int/sites/default/files/2025-07/Table_Records_25Jul2025.pdf
    # so we'll say anything under -100 is outlandishly low
    outlandishly_low_threshold = -100 + 273.15  # convert to kelvin
    return (da < outlandishly_low_threshold).sum().values


def negative_precip(ds):
    return (ds["pr"] < 0).sum().values


def check_temperature_monotonic(ds):
    min_exceeds_mean = (ds["tasmin"] > ds["tas"]).sum().values
    mean_exceeds_max = (ds["tas"] > ds["tasmax"]).sum().values
    min_exceeds_max = (ds["tasmin"] > ds["tasmax"]).sum().values
    return min_exceeds_mean, mean_exceeds_max, min_exceeds_max


def check_physical_constraints(ds):
    print(f"Number of negative precipitation values: {negative_precip(ds)}")
    print(f"Number of outlandishly high precipitation values: {outlandishly_high_precip(ds)}")
    print(f"Number of outlandishly high tas values: {outlandishly_high_temp(ds['tas'])}")
    print(f"Number of outlandishly high tasmax values: {outlandishly_high_temp(ds['tasmax'])}")
    print(f"Number of outlandishly high tasmin values: {outlandishly_high_temp(ds['tasmin'])}")
    print(f"Number of outlandishly low tas values: {outlandishly_low_temp(ds['tas'])}")
    print(f"Number of outlandishly low tasmax values: {outlandishly_low_temp(ds['tasmax'])}")
    print(f"Number of outlandishly low tasmin values: {outlandishly_low_temp(ds['tasmin'])}")
    min_exceeds_mean, mean_exceeds_max, min_exceeds_max = check_temperature_monotonic(ds)
    print(f"Number of times tasmin exceeds tas: {min_exceeds_mean}")
    print(f"Number of times tas exceeds tasmax: {mean_exceeds_max}")
    print(f"Number of times tasmin exceeds tasmax: {min_exceeds_max}")


def confirm_coords(ds, obs_dataset: str = "ERA5"):
    obs = catalog.get(obs_dataset).to_xarray()
    xr.testing.assert_equal(obs[["lat", "lon"]].coords, ds[["lat", "lon"]].coords)
    # TODO: add in the expected time coordinates
    return "Latitude and longitude match expectation"
