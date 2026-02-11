import xarray as xr

from srm import catalog


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


def confirm_coords(ds):
    era5 = catalog.get("ERA5").to_xarray()
    xr.testing.assert_equal(era5[["lat", "lon"]].coords, ds[["lat", "lon"]].coords)
    # TODO: add in the expected time coordinates
    return "Latitude and longitude match expectation"
