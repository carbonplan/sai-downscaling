import icechunk
import xarray as xr
from .datasets import catalog
from .utils import icechunk_store_to_dataset, clean_up_dataset
import rasterix



def check_nans(ds):
    return ds.isnull().sum().values

def outlandishly_high_precip(ds):
    # highest ever recorded daily precipitation value was 1825 mm/day
    # on Reunion in 1966 according to https://www.weather.gov/owp/hdsc_world_record
    # so we'll say anything over 2000 is outlandishly high
    outlandishly_high_threshold = 2000
    return (ds['PREC']>outlandishly_high_threshold).sum().values

def outlandishly_high_temp(da):
    # highest ever recorded temperature value was 56.7 at Furnace Creek, CA in 1913
    # according to https://wmo.int/sites/default/files/2025-07/Table_Records_25Jul2025.pdf
    # so we'll say anything over 65 degC is outlandishly high
    outlandishly_high_threshold = 65 + 273.15 # convert to kelvin
    return (da>outlandishly_high_threshold).sum().values

def outlandishly_low_temp(da):
    # highest ever recorded temperature value was -89.2 at Vostok, Antarctica in 1983
    # according to https://wmo.int/sites/default/files/2025-07/Table_Records_25Jul2025.pdf
    # so we'll say anything under -100 is outlandishly low
    outlandishly_low_threshold = -100 + 273.15 # convert to kelvin
    return (da<outlandishly_low_threshold).sum().values

def negative_precip(ds):
    return (ds['PREC']<0).sum().values

def check_temperature_monotonic(ds):
    min_exceeds_mean = (ds['TASMIN']>ds['TASMEAN']).sum().values
    mean_exceeds_max = (ds['TASMEAN']>ds['TASMAX']).sum().values
    min_exceeds_max = (ds['TASMIN']>ds['TASMAX']).sum().values
    return min_exceeds_mean, mean_exceeds_max, min_exceeds_max

def check_physical_constraints(ds):
    print(f'Number of negative precipitation values: {negative_precip(ds)}')
    print(f'Number of outlandishly high precipitation values: {outlandishly_high_precip(ds)}')
    print(f'Number of outlandishly high TASMEAN values: {outlandishly_high_temp(ds['TASMEAN'])}')
    print(f'Number of outlandishly high TASMAX values: {outlandishly_high_temp(ds['TASMAX'])}')
    print(f'Number of outlandishly high TASMIN values: {outlandishly_high_temp(ds['TASMIN'])}')
    print(f'Number of outlandishly low TASMEAN values: {outlandishly_low_temp(ds['TASMEAN'])}')
    print(f'Number of outlandishly low TASMAX values: {outlandishly_low_temp(ds['TASMAX'])}')
    print(f'Number of outlandishly low TASMIN values: {outlandishly_low_temp(ds['TASMIN'])}')
    min_exceeds_mean, mean_exceeds_max, min_exceeds_max = check_temperature_monotonic(ds)
    print(f'Number of times TASMIN exceeds TASMEAN: {min_exceeds_mean}')    
    print(f'Number of times TASMEAN exceeds TASMAX: {mean_exceeds_max}')   
    print(f'Number of times TASMIN exceeds TASMAX: {min_exceeds_max}')

def confirm_coords(ds):
    era5 = icechunk_store_to_dataset("ERA5", catalog)
    era5 = clean_up_dataset(era5, "ERA5").pipe(rasterix.assign_index)
    xr.testing.assert_equal(era5[['latitude', 'longitude']].coords, ds[['latitude', 'longitude']].coords)
    # TODO: add in the expected time coordinates
    return 'Latitude and longitude match expectation'