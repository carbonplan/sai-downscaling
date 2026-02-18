import icechunk
import xarray as xr

from srm import catalog
from srm.downscaling_utils import get_experiment


def calculate_dtr(gcm: str = "CESM2-WACCM", scenario: str = "SSP245"):
    tasmin = get_experiment(gcm=gcm, scenario=scenario, var="tasmin")
    tasmax = get_experiment(gcm=gcm, scenario=scenario, var="tasmax")
    dtr = tasmax - tasmin
    dtr.attrs["long_name"] = "diurnal temperature range"
    dtr.attrs["units"] = "K"
    return dtr


def append_dtr_to_icechunk(dtr: xr.DataArray, gcm: str = "CESM2-WACCM", scenario: str = "SSP245"):
    cat_name = gcm + "-" + scenario + "-icechunk"
    path = catalog.get(cat_name).path

    repo = icechunk.Repository.open(path)
    session = repo.writable_session(branch="main")

    new_var = dtr.to_dataset(name="dtr")

    new_var.to_zarr(
        session.store,
        mode="a",
    )

    session.commit("append DTR")


def add_dtr(gcm: str = "CESM2-WACCM", scenario: str = "SSP245"):
    dtr = calculate_dtr(gcm=gcm, scenario=scenario)
    append_dtr_to_icechunk(dtr=dtr, gcm=gcm, scenario=scenario)
