import icechunk
import xarray as xr

from srm import catalog
from srm.downscaling_utils import get_experiment, get_obs


def calculate_dtr(data_source: str = "CESM2-WACCM", scenario: str = "SSP245"):
    if data_source == "ERA5":
        if scenario == "historical":
            tasmin = get_obs(var="tasmin")
            tasmax = get_obs(var="tasmax")
        else:
            raise ValueError(
                "ERA5 only contains historical data, cannot calculate DTR for future scenarios."
            )
    else:
        tasmin = get_experiment(data_source=data_source, scenario=scenario, var="tasmin")
        tasmax = get_experiment(data_source=data_source, scenario=scenario, var="tasmax")
    dtr = tasmax - tasmin
    dtr.attrs["long_name"] = "diurnal temperature range"
    dtr.attrs["units"] = "K"
    return dtr


def append_dtr_to_icechunk(
    dtr: xr.DataArray, data_source: str = "CESM2-WACCM", scenario: str = "SSP245"
):
    if data_source == "ERA5":
        if scenario == "historical":
            cat_name = "ERA5"
        else:
            raise ValueError(
                "ERA5 only contains historical data, cannot calculate DTR for future scenarios."
            )

    else:
        cat_name = data_source + "-" + scenario + "-icechunk"

    path = catalog.get(cat_name).path

    repo = icechunk.Repository.open(path)
    session = repo.writable_session(branch="main")

    new_var = dtr.to_dataset(name="dtr")

    new_var.to_zarr(
        session.store,
        mode="a",
    )

    session.commit("append DTR")


def add_dtr(data_source: str = "CESM2-WACCM", scenario: str = "SSP245"):
    dtr = calculate_dtr(data_source=data_source, scenario=scenario)
    append_dtr_to_icechunk(dtr=dtr, data_source=data_source, scenario=scenario)


def process_all_data():
    for data_source in ["CESM2-WACCM", "UKESM", "MIROC-ES2H"]:
        for scenario in ["SSP245", "historical", "G6-1.5K"]:
            add_dtr(data_source=data_source, scenario=scenario)

    add_dtr(data_source="ERA5", scenario="historical")
