import re

import matplotlib.pyplot as plt
import numpy as np
from compare_to_coarse import get_coarse_data, open_icechunk


def make_maps(paths, variables, output_names, dir_out):
    for i, fpath in enumerate(paths):
        print(i)
        var = variables[i]
        output_fpath = output_names[i]
        da = open_icechunk(fpath)[var]

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        da.mean(dim="time").plot()
        plt.title("Mean")
        plt.subplot(1, 3, 2)
        da.min(dim="time").plot()
        plt.title("Min")
        plt.subplot(1, 3, 3)
        da.max(dim="time").plot()
        plt.title("Max")
        plt.tight_layout()
        plt.savefig(dir_out + output_fpath, dpi=300)
        plt.close()


def make_timeseries(
    paths,
    dir_out,
    output_names,
    variables,
    gcms,
    scenarios,
    ensembles,
    plot_coarse_comparison=True,
    lat=-30,
    lon=25,
):
    for i, fpath in enumerate(paths):
        print(i)
        var = variables[i]
        scenario = scenarios[i]
        gcm = gcms[i]
        ens = ensembles[i]

        if scenario != "obs":
            output_fpath = output_names[i]
            da = open_icechunk(fpath)[var]
            tseries_1pt = da.sel(lon=lon, lat=lat, method="nearest").load()

            if plot_coarse_comparison:
                da_coarse = get_coarse_data(
                    scenario=scenario, ens=ens, gcm=gcm, future_time_slice=False
                )[var]
                tseries_1pt_coarse = da_coarse.sel(lon=lon, lat=lat, method="nearest").load()

            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            if not np.isnan(np.nanmax(tseries_1pt)):
                plt.hist(tseries_1pt, alpha=0.3, label="Downscaled", color="tab:blue")
            if plot_coarse_comparison:
                if not np.isnan(np.nanmax(tseries_1pt_coarse)):
                    plt.hist(tseries_1pt_coarse, alpha=0.3, label="Coarse", color="tab:orange")
            plt.legend()
            plt.subplot(1, 2, 2)
            tseries_1pt.rolling(time=365).mean().plot(color="tab:blue")
            if plot_coarse_comparison:
                tseries_1pt_coarse.rolling(time=365).mean().plot(color="tab:orange")
            plt.tight_layout()
            if plot_coarse_comparison:
                plt.savefig(dir_out + "comparison_timeseries_" + output_fpath, dpi=300)
            else:
                plt.savefig(dir_out + "comparison_timeseries_" + output_fpath, dpi=300)
            plt.close()


def make_quick_check_figs(fname_file_list="output_files.txt", dir_out="quick_look/"):
    with open(fname_file_list) as f:
        text = f.read()

    paths = re.findall(r"s3://\S+", text)

    models = {"CESM2-WACCM", "MIROC-ES2H", "UKESM"}

    variables = []
    output_names = []
    scenarios = []
    gcms = []
    ensembles = []
    for path in paths:
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part in models:
                variables.append(parts[i + 1])
                scenarios.append(parts[i - 1])
                gcms.append(parts[i])
                ensembles.append(parts[i + 2])
                break

    for path in paths:
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part in models:
                if "/output/" in path:
                    scenario = parts[i - 1]
                    gcm = parts[i]
                    variable = parts[i + 1]
                    ens = parts[i + 2]
                    output_names.append(f"{variable}_{scenario}_{gcm}_{ens}.png")
                else:
                    scenario = parts[i - 1]
                    gcm = parts[i]
                    variable = parts[i + 1]
                    # region = parts[i + 2]
                    output_names.append(f"{variable}_{scenario}_{gcm}.png")
                break

    make_maps(paths, variables=variables, output_names=output_names, dir_out=dir_out)

    make_timeseries(
        paths,
        lat=-30,
        lon=25,
        dir_out=dir_out,
        variables=variables,
        scenarios=scenarios,
        gcms=gcms,
        ensembles=ensembles,
        output_names=output_names,
    )
