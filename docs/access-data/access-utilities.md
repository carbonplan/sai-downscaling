# Access utilities

We've made a set of utilities to help you access the data stored on Source Cooperative. We have
tried to accommodate users with a range of levels of experience with Python, including those who
have never worked in the cloud or used packages like {term}`Zarr` before. The utilities are all
housed in the
[sai-downscaling-data-utils](https://github.com/carbonplan/sai-downscaling-data-utils) repository.
Follow the instructions in that repository's
[README](https://github.com/carbonplan/sai-downscaling-data-utils/blob/main/README.md) to learn how
to install and use the utilities.

The table below can help you find the right utility for what you want to do, whether you want the
downscaled output or the global climate model (GCM) input it came from. If you're new to the data,
we recommend starting with the quickstart notebook, which walks through one short example from
start to finish.

| What do you want to do? | Use |
| --- | --- |
| Try a short example that loads data for one region and season and saves it to a file | [`notebooks/quickstart.ipynb`](https://github.com/carbonplan/sai-downscaling-data-utils/blob/main/notebooks/quickstart.ipynb) |
| Inspect the data interactively in a Jupyter notebook, with more detail on each step, including quality flags and the bias-corrected data | [`notebooks/subsetting-and-exporting.ipynb`](https://github.com/carbonplan/sai-downscaling-data-utils/blob/main/notebooks/subsetting-and-exporting.ipynb) |
| Run a global (huge!) analysis without running out of memory | [`notebooks/compute-resources.ipynb`](https://github.com/carbonplan/sai-downscaling-data-utils/blob/main/notebooks/compute-resources.ipynb) |
| Explore the GCM input data that the downscaling started from | [`notebooks/input-data.ipynb`](https://github.com/carbonplan/sai-downscaling-data-utils/blob/main/notebooks/input-data.ipynb) |
| Download a subset of the data with Python | [`scripts/download.py`](https://github.com/carbonplan/sai-downscaling-data-utils/blob/main/scripts/download.py) |
| Download a subset of the data with Bash | [`scripts/download.sh`](https://github.com/carbonplan/sai-downscaling-data-utils/blob/main/scripts/download.sh) |

If you come across a term you don't know, check the [glossary](whats-available.md#glossary). The
access utilities repository also has its own
[glossary](https://github.com/carbonplan/sai-downscaling-data-utils/blob/main/GLOSSARY.md),
which covers terms from the notebooks and the command-line tool.
