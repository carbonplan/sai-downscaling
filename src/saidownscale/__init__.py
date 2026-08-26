"""
SAI Downscale package for bias-correcting and spatially disaggregating GCM outputs.

Exposes the dataset :data:`catalog` as the primary public interface. Pipeline
functionality is accessed through the submodules.
"""

# flake8: noqa
import importlib.metadata

from saidownscale.datasets import catalog as catalog

# get the version of the package
__version__ = importlib.metadata.version("saidownscale")
