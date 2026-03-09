# flake8: noqa
import importlib.metadata

from srm.datasets import catalog as catalog

# get the version of the package
__version__ = importlib.metadata.version("srm")
