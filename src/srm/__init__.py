# flake8: noqa
import importlib.metadata

# get the version of the package
__version__ = importlib.metadata.version("srm")
from srm.datasets import catalog as catalog
