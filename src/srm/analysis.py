from functools import cached_property

import icechunk
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

from srm import catalog
from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.cache import ArtifactCache
from srm.utils import lon_to_180


def load_cached_data(s3_uri: str, branch: str = "main") -> xr.Dataset:
    """
    Load an icechunk-backed xarray Dataset from an S3 URI.

    Parameters
    ----------
    s3_uri : str
        Full S3 URI to an icechunk repository, e.g.
        ``s3://my-bucket/path/to/repo``.
    branch : str
        icechunk branch to read from. Defaults to ``"main"``.

    Returns
    -------
    xr.Dataset
        Lazily loaded dataset from the icechunk store.
    """
    parts = s3_uri.split("/")
    storage = icechunk.s3_storage(bucket=parts[2], prefix="/".join(parts[3:]), from_env=True)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch=branch)
    ds = xr.open_dataset(session.store, engine="zarr", chunks={})
    return ds


class BCSDRun:
    """
    Wrapper around a BCSD downscaling run for interactive analysis.

    Loads cached obs, historical, and scenario data and provides
    convenience methods for point extraction and plotting.
    """

    COLORS = {"obs": "#1b1e23", "scenario": "#bc85d9", "historical": "#e587b6"}

    def __init__(self, bcsd_config: BCSDConfig, options: PipelineOptions | None = None):
        self.config = bcsd_config
        self.options = options or PipelineOptions()
        self._location_cache = {}

        self._hist_member = bcsd_config.ensemble_member
        if bcsd_config.scenario is not None:
            from srm.lineage import resolve_member_lineage

            try:
                self._hist_member, _ = resolve_member_lineage(
                    bcsd_config.gcm,
                    bcsd_config.scenario,
                    bcsd_config.ensemble_member,
                    bcsd_config.variable,
                )
            except KeyError:
                pass

    def __repr__(self):
        return f"BCSDRun(gcm={self.config.gcm}, ensemble={self.config.ensemble_member}, var={self.config.variable}, scenario={self.config.scenario})"

    @cached_property
    def _cache(self) -> ArtifactCache:
        return ArtifactCache.from_config(self.config, self.options)

    @cached_property
    def obs(self) -> xr.Dataset:
        return load_cached_data(self._cache.obs_loc.store_path, branch=self._cache.branch)

    @cached_property
    def historical(self) -> xr.Dataset:
        return load_cached_data(
            self._cache.historical_loc(self._hist_member).store_path,
            branch=self._cache.branch,
        )

    @cached_property
    def scenario(self) -> xr.Dataset:
        return load_cached_data(self._cache.scenario_loc.store_path, branch=self._cache.branch)

    def get_location_data(self, lat, lon):
        """
        Extract variable timeseries at a single grid point.

        Results are cached via ``lru_cache`` for repeated access.

        Parameters
        ----------
        lat, lon : float
            Coordinates of the location (nearest grid cell is used).

        Returns
        -------
        dict
            Mapping of ``{'obs', 'historical', 'scenario'}`` to 1-D numpy arrays.
        """
        try:
            return self._location_cache[(lat, lon)]
        except KeyError:
            sel = dict(lon=lon, lat=lat, method="nearest")
            v = self.config.variable
            self._location_cache[(lat, lon)] = {
                "obs": self.obs[v].sel(**sel).to_numpy(),
                "scenario": self.scenario[v].sel(**sel).to_numpy(),
                "historical": self.historical[v].sel(**sel).to_numpy(),
            }
            return self._location_cache[(lat, lon)]

    def plot_location_cdf(self, lat, lon):
        """
        Plot empirical CDFs for obs, historical, and scenario at a location.

        Left panel shows the full CDF; right panel zooms into the upper tail.

        Parameters
        ----------
        lat, lon : float
            Coordinates of the location.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : ndarray of matplotlib.axes.Axes
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        plt_data = self.get_location_data(lat, lon)

        for k, arr in plt_data.items():
            sns.ecdfplot(data=arr, color=self.COLORS[k], ax=axes[0], label=k)
            sns.ecdfplot(data=arr, color=self.COLORS[k], ax=axes[1])

        axes[1].set_ylim(0.9, 1.01)
        axes[1].set_xlim(axes[1].get_xlim()[1] * 0.95, axes[1].get_xlim()[1])

        axes[0].legend(frameon=False)

        fig.suptitle(f"{self.config.variable} {self.config.scenario}")
        return fig, axes


def load_nasa_nex(*, dataset: str, reindex_coords_to_ERA5: bool = True):
    match dataset:
        case "ssp245":
            ds = catalog.get("NASA-NEX-SSP245").to_xarray()
        case "historical":
            ds = catalog.get("NASA-NEX-historical").to_xarray()
        case _:
            raise ValueError("dataset must take value `ssp245` or `historical`")

    from srm.utils import to_proleptic_gregorian

    ds = lon_to_180(ds)
    ds = to_proleptic_gregorian(ds)
    if reindex_coords_to_ERA5:
        obs_ds = catalog.get("ERA5").to_xarray()
        ds = ds.reindex(lat=obs_ds.lat, lon=obs_ds.lon, method="nearest", tolerance=0.15)
    return ds
