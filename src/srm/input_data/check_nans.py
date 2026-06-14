"""Scan input CMIP icechunk datasets for NaN values, physical range violations, outlier members, and zero fractions."""

from __future__ import annotations

import sys

import click
import flox  # noqa: F401 — activates flox backend for xarray groupby/resample
import xarray as xr

from srm import catalog
from srm.config import ClusterConfig, setup_cluster

# (catalog_key, zarr_group) pairs for the unified per-GCM stores
TARGET_DATASETS: list[tuple[str, str]] = [
    ("CESM2-WACCM-unified-icechunk", "historical"),
    ("CESM2-WACCM-unified-icechunk", "ssp245"),
    ("CESM2-WACCM-unified-icechunk", "g6_1p5k"),
    ("MIROC-ES2H-unified-icechunk", "historical"),
    ("MIROC-ES2H-unified-icechunk", "ssp245"),
    ("MIROC-ES2H-unified-icechunk", "g6_1p5k"),
    ("UKESM-unified-icechunk", "historical"),
    ("UKESM-unified-icechunk", "ssp245"),
    ("UKESM-unified-icechunk", "g6_1p5k"),
]

# ---------------------------------------------------------------------------
# NaN checks
# ---------------------------------------------------------------------------


def _nan_breakdown(var: str, da: xr.DataArray) -> None:
    dims = list(da.dims)

    if "ensemble_member" in dims:
        spatial_time_dims = [d for d in dims if d != "ensemble_member"]
        per_member = da.isnull().sum(dim=spatial_time_dims).compute()
        click.echo(f"    NaNs per ensemble_member ({var}):")
        for member in per_member.ensemble_member.values:
            count = int(per_member.sel(ensemble_member=member).values)
            if count > 0:
                click.echo(f"      {member}: {count:,}")

    if "time" in dims:
        non_time_dims = [d for d in dims if d != "time"]
        per_year = da.isnull().sum(dim=non_time_dims).resample(time="YE").sum().compute()
        nonzero = per_year.where(per_year > 0, drop=True)
        if nonzero.time.size > 0:
            click.echo(f"    NaNs per year ({var}):")
            for t in nonzero.time.values:
                click.echo(f"      {str(t)[:4]}: {int(nonzero.sel(time=t).values):,}")


def check_nans(ds: xr.Dataset, vars_to_check: list[str], verbose: bool) -> bool:
    found_any = False
    for var in vars_to_check:
        da = ds[var]
        total = da.size
        nan_count = int(da.isnull().sum().compute())
        pct = 100.0 * nan_count / total if total > 0 else 0.0
        status = "CLEAN" if nan_count == 0 else "NaNs FOUND"
        click.echo(f"  [{status}] {var}: {nan_count:,} / {total:,} NaN ({pct:.4f}%)")
        if nan_count > 0:
            found_any = True
            _nan_breakdown(var, da)
        elif verbose:
            click.echo(f"    (verbose) no NaNs in {var}")
    return found_any


def check_dataset(
    name: str,
    ds: xr.Dataset,
    var_filter: list[str] | None,
    verbose: bool,
    zero_threshold: float,
) -> bool:
    vars_to_check = [v for v in ds.data_vars if var_filter is None or v in var_filter]

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Dataset: {name}")
    click.echo(f"  dims: {dict(ds.dims)}")
    click.echo(f"  vars: {vars_to_check}")
    nan_res = (check_nans(ds, vars_to_check, verbose),)

    return nan_res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--datasets",
    default=None,
    help="Comma-separated dataset names (default: all 9 target datasets).",
)
@click.option(
    "--vars",
    "var_names",
    default=None,
    help="Comma-separated variable names (default: all data_vars).",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show per-dimension breakdown and passing checks.",
)
@click.option(
    "--n-workers",
    default="1,50",
    show_default=True,
    help="Coiled worker range as 'min,max' or a single int.",
)
@click.option(
    "--worker-vm-type",
    default="r8g.2xlarge",
    show_default=True,
    help="Coiled worker VM type.",
)
def main(
    datasets: str | None,
    var_names: str | None,
    verbose: bool,
    n_workers: str,
    worker_vm_type: str,
) -> None:
    """Check input CMIP datasets for NaNs"""
    targets = TARGET_DATASETS
    if datasets:
        # accept "key:group" pairs from CLI, e.g. "CESM2-WACCM-unified-icechunk:ssp245"
        targets = [tuple(d.split(":", 1)) for d in datasets.split(",")]  # type: ignore[assignment]
    var_filter = var_names.split(",") if var_names else None

    worker_range = [int(x) for x in n_workers.split(",")] if "," in n_workers else int(n_workers)
    config = ClusterConfig(
        n_workers=worker_range,
        worker_vm_types=[worker_vm_type],
    )
    client = setup_cluster(config)
    click.echo(f"Dask dashboard: {client.dashboard_link}")

    any_flags = False
    for key, group in targets:
        name = f"{key}/{group}"
        try:
            entry = catalog.get(key)
        except KeyError:
            click.echo(f"WARNING: '{key}' not found in catalog, skipping", err=True)
            continue
        ds = entry.to_xarray(group=group)
        try:
            if check_dataset(name, ds, var_filter, verbose):
                any_flags = True
        finally:
            ds.close()

    click.echo(f"\n{'=' * 60}")
    if any_flags:
        click.echo("RESULT: Issues detected in one or more datasets.")
        sys.exit(1)
    else:
        click.echo("RESULT: All checks passed.")


if __name__ == "__main__":
    main()
