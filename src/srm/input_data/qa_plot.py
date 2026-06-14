"""Plot first and last timestep for each scenario/variable/member combo from an S3 prefix."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import boto3
import click
import icechunk
import matplotlib.pyplot as plt
import xarray as xr

# pattern: .../output/qa/<version>/<scenario>/<gcm>/<variable>/<member>/.../<scenario>.icechunk
_URL_RE = re.compile(r"/([^/]+)/([^/]+)/([^/]+)/([^/]+)/[^/]+/[^/]+/[^/]+\.icechunk$")


def list_icechunk_urls(s3_prefix: str) -> list[str]:
    """Return unique icechunk store URLs under s3_prefix."""
    prefix_no_scheme = s3_prefix.removeprefix("s3://")
    bucket, _, key_prefix = prefix_no_scheme.partition("/")

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    store_prefixes: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # match the store root: everything up to and including *.icechunk
            m = re.search(r"^(.*\.icechunk)/", key)
            if m:
                store_prefixes.add(m.group(1))

    return sorted(f"s3://{bucket}/{p}" for p in store_prefixes)


def parse_url(url: str) -> tuple[str, str, str, str]:
    """Return (scenario, gcm, variable, member) from an icechunk URL."""
    m = _URL_RE.search(url)
    if not m:
        raise ValueError(f"Cannot parse URL: {url}")
    scenario, gcm, variable, member = m.groups()
    return scenario, gcm, variable, member


def open_icechunk(path: str) -> xr.Dataset:
    path_no_scheme = path.removeprefix("s3://")
    bucket, _, prefix = path_no_scheme.partition("/")
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    return xr.open_dataset(session.store, engine="zarr", consolidated=False, chunks={})


def upload_plot(local_path: Path, output_bucket: str, filename: str) -> None:
    no_scheme = output_bucket.removeprefix("s3://").rstrip("/")
    bucket, _, prefix = no_scheme.partition("/")
    key = f"{prefix}/{filename}" if prefix else filename
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    print(f"  uploaded s3://{bucket}/{key}")


def make_plots(urls: list[str], out_dir: Path, output_bucket: str | None) -> None:
    groups: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for url in urls:
        try:
            scenario, gcm, variable, member = parse_url(url)
        except ValueError as e:
            print(f"skip (unparseable): {e}")
            continue
        groups[(scenario, gcm, variable)].append((member, url))

    out_dir.mkdir(parents=True, exist_ok=True)

    for (scenario, gcm, variable), members in sorted(groups.items()):
        members = sorted(members, key=lambda x: x[0])
        n = len(members)
        fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n), squeeze=False)
        fig.suptitle(f"{gcm} | {scenario} | {variable}", fontsize=14, y=1.01)

        for row, (member, url) in enumerate(members):
            title = f"{gcm} | {scenario} | {member} | {variable}"
            try:
                ds = open_icechunk(url)
                da = ds[variable]
                da.isel(time=0).plot(ax=axes[row, 0])
                axes[row, 0].set_title(f"{title}\ntime=first")
                da.isel(time=-1).plot(ax=axes[row, 1])
                axes[row, 1].set_title(f"{title}\ntime=last")
                da.sel(lat=-28, lon=24, method="nearest").plot(ax=axes[row, 2])
                axes[row, 2].set_title(f"{title}\nlat=-28 lon=24")
                ds.close()
            except Exception as e:
                axes[row, 0].set_title(f"{member} ERROR: {e}", fontsize=7)
                axes[row, 1].set_visible(False)
                axes[row, 2].set_visible(False)

        fig.tight_layout()
        fname = out_dir / f"{gcm}_{scenario}_{variable}.png"
        fig.savefig(fname, dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fname}")

        if output_bucket:
            upload_plot(fname, output_bucket, fname.name)


@click.command()
@click.argument("s3_prefix")
@click.option(
    "--output-dir", default="qa_plots", show_default=True, help="Local directory for plots."
)
@click.option(
    "--output-bucket",
    default=None,
    help="S3 prefix to upload plots to, e.g. s3://carbonplan-scratch/srm/qa_plots.",
)
def cli(s3_prefix: str, output_dir: str, output_bucket: str | None) -> None:
    """Generate QA plots for all icechunk stores under S3_PREFIX."""
    print(f"listing icechunk stores under {s3_prefix} ...")
    urls = list_icechunk_urls(s3_prefix)
    print(f"found {len(urls)} stores")
    make_plots(urls, Path(output_dir), output_bucket)


if __name__ == "__main__":
    cli()
