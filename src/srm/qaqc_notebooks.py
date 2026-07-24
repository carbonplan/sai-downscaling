"""
Papermill runner for QA/QC notebooks.

Executes QA/QC notebooks in place against a production output store at a given
icechunk ``branch`` (injected as a papermill parameter), keeping their rendered
cells, then uploads the executed ``.ipynb`` to S3 so an off-VM caller (CI) can
retrieve them. Mirrors :mod:`srm.batch_runner`: a typer entrypoint reads
``QAQC_BRANCH`` / ``QAQC_NOTEBOOKS`` from the environment for Coiled batch jobs.
"""

import logging
import os
import subprocess
from pathlib import Path

import obstore as obs
import papermill as pm
import typer
from obstore.store import from_url

from srm.input_data.etl_utils import get_aws_creds

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Repo-relative pilot notebooks.
PILOT_NOTEBOOKS: list[str] = [
    "notebooks/QA_QC/qaqc_pilot_dummy.ipynb",
]

SCRATCH_BUCKET = "carbonplan-scratch"
S3_PREFIX = "srm/qaqc-notebooks"


def _git_sha() -> str:
    """Resolve the S3-prefix SHA.

    Prefers ``QAQC_SHA`` then ``GITHUB_SHA`` so the dispatcher (CI runner) and the
    Coiled VM agree on the upload/download location, since the VM may not be a git
    checkout. Falls back to the local short SHA, or ``"nogit"``.
    """
    sha = os.environ.get("QAQC_SHA") or os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "nogit"


def _upload(local: Path, nb: str, branch: str, sha: str) -> str:
    """Upload an executed notebook to S3 and return its ``s3://`` URI.

    Keyed by the repo-relative notebook path so a caller can recursive-download
    the prefix straight back onto the working tree.
    """
    aws = get_aws_creds()
    region = aws.pop("region")
    store = from_url(f"s3://{SCRATCH_BUCKET}", region=region, **aws)
    key = f"{S3_PREFIX}/{branch}/{sha}/{nb}"
    obs.put(store, key, local.read_bytes())
    return f"s3://{SCRATCH_BUCKET}/{key}"


def run_notebooks(
    branch: str,
    notebooks: list[str] | None = None,
    upload: bool = True,
) -> list[tuple[str, str | None]]:
    """Execute each notebook in place at ``branch``; optionally upload to S3.

    Returns ``(notebook, s3_uri | None)`` pairs.
    """
    notebooks = notebooks or PILOT_NOTEBOOKS
    sha = _git_sha()
    results: list[tuple[str, str | None]] = []
    for nb in notebooks:
        path = REPO_ROOT / nb
        logger.info(f"Executing {nb} at branch {branch}")
        pm.execute_notebook(str(path), str(path), parameters={"branch": branch})
        uri = _upload(path, nb, branch, sha) if upload else None
        if uri:
            logger.info(f"Uploaded {nb} -> {uri}")
        results.append((nb, uri))
    return results


@app.command()
def run(
    branch: str = typer.Option(None, envvar="QAQC_BRANCH", help="Icechunk branch/tag."),
    notebooks: str = typer.Option(
        None,
        envvar="QAQC_NOTEBOOKS",
        help="Comma-separated repo-relative notebook paths (default: pilots).",
    ),
    upload: bool = typer.Option(True, help="Upload executed notebooks to S3."),
):
    """Entry point for Coiled batch jobs (reads QAQC_BRANCH / QAQC_NOTEBOOKS)."""
    if not branch:
        raise ValueError("branch is required (set --branch or QAQC_BRANCH)")
    nb_list = [n.strip() for n in notebooks.split(",") if n.strip()] if notebooks else None
    for nb, uri in run_notebooks(branch, nb_list, upload=upload):
        print(f"{nb} -> {uri}")


if __name__ == "__main__":
    app()
