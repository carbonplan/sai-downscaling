"""
Papermill runner for QA/QC notebooks.

Executes QA/QC notebooks against a production output store at a given icechunk
``branch`` (injected as a papermill parameter), keeping their rendered cells,
then uploads the executed ``.ipynb`` to S3 so an off-VM caller (CI) can retrieve
them.

Coiled batch VMs get ``srm`` as a site-packages wheel but *not* the repo's
``notebooks/`` tree, so a notebook absent locally is seeded from S3 by the
dispatcher (see :func:`seed_sources`) and downloaded here. Layout under
``s3://carbonplan-scratch/srm/qaqc-notebooks/<branch>/<sha>/``:

    src/<repo-relative-nb>   source notebooks uploaded by the dispatcher
    out/<repo-relative-nb>   executed notebooks uploaded after render

Mirrors :mod:`srm.batch_runner`: a typer entrypoint reads ``QAQC_BRANCH`` /
``QAQC_NOTEBOOKS`` from the environment for Coiled batch jobs.
"""

import logging
import os
import subprocess
import tempfile
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


def _store():
    """obstore handle for the scratch bucket using the active AWS session."""
    aws = get_aws_creds()
    region = aws.pop("region")
    return from_url(f"s3://{SCRATCH_BUCKET}", region=region, **aws)


def _src_key(nb: str, branch: str, sha: str) -> str:
    return f"{S3_PREFIX}/{branch}/{sha}/src/{nb}"


def _out_key(nb: str, branch: str, sha: str) -> str:
    return f"{S3_PREFIX}/{branch}/{sha}/out/{nb}"


def seed_sources(notebooks: list[str], branch: str, sha: str) -> None:
    """Upload source notebooks (from the local repo) to the S3 ``src/`` prefix.

    Called by the dispatcher, which has the repo checked out, so the Coiled VM
    can fetch notebooks it does not have locally.
    """
    store = _store()
    for nb in notebooks:
        path = REPO_ROOT / nb
        obs.put(store, _src_key(nb, branch, sha), path.read_bytes())
        logger.info(f"Seeded source {nb} -> s3://{SCRATCH_BUCKET}/{_src_key(nb, branch, sha)}")


def _resolve_source(nb: str, branch: str, sha: str, workdir: Path) -> Path:
    """Return a local path to the source notebook.

    Uses the repo copy when present (local runs); otherwise downloads the
    S3-seeded source into ``workdir`` (Coiled VM runs).
    """
    local = REPO_ROOT / nb
    if local.exists():
        return local
    dest = workdir / Path(nb).name
    result = obs.get(_store(), _src_key(nb, branch, sha))
    dest.write_bytes(bytes(result.bytes()))
    logger.info(f"Downloaded seeded source {nb} from S3")
    return dest


def run_notebooks(
    branch: str,
    notebooks: list[str] | None = None,
    upload: bool = True,
) -> list[tuple[str, str | None]]:
    """Execute each notebook at ``branch``; optionally upload the executed copy.

    Returns ``(notebook, s3_uri | None)`` pairs.
    """
    notebooks = notebooks or PILOT_NOTEBOOKS
    sha = _git_sha()
    store = _store() if upload else None
    results: list[tuple[str, str | None]] = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for nb in notebooks:
            src = _resolve_source(nb, branch, sha, workdir)
            out = workdir / f"executed_{Path(nb).name}"
            logger.info(f"Executing {nb} at branch {branch}")
            pm.execute_notebook(str(src), str(out), parameters={"branch": branch})
            uri = None
            if upload:
                key = _out_key(nb, branch, sha)
                obs.put(store, key, out.read_bytes())
                uri = f"s3://{SCRATCH_BUCKET}/{key}"
                logger.info(f"Uploaded executed {nb} -> {uri}")
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
