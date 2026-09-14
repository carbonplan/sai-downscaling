"""
Runs srm.qa_flags.run_step2 as a standalone script, so it can be submitted as a
Coiled batch job and left to finish on its own -- no notebook, no one watching it --
instead of running the same steps interactively in
notebooks/QA_QC/qa_flags/flag_step2_generate-intermediate-qa-flags_inprogress.ipynb.

Meant to be submitted as a Coiled batch job so it runs without a notebook open and its
Dask cluster shuts down automatically when done, e.g. from a Python shell or another
script:

    import coiled
    coiled.batch.run(
        command=["python", "-m", "srm.qa_flags_runner"],
        name="qa-flags-step2",
        vm_type=["c8g.xlarge"],  # small driver VM; the actual flag computation runs on
                                  # the multi-worker Dask cluster this script creates
        region="us-west-2",
        forward_aws_credentials=False,
        spot_policy="on-demand",  # keep the driver itself stable; the Dask workers it
                                   # spins up below still use spot_with_fallback
        tag={"Project": "SRM"},
    )

or run directly on any machine with Coiled credentials configured:

    >> uv run python src/srm/qa_flags_runner.py

Plots
-----
Flag-map figures are uploaded straight to S3 by qa_flags.py itself (plot_flags's
bucket/s3_key params, via save_figure_to_s3) whenever PLOT_FLAG_MAPS and SAVE_PLOTS are
both True -- this script doesn't do anything plot-specific beyond passing those two
flags through to run_step2.

Logging
-------
Both this script's own progress and qa_flags.py's internal logger.info() calls
(qa_flags.py logs via logging.getLogger(__name__)) go to the same two places: the console (so everything is
still visible live via `coiled.batch.logs(job_id)` / the Coiled dashboard while the job
is running) and a local file, via the two handlers set up below. That local file is uploaded to S3 once the run finishes,
successfully or not, so there's a permanent record afterward at LOG_S3_KEY.
"""

import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3
import coiled
import matplotlib
from frisky import hijack

os.environ.setdefault("FRISKY_SUMMARY", "off")

RUN_ID = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
LOG_PATH = Path(f"run_step2_{RUN_ID}.log")

# Both this module's own logger and qa_flags.py's (via propagation to the root logger)
# go to these two handlers -- console and a local file, no redirection needed.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
)
logger = logging.getLogger(__name__)

matplotlib.use("Agg")  # headless backend; must happen before any `import matplotlib.pyplot`,
# including the one inside srm.qa_flags -- hence importing that only after this line

from srm.qa_flags import run_step2  # noqa: E402 -- must come after matplotlib.use("Agg") above

# --- Run parameters, matching flag_step2_generate-intermediate-qa-flags_inprogress.ipynb ---
# Update these per experiment before submitting.
VARIABLES = ["tas", "tasmax", "tasmin", "pr", "rsds"]
GCMS = ["CESM2-WACCM6", "UKESM1-1-LL"]

METHODS = ["bcsd", "qdmsd"]

BRANCH = "v1.0.0"
ROOT_DIR = "s3://us-west-2.opendata.source.coop/carbonplan/srm-downscaling/output/production/"
STORE_SUBSET_ID = "global"

BUCKET = "carbonplan-srm"
PREFIX = "scratch/output/qa-intermediate-flags-v1.0.0-qa-run2"

PLOT_FLAG_MAPS = True  # compute flag-map figures
SAVE_PLOTS = True  # and upload them to S3 under {BUCKET}/{PREFIX}/_plots/ (see qa_flags.plot_flags)
VERBOSE = True

LOG_S3_KEY = f"{PREFIX}/_logs/run_step2_{RUN_ID}.log"

# --- Cluster config, same tuning as the notebook's "Set up cluster" cell ---
CLUSTER_KWARGS = dict(
    name="srm-qaqc-flags-step2-job",
    region="us-west-2",
    n_workers=12,
    worker_vm_types=["c9g.2xlarge"],
    scheduler_vm_types="c8g.xlarge",
    spot_policy="spot_with_fallback",  # sometimes switch to "on-demand" if it crashes
    use_best_zone=True,
    tags={"Project": "SRM"},
    environ={"ZARR_ASYNC__CONCURRENCY": "128"},
    idle_timeout="30 minutes",
)


def main() -> None:
    t_start = time.time()
    try:
        logger.info("Starting run_step2 job: gcms=%s, methods=%s, branch=%r", GCMS, METHODS, BRANCH)

        # The `with` block guarantees the cluster shuts down when run_step2 returns --
        # or if it raises, so a failed run doesn't leave workers billing in the background.
        with coiled.Cluster(**CLUSTER_KWARGS) as cluster:
            hijack(cluster.get_client())

            run_step2(
                variables=VARIABLES,
                gcms=GCMS,
                methods=METHODS,
                branch=BRANCH,
                root_dir=ROOT_DIR,
                store_subset_id=STORE_SUBSET_ID,
                bucket=BUCKET,
                prefix=PREFIX,
                plot_flag_maps=PLOT_FLAG_MAPS,
                save_plots=SAVE_PLOTS,
                verbose=VERBOSE,
                mode="both",
            )

        logger.info("run_step2 job finished in %.1fs", time.time() - t_start)
    except Exception:
        # logger.exception logs this message plus the full traceback through the same
        # handlers as everything else (console and LOG_PATH) -- without this, an uncaught
        # exception would only ever print to stderr, bypassing the log file entirely.
        logger.exception("run_step2 job failed")
        raise
    finally:
        boto3.client("s3").upload_file(str(LOG_PATH), BUCKET, LOG_S3_KEY)
        logger.info("Uploaded log to s3://%s/%s", BUCKET, LOG_S3_KEY)


if __name__ == "__main__":
    main()
