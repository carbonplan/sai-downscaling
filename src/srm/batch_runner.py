"""
Batch job runner for individual BCSD pipeline stages on remote task VMs.

Reads configuration from the environment and invokes the requested stage via
:class:`~srm.pipeline.BCSDPipeline`. This is the entry point for all distributed
remote execution, on Coiled Batch and on AWS Batch alike; see
:func:`_load_config_dict` for how each delivers a task's config.
"""

import json
import logging
import os

import typer

from srm.batch_manifest import read_manifest_entry
from srm.bcsd_config import BCSDConfig, PipelineOptions
from srm.pipeline import BCSDPipeline

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_config_dict() -> dict:
    """
    Obtain this task's configuration payload from the environment.

    Two delivery mechanisms are supported. Coiled Batch sets a distinct ``CONFIG_JSON``
    per task through ``map_over_task_var_dicts``. AWS Batch array jobs cannot vary the
    environment per child, so the orchestrator writes one manifest and each child reads
    its own entry by ``AWS_BATCH_JOB_ARRAY_INDEX``.

    ``CONFIG_JSON`` takes precedence so that a single-task AWS Batch job, which is
    submitted as a plain job rather than an array, works without a manifest.

    Returns
    -------
    dict
        Payload holding ``BCSDConfig`` fields plus an ``options`` sub-dict.

    Raises
    ------
    ValueError
        If neither delivery mechanism is fully configured.
    """
    config_json = os.environ.get("CONFIG_JSON")
    if config_json:
        return json.loads(config_json)

    manifest_uri = os.environ.get("CONFIG_MANIFEST_URI")
    if not manifest_uri:
        raise ValueError(
            "No task config found: set CONFIG_JSON, or CONFIG_MANIFEST_URI together with "
            "AWS_BATCH_JOB_ARRAY_INDEX"
        )

    index = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
    if index is None:
        raise ValueError(
            "CONFIG_MANIFEST_URI is set but AWS_BATCH_JOB_ARRAY_INDEX is not; "
            "array children must know which entry to read"
        )
    return read_manifest_entry(manifest_uri, int(index))


@app.command()
def run_stage(
    stage: str = typer.Argument(
        ..., help="Stage name: prepare_observations, fit_historical, or transform_scenario"
    ),
):
    """
    Run a single BCSD pipeline stage with configuration from the environment.

    This is the entry point for both remote executors. See :func:`_load_config_dict`
    for the two ways a task's configuration reaches it.
    """
    try:
        # Read config from environment (CONFIG_JSON, or manifest plus array index)
        config_dict = _load_config_dict()
        options_dict = config_dict.pop("options", {})
        config = BCSDConfig(**config_dict)
        options = PipelineOptions(**options_dict)

        logger.info(f"Running {stage} for {config.run_id}")

        # Create pipeline and run stage
        pipeline = BCSDPipeline(config, options)

        if stage == "prepare_observations":
            result_path = pipeline.prepare_observations()
        elif stage == "fit_historical":
            result_path = pipeline.fit_historical()
        elif stage == "transform_scenario":
            result_path = pipeline.transform_scenario()
        else:
            raise ValueError(f"Unknown stage: {stage}")

        logger.info(f"✓ Completed {stage}: {result_path}")

        # Print result path for capture
        print(result_path)
        return result_path

    except Exception as e:
        logger.error(f"✗ Failed {stage}: {e}")
        raise


if __name__ == "__main__":
    app()
