"""
Batch job runner for BCSD pipeline stages.

This module provides a CLI entry point for running individual BCSD pipeline stages
from within Coiled batch jobs. It reads configuration from the CONFIG_JSON
environment variable set by coiled.batch.run().
"""

import json
import logging
import os

import typer

from srm.bcsd_config import BCSDConfig
from srm.pipeline import BCSDPipeline

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.command()
def run_stage(
    stage: str = typer.Argument(
        ..., help="Stage name: prepare_observations, fit_historical, or transform_scenario"
    ),
):
    """
    Run a single BCSD pipeline stage with configuration from CONFIG_JSON env var.

    This is the entry point for Coiled batch jobs. The configuration is passed
    via the CONFIG_JSON environment variable (set by coiled.batch.run's
    map_over_task_var_dicts parameter).
    """
    try:
        # Read config from environment variable
        config_json = os.environ.get("CONFIG_JSON")
        if not config_json:
            raise ValueError("CONFIG_JSON environment variable not set")

        # Parse config from JSON
        config_dict = json.loads(config_json)
        config = BCSDConfig(**config_dict)

        logger.info(f"Running {stage} for {config.run_id}")

        # Create pipeline and run stage
        pipeline = BCSDPipeline(config)

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
