# BCSD Pipeline Documentation

The BCSD pipeline provides a command-line interface for running SRM downscaling workflows with automatic caching, resumability, and distributed execution via Coiled.

## How-To Guides

Task-oriented guides for common workflows:

- [Run the pipeline](how-to/run-pipeline.md) — installation, quick start, batch processing, local execution
- [Manage the cache](how-to/manage-cache.md) — resumability, force recompute, cache inspection and clearing
- [Compare outputs across code versions](how-to/compare-outputs-across-versions.md) — validate pipeline changes on a test region
- [Run a multi-model ensemble](how-to/run-multi-model-ensemble.md) — run across multiple GCMs, variables, members, and scenarios

## Reference

Exhaustive reference material:

- [CLI reference](reference/cli.md) — all commands, options, and usage examples
- [Configuration reference](reference/configuration.md) — all config fields, environment variable overrides, and validation rules

## Explanation

Background and design rationale:

- [Pipeline architecture](explanation/pipeline-architecture.md) — three-stage design, caching strategy, Coiled execution, code organization
