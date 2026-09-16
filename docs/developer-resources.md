# Developer resources

If you want to run the downscaling pipeline yourself or change how it works, start here. This
repository reflects the code and infrastructure we used for this project. We don't maintain it as a
general-purpose downscaling tool, so treat it as a reference to borrow patterns from rather than
something to run as-is.

## Infrastructure

The code assumes the cloud setup we ran it on. The table below lists each assumption, its default,
and the setting or file you change to point the pipeline somewhere else.

| Assumption | Default | Where to change it |
| --- | --- | --- |
| Compute runs in the same AWS region as the data | `us-west-2` | `batch_region` |
| Input data is in Icechunk stores on S3 | `s3://carbonplan-srm/input/processed/` | Dataset catalog in `src/saidownscale/datasets.py`; see the [input data catalog](input-data.md) |
| Cached intermediates and outputs are on S3 | `s3://carbonplan-srm/scratch/cache/` and `s3://carbonplan-srm/scratch/output/` | `scratch_dir` and `output_dir` |
| Each package version writes to its own Icechunk branch | A branch named after the installed package version | `branch`, or the `SAIDOWNSCALE_BRANCH` environment variable |
| Deploys run remote tasks on AWS Batch | The code defaults to `coiled`, and deploys pass `--executor aws-batch` | `executor`, `batch_job_queue` (default `srm-production`), and `batch_job_definition` (default `srm-downscaling`) |
| Tasks run from a container image in Amazon ECR | `Dockerfile`, built by `.github/workflows/build-image.yml` into the `srm-downscaling` repository | `.github/workflows/build-image.yml` |
| Deploys run from GitHub Actions and assume an AWS role through OpenID Connect (OIDC) | `.github/workflows/deploy.yml`, with the IAM policies it needs recorded in `infra/iam/` | [Deploy the pipeline](how-to/deploy.md) |

Settings in the last column that aren't file paths are `PipelineOptions` fields. You can set them
in a config file or with `SAIDOWNSCALE_*` environment variables, as the
[configuration reference](reference/configuration.md) describes.

## How-to guides

We wrote step-by-step guides for the tasks that come up when you run or change the pipeline. Find
what you want to do in the table below, then follow the link.

| What do you want to do? | Guide |
| --- | --- |
| Set up a development environment, run the linters, and run the tests | [Contributing](contributing.md) |
| Install the pipeline and run it, locally or in batch | [Run the pipeline](how-to/run-pipeline.md) |
| Look up a CLI command or a config field | [CLI reference](reference/cli.md), [configuration reference](reference/configuration.md) |
| Inspect, resume, or clear cached artifacts | [Manage the cache](how-to/manage-cache.md) |
| Run a QA or production deploy | [Deploy the pipeline](how-to/deploy.md) |
| Regenerate the processed input data, or open it from the catalog | [Regenerate input data](how-to/regenerate-input-data.md), [input data catalog](input-data.md) |
| Check a modeling change against the snapshot before merging | [Compare a run against the snapshot](how-to/run-snapshot-tests.md) |
| Browse the Python API | [API reference](reference/api/index.md) |

### Notebooks

These notebooks walk through parts of the pipeline interactively. Most of them read data from S3,
so run them where you have access to the project's buckets.

| Notebook | Use it to |
| --- | --- |
| [Pipeline demo](how-to/demo-new-pipeline.ipynb) | Run a small South Africa example end to end and plot the results |
| [Pipeline stage-by-stage debugger](how-to/pipeline-stage-debugger.ipynb) | Inspect the intermediate output of each pipeline step, one cell at a time |
| [Validate input data stores](how-to/validate-input-data-stores.ipynb) | Sanity-check the processed input stores that the ETL pipelines write |
| [Ensemble member lineage](how-to/ensemble-member-lineage.ipynb) | See which simulations each downscaled run is assembled from |
