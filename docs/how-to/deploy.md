# Deploy the Pipeline

The BCSD pipeline is deployed via GitHub Actions using pre-defined config files in `configs/`. `.github/workflows/deploy.yml` holds six jobs:

| Job | Purpose | Trigger |
|---|---|---|
| `image` | Build the arm64 task container and pin a job definition to it | Every deploy |
| `qa` | Fast regional validation | Manual (`workflow_dispatch` with `environment: qa`) |
| `snapshot` | Rebuild the regional snapshot baseline, then freeze it under an icechunk tag | Automatic on GitHub release |
| `models` | Resolve which GCMs the release runs | Whenever `production` would run |
| `plan` | Write the production cost estimate to the run summary, one job per GCM | Whenever `production` would run |
| `production` | Full global run, one job per GCM | Automatic on GitHub release, or manual (`workflow_dispatch` with `environment: production`) |

`snapshot` and `production` run in parallel, so the six-hour global run does not hold up the baseline the next pull request compares against.

`production` is itself a matrix with one job per GCM, so the models run concurrently and each gets its own six-hour AWS session instead of sharing one.

## Where the tasks run

All three run jobs dispatch to AWS Batch (`--executor aws-batch`) rather than Coiled, which removes Coiled's $0.05 per CPU-hour platform fee. Coiled still backs the `validate` and `validate-output` steps, which spin up a short-lived Dask cluster rather than submitting batch tasks, so `DASK_COILED__TOKEN` is still required.

| Job | AWS Batch queue |
|---|---|
| `qa`, `snapshot` | `srm-qa` (priority 1) |
| `production` | `srm-production` (priority 10) |

Both queues are served by the one `srm-production` compute environment. Splitting them means a qa dispatch that overlaps a release cannot queue ahead of the global run, because AWS Batch schedules the higher-priority queue first.

### The image is a dependency, not a side effect

`qa`, `snapshot`, and `production` all declare `needs: image`. That job builds the container from `uv.lock` and pushes it to the `srm-downscaling` ECR repository, then registers an AWS Batch job definition revision pointing at that exact image and returns its `name:revision`. Each run job passes that value through `BCSD_BATCH_JOB_DEFINITION`.

Every build tags the image with the commit SHA. A release additionally tags it with the package version and moves `latest`. The version tag is what lets you walk backward from data to code: every output store is stamped `srm_downscaling:version` and written to an icechunk branch of that version, so the tag names the image that produced it. Only a release moves `latest`, because the job definition's fallback image is `latest` and a dispatch from a feature branch would otherwise repoint unpinned runs at its code.

The indirection is necessary because AWS Batch `containerOverrides` cannot override a job's image. Building before deploying would not be enough on its own: a concurrent build could replace the `latest` tag mid-run, so a run is bound to a specific revision instead.

### Cost approval

The `plan` job runs `bcsd run --dry-run` and writes the per-stage cost estimate into the workflow run summary. Enabling **required reviewers** on the `production` environment (Settings → Environments → production) pauses the run there, so the reviewer approves against a concrete number rather than a blank prompt. Without that setting the job is informational only, and the run proceeds unattended.

## Config structure

Configs are organized by environment under `configs/`, with one subdirectory per GCM holding one or more YAML files:

```
configs/
  qa/                    # regional (South Africa subset) end-to-end checks
    cesm2-waccm/         # e.g. cesm2-waccm-ssp245-std-southafrica.yaml, ...-g6-southafrica.yaml
    miroc-es2h/
    ukesm/
    obs-comparison/      # ERA5 vs GDEX observation-dataset comparison configs
  production/            # global runs
    cesm2-waccm/         # e.g. cesm2-waccm-ssp245-std.yaml, cesm2-waccm-g6.yaml, ...
    miroc-es2h/
    ukesm/
  snapshot/              # configs used by the snapshot regression tests
    cesm2-waccm/
```

Pointing `--config-path` at a directory (e.g. `configs/qa/`) loads every YAML beneath it. Each file is a [BCSD config](../reference/configuration.md) and supports the matrix format: list values for `gcm`/`variables`/`ensemble_members`/`scenarios`/`downscaling_methods` are expanded into one run per cartesian-product combination. For example, `ensemble_members: ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]` in a single file produces three runs without any extra files.

The key difference between environments is `environment: "qa"` vs `environment: "production"` and the presence of `subset_bounds` in QA configs. The `branch` field is intentionally left unset in all deploy configs — it defaults to the installed package version at runtime, so the cache namespace automatically tracks the released version.

## QA runs

QA runs execute all configs in `configs/qa/` against a small South Africa spatial subset. They are designed to complete quickly and validate the full pipeline end-to-end.

**To trigger a QA run:**

1. Go to **Actions → deploy → Run workflow**
2. Leave **environment** set to `qa`, which is the default
3. Optionally name a **model** to run one GCM subfolder (e.g. `miroc-es2h`), or a single config file relative to `configs/qa/` (e.g. `cesm2-waccm/cesm2-waccm-g6-southafrica.yaml`). Leave it blank to run every model.
4. Optionally enable **Force recompute** to bypass the S3 cache
5. Optionally provide a **branch** override to pin a specific cache namespace (passed to the pipeline's `--branch` flag)
6. Click **Run workflow**

The job runs three steps in order, against `configs/qa/` or the narrower path implied by **model**:

1. `bcsd validate --config-path configs/qa/` — checks input datasets for the GCMs and scenarios referenced by the configs. Exits with code 1 on any blocking failure before any compute is spent.
2. `bcsd run --config-path configs/qa/` — runs the full pipeline.
3. `bcsd validate-output --config-path configs/qa/` — checks the written output datasets.

## Production runs

Production runs execute the configs under `configs/production/{model}/` globally, as one GitHub Actions job per GCM. Each GCM owns a separate icechunk store keyed on `(gcm, obs_dataset, subset_id)`, so parallel jobs never commit to the same branch and cannot race each other. `fail-fast` is disabled, so one model failing does not cancel the others mid-run.

The matrix is an explicit list in `deploy.yml` rather than a directory listing, so adding a folder under `configs/production/` cannot silently start a global run:

| GCM | In the release matrix | Reason |
| --- | --- | --- |
| `cesm2-waccm` | Yes | |
| `miroc-es2h` | Yes | |
| `ukesm` | No | Issue #529 leaves a 0.70 K discontinuity at 2015 between the UKESM1.0 historical and the UKESM1.1 ARISE runs |

**To run every model in the matrix:**

1. Merge all intended changes to `main`
2. Create and publish a GitHub release with a SemVer tag (e.g., `v1.2.3`)
3. The `production` and `snapshot` workflow jobs fire automatically

**To run one model without cutting a release:**

1. Go to **Actions → deploy → Run workflow**
2. Set **environment** to `production`
3. Set **model** to a GCM subfolder (e.g. `miroc-es2h`), or to a single config file relative to `configs/production/`. Leave it blank to run every model in the matrix.
4. Set **branch** explicitly. At a release tag the default resolves to the release version, but off any other ref `setuptools_scm` resolves a development version such as `v0.12.0.post28`, which no documentation page or `baselines.py` entry cites.
5. Click **Run workflow**

Each job checks out the ref, installs the package at it (so the `branch` in all configs resolves to the package version), then runs:

1. `bcsd validate --config-path configs/production/{model}/`
2. `bcsd run --config-path configs/production/{model}/`
3. `bcsd validate-output --config-path configs/production/{model}/`

## Snapshot baseline runs

The `snapshot` job runs `configs/snapshot/` at the release tag and produces the regional baseline the per-pull-request check compares against. It runs three steps:

1. `bcsd run --config-path configs/snapshot/cesm2-waccm/` — writes to the branch named for the release's package version.
2. `bcsd validate-output --config-path configs/snapshot/cesm2-waccm/`
3. `bcsd release --config-path configs/snapshot/cesm2-waccm/ --tag snapshot-<release tag>` — creates an icechunk tag so the state cannot be overwritten by a later run on the same branch.

Repointing `CESM2_WACCM_SOUTH_AFRICA` in `src/srm/snapshot/baselines.py` at the new release is manual. The job prints both fields in its workflow summary, the store URI as well as the branch, because a release can move either one. See [How to Compare a Run Against the Snapshot](run-snapshot-tests.md).

## Adding a new production config

To add a variable, member, or scenario to a GCM that already runs in production:

1. Edit a file under `configs/production/{model}/` to add a value to a list (e.g. append to `ensemble_members`), or add another YAML file to that folder.
2. Set `environment: "production"` and omit `branch`.
3. Omit `subset_bounds` for a global run.
4. Open a pull request. The config is picked up on the next release, because each job loads every YAML under its own GCM folder.

To add a **new GCM**, do the same in a new `configs/production/{model}/` folder, then add that folder name to the list emitted by the `models` job in `.github/workflows/deploy.yml`. Both steps are required: that list is a deliberate allowlist, so a config folder not named there is never run. The `plan` and `production` jobs both read it, which keeps the cost estimate covering exactly the models that then run.

Two GCMs are deliberately excluded. `ukesm` is held back by issue #529, which leaves a 0.70 K discontinuity at 2015 between our UKESM1.0 historical and the UKESM1.1 ARISE runs. `miroc-es2h` is out of scope for the deliverable, and its configs are intentionally left without `downscaling_method`, so they no longer load.

## Prerequisites

The deploy workflow requires the following to be configured in the GitHub repository settings:

- **GitHub environments**: `qa` and `production` must exist (Settings → Environments)
- **`DASK_COILED__TOKEN` secret**: must be set in both the `qa` and `production` environments. Only the `validate` and `validate-output` steps need it now; the pipeline runs themselves go through AWS Batch.
- **AWS Batch resources** in `us-west-2`: the `srm-qa` and `srm-production` job queues, the `srm-production` compute environment, the `srm-downscaling` ECR repository, and the `srm-batch-job-role`, `srm-batch-execution-role`, and `srm-batch-instance-role` IAM roles. None of this is defined in the repository, so it must be recreated by hand if lost.
- **Batch permissions on the deploy role**: granted by the `SrmAwsBatchDeployPolicy` inline policy on `github-action-role`. It allows `batch:SubmitJob` on the two `srm-*` queues, `batch:RegisterJobDefinition` on `srm-downscaling*` only, the read actions the poll loop needs, and `iam:PassRole` restricted to `srm-batch-job-role` and `srm-batch-execution-role` when passed to `ecs-tasks.amazonaws.com`. ECR push comes from the role's pre-existing inline policy. `batch:TerminateJob` is deliberately not granted, since nothing calls it; add it alongside any orphan-cleanup step. Verify with:

  ```bash
  aws iam simulate-principal-policy \
    --policy-source-arn arn:aws:iam::631969445205:role/github-action-role \
    --action-names batch:SubmitJob batch:RegisterJobDefinition iam:PassRole \
    --query 'EvaluationResults[].{action:EvalActionName,decision:EvalDecision}'
  ```

  Note that a local `aws` session usually authenticates as the `github-action` IAM **user**, which is a different principal with a different policy set. A run that works locally says nothing about whether the deploy role can do the same, and the role cannot be assumed from a workstation because its trust policy admits only the GitHub OIDC provider. `simulate-principal-policy` is the way to check it.
- **AWS OIDC role**: `arn:aws:iam::631969445205:role/github-action-role` is assumed via the [setup action](../../.github/actions/setup/action.yml) — the role must trust the repository's GitHub Actions OIDC provider
