# Deploy the Pipeline

The BCSD pipeline is deployed via GitHub Actions using pre-defined config files in `configs/`. There are two environments:

| Environment | Purpose | Trigger |
|---|---|---|
| `qa` | Fast regional validation | Manual (`workflow_dispatch`) |
| `production` | Full global run | Automatic on GitHub release |

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

Pointing `--config-path` at a directory (e.g. `configs/qa/`) loads every YAML beneath it. Each file is a [BCSD config](../reference/configuration.md) and supports the matrix format — list values for `gcm`/`variables`/`ensemble_members`/`scenarios` are expanded into one run per cartesian-product combination. For example, `ensemble_members: ["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"]` in a single file produces three runs without any extra files.

The key difference between environments is `environment: "qa"` vs `environment: "production"` and the presence of `subset_bounds` in QA configs. The `branch` field is intentionally left unset in all deploy configs — it defaults to the installed package version at runtime, so the cache namespace automatically tracks the released version.

## QA runs

QA runs execute all configs in `configs/qa/` against a small South Africa spatial subset. They are designed to complete quickly and validate the full pipeline end-to-end.

**To trigger a QA run:**

1. Go to **Actions → deploy → Run workflow**
2. Optionally enable **Force recompute** to bypass the S3 cache
3. Optionally provide a **branch** override to pin a specific cache namespace (passed to the pipeline's `--branch` flag)
4. Click **Run workflow**

The job runs two steps in order:
1. `bcsd validate --config-path configs/qa/` — checks input datasets for the GCMs and scenarios referenced by the configs. Exits with code 1 on any blocking failure before Coiled compute is spent.
2. `bcsd run --config-path configs/qa/` — runs the full pipeline.

## Production runs

Production runs execute all configs in `configs/production/` globally. They trigger automatically when a GitHub release is published.

**To trigger a production run:**

1. Merge all intended changes to `main`
2. Create and publish a GitHub release with a SemVer tag (e.g., `v1.2.3`)
3. The `production` workflow job fires automatically

The job checks out the release tag, installs the package at that tag (so the `branch` in all configs resolves to the release's package version), then runs:
1. `bcsd validate --config-path configs/production/`
2. `bcsd run --config-path configs/production/`

## Adding a new production config

To add a new GCM, variable, member, or scenario to future production runs:

1. Edit an existing file under `configs/production/` to add a value to a list (e.g. append to `ensemble_members`), or create a new YAML file for a new GCM.
2. Set `environment: "production"` and omit `branch`.
3. Omit `subset_bounds` for a global run.
4. Open a PR — the config will be picked up automatically on the next release.

## Prerequisites

The deploy workflow requires the following to be configured in the GitHub repository settings:

- **GitHub environments**: `qa` and `production` must exist (Settings → Environments)
- **`DASK_COILED__TOKEN` secret**: must be set in both the `qa` and `production` environments
- **AWS OIDC role**: `arn:aws:iam::631969445205:role/github-action-role` is assumed via the [setup action](../../.github/actions/setup/action.yml) — the role must trust the repository's GitHub Actions OIDC provider
